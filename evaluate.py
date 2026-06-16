import argparse
import csv
import os

import torch

from config import (
    BEST_MODEL_SAVE_PATH,
    METRICS_DIR,
    RANDOM_SEED,
    SAMPLES_DIR,
)
from data_utils import build_dataloaders
from generate import (
    build_rhyme_vocab_map,
    format_poem_for_display,
    generate_acrostic,
    generate_from_first_line,
    load_model_from_checkpoint,
)
from metrics import batch_diversity, batch_rhyme_rate
from trainer import evaluate_one_epoch
from utils import configure_torch_runtime, ensure_dir, get_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="诗词生成采样参数横评")
    parser.add_argument("--checkpoint", type=str, default=BEST_MODEL_SAVE_PATH, help="checkpoint 路径")
    parser.add_argument("--num_samples", type=int, default=30, help="每种条件生成样例数")
    parser.add_argument("--first_line", type=str, default="春风又绿江南岸", help="首句续写输入")
    parser.add_argument("--acrostic", type=str, default="春江花月", help="藏头诗输入")
    return parser.parse_args()


def run_one_batch(model, stoi, itos, device, mode, text, n, temp, topk, rhyme_map):
    poems = []
    for _ in range(n):
        if mode == "first_line":
            poem = generate_from_first_line(
                model=model, first_line=text, stoi=stoi, itos=itos,
                device=device, temperature=temp, top_k=topk, rhyme_map=rhyme_map,
            )
        else:
            poem = generate_acrostic(
                model=model, head_chars=text, stoi=stoi, itos=itos,
                device=device, temperature=temp, top_k=topk, rhyme_map=rhyme_map,
            )
        poems.append(poem)
    return poems


def evaluate_one_config(model, stoi_ckpt, itos_ckpt, device, rhyme_map,
                        temperature, top_k, first_line, acrostic, num_samples):
    fl_free = run_one_batch(model, stoi_ckpt, itos_ckpt, device,
                            "first_line", first_line, num_samples, temperature, top_k, None)
    ac_free = run_one_batch(model, stoi_ckpt, itos_ckpt, device,
                            "acrostic", acrostic, num_samples, temperature, top_k, None)

    fl_guided = run_one_batch(model, stoi_ckpt, itos_ckpt, device,
                              "first_line", first_line, num_samples, temperature, top_k, rhyme_map)
    ac_guided = run_one_batch(model, stoi_ckpt, itos_ckpt, device,
                              "acrostic", acrostic, num_samples, temperature, top_k, rhyme_map)

    fl_div = batch_diversity(fl_free)
    ac_div = batch_diversity(ac_free)

    fl_rhyme_free = batch_rhyme_rate(fl_free)
    ac_rhyme_free = batch_rhyme_rate(ac_free)
    fl_rhyme_g = batch_rhyme_rate(fl_guided)
    ac_rhyme_g = batch_rhyme_rate(ac_guided)

    fl_poems_display = [format_poem_for_display(p) for p in fl_free]
    ac_poems_display = [format_poem_for_display(p) for p in ac_free]

    return {
        "temperature": temperature,
        "top_k": top_k,
        "fl_dens": fl_div["inter_dens"],
        "fl_rpt": fl_div["repeat_pct"],
        "fl_uniq": fl_div["uniq_rate"],
        "ac_dens": ac_div["inter_dens"],
        "ac_rpt": ac_div["repeat_pct"],
        "ac_uniq": ac_div["uniq_rate"],
        "fl_rhyme_free24": fl_rhyme_free["rate_2_4"],
        "fl_rhyme_free124": fl_rhyme_free["rate_1_2_4"],
        "fl_rhyme_g24": fl_rhyme_g["rate_2_4"],
        "fl_rhyme_g124": fl_rhyme_g["rate_1_2_4"],
        "ac_rhyme_free24": ac_rhyme_free["rate_2_4"],
        "ac_rhyme_free124": ac_rhyme_free["rate_1_2_4"],
        "ac_rhyme_g24": ac_rhyme_g["rate_2_4"],
        "ac_rhyme_g124": ac_rhyme_g["rate_1_2_4"],
        "first_line_poems": fl_poems_display,
        "acrostic_poems": ac_poems_display,
    }


# ========================================================================
#  CSV
# ========================================================================

CSV_COLUMNS = [
    "temperature", "top_k",
    "fl_dens", "fl_rpt", "fl_uniq",
    "ac_dens", "ac_rpt", "ac_uniq",
    "fl_rhyme_free24", "fl_rhyme_free124",
    "fl_rhyme_g24", "fl_rhyme_g124",
    "ac_rhyme_free24", "ac_rhyme_free124",
    "ac_rhyme_g24", "ac_rhyme_g124",
]

CSV_HEADERS = [
    "Temperature", "Top-K",
    "首句-密度", "首句-重复度%", "首句-去重率",
    "藏头-密度", "藏头-重复度%", "藏头-去重率",
    "首句-自由2-4押韵", "首句-自由1-2-4押韵",
    "首句-引导2-4押韵", "首句-引导1-2-4押韵",
    "藏头-自由2-4押韵", "藏头-自由1-2-4押韵",
    "藏头-引导2-4押韵", "藏头-引导1-2-4押韵",
]


def save_csv(file_path, results):
    with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADERS)
        for r in results:
            writer.writerow([r.get(c, "") for c in CSV_COLUMNS])


# ========================================================================
#  对比组
# ========================================================================

TEMPERATURE_GRID = [
    {"temperature": 0.6, "top_k": 0},
    {"temperature": 0.8, "top_k": 0},
    {"temperature": 1.0, "top_k": 0},
    {"temperature": 1.2, "top_k": 0},
]

TOPK_GRID = [
    {"temperature": 0.8, "top_k": 5},
    {"temperature": 0.8, "top_k": 10},
    {"temperature": 0.8, "top_k": 20},
    {"temperature": 1.0, "top_k": 5},
    {"temperature": 1.0, "top_k": 10},
    {"temperature": 1.0, "top_k": 20},
]

FULL_GRID = TEMPERATURE_GRID + TOPK_GRID


def _label(r):
    t, k = r["temperature"], r["top_k"]
    return f"t={t:.1f} 全表" if k == 0 else f"t={t:.1f} k={k}"


# ========================================================================
#  主函数
# ========================================================================

def main():
    print("=" * 50)
    print(" 采样参数横评 temperature × top-k + 韵律")
    print("=" * 50)

    args = parse_args()
    configure_torch_runtime()
    set_seed(RANDOM_SEED)
    ensure_dir(METRICS_DIR)
    ensure_dir(SAMPLES_DIR)

    device = get_device()
    if device == "cuda":
        print(f"[设备] CUDA GPU {torch.cuda.get_device_name(0)} PyTorch {torch.__version__}")
    else:
        print(f"[设备] CPU PyTorch {torch.__version__}")

    loaders = build_dataloaders()
    _, _, test_loader, stoi_data, itos_data = loaders
    model, checkpoint, stoi_ckpt, itos_ckpt = load_model_from_checkpoint(args.checkpoint, device)
    assert len(itos_data) == len(itos_ckpt), "数据集词表大小与 checkpoint 不一致"

    rhyme_map = build_rhyme_vocab_map(itos_ckpt)
    print(f"[韵律] 韵母种类 {len(rhyme_map)}  pypinyin {'OK' if rhyme_map else 'MISSING'}")

    _, test_ppl = evaluate_one_epoch(model, test_loader, device)
    print(f"[模型] Test PPL {test_ppl:.1f}")

    results = []
    for idx, cfg in enumerate(FULL_GRID, start=1):
        t, k = cfg["temperature"], cfg["top_k"]
        print(f"[{idx}/{len(FULL_GRID)}] {_label({'temperature': t, 'top_k': k})} ...",
              end=" ", flush=True)

        result = evaluate_one_config(
            model, stoi_ckpt, itos_ckpt, device, rhyme_map,
            temperature=t, top_k=k,
            first_line=args.first_line, acrostic=args.acrostic,
            num_samples=args.num_samples,
        )
        results.append(result)
        print("OK")

    # ====== 表1: 采样策略对比 自由生成 ======
    print(f"\n{'=' * 80}")
    print(" 表1: 采样策略对比 — 自由生成 无韵律引导")
    print(f"{'=' * 80}")
    print(f"  ← 密度和重复度越低越多样 →")
    print(f"  {'t':>4}  {'top_k':>5}  {'首句密度':>8}  {'藏头密度':>8}  {'首句重复%':>9}  {'藏头重复%':>9}  {'首句去重':>8}")
    print(f"  {'─' * 65}")
    for r in results:
        t_label = f"t={r['temperature']:.1f}"
        k_label = "全" if r["top_k"] == 0 else str(r["top_k"])
        print(f"  {t_label:>4}  {k_label:>5}  {r['fl_dens']:>8.3f}  {r['ac_dens']:>8.3f}  "
              f"{r['fl_rpt']:>8.1f}  {r['ac_rpt']:>8.1f}  {r['fl_uniq']:>7.2f}")
    print(f"{'=' * 80}")

    # ====== 表2: 韵律引导对比 ======
    print(f"\n{'=' * 80}")
    print(" 表2: 韵律引导对比 — 无引导 vs 有引导 2-4句押韵")
    print(f"{'=' * 80}")
    print(f"  {'t':>4}  {'top_k':>5}  {'首-无引导':>10}  {'首-有引导':>10}  {'藏-无引导':>10}  {'藏-有引导':>10}  {'平均提升':>8}")
    print(f"  {'─' * 65}")
    for r in results:
        t_label = f"t={r['temperature']:.1f}"
        k_label = "全" if r["top_k"] == 0 else str(r["top_k"])
        avg_imp = (r["fl_rhyme_g24"] - r["fl_rhyme_free24"] + r["ac_rhyme_g24"] - r["ac_rhyme_free24"]) / 2
        print(f"  {t_label:>4}  {k_label:>5}  {r['fl_rhyme_free24']:>9.1%}  {r['fl_rhyme_g24']:>9.1%}  "
              f"{r['ac_rhyme_free24']:>9.1%}  {r['ac_rhyme_g24']:>9.1%}  {avg_imp:>+7.1%}")
    print(f"{'=' * 80}")

    # ====== 保存 ======
    json_path = os.path.join(METRICS_DIR, "comparison_results.json")
    save_json({
        "checkpoint_path": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch", None),
        "best_val_ppl": checkpoint.get("best_val_ppl", None),
        "test_ppl": test_ppl,
        "num_samples": args.num_samples,
        "first_line": args.first_line,
        "acrostic": args.acrostic,
        "results": results,
    }, json_path)

    csv_path = os.path.join(METRICS_DIR, "comparison_results.csv")
    save_csv(csv_path, results)

    # ====== 推荐 ======
    _score = lambda r: r["fl_dens"] + r["ac_dens"] + r["fl_rpt"] / 100 + r["ac_rpt"] / 100
    best_div = min(results, key=_score)
    best_rhyme = max(results, key=lambda r: (r["fl_rhyme_g24"] + r["ac_rhyme_g24"]) / 2)
    print(f"\n推荐: 多样性 t={best_div['temperature']:.1f} k={best_div['top_k']}  "
          f"| 押韵 t={best_rhyme['temperature']:.1f} k={best_rhyme['top_k']} "
          f"引导后 fl={best_rhyme['fl_rhyme_g24']:.0%} ac={best_rhyme['ac_rhyme_g24']:.0%}")

    print(f"\n[JSON] {json_path}")
    print(f"[CSV]  {csv_path}")
    print("测评完成")


if __name__ == "__main__":
    main()

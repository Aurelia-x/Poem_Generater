"""
消融实验 - 评测脚本

加载训练好的消融模型，逐项评测 PPL + 生成质量，输出 CSV + 总结分析。
"""
import csv
import os

import torch

from config import (
    ABLATION_DIR,
    ABLATION_MANIFEST_PATH,
    DEFAULT_TEMPERATURE,
    RANDOM_SEED,
)
from data_utils import build_dataloaders
from generate import generate_acrostic, generate_from_first_line, load_model_from_checkpoint
from metrics import batch_diversity
from trainer import evaluate_one_epoch
from utils import configure_torch_runtime, ensure_dir, get_device, load_json, save_json, set_seed


# ==========================================
#  1. 单模型测评
# ==========================================

def evaluate_ablation_model(checkpoint_path, loaders, device, num_samples=5):
    train_loader, val_loader, test_loader, stoi_data, itos_data = loaders

    model, checkpoint, stoi_ckpt, itos_ckpt = load_model_from_checkpoint(checkpoint_path, device, verbose=False)

    train_loss, train_ppl = evaluate_one_epoch(model, train_loader, device)
    val_loss, val_ppl = evaluate_one_epoch(model, val_loader, device)
    test_loss, test_ppl = evaluate_one_epoch(model, test_loader, device)

    # 首句续写
    first_line_poems = []
    for _ in range(num_samples):
        poem = generate_from_first_line(
            model=model, first_line="春风又绿江南岸",
            stoi=stoi_ckpt, itos=itos_ckpt, device=device,
            temperature=DEFAULT_TEMPERATURE, top_k=0,
        )
        first_line_poems.append(poem)
    fl_div = batch_diversity(first_line_poems)

    # 藏头诗
    acrostic_poems = []
    for _ in range(num_samples):
        poem = generate_acrostic(
            model=model, head_chars="春江花月",
            stoi=stoi_ckpt, itos=itos_ckpt, device=device,
            temperature=DEFAULT_TEMPERATURE, top_k=0,
        )
        acrostic_poems.append(poem)
    ac_div = batch_diversity(acrostic_poems)

    return {
        "train_loss": train_loss,
        "train_ppl": train_ppl,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "test_loss": test_loss,
        "test_ppl": test_ppl,
        "fl_uniq_rate": fl_div["uniq_rate"],
        "fl_repeat_pct": fl_div["repeat_pct"],
        "fl_inter_dens": fl_div["inter_dens"],
        "ac_uniq_rate": ac_div["uniq_rate"],
        "ac_repeat_pct": ac_div["repeat_pct"],
        "ac_inter_dens": ac_div["inter_dens"],
    }


# ==========================================
#  2. CSV 输出
# ==========================================

ABLATION_CSV_COLUMNS = [
    "experiment_name", "model_type", "num_layers", "embed_dim", "hidden_dim",
    "dropout", "weight_decay", "parameter_count",
    "best_epoch", "best_val_ppl",
    "test_loss", "test_ppl",
    "fl_uniq_rate", "fl_repeat_pct", "fl_inter_dens",
    "ac_uniq_rate", "ac_repeat_pct", "ac_inter_dens",
]

ABLATION_CSV_HEADERS = [
    "实验名称", "模型类型", "层数", "Embed", "Hidden",
    "Dropout", "Weight Decay", "参数量",
    "最佳 Epoch", "最佳 Val PPL",
    "Test Loss", "Test PPL",
    "首句-去重率", "首句-重复度(%)", "首句-跨样本密度",
    "藏头-去重率", "藏头-重复度(%)", "藏头-跨样本密度",
]

EXPERIMENT_LABELS = {
    "ablation_a_baseline": "A: 基线 (LSTM-128)",
    "ablation_b_deeper": "B: 加深 (LSTM-128×2)",
    "ablation_c_wider": "C: 加宽 (LSTM-256)",
    "ablation_d_regularized": "D: 正则化 (LSTM-128+WD)",
    "ablation_e_full": "E: 全量 (LSTM-256×2)",
    "ablation_f_gru": "F: GRU (GRU-128)",
}


# ==========================================
#  3. 主函数
# ==========================================

def main():
    print("=" * 50)
    print(" 消融实验 - 横评")
    print("=" * 50)

    configure_torch_runtime()
    set_seed(RANDOM_SEED)

    device = get_device()
    if device == "cuda":
        print(f"[设备] CUDA | GPU: {torch.cuda.get_device_name(0)} | PyTorch: {torch.__version__}")
    else:
        print(f"[设备] CPU | PyTorch: {torch.__version__}")

    # 加载训练清单
    manifest = load_json(ABLATION_MANIFEST_PATH)
    print(f"[清单] {len(manifest)} 个模型")

    # 重建数据集
    loaders = build_dataloaders()

    results = []
    for entry in manifest:
        cp = entry["best_checkpoint"]
        if not os.path.exists(cp):
            print(f"[跳过] {entry['experiment_name']}: checkpoint 不存在 ({cp})")
            continue

        print(f"[{entry['experiment_name']}] ...", end=" ", flush=True)
        eval_result = evaluate_ablation_model(cp, loaders, device, num_samples=5)
        combined = {**entry, **eval_result}
        results.append(combined)
        print("OK")

    if not results:
        print("\n没有可评测的模型，请先运行 ablation_train.py")
        return

    # 保存
    ensure_dir(ABLATION_DIR)

    json_path = os.path.join(ABLATION_DIR, "ablation_results.json")
    save_json(results, json_path)

    csv_path = os.path.join(ABLATION_DIR, "ablation_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(ABLATION_CSV_HEADERS)
        for r in results:
            writer.writerow([r.get(c, "") for c in ABLATION_CSV_COLUMNS])

    # 对比总表
    print(f"\n{'=' * 80}")
    print(" 消融实验对比总表 (PPL + 多样性)")
    print(f"{'=' * 80}")
    header = (f"{'实验':<28s} {'Test PPL':<10s} {'首-密度':<8s} {'首-重复%':<9s} {'藏-密度':<8s} {'藏-重复%':<9s}")
    print(header)
    print("-" * 80)
    for r in sorted(results, key=lambda r: r["test_ppl"]):
        label = EXPERIMENT_LABELS.get(r["experiment_name"], r["experiment_name"])
        print(f"{label:<28s} {r['test_ppl']:<10.2f} "
              f"{r['fl_inter_dens']:<8.3f} {r['fl_repeat_pct']:<8.1f} "
              f"{r['ac_inter_dens']:<8.3f} {r['ac_repeat_pct']:<8.1f}")
    print(f"{'=' * 80}")

    # ====== 推荐 ======
    best_ppl = min(results, key=lambda r: r["test_ppl"])
    _div_score = lambda r: r["fl_inter_dens"] + r["ac_inter_dens"] + r["fl_repeat_pct"] / 100 + r["ac_repeat_pct"] / 100
    best_div = min(results, key=_div_score)
    lbl_ppl = EXPERIMENT_LABELS.get(best_ppl["experiment_name"], best_ppl["experiment_name"])
    lbl_div = EXPERIMENT_LABELS.get(best_div["experiment_name"], best_div["experiment_name"])
    print(f"\n推荐 PPL最优: {lbl_ppl} PPL={best_ppl['test_ppl']:.2f} | 多样最优: {lbl_div} 密度={best_div['fl_inter_dens']:.3f}/{best_div['ac_inter_dens']:.3f}")

    print(f"\n[保存 JSON] {json_path}")
    print(f"[保存 CSV]  {csv_path}")
    print("\n横评完成")


if __name__ == "__main__":
    main()

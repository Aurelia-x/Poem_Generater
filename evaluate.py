import argparse
import os

import torch

from config import (
    BASE_DIR,
    BEST_MODEL_SAVE_PATH,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    METRICS_DIR,
    RANDOM_SEED,
    SAMPLES_DIR,
)
from data_utils import build_dataloaders
from generate import (
    format_poem_for_display,
    generate_acrostic,
    generate_from_first_line,
    load_model_from_checkpoint,
)
from metrics import batch_format_compliance
from trainer import evaluate_one_epoch
from utils import configure_torch_runtime, ensure_dir, get_device, save_json, set_seed


# ==========================================
#  1. 基础测评主函数
# ==========================================

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="诗词生成基础测评入口")
    parser.add_argument("--checkpoint", type=str, default=BEST_MODEL_SAVE_PATH, help="checkpoint 路径")
    parser.add_argument("--num_samples", type=int, default=5, help="每种条件生成多少组样例")
    parser.add_argument("--strategy", type=str, default="temperature", choices=["temperature", "top_k"], help="采样策略")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="temperature 参数")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="top-k 参数")
    parser.add_argument("--first_line", type=str, default="春风又绿江南岸", help="首句续写输入")
    parser.add_argument("--acrostic", type=str, default="春江花月", help="藏头诗输入")
    return parser.parse_args()


def save_evaluation_summary(file_path, summary_dict):
    """
    保存基础测评结果为 JSON

    参数:
        file_path:     保存路径
        summary_dict:  结果字典
    """
    save_json(summary_dict, file_path)
    print(f"[输出] 测评结果已保存到: {file_path}")


def run_batch_generation(model, stoi, itos, device, mode,
                         text, num_samples, strategy, temperature, top_k):
    """
    批量生成样例并统计格式合规率

    参数:
        model:        已加载模型
        stoi, itos:   词表映射
        device:       计算设备
        mode:         "first_line" 或 "acrostic"
        text:         条件输入
        num_samples:  生成样例数量
        strategy:     采样策略
        temperature:  temperature 参数
        top_k:        top-k 参数

    返回:
        poems:    生成的诗歌列表
        summary:  批量统计结果
    """
    poems = []

    print(f"\n[批量生成] mode={mode}, text={text}, num_samples={num_samples}")

    for idx in range(1, num_samples + 1):
        print(f"  [生成样例 {idx}/{num_samples}]")

        if mode == "first_line":
            poem = generate_from_first_line(
                model=model,
                first_line=text,
                stoi=stoi,
                itos=itos,
                device=device,
                strategy=strategy,
                temperature=temperature,
                top_k=top_k,
            )
        elif mode == "acrostic":
            poem = generate_acrostic(
                model=model,
                head_chars=text,
                stoi=stoi,
                itos=itos,
                device=device,
                strategy=strategy,
                temperature=temperature,
                top_k=top_k,
            )
        else:
            raise ValueError(f"[run_batch_generation] 不支持的 mode: {mode}")

        poems.append(poem)

    summary = batch_format_compliance(poems, expected_len=28)
    print(f"\n[批量生成统计] 完全 28 字合规数: {summary['exact_match_count']}")
    print(f"[批量生成统计] 格式合规率: {summary['exact_match_rate']:.4f}")
    print(f"[批量生成统计] 平均 relaxed score: {summary['avg_relaxed_score']:.4f}")

    return poems, summary


def main():
    print("=" * 60)
    print(" 诗词生成基础测评")
    print(" 测试集 PPL + 批量生成 + 格式合规率")
    print("=" * 60)

    args = parse_args()
    configure_torch_runtime()
    set_seed(RANDOM_SEED)
    ensure_dir(METRICS_DIR)
    ensure_dir(SAMPLES_DIR)

    # ==========================================
    #  1.1 检测计算设备
    # ==========================================
    print("\n" + "-" * 55)
    print(" 1. 检测计算设备")
    print("-" * 55)

    device = get_device()
    print(f"使用设备: {device}")
    if device == "cuda":
        print(f"  GPU 型号: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA 版本: {torch.version.cuda}")
    else:
        print(f"  未检测到 GPU，使用 CPU 进行评测")
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  项目目录: {BASE_DIR}")

    # ==========================================
    #  1.2 重建数据集划分
    # ==========================================
    print("\n" + "-" * 55)
    print(" 2. 重建数据集与 DataLoader")
    print("-" * 55)

    train_loader, val_loader, test_loader, stoi_data, itos_data = build_dataloaders()

    # ==========================================
    #  1.3 加载最佳模型
    # ==========================================
    print("\n" + "-" * 55)
    print(" 3. 加载最佳模型 checkpoint")
    print("-" * 55)

    model, checkpoint, stoi_ckpt, itos_ckpt = load_model_from_checkpoint(args.checkpoint, device)

    print(f"\n[一致性检查] DataLoader 词表大小: {len(itos_data)}")
    print(f"[一致性检查] Checkpoint 词表大小: {len(itos_ckpt)}")
    assert len(itos_data) == len(itos_ckpt), \
        "[一致性检查] 数据集重建得到的词表大小与 checkpoint 不一致!"

    # ==========================================
    #  1.4 测试集 PPL 评测
    # ==========================================
    print("\n" + "-" * 55)
    print(" 4. 测试集 PPL 评测")
    print("-" * 55)

    train_loss, train_ppl = evaluate_one_epoch(model, train_loader, device)
    val_loss, val_ppl = evaluate_one_epoch(model, val_loader, device)
    test_loss, test_ppl = evaluate_one_epoch(model, test_loader, device)

    print(f"\n{'=' * 48}")
    print(" 基础测评：数据集级指标")
    print(f"{'=' * 48}")
    print(f"  Train Loss: {train_loss:.6f},  Train PPL: {train_ppl:.4f}")
    print(f"  Val   Loss: {val_loss:.6f},  Val   PPL: {val_ppl:.4f}")
    print(f"  Test  Loss: {test_loss:.6f},  Test  PPL: {test_ppl:.4f}")
    print(f"{'=' * 48}")

    # ==========================================
    #  1.5 首句续写批量生成
    # ==========================================
    print("\n" + "-" * 55)
    print(" 5. 首句续写基础测评")
    print("-" * 55)

    first_line_poems, first_line_summary = run_batch_generation(
        model=model,
        stoi=stoi_ckpt,
        itos=itos_ckpt,
        device=device,
        mode="first_line",
        text=args.first_line,
        num_samples=args.num_samples,
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    # ==========================================
    #  1.6 藏头诗批量生成
    # ==========================================
    print("\n" + "-" * 55)
    print(" 6. 藏头诗基础测评")
    print("-" * 55)

    acrostic_poems, acrostic_summary = run_batch_generation(
        model=model,
        stoi=stoi_ckpt,
        itos=itos_ckpt,
        device=device,
        mode="acrostic",
        text=args.acrostic,
        num_samples=args.num_samples,
        strategy=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    # ==========================================
    #  1.7 汇总并保存测评结果
    # ==========================================
    print("\n" + "-" * 55)
    print(" 7. 汇总并保存测评结果")
    print("-" * 55)

    result_summary = {
        "checkpoint_path": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch", None),
        "best_val_ppl": checkpoint.get("best_val_ppl", None),
        "evaluation_config": {
            "strategy": args.strategy,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "num_samples": args.num_samples,
            "first_line": args.first_line,
            "acrostic": args.acrostic,
        },
        "dataset_metrics": {
            "train_loss": train_loss,
            "train_ppl": train_ppl,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "test_loss": test_loss,
            "test_ppl": test_ppl,
        },
        "generation_metrics": {
            "first_line_summary": first_line_summary,
            "acrostic_summary": acrostic_summary,
        },
        "generation_examples": {
            "first_line_poems": [format_poem_for_display(poem) for poem in first_line_poems],
            "acrostic_poems": [format_poem_for_display(poem) for poem in acrostic_poems],
        },
    }

    result_file_name = (
        f"basic_eval_{args.strategy}_"
        f"t{str(args.temperature).replace('.', '')}_"
        f"k{args.top_k}.json"
    )
    result_file_path = os.path.join(METRICS_DIR, result_file_name)
    save_evaluation_summary(result_file_path, result_summary)

    print(f"\n[输出] 样例目录: {SAMPLES_DIR}")
    print(f"[输出] 指标目录: {METRICS_DIR}")

    print("\n" + "=" * 60)
    print(" 诗词生成基础测评完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

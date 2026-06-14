import os

import torch

from config import (
    ABLATION_DIR,
    ABLATION_EXPERIMENTS,
    BASE_DIR,
    BATCH_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    RANDOM_SEED,
)
from data_utils import build_dataloaders
from generate import generate_acrostic, generate_from_first_line
from metrics import batch_format_compliance
from model import build_model
from trainer import evaluate_one_epoch, fit
from utils import configure_torch_runtime, count_parameters, ensure_dir, get_device, save_json, set_seed


# ==========================================
#  1. 批量生成并统计格式合规率
# ==========================================

def evaluate_generation_quality(model, stoi, itos, device,
                                num_samples=5,
                                strategy="temperature",
                                temperature=DEFAULT_TEMPERATURE,
                                top_k=DEFAULT_TOP_K):
    """
    对当前模型做基础生成质量评估

    参数:
        model:        当前训练好的模型
        stoi, itos:   词表映射
        device:       计算设备
        num_samples:  每种条件生成多少首诗
        strategy:     采样策略
        temperature:  temperature 参数
        top_k:        top-k 参数

    返回:
        quality_summary: 生成质量汇总
    """
    print(f"\n[生成质量评估] 开始统计格式合规率...")
    print(f"  [生成质量评估] 每种条件样例数 = {num_samples}")
    print(f"  [生成质量评估] strategy = {strategy}, temperature = {temperature}, top_k = {top_k}")

    first_line_poems = []
    acrostic_poems = []

    for idx in range(num_samples):
        print(f"\n  [生成质量评估] 首句续写样例 {idx + 1}/{num_samples}")
        first_line_poem = generate_from_first_line(
            model=model,
            first_line="春风又绿江南岸",
            stoi=stoi,
            itos=itos,
            device=device,
            strategy=strategy,
            temperature=temperature,
            top_k=top_k,
        )
        first_line_poems.append(first_line_poem)

    for idx in range(num_samples):
        print(f"\n  [生成质量评估] 藏头诗样例 {idx + 1}/{num_samples}")
        acrostic_poem = generate_acrostic(
            model=model,
            head_chars="春江花月",
            stoi=stoi,
            itos=itos,
            device=device,
            strategy=strategy,
            temperature=temperature,
            top_k=top_k,
        )
        acrostic_poems.append(acrostic_poem)

    first_line_summary = batch_format_compliance(first_line_poems, expected_len=28)
    acrostic_summary = batch_format_compliance(acrostic_poems, expected_len=28)

    print(f"\n[生成质量评估] 首句续写格式合规率: {first_line_summary['exact_match_rate']:.4f}")
    print(f"[生成质量评估] 藏头诗格式合规率:   {acrostic_summary['exact_match_rate']:.4f}")

    return {
        "first_line_summary": first_line_summary,
        "acrostic_summary": acrostic_summary,
        "first_line_poems": first_line_poems,
        "acrostic_poems": acrostic_poems,
    }


# ==========================================
#  2. 主函数
# ==========================================

def main():
    print("=" * 60)
    print(" 诗词生成消融实验")
    print(" 控制变量：深度 / 宽度 / 正则化 / 完整模型")
    print("=" * 60)

    ensure_dir(ABLATION_DIR)
    configure_torch_runtime()
    set_seed(RANDOM_SEED)

    print("\n" + "-" * 55)
    print(" 1. 检测计算设备")
    print("-" * 55)
    device = get_device()
    print(f"使用设备: {device}")
    if device == "cuda":
        print(f"  GPU 型号: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA 版本: {torch.version.cuda}")
    else:
        print(f"  未检测到 GPU，使用 CPU 进行消融实验")
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  项目目录: {BASE_DIR}")

    print("\n" + "-" * 55)
    print(" 2. 构建统一数据集")
    print("-" * 55)
    train_loader, val_loader, test_loader, stoi, itos = build_dataloaders()

    ablation_results = []

    print("\n" + "-" * 55)
    print(" 3. 开始逐个运行消融实验")
    print("-" * 55)

    for exp_idx, exp_config in enumerate(ABLATION_EXPERIMENTS, start=1):
        print("\n" + "=" * 60)
        print(f" [消融实验 {exp_idx}/{len(ABLATION_EXPERIMENTS)}] {exp_config['experiment_name']}")
        print("=" * 60)

        merged_config = dict(exp_config)
        merged_config["batch_size"] = BATCH_SIZE
        merged_config["random_seed"] = RANDOM_SEED
        merged_config["vocab_size"] = len(stoi)

        print(f"\n[实验配置]")
        for key in [
            "experiment_name", "model_type", "embed_dim", "hidden_dim",
            "num_layers", "dropout", "learning_rate", "weight_decay",
            "num_epochs", "grad_clip"
        ]:
            print(f"  {key}: {merged_config[key]}")

        model = build_model(
            vocab_size=len(stoi),
            model_type=merged_config["model_type"],
            embed_dim=merged_config["embed_dim"],
            hidden_dim=merged_config["hidden_dim"],
            num_layers=merged_config["num_layers"],
            dropout=merged_config["dropout"],
        ).to(device)

        total_params, trainable_params = count_parameters(model)
        print(f"\n[模型] 总参数量: {total_params:,}")
        print(f"[模型] 可训练参数量: {trainable_params:,}")

        history, best_epoch, best_val_ppl, _ = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            stoi=stoi,
            itos=itos,
            config_dict=merged_config,
        )

        print(f"\n[实验评估] 使用最佳模型进行 train / val / test 评测")
        train_loss, train_ppl = evaluate_one_epoch(model, train_loader, device)
        val_loss, val_ppl = evaluate_one_epoch(model, val_loader, device)
        test_loss, test_ppl = evaluate_one_epoch(model, test_loader, device)

        generation_quality = evaluate_generation_quality(
            model=model,
            stoi=stoi,
            itos=itos,
            device=device,
            num_samples=5,
            strategy="temperature",
            temperature=DEFAULT_TEMPERATURE,
            top_k=DEFAULT_TOP_K,
        )

        exp_result = {
            "experiment_name": merged_config["experiment_name"],
            "model_type": merged_config["model_type"],
            "embed_dim": merged_config["embed_dim"],
            "hidden_dim": merged_config["hidden_dim"],
            "num_layers": merged_config["num_layers"],
            "dropout": merged_config["dropout"],
            "learning_rate": merged_config["learning_rate"],
            "weight_decay": merged_config["weight_decay"],
            "num_epochs": merged_config["num_epochs"],
            "parameter_count": total_params,
            "train_loss": train_loss,
            "train_ppl": train_ppl,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "test_loss": test_loss,
            "test_ppl": test_ppl,
            "best_epoch": best_epoch,
            "best_val_ppl": best_val_ppl,
            "first_line_format_rate": generation_quality["first_line_summary"]["exact_match_rate"],
            "acrostic_format_rate": generation_quality["acrostic_summary"]["exact_match_rate"],
            "history": history,
            "generation_quality": generation_quality,
        }
        ablation_results.append(exp_result)

    print("\n" + "-" * 55)
    print(" 4. 保存消融实验结果")
    print("-" * 55)

    result_path = os.path.join(ABLATION_DIR, "ablation_results.json")
    save_json({"results": ablation_results}, result_path)
    print(f"[输出] 消融实验汇总结果已保存到: {result_path}")

    print("\n" + "=" * 60)
    print(" 消融实验完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

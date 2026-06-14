import os

import torch

from config import (
    BASE_DIR,
    BEST_MODEL_SAVE_PATH,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    RANDOM_SEED,
    SKIP_TRAINING,
    build_runtime_config,
)
from data_utils import build_dataloaders
from model import build_model
from trainer import evaluate_one_epoch, fit
from utils import count_parameters, ensure_dir, get_device, set_seed


# ==========================================
#  1. 主函数
# ==========================================

def main():
    print("=" * 60)
    print(" 诗词生成最小训练闭环")
    print(" 字符级语言模型（LSTM / GRU）")
    print("=" * 60)

    # ==========================================
    #  1.1 准备目录与随机种子
    # ==========================================
    ensure_dir(CHECKPOINT_DIR)
    set_seed(RANDOM_SEED)

    # ==========================================
    #  1.2 检测计算设备
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
        print(f"  未检测到 GPU，使用 CPU 进行训练")
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  项目目录: {BASE_DIR}")

    # ==========================================
    #  1.3 构建数据集与 DataLoader
    # ==========================================
    train_loader, val_loader, test_loader, stoi, itos = build_dataloaders()

    # ==========================================
    #  1.4 构建模型
    # ==========================================
    print("\n" + "-" * 55)
    print(" 2. 构建诗歌生成模型")
    print("-" * 55)

    runtime_config = build_runtime_config()

    model = build_model(
        vocab_size=len(stoi),
        model_type=runtime_config["model_type"],
        embed_dim=runtime_config["embed_dim"],
        hidden_dim=runtime_config["hidden_dim"],
        num_layers=runtime_config["num_layers"],
        dropout=runtime_config["dropout"],
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"\n[模型] 总参数量: {total_params:,}")
    print(f"[模型] 可训练参数量: {trainable_params:,}")
    print(f"[模型] 参数明细:")
    for name, param in model.named_parameters():
        print(f"  {name}: {tuple(param.shape)}  ({param.numel():,} 个参数)")

    # ==========================================
    #  1.5 模型维度自检
    # ==========================================
    print(f"\n[模型维度自检]")
    with torch.no_grad():
        sample_x, sample_y = next(iter(train_loader))
        sample_x = sample_x.to(device)
        sample_logits, _ = model(sample_x)

    print(f"  输入 x:      {tuple(sample_x.shape)}  (期望: ({BATCH_SIZE} 或更小, 29))")
    print(f"  目标 y:      {tuple(sample_y.shape)}  (期望: ({BATCH_SIZE} 或更小, 29))")
    print(f"  输出 logits: {tuple(sample_logits.shape)}  (期望: ({sample_x.shape[0]}, 29, {len(stoi)}))")
    assert sample_logits.shape == (sample_x.shape[0], sample_x.shape[1], len(stoi)), \
        "[模型维度自检] logits 形状异常!"

    config_dict = dict(runtime_config)
    config_dict["batch_size"] = BATCH_SIZE
    config_dict["random_seed"] = RANDOM_SEED
    config_dict["vocab_size"] = len(stoi)

    # ==========================================
    #  1.6 训练或加载最佳模型
    # ==========================================
    if SKIP_TRAINING:
        print("\n" + "-" * 55)
        print(" 3. 跳过训练，加载最佳模型")
        print("-" * 55)

        print(f"\n[模型加载] 加载路径: {BEST_MODEL_SAVE_PATH}")
        checkpoint = torch.load(BEST_MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[模型加载] 模型参数加载成功")
        print(f"[模型加载] 已保存最佳 Val PPL: {checkpoint.get('best_val_ppl', 'N/A')}")
    else:
        history, best_epoch, best_val_ppl, _ = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            stoi=stoi,
            itos=itos,
            config_dict=config_dict,
        )
        print(f"\n[训练总结] history 长度: {len(history['train_loss'])}")
        print(f"[训练总结] 最佳 Epoch: {best_epoch}")
        print(f"[训练总结] 最佳 Val PPL: {best_val_ppl:.4f}")

    # ==========================================
    #  1.7 最终评估
    # ==========================================
    print("\n" + "-" * 55)
    print(" 4. 最终评估（使用当前模型）")
    print("-" * 55)

    train_loss, train_ppl = evaluate_one_epoch(model, train_loader, device)
    val_loss, val_ppl = evaluate_one_epoch(model, val_loader, device)
    test_loss, test_ppl = evaluate_one_epoch(model, test_loader, device)

    print(f"\n{'=' * 44}")
    print(" 诗歌语言模型最终评估结果")
    print(f"{'=' * 44}")
    print(f"  Train Loss: {train_loss:.6f},  Train PPL: {train_ppl:.4f}")
    print(f"  Val   Loss: {val_loss:.6f},  Val   PPL: {val_ppl:.4f}")
    print(f"  Test  Loss: {test_loss:.6f},  Test  PPL: {test_ppl:.4f}")
    print(f"{'=' * 44}")

    print("\n" + "=" * 60)
    print(" 诗词生成最小训练闭环完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

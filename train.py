import torch

from config import (
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
from utils import configure_torch_runtime, count_parameters, ensure_dir, get_device, set_seed


# ==========================================
#  1. 主函数
# ==========================================

def main():
    print("=" * 50)
    print(" 诗词生成训练 (LSTM/GRU)")
    print("=" * 50)


    ensure_dir(CHECKPOINT_DIR)
    configure_torch_runtime()
    set_seed(RANDOM_SEED)

    device = get_device()
    if device == "cuda":
        print(f"[设备] CUDA | GPU: {torch.cuda.get_device_name(0)} | PyTorch: {torch.__version__}")
    else:
        print(f"[设备] CPU | PyTorch: {torch.__version__}")

    # ==========================================
    #  1.3 构建数据集与 DataLoader
    # ==========================================
    train_loader, val_loader, test_loader, stoi, itos = build_dataloaders()

    # ==========================================
    #  1.4 构建模型
    # ==========================================
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
    print(f"[模型] 参数量: total={total_params:,} trainable={trainable_params:,}")

    print(f"[自检] 维度校验...")
    with torch.no_grad():
        sample_x, sample_y = next(iter(train_loader))
        sample_x = sample_x.to(device)
        sample_logits, _ = model(sample_x)

    assert sample_logits.shape == (sample_x.shape[0], sample_x.shape[1], len(stoi)), \
        f"[自检] logits 形状异常! 期望 ({sample_x.shape[0]}, {sample_x.shape[1]}, {len(stoi)}), 实际 {tuple(sample_logits.shape)}"
    print(f"[自检] 通过 x={tuple(sample_x.shape)} -> logits={tuple(sample_logits.shape)}")

    config_dict = dict(runtime_config)
    config_dict["batch_size"] = BATCH_SIZE
    config_dict["random_seed"] = RANDOM_SEED
    config_dict["vocab_size"] = len(stoi)

    # ==========================================
    #  1.6 训练或加载最佳模型
    # ==========================================
    if SKIP_TRAINING:
        print(f"\n[加载] 跳过训练，加载最佳模型: {BEST_MODEL_SAVE_PATH}")
        checkpoint = torch.load(BEST_MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_epoch = checkpoint.get("epoch", 0)
        best_val_ppl = checkpoint.get("best_val_ppl", float("inf"))
        saved_history = checkpoint.get("history", {})
        train_loss = (saved_history.get("train_loss", [0])[best_epoch - 1]
                      if saved_history.get("train_loss") and best_epoch > 0
                      else 0.0)
        train_ppl = (saved_history.get("train_ppl", [0])[best_epoch - 1]
                     if saved_history.get("train_ppl") and best_epoch > 0
                     else 0.0)
        print(f"[加载] 完成, best_epoch={best_epoch}, best_val_ppl={best_val_ppl:.4f}")
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
        train_loss = history["train_loss"][best_epoch - 1]
        train_ppl = history["train_ppl"][best_epoch - 1]
        print(f"\n[总结] epochs={len(history['train_loss'])} best_epoch={best_epoch} best_val_ppl={best_val_ppl:.4f}")

    print(f"\n{'=' * 44}")
    print(" 最终评估 (使用最佳模型)")
    print(f"{'=' * 44}")

    val_loss, val_ppl = evaluate_one_epoch(model, val_loader, device)
    test_loss, test_ppl = evaluate_one_epoch(model, test_loader, device)

    print(f"  Train Loss: {train_loss:.6f}  PPL: {train_ppl:.4f}")
    print(f"  Val   Loss: {val_loss:.6f}  PPL: {val_ppl:.4f}")
    print(f"  Test  Loss: {test_loss:.6f}  PPL: {test_ppl:.4f}")
    print(f"{'=' * 44}")

    print("\n训练完成")


if __name__ == "__main__":
    main()

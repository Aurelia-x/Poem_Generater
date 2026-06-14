"""
断点续训脚本：从已保存的 checkpoint 加载模型、优化器、训练历史，继续训练指定的额外 epoch。
"""

import copy
import math
import os

import torch
import torch.optim as optim

from config import (
    BEST_MODEL_SAVE_PATH,
    CHECKPOINT_DIR,
    RANDOM_SEED,
)
from data_utils import build_dataloaders
from model import build_model
from trainer import evaluate_one_epoch, save_checkpoint, train_one_epoch
from utils import configure_torch_runtime, ensure_dir, get_device, set_seed


EXTRA_EPOCHS = 25
CHECKPOINT_TO_RESUME = BEST_MODEL_SAVE_PATH
LAST_MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "poem_lstm_final_last.pth")


def resume_training(resume_path, extra_epochs):
    print("=" * 55)
    print(" 断点续训")
    print(f" 从 checkpoint 加载: {resume_path}")
    print(f" 额外训练轮数: {extra_epochs}")
    print("=" * 55)

    ensure_dir(CHECKPOINT_DIR)
    configure_torch_runtime()
    set_seed(RANDOM_SEED)

    device = get_device()
    if device == "cuda":
        print(f"[设备] CUDA | GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"[设备] CPU")
    print(f"[设备] PyTorch {torch.__version__}")

    # 1. 构建数据集（缓存命中后直接加载）
    train_loader, val_loader, test_loader, stoi, itos = build_dataloaders()

    # 2. 加载 checkpoint
    checkpoint = torch.load(resume_path, map_location=device)
    config_dict = checkpoint["config"]
    start_epoch = checkpoint["epoch"]
    history = checkpoint["history"]
    best_val_ppl = checkpoint["best_val_ppl"]
    best_epoch = start_epoch

    print(f"\n[加载] 起始 epoch = {start_epoch}")
    print(f"[加载] 历史 best_val_ppl = {best_val_ppl:.4f} (epoch {best_epoch})")
    print(f"[加载] 历史训练记录: {len(history['train_loss'])} 轮已保存")

    # 3. 重建模型并加载权重
    model = build_model(
        vocab_size=config_dict["vocab_size"],
        model_type=config_dict["model_type"],
        embed_dim=config_dict["embed_dim"],
        hidden_dim=config_dict["hidden_dim"],
        num_layers=config_dict["num_layers"],
        dropout=config_dict["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[加载] 模型权重已恢复")

    # 4. 重建优化器并加载状态
    learning_rate = config_dict.get("learning_rate", 3e-4)
    weight_decay = config_dict.get("weight_decay", 0.0)
    grad_clip = config_dict.get("grad_clip", 1.0)
    log_every_n_batches = config_dict.get("log_every_n_batches", 0)
    scheduler_factor = config_dict.get("scheduler_factor", 0.5)
    scheduler_patience = config_dict.get("scheduler_patience", 1)
    scheduler_min_lr = config_dict.get("scheduler_min_lr", 1e-5)
    early_stopping_patience = config_dict.get("early_stopping_patience", 3)
    early_stopping_min_delta = config_dict.get("early_stopping_min_delta", 0.1)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"[加载] 优化器状态已恢复 (lr={optimizer.param_groups[0]['lr']:.6f})")

    # 5. 重建 scheduler（ReduceLROnPlateau 需要手动恢复内部状态）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr,
    )
    # 用历史 val_ppl 序列重建 scheduler 状态
    for i in range(len(history["val_ppl"])):
        scheduler.step(history["val_ppl"][i])
    print(f"[加载] Scheduler 状态已重建, 当前 lr={optimizer.param_groups[0]['lr']:.6f}")

    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    print(f"\n[续训配置]")
    print(f"  额外轮数: {extra_epochs}, 预计完成 epoch {start_epoch + extra_epochs}")
    print(f"  梯度裁剪: {grad_clip}")
    print(f"  学习率调度: ReduceLROnPlateau (factor={scheduler_factor}, patience={scheduler_patience})")
    print(f"  早停耐心: {early_stopping_patience}, min_delta={early_stopping_min_delta}")

    print(f"\n{'Epoch':>6s}  {'LR':>10s}  {'Train Loss':>12s}  {'Train PPL':>12s}  {'Val Loss':>12s}  {'Val PPL':>12s}")
    print(f"{'------':>6s}  {'--------':>10s}  {'----------':>12s}  {'----------':>12s}  {'--------':>12s}  {'--------':>12s}")

    end_epoch = start_epoch + extra_epochs
    for epoch in range(start_epoch + 1, end_epoch + 1):
        train_loss, train_ppl, avg_grad_norm = train_one_epoch(
            model, train_loader, optimizer, device, epoch, grad_clip, log_every_n_batches
        )
        val_loss, val_ppl = evaluate_one_epoch(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_ppl"].append(train_ppl)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        history["lr"].append(current_lr)
        history["grad_norm"].append(avg_grad_norm)

        if best_val_ppl - val_ppl > early_stopping_min_delta:
            best_val_ppl = val_ppl
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            save_checkpoint(BEST_MODEL_SAVE_PATH, model, optimizer, epoch, history,
                            best_val_ppl, stoi, itos, config_dict)
        else:
            epochs_without_improvement += 1

        print(f"{epoch:>6d}  {current_lr:>10.6f}  {train_loss:>12.6f}  {train_ppl:>12.4f}  {val_loss:>12.6f}  {val_ppl:>12.4f}")

        scheduler.step(val_ppl)

        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n[续训] 验证集 PPL 已连续 {epochs_without_improvement} 轮无提升，提前停止")
            break

    print(f"{'------':>6s}  {'--------':>10s}  {'----------':>12s}  {'----------':>12s}  {'--------':>12s}  {'--------':>12s}")

    finished_epochs = len(history["train_loss"])
    save_checkpoint(LAST_MODEL_SAVE_PATH, model, optimizer, finished_epochs, history,
                    best_val_ppl, stoi, itos, config_dict)

    model.load_state_dict(best_state)

    # 最终评估
    print(f"\n[续训] 完成! epochs={finished_epochs} best_epoch={best_epoch} best_val_ppl={best_val_ppl:.4f}")

    print(f"\n{'=' * 44}")
    print(" 最终评估 (使用最佳模型)")
    print(f"{'=' * 44}")

    train_loss, train_ppl = evaluate_one_epoch(model, train_loader, device)
    val_loss, val_ppl = evaluate_one_epoch(model, val_loader, device)
    test_loss, test_ppl = evaluate_one_epoch(model, test_loader, device)

    print(f"  Train Loss: {train_loss:.6f}  PPL: {train_ppl:.4f}")
    print(f"  Val   Loss: {val_loss:.6f}  PPL: {val_ppl:.4f}")
    print(f"  Test  Loss: {test_loss:.6f}  PPL: {test_ppl:.4f}")
    print(f"{'=' * 44}")

    print("\n续训完成")


if __name__ == "__main__":
    assert os.path.exists(CHECKPOINT_TO_RESUME), \
        f"未找到要恢复的 checkpoint: {CHECKPOINT_TO_RESUME}"

    resume_training(CHECKPOINT_TO_RESUME, EXTRA_EPOCHS)

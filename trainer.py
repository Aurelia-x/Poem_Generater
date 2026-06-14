import copy
import math

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    BEST_MODEL_SAVE_PATH,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    GRAD_CLIP,
    LAST_MODEL_SAVE_PATH,
    LEARNING_RATE,
    LOG_EVERY_N_BATCHES,
    NUM_EPOCHS,
    SCHEDULER_FACTOR,
    SCHEDULER_MIN_LR,
    SCHEDULER_PATIENCE,
    WEIGHT_DECAY,
)


# ==========================================
#  1. 训练与评估函数
# ==========================================

def sequence_cross_entropy(logits, targets):
    """
    计算序列级交叉熵损失

    参数:
        logits:  (B, T, V)
        targets: (B, T)

    返回:
        loss: 标量损失
    """
    B, T, V = logits.shape
    assert targets.shape == (B, T), \
        f"[sequence_cross_entropy] targets 形状异常: 期望 ({B},{T}), 实际 {tuple(targets.shape)}"

    logits_2d = logits.reshape(B * T, V)
    targets_1d = targets.reshape(B * T)

    loss = nn.CrossEntropyLoss()(logits_2d, targets_1d)
    return loss


@torch.no_grad()
def evaluate_one_epoch(model, dataloader, device):
    """
    在一个数据集上评估平均损失与 PPL

    参数:
        model:      模型实例
        dataloader: 数据加载器
        device:     计算设备

    返回:
        avg_loss: 平均交叉熵损失
        ppl:      困惑度
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        logits, _ = model(x_batch)
        loss = sequence_cross_entropy(logits, y_batch)
        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def train_one_epoch(model, dataloader, optimizer, device, epoch,
                     grad_clip=GRAD_CLIP, log_every_n_batches=LOG_EVERY_N_BATCHES):
    """
    训练一个 Epoch

    参数:
        model:               模型实例
        dataloader:          训练集数据加载器
        optimizer:           优化器
        device:              计算设备
        epoch:               当前 Epoch 编号
        grad_clip:           梯度裁剪阈值
        log_every_n_batches: 每 N 个 batch 输出一次日志

    返回:
        avg_loss:      平均训练损失
        ppl:           训练集困惑度
        avg_grad_norm: 平均梯度范数
    """
    model.train()
    running_loss = 0.0
    running_grad_norm = 0.0
    num_batches = len(dataloader)

    for batch_idx, (x_batch, y_batch) in enumerate(dataloader, start=1):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits, _ = model(x_batch)

        loss = sequence_cross_entropy(logits, y_batch)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += loss.item()
        running_grad_norm += float(grad_norm)

        should_log_batch = (
            log_every_n_batches is not None
            and log_every_n_batches > 0
            and (batch_idx == 1 or batch_idx % log_every_n_batches == 0 or batch_idx == num_batches)
        )
        if should_log_batch:
            print(f"  [Epoch {epoch:02d} | Batch {batch_idx:03d}/{num_batches:03d}] "
                  f"loss = {loss.item():.6f}, grad_norm = {float(grad_norm):.6f}")

    avg_loss = running_loss / num_batches
    avg_grad_norm = running_grad_norm / num_batches
    ppl = math.exp(avg_loss)
    return avg_loss, ppl, avg_grad_norm


# ==========================================
#  2. Checkpoint 保存函数
# ==========================================

def save_checkpoint(file_path, model, optimizer, epoch, history, best_val_ppl, stoi, itos, config_dict):
    """
    保存 checkpoint

    参数:
        file_path:     checkpoint 保存路径
        model:         模型实例
        optimizer:     优化器
        epoch:         当前 epoch
        history:       训练历史字典
        best_val_ppl:  当前最佳验证集 PPL
        stoi, itos:    词表映射
        config_dict:   超参数配置字典
    """
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "history": history,
        "best_val_ppl": best_val_ppl,
        "stoi": stoi,
        "itos": itos,
        "config": config_dict,
    }, file_path)

    # print(f"[checkpoint] 已保存模型到: {file_path}")


# ==========================================
#  3. 完整训练主循环
# ==========================================

def fit(model, train_loader, val_loader, device, stoi, itos, config_dict):
    """
    完整训练主循环

    参数:
        model:        模型实例
        train_loader: 训练集 DataLoader
        val_loader:   验证集 DataLoader
        device:       计算设备
        stoi, itos:   词表映射
        config_dict:  超参数配置字典

    返回:
        history:       训练历史
        best_epoch:    最佳 epoch
        best_val_ppl:  最佳验证集 PPL
        best_state:    最佳模型参数
    """
    learning_rate = config_dict.get("learning_rate", LEARNING_RATE)
    num_epochs = config_dict.get("num_epochs", NUM_EPOCHS)
    grad_clip = config_dict.get("grad_clip", GRAD_CLIP)
    weight_decay = config_dict.get("weight_decay", WEIGHT_DECAY)
    log_every_n_batches = config_dict.get("log_every_n_batches", LOG_EVERY_N_BATCHES)
    scheduler_factor = config_dict.get("scheduler_factor", SCHEDULER_FACTOR)
    scheduler_patience = config_dict.get("scheduler_patience", SCHEDULER_PATIENCE)
    scheduler_min_lr = config_dict.get("scheduler_min_lr", SCHEDULER_MIN_LR)
    early_stopping_patience = config_dict.get("early_stopping_patience", EARLY_STOPPING_PATIENCE)
    early_stopping_min_delta = config_dict.get("early_stopping_min_delta", EARLY_STOPPING_MIN_DELTA)
    best_model_save_path = config_dict.get("best_model_save_path", BEST_MODEL_SAVE_PATH)
    last_model_save_path = config_dict.get("last_model_save_path", LAST_MODEL_SAVE_PATH)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=scheduler_min_lr,
    )
    print(f"\n[训练配置]")
    print(f"  损失函数: CrossEntropyLoss (序列展平后计算)")
    print(f"  优化器:   Adam (lr={learning_rate}, weight_decay={weight_decay})")
    print(f"  最大轮数: {num_epochs}")
    print(f"  梯度裁剪: {grad_clip}")
    print(f"  学习率调度: ReduceLROnPlateau (factor={scheduler_factor}, patience={scheduler_patience}, min_lr={scheduler_min_lr})")
    print(f"  提前停止: 连续 {early_stopping_patience} 轮验证集无提升则停止")
    print(f"  模型选择策略: 根据 val ppl 选择最佳模型，PPL 越低越好")

    history = {
        "train_loss": [],
        "train_ppl": [],
        "val_loss": [],
        "val_ppl": [],
        "lr": [],
        "grad_norm": [],
    }

    best_val_ppl = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    print(f"\n{'Epoch':>6s}  {'LR':>10s}  {'Train Loss':>12s}  {'Train PPL':>12s}  {'Val Loss':>12s}  {'Val PPL':>12s}")
    print(f"{'------':>6s}  {'--------':>10s}  {'----------':>12s}  {'----------':>12s}  {'--------':>12s}  {'--------':>12s}")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_ppl, avg_grad_norm = train_one_epoch(model, train_loader, optimizer, device, epoch, grad_clip, log_every_n_batches)
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
            save_checkpoint(best_model_save_path, model, optimizer, epoch, history, best_val_ppl, stoi, itos, config_dict)
        else:
            epochs_without_improvement += 1

        print(f"{epoch:>6d}  {current_lr:>10.6f}  {train_loss:>12.6f}  {train_ppl:>12.4f}  {val_loss:>12.6f}  {val_ppl:>12.4f}")

        scheduler.step(val_ppl)

        if epochs_without_improvement >= early_stopping_patience:
            print(f"\n[训练] 验证集 PPL 已连续 {epochs_without_improvement} 轮没有提升，提前停止训练")
            break

    print(f"{'------':>6s}  {'--------':>10s}  {'----------':>12s}  {'----------':>12s}  {'--------':>12s}  {'--------':>12s}")

    assert best_state is not None, "[fit] best_state 为空，说明训练过程未正确更新最佳模型!"

    finished_epochs = len(history["train_loss"])
    save_checkpoint(last_model_save_path, model, optimizer, finished_epochs, history, best_val_ppl, stoi, itos, config_dict)

    model.load_state_dict(best_state)

    print(f"\n[训练] 完成! epochs={finished_epochs} best_epoch={best_epoch} best_val_ppl={best_val_ppl:.4f}")

    return history, best_epoch, best_val_ppl, best_state

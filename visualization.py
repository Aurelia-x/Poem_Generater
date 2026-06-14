import logging
import os
import warnings

import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["mathtext.default"] = "regular"
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

from config import BEST_MODEL_SAVE_PATH, OUTPUT_DIR, RANDOM_SEED
from utils import ensure_dir, set_seed

logging.getLogger("matplotlib").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*glyph.*")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

set_seed(RANDOM_SEED)

def format_sci_tick(value):
    if value is None:
        return ""
    if not np.isfinite(value):
        return ""
    if value == 0:
        return "0"
    text = f"{value:.0e}"
    text = text.replace("e+0", "e+").replace("e-0", "e-")
    return text

def set_discrete_log_ticks(ax, values):
    if not values:
        return
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    if arr.size == 0:
        return
    unique = np.unique(np.round(arr, 12))
    unique = np.sort(unique)
    ax.set_yticks(unique.tolist())
    ax.set_ylim(unique.min() * 0.9, unique.max() * 1.1)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: format_sci_tick(y)))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())


def load_history(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    history = checkpoint["history"]
    best_epoch = checkpoint.get("epoch", len(history["train_loss"]))
    best_val_ppl = checkpoint.get("best_val_ppl", float("inf"))
    return history, best_epoch, best_val_ppl


def plot_loss_ppl_curve(history, best_epoch, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_train = "#2C7BB6"
    color_val = "#D7191C"
    ax1.plot(epochs, history["train_loss"], color=color_train, marker="o", markersize=4,
             linewidth=1.5, label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   color=color_val,   marker="s", markersize=4,
             linewidth=1.5, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color_train_ppl = "#5AAE61"
    color_val_ppl = "#FAA43A"
    ax2.plot(epochs, history["train_ppl"], color=color_train_ppl, marker="^", markersize=4,
             linewidth=1.5, linestyle="--", label="Train PPL")
    ax2.plot(epochs, history["val_ppl"],   color=color_val_ppl,   marker="v", markersize=4,
             linewidth=1.5, linestyle="--", label="Val PPL")
    ax2.set_ylabel("PPL")
    ax2.legend(loc="upper right")

    if best_epoch in epochs:
        ax1.axvline(x=best_epoch, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
        ax1.text(best_epoch + 0.1, ax1.get_ylim()[1] * 0.95, f"Best Epoch={best_epoch}",
                 fontsize=9, color="gray")

    plt.title("Training & Validation Loss / PPL")
    fig.tight_layout()

    save_path = os.path.join(save_dir, "fig_loss_ppl.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表] 已保存: {save_path}")


def plot_lr_curve(history, best_epoch, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(epochs, history["train_loss"], color="#2C7BB6", marker="o", markersize=4,
             linewidth=1.5, label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   color="#D7191C", marker="s", markersize=4,
             linewidth=1.5, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    lr_values = history.get("lr", [])
    if lr_values:
        ax2.step(epochs, lr_values, color="#762A83", linewidth=2, where="post", label="Learning Rate")
        ax2.set_ylabel("Learning Rate")
        ax2.set_yscale("log")
        set_discrete_log_ticks(ax2, lr_values)
        ax2.legend(loc="upper right")

    if best_epoch in epochs:
        ax1.axvline(x=best_epoch, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)

    plt.title("Loss & Learning Rate Decay")
    fig.tight_layout()

    save_path = os.path.join(save_dir, "fig_lr_decay.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表] 已保存: {save_path}")


def plot_grad_norm(history, save_dir):
    grad_norms = history.get("grad_norm", [])
    if not grad_norms:
        print("[图表] history 中无 grad_norm 数据，跳过梯度范数图")
        return

    epochs = range(1, len(grad_norms) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(epochs, grad_norms, color="#5AAE61", alpha=0.7, width=0.6)
    ax.axhline(y=1.0, color="#D7191C", linestyle="--", linewidth=1.2, alpha=0.8, label="Grad Clip Threshold=1.0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Gradient Norm")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.title("Average Gradient Norm per Epoch")
    fig.tight_layout()

    save_path = os.path.join(save_dir, "fig_grad_norm.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表] 已保存: {save_path}")


def plot_combined_dashboard(history, best_epoch, save_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    n = len(epochs)
    lr_values = history.get("lr", [])
    grad_norms = history.get("grad_norm", [])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def add_best_line(ax):
        if best_epoch in epochs:
            ax.axvline(x=best_epoch, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)

    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], color="#2C7BB6", marker="o", markersize=3, linewidth=1.5, label="Train Loss")
    ax.plot(epochs, history["val_loss"],   color="#D7191C", marker="s", markersize=3, linewidth=1.5, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    add_best_line(ax)

    ax = axes[0, 1]
    ax.plot(epochs, history["train_ppl"], color="#5AAE61", marker="^", markersize=3, linewidth=1.5,
            linestyle="--", label="Train PPL")
    ax.plot(epochs, history["val_ppl"],   color="#FAA43A", marker="v", markersize=3, linewidth=1.5,
            linestyle="--", label="Val PPL")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PPL")
    ax.set_title("Perplexity Curve")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    add_best_line(ax)

    ax = axes[1, 0]
    if lr_values:
        ax.step(epochs, lr_values, color="#762A83", linewidth=2, where="post")
        ax.set_yscale("log")
        set_discrete_log_ticks(ax, lr_values)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate (log scale)")
    ax.set_title("Learning Rate Decay")
    ax.grid(True, alpha=0.3)
    add_best_line(ax)

    ax = axes[1, 1]
    if grad_norms:
        ax.bar(epochs, grad_norms, color="#5AAE61", alpha=0.7, width=0.6)
        ax.axhline(y=1.0, color="#D7191C", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Average Gradient Norm")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Training Dashboard (Best Epoch = {best_epoch})", fontsize=14, fontweight="bold")
    fig.tight_layout()

    save_path = os.path.join(save_dir, "fig_dashboard.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[图表] 已保存: {save_path}")


def main():
    checkpoint_path = BEST_MODEL_SAVE_PATH
    if not os.path.exists(checkpoint_path):
        print(f"[错误] checkpoint 不存在: {checkpoint_path}")
        print("[提示] 请先运行 train.py 完成训练")
        return

    charts_dir = os.path.join(OUTPUT_DIR, "charts")
    ensure_dir(charts_dir)

    history, best_epoch, best_val_ppl = load_history(checkpoint_path)
    n_epochs = len(history["train_loss"])
    print(f"[数据] 加载 checkpoint: {checkpoint_path}")
    print(f"[数据] 共 {n_epochs} 个 epoch, best_epoch={best_epoch}, best_val_ppl={best_val_ppl:.4f}")

    plot_loss_ppl_curve(history, best_epoch, charts_dir)
    plot_lr_curve(history, best_epoch, charts_dir)
    plot_grad_norm(history, charts_dir)
    plot_combined_dashboard(history, best_epoch, charts_dir)

    print(f"\n[完成] 共生成 4 张图表，保存在 {charts_dir}")


if __name__ == "__main__":
    main()

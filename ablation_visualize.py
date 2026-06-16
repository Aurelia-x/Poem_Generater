"""消融实验可视化 — 从 ablation_results.json 生成对比图表"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import ABLATION_DIR, OUTPUT_DIR

FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

EXPERIMENT_LABELS = {
    "ablation_a_baseline": "A: 基线\nLSTM-128",
    "ablation_b_deeper": "B: 加深\nLSTM-128×2",
    "ablation_c_wider": "C: 加宽\nLSTM-256",
    "ablation_d_regularized": "D: 正则化\nLSTM-128+WD",
    "ablation_e_full": "E: 全量\nLSTM-256×2",
    "ablation_f_gru": "F: GRU\nGRU-128",
}

EXPERIMENT_LABELS_SHORT = {
    "ablation_a_baseline": "A: 基线",
    "ablation_b_deeper": "B: 加深",
    "ablation_c_wider": "C: 加宽",
    "ablation_d_regularized": "D: 正则化",
    "ablation_e_full": "E: 全量",
    "ablation_f_gru": "F: GRU",
}


def load_results():
    path = os.path.join(ABLATION_DIR, "ablation_results.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================
#  图1: 各模型 Test PPL 柱状图
# =============================================
def plot_ppl(results):
    sorted_r = sorted(results, key=lambda r: r["test_ppl"])
    labels = [EXPERIMENT_LABELS[r["experiment_name"]] for r in sorted_r]
    ppls = [r["test_ppl"] for r in sorted_r]
    val_ppls = [r["best_val_ppl"] for r in sorted_r]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - w / 2, val_ppls, w, label="验证集 PPL", color="#E8C170", edgecolor="white")
    bars2 = ax.bar(x + w / 2, ppls, w, label="测试集 PPL", color="#4C72B0", edgecolor="white")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("PPL", fontsize=11)
    ax.set_title("图1: 消融实验 PPL 对比", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(ppls), max(val_ppls)) * 1.15)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ablation_ppl.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图1] {path}")


# =============================================
#  图2: 多样性对比
# =============================================
def plot_diversity(results):
    sorted_r = sorted(results, key=lambda r: r["test_ppl"])
    labels = [EXPERIMENT_LABELS[r["experiment_name"]] for r in sorted_r]
    fl_dens = [r["fl_inter_dens"] for r in sorted_r]
    ac_dens = [r["ac_inter_dens"] for r in sorted_r]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - w / 2, fl_dens, w, label="首句续写", color="#4C72B0", edgecolor="white")
    ax.bar(x + w / 2, ac_dens, w, label="藏头诗", color="#DD8452", edgecolor="white")

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("跨样本 2-gram 密度", fontsize=11)
    ax.set_title("图2: 消融实验生成多样性对比", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0.9, max(max(fl_dens), max(ac_dens)) * 1.1)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ablation_diversity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图2] {path}")


# =============================================
#  图3: 参数量 vs PPL 散点图
# =============================================
def plot_params_vs_ppl(results):
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for i, r in enumerate(results):
        x = r["parameter_count"] / 1e6
        y = r["test_ppl"]
        label = EXPERIMENT_LABELS_SHORT[r["experiment_name"]]
        ax.scatter(x, y, c=[colors[i]], s=120, edgecolors="white", linewidth=0.8, zorder=5)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 4),
                    fontsize=8)

    ax.set_xlabel("参数量 (百万)", fontsize=11)
    ax.set_ylabel("Test PPL", fontsize=11)
    ax.set_title("图3: 参数量与 PPL 关系", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ablation_params.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图3] {path}")


# =============================================
#  图4: 训练/验证/测试 PPL 三线图
# =============================================
def plot_ppl_split(results):
    sorted_r = sorted(results, key=lambda r: r["test_ppl"])
    labels = [EXPERIMENT_LABELS_SHORT[r["experiment_name"]] for r in sorted_r]
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    train_ppl = [r["train_ppl"] for r in sorted_r]
    val_ppl = [r["val_ppl"] for r in sorted_r]
    test_ppl = [r["test_ppl"] for r in sorted_r]

    ax.bar(x - w, train_ppl, w, label="训练集 PPL", color="#55A868", edgecolor="white")
    ax.bar(x, val_ppl, w, label="验证集 PPL", color="#E8C170", edgecolor="white")
    ax.bar(x + w, test_ppl, w, label="测试集 PPL", color="#4C72B0", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("PPL", fontsize=11)
    ax.set_title("图4: 训练/验证/测试集 PPL 对比", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ablation_ppl_split.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图4] {path}")


# =============================================
#  图5: 模型架构配置总览
# =============================================
def plot_config_table(results):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.axis("off")

    columns = ["模型", "Embed", "Hidden", "层数", "Dropout", "参数量(M)", "Test PPL"]
    cell_text = []
    cell_colors = []
    for r in sorted(results, key=lambda r: r["test_ppl"]):
        name = EXPERIMENT_LABELS_SHORT[r["experiment_name"]]
        cell_text.append([
            name,
            str(r["embed_dim"]),
            str(r["hidden_dim"]),
            str(r["num_layers"]),
            f"{r['dropout']:.1f}",
            f"{r['parameter_count'] / 1e6:.2f}",
            f"{r['test_ppl']:.1f}",
        ])
        ppl_color = "#D4E6F1" if r["test_ppl"] < 180 else "#FDEBD0" if r["test_ppl"] < 210 else "#FADBD8"
        cell_colors.append(["#F8F9FA"] * 6 + [ppl_color])

    table = ax.table(
        cellText=cell_text, colLabels=columns,
        cellLoc="center", loc="center",
        cellColours=cell_colors,
        colColours=["#4C72B0"] * len(columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color="white", fontweight="bold")

    ax.set_title("表1: 消融实验模型配置与 PPL 总览", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ablation_config.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图5] {path}")


# =============================================
#  主入口
# =============================================
if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    results = load_results()
    print(f"[数据] {len(results)} 个消融模型\n")

    plot_ppl(results)
    plot_diversity(results)
    plot_params_vs_ppl(results)
    plot_ppl_split(results)
    plot_config_table(results)

    print(f"\n图表已保存至 {FIGURES_DIR}")

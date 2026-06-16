"""采样策略对比可视化 — 从 comparison_results.json 生成图表"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from config import METRICS_DIR, OUTPUT_DIR

FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# =============================================
#  加载数据
# =============================================
def load_results():
    path = os.path.join(METRICS_DIR, "comparison_results.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def config_label(r):
    t, k = r["temperature"], r["top_k"]
    return f"t={t:.1f}\nk={k}" if k > 0 else f"t={t:.1f}\n全表"


# =============================================
#  图1: 多样性对比 — 跨样本 2-gram 密度
# =============================================
def plot_diversity(results):
    labels = [config_label(r) for r in results]
    x = np.arange(len(labels))
    fl_dens = [r["fl_dens"] for r in results]
    ac_dens = [r["ac_dens"] for r in results]

    fig, ax = plt.subplots(figsize=(12, 5))
    w = 0.35
    bars1 = ax.bar(x - w / 2, fl_dens, w, label="首句续写", color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + w / 2, ac_dens, w, label="藏头诗", color="#DD8452", edgecolor="white")

    # 参考线 1.0 = 完美多样
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("跨样本 2-gram 密度 (↓ 越低越多样)", fontsize=11)
    ax.set_title("图1: 采样策略多样性对比", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0.9, max(max(fl_dens), max(ac_dens)) * 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "diversity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图1] {path}")


# =============================================
#  图2: 押韵率对比 — 无引导 vs 有引导
# =============================================
def plot_rhyme(results):
    labels = [config_label(r) for r in results]
    x = np.arange(len(labels))
    fl_free = [r["fl_rhyme_free24"] * 100 for r in results]
    fl_guided = [r["fl_rhyme_g24"] * 100 for r in results]
    ac_free = [r["ac_rhyme_free24"] * 100 for r in results]
    ac_guided = [r["ac_rhyme_g24"] * 100 for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # 左: 首句续写
    ax = axes[0]
    w = 0.35
    ax.bar(x - w / 2, fl_free, w, label="无引导", color="#E8C170", edgecolor="white")
    ax.bar(x + w / 2, fl_guided, w, label="有引导", color="#6BAED6", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("首句续写", fontsize=12)
    ax.set_ylabel("2-4句押韵率 (%)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 105)

    # 右: 藏头诗
    ax = axes[1]
    ax.bar(x - w / 2, ac_free, w, label="无引导", color="#E8C170", edgecolor="white")
    bars = ax.bar(x + w / 2, ac_guided, w, label="有引导", color="#6BAED6", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("藏头诗", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # 标注提升幅度
    for i in range(len(labels)):
        imp = fl_guided[i] - fl_free[i]
        axes[0].annotate(f"+{imp:.0f}%", (x[i], max(fl_free[i], fl_guided[i]) + 3),
                         ha="center", fontsize=7, color="#2171B5", fontweight="bold")
    for i in range(len(labels)):
        imp = ac_guided[i] - ac_free[i]
        axes[1].annotate(f"+{imp:.0f}%", (x[i], max(ac_free[i], ac_guided[i]) + 3),
                         ha="center", fontsize=7, color="#2171B5", fontweight="bold")

    fig.suptitle("图2: 韵律引导效果对比 (2-4句押韵)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "rhyme.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图2] {path}")


# =============================================
#  图3: 多样性 vs 押韵率 trade-off 散点图
# =============================================
def plot_tradeoff(results):
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))

    for i, r in enumerate(results):
        # x = 首句密度 (越低越多样, 所以取倒数或直接用), y = 引导后押韵率
        x = r["fl_dens"]
        y = (r["fl_rhyme_g24"] + r["ac_rhyme_g24"]) / 2 * 100
        label = config_label(r).replace("\n", " ")
        ax.scatter(x, y, c=[colors[i]], s=100, edgecolors="white", linewidth=0.8, zorder=5)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4),
                    fontsize=7, alpha=0.85)

    ax.set_xlabel("跨样本 2-gram 密度 (← 越多样)", fontsize=11)
    ax.set_ylabel("引导后平均押韵率 (%)", fontsize=11)
    ax.set_title("图3: 多样性与押韵率 Trade-off", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    # 标注理想区域 (左上角)
    ax.annotate("理想方向\n(低密度 + 高押韵)", xy=(0.05, 0.92), xycoords="axes fraction",
                fontsize=10, color="#2171B5", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#DEEBF7", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "tradeoff.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图3] {path}")


# =============================================
#  图4: 诗内重复度对比
# =============================================
def plot_repeat(results):
    labels = [config_label(r) for r in results]
    x = np.arange(len(labels))
    fl_rpt = [r["fl_rpt"] for r in results]
    ac_rpt = [r["ac_rpt"] for r in results]

    fig, ax = plt.subplots(figsize=(12, 5))
    w = 0.35
    ax.bar(x - w / 2, fl_rpt, w, label="首句续写", color="#4C72B0", edgecolor="white")
    ax.bar(x + w / 2, ac_rpt, w, label="藏头诗", color="#DD8452", edgecolor="white")

    for bar, val in zip(ax.patches, fl_rpt + ac_rpt):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("诗内 2-gram 重复率 % (↓ 越低越好)", fontsize=11)
    ax.set_title("图4: 诗内重复度对比", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "repeat.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[图4] {path}")


# =============================================
#  主入口
# =============================================
if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    data = load_results()
    results = data["results"]
    print(f"[数据] 测试PPL={data['test_ppl']:.1f} | {len(results)} 组配置 | 每组{data['num_samples']}样本\n")

    plot_diversity(results)
    plot_rhyme(results)
    plot_tradeoff(results)
    plot_repeat(results)

    print(f"\n图表已保存至 {FIGURES_DIR}")

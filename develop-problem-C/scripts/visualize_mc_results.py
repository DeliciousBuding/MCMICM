"""
蒙特卡洛结果可视化（高信息密度版）。
输出论文用 PDF 图表与 LaTeX 汇总表。
"""
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 论文配色（严格遵循指定色板）
PALETTE = {
    "light_blue": "#90C9E7",
    "cyan_blue": "#219EBC",
    "deep_cyan": "#136783",
    "navy": "#02304A",
    "yellow": "#FEB705",
    "orange": "#FF9E02",
    "dark_orange": "#FA8600",
}

# 全局绘图风格（简洁 + 高对比）
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.edgecolor": PALETTE["navy"],
        "axes.labelcolor": PALETTE["navy"],
        "text.color": PALETTE["navy"],
    }
)


def _make_cmap():
    return LinearSegmentedColormap.from_list(
        "dwts_cmap",
        [PALETTE["light_blue"], PALETTE["cyan_blue"], PALETTE["deep_cyan"]],
    )


def load_results(csv_path: Optional[str] = None) -> pd.DataFrame:
    """读取蒙特卡洛结果"""
    if csv_path is None:
        csv_path = OUTPUT_DIR / "mc_robustness_results.csv"
    return pd.read_csv(csv_path)


def _style_axes(ax):
    """统一坐标轴风格"""
    ax.grid(True, alpha=0.2, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_probability_distribution(df: pd.DataFrame, output_path: Optional[str] = None):
    """P(Wrongful) 分布：直方图 + KDE + 累积分布"""
    if output_path is None:
        output_path = FIGURES_DIR / "mc_probability_distribution.pdf"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax1 = axes[0]
    ax1.hist(
        df["p_wrongful"],
        bins=30,
        density=True,
        alpha=0.8,
        color=PALETTE["cyan_blue"],
        edgecolor=PALETTE["navy"],
        linewidth=0.6,
    )
    from scipy import stats

    kde = stats.gaussian_kde(df["p_wrongful"])
    x_range = np.linspace(0, 1, 200)
    ax1.plot(x_range, kde(x_range), color=PALETTE["dark_orange"], linewidth=2, label="KDE")

    mean_val = df["p_wrongful"].mean()
    ax1.axvline(0.5, color=PALETTE["navy"], linestyle="--", alpha=0.5, label="50%")
    ax1.axvline(mean_val, color=PALETTE["orange"], linestyle="-", linewidth=2, label=f"Mean={mean_val:.3f}")
    ax1.set_xlabel("P(Wrongful Elimination)")
    ax1.set_ylabel("Density")
    ax1.set_title("Overall Distribution")
    ax1.legend(frameon=False)
    _style_axes(ax1)

    ax2 = axes[1]
    sorted_probs = np.sort(df["p_wrongful"])
    cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
    ax2.plot(sorted_probs, cumulative, color=PALETTE["deep_cyan"], linewidth=2)
    ax2.axvline(0.5, color=PALETTE["navy"], linestyle="--", alpha=0.5)
    ax2.axhline(0.5, color=PALETTE["navy"], linestyle="--", alpha=0.5)
    for p in [0.25, 0.5, 0.75]:
        val = np.percentile(df["p_wrongful"], p * 100)
        ax2.plot(val, p, "o", color=PALETTE["orange"], markersize=6)
        ax2.text(val + 0.02, p, f"{val:.2f}", fontsize=9, va="center")
    ax2.set_xlabel("P(Wrongful Elimination)")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("Cumulative Distribution")
    _style_axes(ax2)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"OK Saved: {output_path}")
    plt.close()


def plot_season_evolution(df: pd.DataFrame, output_path: Optional[str] = None):
    """赛季演化：均值 + 95% CI + 样本量"""
    if output_path is None:
        output_path = FIGURES_DIR / "mc_season_evolution.pdf"

    season_stats = (
        df.groupby("season")
        .agg(
            mean=("p_wrongful", "mean"),
            std=("p_wrongful", "std"),
            count=("p_wrongful", "count"),
            median=("p_wrongful", "median"),
        )
        .reset_index()
    )

    seasons = season_stats["season"]
    means = season_stats["mean"]
    stds = season_stats["std"]
    counts = season_stats["count"]
    ci = 1.96 * stds / np.sqrt(counts)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [2, 1]})

    ax1 = axes[0]
    ax1.plot(seasons, means, "o-", color=PALETTE["cyan_blue"], linewidth=2, markersize=5)
    ax1.fill_between(seasons, means - ci, means + ci, color=PALETTE["light_blue"], alpha=0.5)
    ax1.axhline(0.5, color=PALETTE["navy"], linestyle="--", alpha=0.5)

    # 规则变更标注
    for season, label in {3: "Percent Start", 28: "Judge Save", 32: "Mismatch"}.items():
        if season in seasons.values:
            ax1.axvline(season, color=PALETTE["dark_orange"], linestyle=":", alpha=0.6)
            ax1.text(season, ax1.get_ylim()[1] * 0.95, label, ha="center", fontsize=8)

    ax1.set_xlabel("Season")
    ax1.set_ylabel("Mean P(Wrongful)")
    ax1.set_title("Season Evolution with 95% CI")
    _style_axes(ax1)

    ax2 = axes[1]
    ax2.bar(seasons, counts, color=PALETTE["deep_cyan"], edgecolor=PALETTE["navy"], linewidth=0.5)
    ax2.set_xlabel("Season")
    ax2.set_ylabel("Eliminations")
    ax2.set_title("Sample Size per Season")
    _style_axes(ax2)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"OK Saved: {output_path}")
    plt.close()


def plot_confidence_intervals(df: pd.DataFrame, top_n: int = 20, output_path: Optional[str] = None):
    """Top N 案例置信区间"""
    if output_path is None:
        output_path = FIGURES_DIR / "mc_confidence_intervals.pdf"

    top_cases = df.nlargest(top_n, "p_wrongful").copy()
    top_cases["label"] = (
        top_cases["contestant"].str[:15]
        + " (S"
        + top_cases["season"].astype(str)
        + "W"
        + top_cases["week"].astype(str)
        + ")"
    )

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.38)))
    y_pos = np.arange(len(top_cases))

    for i, (_, row) in enumerate(top_cases.iterrows()):
        color = PALETTE["dark_orange"] if row["p_wrongful"] > 0.9 else PALETTE["cyan_blue"]
        ax.errorbar(
            row["p_wrongful"],
            i,
            xerr=[[row["p_wrongful"] - row["ci_lower"]], [row["ci_upper"] - row["p_wrongful"]]],
            fmt="o",
            markersize=6,
            capsize=4,
            capthick=1.5,
            color=color,
            ecolor=color,
            alpha=0.8,
        )
        ax.text(row["ci_upper"] + 0.02, i, f"{row['p_wrongful']:.3f}", fontsize=8, va="center")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_cases["label"], fontsize=8)
    ax.set_xlabel("P(Wrongful Elimination)")
    ax.set_title(f"Top {top_n} Cases with 95% CI")
    ax.axvline(0.5, color=PALETTE["navy"], linestyle="--", alpha=0.5)
    ax.set_xlim([0, 1.05])
    _style_axes(ax)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"OK Saved: {output_path}")
    plt.close()


def plot_voting_method_comparison(df: pd.DataFrame, output_path: Optional[str] = None):
    """投票机制对比：箱线 + 小提琴"""
    if output_path is None:
        output_path = FIGURES_DIR / "mc_voting_method_comparison.pdf"

    df_plot = df[df["voting_method"].isin(["percent", "rank"])]
    box_data = [
        df_plot[df_plot["voting_method"] == "percent"]["p_wrongful"],
        df_plot[df_plot["voting_method"] == "rank"]["p_wrongful"],
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1 = axes[0]
    bp = ax1.boxplot(
        box_data,
        tick_labels=["Percent (S3-27)", "Rank (S1-2, S28+)"],
        patch_artist=True,
        widths=0.6,
    )
    for patch, color in zip(bp["boxes"], [PALETTE["light_blue"], PALETTE["orange"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax1.set_ylabel("P(Wrongful)")
    ax1.set_title("Boxplot by Voting Method")
    ax1.axhline(0.5, color=PALETTE["navy"], linestyle="--", alpha=0.5)
    _style_axes(ax1)

    ax2 = axes[1]
    parts = ax2.violinplot(box_data, positions=[0, 1], showmeans=True, showmedians=True, widths=0.7)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor([PALETTE["light_blue"], PALETTE["orange"]][i])
        pc.set_alpha(0.8)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Percent", "Rank"])
    ax2.set_ylabel("P(Wrongful)")
    ax2.set_title("Violin Density")
    ax2.axhline(0.5, color=PALETTE["navy"], linestyle="--", alpha=0.5)
    _style_axes(ax2)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"OK Saved: {output_path}")
    plt.close()


def plot_classification_breakdown(df: pd.DataFrame, output_path: Optional[str] = None):
    """分类分布：饼图 + 赛季堆叠"""
    if output_path is None:
        output_path = FIGURES_DIR / "mc_classification_breakdown.pdf"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1 = axes[0]
    class_counts = df["classification"].value_counts()
    colors = {
        "Definite-Wrongful": PALETTE["dark_orange"],
        "Uncertain": PALETTE["yellow"],
        "Definite-Correct": PALETTE["deep_cyan"],
    }
    wedges, texts, autotexts = ax1.pie(
        class_counts.values,
        labels=class_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=[colors.get(c, PALETTE["light_blue"]) for c in class_counts.index],
    )
    for text in texts:
        text.set_fontsize(9)
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(9)
        autotext.set_weight("bold")
    ax1.set_title("Classification Distribution")

    ax2 = axes[1]
    pivot = pd.crosstab(df["season"], df["classification"], normalize="index") * 100
    col_order = ["Definite-Correct", "Uncertain", "Definite-Wrongful"]
    pivot = pivot[[c for c in col_order if c in pivot.columns]]
    pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax2,
        color=[colors.get(c, PALETTE["light_blue"]) for c in pivot.columns],
        width=0.8,
        edgecolor=PALETTE["navy"],
        linewidth=0.3,
    )
    ax2.set_xlabel("Season")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Classification by Season")
    ax2.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
    _style_axes(ax2)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"OK Saved: {output_path}")
    plt.close()


def plot_interval_width_analysis(df: pd.DataFrame, output_path: Optional[str] = None):
    """区间宽度分析：相关性 + 分组对比"""
    if output_path is None:
        output_path = FIGURES_DIR / "mc_interval_width_analysis.pdf"

    df = df.copy()
    df["interval_width"] = df["fan_vote_upper"] - df["fan_vote_lower"]
    df["ci_width"] = df["ci_upper"] - df["ci_lower"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax1 = axes[0, 0]
    cmap = _make_cmap()
    scatter = ax1.scatter(
        df["interval_width"],
        df["p_wrongful"],
        c=df["season"],
        cmap=cmap,
        alpha=0.7,
        s=20,
        edgecolors=PALETTE["navy"],
        linewidth=0.2,
    )
    ax1.set_xlabel("Interval Width")
    ax1.set_ylabel("P(Wrongful)")
    ax1.set_title("Width vs P(W)")
    _style_axes(ax1)
    plt.colorbar(scatter, ax=ax1, label="Season")

    ax2 = axes[0, 1]
    ax2.hist(
        df["ci_width"],
        bins=30,
        color=PALETTE["cyan_blue"],
        edgecolor=PALETTE["navy"],
        alpha=0.8,
    )
    ax2.axvline(df["ci_width"].mean(), color=PALETTE["orange"], linestyle="--", linewidth=2)
    ax2.set_xlabel("CI Width")
    ax2.set_ylabel("Count")
    ax2.set_title("CI Width Distribution")
    _style_axes(ax2)

    ax3 = axes[1, 0]
    season_widths = df.groupby("season")["interval_width"].mean()
    ax3.bar(
        season_widths.index,
        season_widths.values,
        color=PALETTE["deep_cyan"],
        edgecolor=PALETTE["navy"],
        linewidth=0.4,
    )
    ax3.set_xlabel("Season")
    ax3.set_ylabel("Mean Width")
    ax3.set_title("Interval Width by Season")
    _style_axes(ax3)

    ax4 = axes[1, 1]
    df["width_bin"] = pd.qcut(
        df["interval_width"],
        q=4,
        labels=["Narrow", "Mid-Narrow", "Mid-Wide", "Wide"],
    )
    box_data = [df[df["width_bin"] == cat]["p_wrongful"] for cat in df["width_bin"].cat.categories]
    bp = ax4.boxplot(box_data, tick_labels=df["width_bin"].cat.categories, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE["light_blue"])
        patch.set_alpha(0.85)
    ax4.set_xlabel("Interval Width Bin")
    ax4.set_ylabel("P(Wrongful)")
    ax4.set_title("P(W) by Width Bin")
    ax4.axhline(0.5, color=PALETTE["navy"], linestyle="--", alpha=0.5)
    _style_axes(ax4)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"OK Saved: {output_path}")
    plt.close()


def generate_summary_table(df: pd.DataFrame, output_path: Optional[str] = None):
    """生成 LaTeX 汇总表"""
    if output_path is None:
        output_path = OUTPUT_DIR / "mc_summary_statistics.tex"

    summary_data = [
        {
            "Category": "Overall",
            "N": len(df),
            "Mean P(W)": f"{df['p_wrongful'].mean():.3f}",
            "Median P(W)": f"{df['p_wrongful'].median():.3f}",
            "Std P(W)": f"{df['p_wrongful'].std():.3f}",
            "Mean CI Width": f"{(df['ci_upper'] - df['ci_lower']).mean():.3f}",
        }
    ]

    for method in df["voting_method"].unique():
        subset = df[df["voting_method"] == method]
        summary_data.append(
            {
                "Category": f"Method: {method}",
                "N": len(subset),
                "Mean P(W)": f"{subset['p_wrongful'].mean():.3f}",
                "Median P(W)": f"{subset['p_wrongful'].median():.3f}",
                "Std P(W)": f"{subset['p_wrongful'].std():.3f}",
                "Mean CI Width": f"{(subset['ci_upper'] - subset['ci_lower']).mean():.3f}",
            }
        )

    for cls in df["classification"].unique():
        subset = df[df["classification"] == cls]
        summary_data.append(
            {
                "Category": f"Class: {cls}",
                "N": len(subset),
                "Mean P(W)": f"{subset['p_wrongful'].mean():.3f}",
                "Median P(W)": f"{subset['p_wrongful'].median():.3f}",
                "Std P(W)": f"{subset['p_wrongful'].std():.3f}",
                "Mean CI Width": f"{(subset['ci_upper'] - subset['ci_lower']).mean():.3f}",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    latex_table = summary_df.to_latex(
        index=False,
        caption="Monte Carlo Robustness Analysis Summary Statistics",
        label="tab:mc_summary",
        column_format="lrrrrrr",
        escape=False,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print(f"OK Saved: {output_path}")
    return summary_df


def main():
    """主入口：生成全部图表与汇总表"""
    print("\n" + "=" * 60)
    print("MONTE CARLO RESULTS VISUALIZATION")
    print("=" * 60 + "\n")

    print("加载结果...")
    df = load_results()
    print(f"Loaded {len(df)} eliminations from {df['season'].nunique()} seasons\n")

    print("生成可视化...\n")
    plot_probability_distribution(df)
    plot_season_evolution(df)
    plot_confidence_intervals(df, top_n=20)
    plot_voting_method_comparison(df)
    plot_classification_breakdown(df)
    plot_interval_width_analysis(df)

    print("\n生成汇总表...")
    summary_df = generate_summary_table(df)
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("OK 所有图表生成完成")
    print("=" * 60)
    print("\n输出目录:")
    print(f"  - Figures: {FIGURES_DIR}/")
    print(f"  - Summary: {OUTPUT_DIR / 'mc_summary_statistics.tex'}")
    print(f"  - Raw data: {OUTPUT_DIR / 'mc_robustness_results.csv'}")


if __name__ == "__main__":
    main()

"""
DWTS 论文统一配色方案。

设计原则：
1. 主视觉仅使用 3 个高饱和色承担主要角色（baseline / proposed / warning）。
2. 其他信息通过浅色 + 透明度 + 线型表达。
3. 黄色仅用于小标注、箭头与关键数字点。
"""

# ============================================================
# 核心配色映射
# ============================================================

PALETTE = {
    # 主色（新机制/推荐方案）
    "proposed": "#219EBC",

    # 基准色（现行机制）
    "baseline": "#02304A",

    # 警示色（不一致/失败）
    "warning": "#FA8600",
    "warning2": "#FF9E02",

    # 填充色（区间/误差带）
    "fill": "#90C9E7",

    # 标注色（小标签/强调）
    "accent": "#FEB705",

    # 辅助色（次要序列）
    "aux": "#136783",
}

# ============================================================
# 语义化快捷访问
# ============================================================

VOTING_METHODS = {
    "percent": PALETTE["proposed"],
    "rank": PALETTE["baseline"],
}

MECHANISMS = {
    "current": PALETTE["baseline"],
    "proposed": PALETTE["proposed"],
    "soft_floor": PALETTE["warning"],
}

DATA_STATES = {
    "match": PALETTE["proposed"],
    "mismatch": PALETTE["warning"],
    "uncertain": PALETTE["fill"],
}

LEGEND_LABELS = {
    "percent": "Percent Method (S3-27)",
    "rank": "Rank Method (S1-2, S28+)",
    "mismatch": "Mismatch Detected",
    "proposed": "Proposed (Weighted %)",
    "current": "Current",
}

# ============================================================
# 绘图样式设置
# ============================================================

LINE_STYLES = {
    "proposed": {"color": PALETTE["proposed"], "linewidth": 2.5, "linestyle": "-"},
    "baseline": {"color": PALETTE["baseline"], "linewidth": 2.0, "linestyle": "--"},
    "warning": {"color": PALETTE["warning"], "linewidth": 2.0, "linestyle": "-"},
    "aux": {"color": PALETTE["aux"], "linewidth": 1.5, "linestyle": "-."},
}

BAR_STYLES = {
    "proposed": {"color": PALETTE["proposed"], "edgecolor": PALETTE["aux"], "linewidth": 0.8},
    "baseline": {"color": PALETTE["baseline"], "edgecolor": PALETTE["aux"], "linewidth": 0.8},
    "warning": {"color": PALETTE["warning"], "edgecolor": "#8B4513", "linewidth": 0.8},
}

FILL_STYLES = {
    "default": {"color": PALETTE["fill"], "alpha": 0.30},
    "light": {"color": PALETTE["fill"], "alpha": 0.20},
    "strong": {"color": PALETTE["fill"], "alpha": 0.45},
}

# ============================================================
# 工具函数
# ============================================================


def get_season_color(season: int) -> str:
    """根据赛季返回对应颜色"""
    if season <= 2 or season >= 28:
        return PALETTE["baseline"]
    return PALETTE["proposed"]


def get_season_colors(seasons: list) -> list:
    """批量获取赛季颜色"""
    return [get_season_color(s) for s in seasons]


def get_method_color(method: str) -> str:
    """根据投票方法返回颜色"""
    if method.lower() in ["rank", "ranking"]:
        return PALETTE["baseline"]
    if method.lower() in ["percent", "percentage"]:
        return PALETTE["proposed"]
    return PALETTE["aux"]


def apply_paper_style(ax, grid_alpha: float = 0.3):
    """为 matplotlib 轴应用论文风格"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(colors="#333333", labelsize=10)
    ax.grid(True, alpha=grid_alpha, linestyle="--", linewidth=0.5)


def create_legend_patches():
    """创建标准图例 patches"""
    import matplotlib.patches as mpatches

    patches = {
        "rank": mpatches.Patch(color=PALETTE["baseline"], label=LEGEND_LABELS["rank"]),
        "percent": mpatches.Patch(color=PALETTE["proposed"], label=LEGEND_LABELS["percent"]),
        "mismatch": mpatches.Patch(color=PALETTE["warning"], label=LEGEND_LABELS["mismatch"]),
    }
    return patches


# ============================================================
# 论文盒式颜色定义（LaTeX）
# ============================================================

LATEX_COLORS = """
% DWTS Paper Palette - LaTeX Color Definitions
% Add to main.tex preamble

\definecolor{dwts-proposed}{HTML}{219EBC}   % 青蓝 - 新机制
\definecolor{dwts-baseline}{HTML}{02304A}   % 藏蓝 - 基准
\definecolor{dwts-warning}{HTML}{FA8600}    % 深橙 - 警示
\definecolor{dwts-warning2}{HTML}{FF9E02}   % 亮橙 - 次级警示
\definecolor{dwts-fill}{HTML}{90C9E7}       % 浅蓝 - 填充
\definecolor{dwts-accent}{HTML}{FEB705}     % 黄色 - 标注
\definecolor{dwts-aux}{HTML}{136783}        % 深青 - 辅助

% tcolorbox 预设样式
\newtcolorbox{proposedbox}[1][] {
    colback=dwts-proposed!10!white,
    colframe=dwts-proposed!80!black,
    #1
}
\newtcolorbox{warningbox}[1][] {
    colback=dwts-warning!10!white,
    colframe=dwts-warning!80!black,
    #1
}
\newtcolorbox{baselinebox}[1][] {
    colback=dwts-baseline!5!white,
    colframe=dwts-baseline!70!black,
    #1
}
"""


if __name__ == "__main__":
    print("=" * 60)
    print("DWTS Paper Palette - 论文级配色方案")
    print("=" * 60)
    print()

    for name, color in PALETTE.items():
        print(f"  {name:12s}  {color}  {'█' * 10}")

    print()
    print("LaTeX 定义：")
    print(LATEX_COLORS)

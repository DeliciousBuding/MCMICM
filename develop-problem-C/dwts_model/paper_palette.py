"""
DWTS Paper Palette - 论文级统一配色方案

遵循原则：
1. 全文只用 3 个高饱和色担任角色（baseline / proposed / warning）
2. 其余信息用浅色 + 透明度 + 线型表达
3. 黄色只用于小标注、箭头、关键数字点
"""

# ============================================================
# 核心调色板 - 论文级映射规则
# ============================================================

PALETTE = {
    # 主色 - Proposed / Weighted Percent / New
    "proposed": "#219EBC",   # 青蓝 - 新机制/推荐方案
    
    # 对比色 - Current / Baseline / Original
    "baseline": "#02304A",   # 藏蓝 - 原机制/基准
    
    # 警示色 - Mismatch / Fail / Infeasible
    "warning":  "#FA8600",   # 深橙 - 警示/失败/不匹配
    "warning2": "#FF9E02",   # 亮橙 - 次级警示（更轻）
    
    # 填充色 - 误差带/区间带/背景强调
    "fill":     "#90C9E7",   # 浅蓝 - 配合 alpha 0.25-0.35 使用
    
    # 标注色 - 关键数字/箭头/小点
    "accent":   "#FEB705",   # 黄色 - 仅用于小标注，不做主色
    
    # 辅助色 - 备用第三条线/次要系列
    "aux":      "#136783",   # 深青 - 辅助数据
}

# ============================================================
# 语义化快捷访问
# ============================================================

# 投票方法配色
VOTING_METHODS = {
    "percent":  PALETTE["proposed"],   # 百分比制 = 推荐/新
    "rank":     PALETTE["baseline"],   # 排名制 = 基准/旧
}

# 机制对比配色
MECHANISMS = {
    "current":   PALETTE["baseline"],  # 现行机制
    "proposed":  PALETTE["proposed"],  # 建议机制
    "soft_floor": PALETTE["warning"],  # Soft floor / 警示
}

# 数据状态配色
DATA_STATES = {
    "match":     PALETTE["proposed"],  # 匹配/一致
    "mismatch":  PALETTE["warning"],   # 不匹配/冲突
    "uncertain": PALETTE["fill"],      # 不确定/区间
}

# 图例标签标准化
LEGEND_LABELS = {
    "percent": "Percent Method (S3-27)",
    "rank": "Rank Method (S1-2, S28+)",
    "mismatch": "Mismatch Detected",
    "proposed": "Proposed (Weighted %)",
    "current": "Current",
}

# ============================================================
# Matplotlib 样式设置
# ============================================================

# 默认线条样式
LINE_STYLES = {
    "proposed": {"color": PALETTE["proposed"], "linewidth": 2.5, "linestyle": "-"},
    "baseline": {"color": PALETTE["baseline"], "linewidth": 2.0, "linestyle": "--"},
    "warning":  {"color": PALETTE["warning"],  "linewidth": 2.0, "linestyle": "-"},
    "aux":      {"color": PALETTE["aux"],      "linewidth": 1.5, "linestyle": "-."},
}

# 柱状图样式
BAR_STYLES = {
    "proposed": {"color": PALETTE["proposed"], "edgecolor": PALETTE["aux"], "linewidth": 0.8},
    "baseline": {"color": PALETTE["baseline"], "edgecolor": PALETTE["aux"], "linewidth": 0.8},
    "warning":  {"color": PALETTE["warning"],  "edgecolor": "#8B4513",      "linewidth": 0.8},
}

# 填充区域样式
FILL_STYLES = {
    "default": {"color": PALETTE["fill"], "alpha": 0.30},
    "light":   {"color": PALETTE["fill"], "alpha": 0.20},
    "strong":  {"color": PALETTE["fill"], "alpha": 0.45},
}

# ============================================================
# 便捷函数
# ============================================================

def get_season_color(season: int) -> str:
    """根据赛季获取对应颜色"""
    if season <= 2 or season >= 28:
        return PALETTE["baseline"]  # Rank method
    else:
        return PALETTE["proposed"]  # Percent method


def get_season_colors(seasons: list) -> list:
    """批量获取赛季颜色列表"""
    return [get_season_color(s) for s in seasons]


def get_method_color(method: str) -> str:
    """根据投票方法获取颜色"""
    if method.lower() in ["rank", "ranking"]:
        return PALETTE["baseline"]
    elif method.lower() in ["percent", "percentage"]:
        return PALETTE["proposed"]
    else:
        return PALETTE["aux"]


def apply_paper_style(ax, grid_alpha: float = 0.3):
    """应用论文级样式到matplotlib axes"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    ax.tick_params(colors='#333333', labelsize=10)
    ax.grid(True, alpha=grid_alpha, linestyle='--', linewidth=0.5)


def create_legend_patches():
    """创建标准图例patches"""
    import matplotlib.patches as mpatches
    
    patches = {
        "rank": mpatches.Patch(color=PALETTE["baseline"], label=LEGEND_LABELS["rank"]),
        "percent": mpatches.Patch(color=PALETTE["proposed"], label=LEGEND_LABELS["percent"]),
        "mismatch": mpatches.Patch(color=PALETTE["warning"], label=LEGEND_LABELS["mismatch"]),
    }
    return patches


# ============================================================
# LaTeX tcolorbox 颜色对应表
# ============================================================

# 用于在论文中保持一致性
LATEX_COLORS = """
% DWTS Paper Palette - LaTeX Color Definitions
% Add to main.tex preamble

\\definecolor{dwts-proposed}{HTML}{219EBC}   % 青蓝 - 新机制
\\definecolor{dwts-baseline}{HTML}{02304A}   % 藏蓝 - 基准
\\definecolor{dwts-warning}{HTML}{FA8600}    % 深橙 - 警示
\\definecolor{dwts-warning2}{HTML}{FF9E02}   % 亮橙 - 次级警示
\\definecolor{dwts-fill}{HTML}{90C9E7}       % 浅蓝 - 填充
\\definecolor{dwts-accent}{HTML}{FEB705}     % 黄色 - 标注
\\definecolor{dwts-aux}{HTML}{136783}        % 深青 - 辅助

% tcolorbox 预设样式
\\newtcolorbox{proposedbox}[1][]{
    colback=dwts-proposed!10!white,
    colframe=dwts-proposed!80!black,
    #1
}
\\newtcolorbox{warningbox}[1][]{
    colback=dwts-warning!10!white,
    colframe=dwts-warning!80!black,
    #1
}
\\newtcolorbox{baselinebox}[1][]{
    colback=dwts-baseline!5!white,
    colframe=dwts-baseline!70!black,
    #1
}
"""


if __name__ == "__main__":
    # 打印调色板供参考
    print("=" * 60)
    print("DWTS Paper Palette - 论文级配色方案")
    print("=" * 60)
    print()
    
    for name, color in PALETTE.items():
        print(f"  {name:12s}  {color}  {'█' * 10}")
    
    print()
    print("LaTeX 定义：")
    print(LATEX_COLORS)

"""
机制设计与帕累托前沿分析。
以“评委一致性（技术公平）”与“观众一致性（参与度）”为双目标。
"""
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..paper_palette import PALETTE


@dataclass
class 机制设计结果:
    结果表: pd.DataFrame
    图像路径: str


def _机制淘汰(fan_votes, judge_pct, alpha: float) -> str:
    combined = {k: alpha * judge_pct.get(k, 0.0) + (1 - alpha) * fan_votes.get(k, 0.0) for k in fan_votes}
    return min(combined.items(), key=lambda x: x[1])[0]


def 运行帕累托优化(
    manager,
    posterior_summary: pd.DataFrame,
    alphas: List[float],
    fig_path: str,
    dynamic_alpha_fn: Callable[[int, int], float] = None,
):
    records = []
    grouped = posterior_summary.groupby(["season", "week"])

    for alpha in alphas:
        judge_align = []
        fan_align = []
        for (season, week), group in grouped:
            ctx = manager.get_season_context(int(season))
            week_ctx = ctx.weeks.get(int(week))
            if week_ctx is None or not week_ctx.has_valid_elimination():
                continue

            fan_votes = dict(zip(group["contestant"], group["fan_mean"]))
            judge_pct = week_ctx.judge_percentages

            fan_elim = min(fan_votes.items(), key=lambda x: x[1])[0]
            judge_elim = min(judge_pct.items(), key=lambda x: x[1])[0] if judge_pct else None
            use_alpha = dynamic_alpha_fn(week, ctx.num_weeks) if dynamic_alpha_fn else alpha
            mech_elim = _机制淘汰(fan_votes, judge_pct, use_alpha)

            if judge_elim:
                judge_align.append(1.0 if mech_elim == judge_elim else 0.0)
            fan_align.append(1.0 if mech_elim == fan_elim else 0.0)

        records.append(
            {
                "alpha": alpha,
                "judge_alignment": float(np.mean(judge_align)) if judge_align else np.nan,
                "fan_influence": float(np.mean(fan_align)) if fan_align else np.nan,
                "wrongful_rate": float(1.0 - np.mean(fan_align)) if fan_align else np.nan,
            }
        )

    df = pd.DataFrame(records)

    # 绘图
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(df["judge_alignment"], df["fan_influence"], "o-", color=PALETTE["proposed"])
    for _, row in df.iterrows():
        plt.text(row["judge_alignment"] + 0.002, row["fan_influence"], f"{row['alpha']:.2f}", fontsize=8)
    plt.xlabel("Judge Alignment (Fairness)")
    plt.ylabel("Fan Influence Index")
    plt.title("Pareto Frontier for Dynamic Weighting")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    return 机制设计结果(结果表=df, 图像路径=fig_path)

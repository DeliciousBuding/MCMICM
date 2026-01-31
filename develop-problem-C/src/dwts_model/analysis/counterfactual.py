"""
反事实评估：在固定粉丝意愿与评委分数下替换规则引擎。
"""
from dataclasses import dataclass
from typing import Dict, List, Callable

import pandas as pd

from .metrics import 计算肯德尔相关, 计算逆转率, 计算孔多塞一致性


def _计算淘汰_加权百分比(
    fan_votes: Dict[str, float],
    judge_percentages: Dict[str, float],
    alpha: float,
) -> str:
    combined = {}
    for name, f in fan_votes.items():
        j = judge_percentages.get(name, 0.0)
        combined[name] = alpha * j + (1 - alpha) * f
    return min(combined.items(), key=lambda x: x[1])[0]


def _计算排名(fan_votes: Dict[str, float]) -> Dict[str, float]:
    return fan_votes


def _计算分数_加权百分比(
    fan_votes: Dict[str, float],
    judge_percentages: Dict[str, float],
    alpha: float,
) -> Dict[str, float]:
    return {k: alpha * judge_percentages.get(k, 0.0) + (1 - alpha) * fan_votes.get(k, 0.0) for k in fan_votes}


@dataclass
class 反事实评估结果:
    逐周结果: pd.DataFrame
    汇总结果: pd.DataFrame


def 运行反事实评估(
    manager,
    posterior_summary: pd.DataFrame,
    alpha: float = 0.6,
    dynamic_alpha_fn: Callable[[int, int], float] = None,
) -> 反事实评估结果:
    """
    根据后验均值进行反事实规则评估。
    dynamic_alpha_fn: (week, total_weeks) -> alpha
    """
    records: List[dict] = []

    grouped = posterior_summary.groupby(["season", "week"])

    for (season, week), group in grouped:
        ctx = manager.get_season_context(int(season))
        week_ctx = ctx.weeks.get(int(week))
        if week_ctx is None or not week_ctx.has_valid_elimination():
            continue

        fan_votes = dict(zip(group["contestant"], group["fan_mean"]))
        judge_pct = week_ctx.judge_percentages

        observed = week_ctx.eliminated[0] if week_ctx.eliminated else None
        fan_elim = min(fan_votes.items(), key=lambda x: x[1])[0]

        alpha_dynamic = dynamic_alpha_fn(week, ctx.num_weeks) if dynamic_alpha_fn else alpha
        mech_elim = _计算淘汰_加权百分比(fan_votes, judge_pct, alpha_dynamic)

        tau = 计算肯德尔相关(fan_votes, _计算分数_加权百分比(fan_votes, judge_pct, alpha_dynamic))
        reversal = 计算逆转率(fan_elim, mech_elim)
        condorcet_ok = 计算孔多塞一致性(fan_votes, mech_elim)

        records.append(
            {
                "season": season,
                "week": week,
                "alpha": alpha_dynamic,
                "observed_elim": observed,
                "fan_elim": fan_elim,
                "mech_elim": mech_elim,
                "kendall_tau": tau,
                "reversal": reversal,
                "condorcet_ok": condorcet_ok,
            }
        )

    df = pd.DataFrame(records)
    summary = pd.DataFrame(
        {
            "kendall_tau_mean": [df["kendall_tau"].mean()],
            "reversal_rate": [df["reversal"].mean()],
            "condorcet_consistency": [df["condorcet_ok"].mean()],
        }
    )

    return 反事实评估结果(逐周结果=df, 汇总结果=summary)

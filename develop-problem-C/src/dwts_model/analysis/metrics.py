"""
评估指标：肯德尔相关、逆转率、孔多塞一致性。
"""
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.stats import kendalltau
except Exception:  # pragma: no cover
    kendalltau = None


def _排名列表(scores: Dict[str, float], reverse: bool = True) -> List[str]:
    return sorted(scores.keys(), key=lambda k: scores.get(k, 0.0), reverse=reverse)


def _排名向量(ranking: List[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(ranking)}
    return np.array([index[name] for name in ranking])


def 计算肯德尔相关(score_a: Dict[str, float], score_b: Dict[str, float]) -> float:
    """基于两组分数的排名计算 Kendall's Tau。"""
    if kendalltau is None:
        return float("nan")

    common = list(set(score_a.keys()) & set(score_b.keys()))
    if len(common) < 2:
        return float("nan")

    rank_a = _排名列表({k: score_a[k] for k in common}, reverse=True)
    rank_b = _排名列表({k: score_b[k] for k in common}, reverse=True)

    order_a = {name: i for i, name in enumerate(rank_a)}
    order_b = {name: i for i, name in enumerate(rank_b)}

    a = [order_a[name] for name in common]
    b = [order_b[name] for name in common]

    tau, _ = kendalltau(a, b)
    return float(tau) if tau is not None else float("nan")


def 计算逆转率(真实淘汰: str, 机制淘汰: str) -> float:
    """逆转率：机制淘汰是否与“观众意愿淘汰”不同。"""
    if not 真实淘汰 or not 机制淘汰:
        return float("nan")
    return 1.0 if 真实淘汰 != 机制淘汰 else 0.0


def 计算孔多塞一致性(fan_votes: Dict[str, float], eliminated: str) -> float:
    """
    孔多塞一致性：若被淘汰者是“粉丝票最高者”，则视为不一致。
    返回 1 表示一致（未淘汰孔多塞赢家），0 表示不一致。
    """
    if not fan_votes or eliminated is None:
        return float("nan")
    condorcet = max(fan_votes.items(), key=lambda x: x[1])[0]
    return 0.0 if eliminated == condorcet else 1.0

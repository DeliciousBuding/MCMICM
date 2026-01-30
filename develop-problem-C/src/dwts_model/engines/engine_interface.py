"""
反演引擎统一接口
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class FanVoteEstimate:
    """单个选手-周的投票估计"""

    contestant: str
    week: int
    point_estimate: float
    lower_bound: float
    upper_bound: float
    certainty: float
    method: str

    def get_interval_width(self) -> float:
        """返回区间宽度"""
        return self.upper_bound - self.lower_bound

    def contains_value(self, value: float) -> bool:
        """检查某值是否落在区间内"""
        return self.lower_bound <= value <= self.upper_bound


@dataclass
class InversionResult:
    """反演引擎输出结果"""

    season: int
    method: str  # percent / rank

    inconsistency_score: float
    is_feasible: bool

    week_results: Dict[int, Dict[str, FanVoteEstimate]] = field(default_factory=dict)
    slack_values: Dict[Tuple[int, str, str], float] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)

    solve_time: float = 0.0
    iterations: int = 0

    def get_contestant_trajectory(self, contestant: str) -> List[FanVoteEstimate]:
        """返回选手跨周轨迹"""
        trajectory = []
        for week in sorted(self.week_results.keys()):
            if contestant in self.week_results[week]:
                trajectory.append(self.week_results[week][contestant])
        return trajectory

    def get_week_estimates(self, week: int) -> Dict[str, FanVoteEstimate]:
        """返回某周所有估计"""
        return self.week_results.get(week, {})

    def get_point_estimates_matrix(self) -> Dict[int, Dict[str, float]]:
        """仅返回点估计矩阵"""
        result = {}
        for week, estimates in self.week_results.items():
            result[week] = {c: e.point_estimate for c, e in estimates.items()}
        return result

    def get_uncertainty_matrix(self) -> Dict[int, Dict[str, float]]:
        """返回确定性矩阵"""
        result = {}
        for week, estimates in self.week_results.items():
            result[week] = {c: e.certainty for c, e in estimates.items()}
        return result

    def compute_overall_certainty(self) -> float:
        """计算整体平均确定性"""
        all_certainties = []
        for estimates in self.week_results.values():
            for e in estimates.values():
                all_certainties.append(e.certainty)
        return np.mean(all_certainties) if all_certainties else 0.0


class InversionEngine(ABC):
    """反演引擎抽象基类"""

    @abstractmethod
    def solve(self, season_context) -> InversionResult:
        """求解反演问题"""
        raise NotImplementedError

    @abstractmethod
    def get_method_name(self) -> str:
        """返回方法名"""
        raise NotImplementedError

    def validate_result(self, result: InversionResult, season_context) -> List[str]:
        """验证反演结果是否与淘汰一致"""
        errors = []
        for week, week_ctx in season_context.weeks.items():
            if week not in result.week_results:
                continue

            estimates = result.week_results[week]
            eliminated = week_ctx.eliminated
            survivors = week_ctx.survivors

            for e in eliminated:
                for s in survivors:
                    if e not in estimates or s not in estimates:
                        continue

                    e_total = self._compute_total(
                        estimates[e].point_estimate,
                        week_ctx.judge_percentages.get(e, 0),
                        week_ctx.judge_ranks.get(e, 0),
                        result.method,
                    )
                    s_total = self._compute_total(
                        estimates[s].point_estimate,
                        week_ctx.judge_percentages.get(s, 0),
                        week_ctx.judge_ranks.get(s, 0),
                        result.method,
                    )

                    if result.method == "rank":
                        if e_total < s_total:
                            errors.append(
                                f"Week {week}: {e} (rank {e_total}) beats {s} (rank {s_total})"
                            )
                    else:
                        if e_total > s_total:
                            errors.append(
                                f"Week {week}: {e} ({e_total:.3f}) beats {s} ({s_total:.3f})"
                            )
        return errors

    def _compute_total(
        self,
        fan_value: float,
        judge_pct: float,
        judge_rank: int,
        method: str,
    ) -> float:
        """按规则计算合成分数"""
        if method == "rank":
            return fan_value + judge_rank
        return fan_value + judge_pct


@dataclass
class EngineComparison:
    """不同引擎结果对比"""

    season: int
    results: Dict[str, InversionResult] = field(default_factory=dict)

    def add_result(self, method: str, result: InversionResult):
        self.results[method] = result

    def compare_eliminations(self) -> Dict[int, Dict[str, str]]:
        """比较不同方法的淘汰结果"""
        comparisons = {}
        all_weeks = set()
        for result in self.results.values():
            all_weeks.update(result.week_results.keys())

        for week in sorted(all_weeks):
            comparisons[week] = {}
            for method, result in self.results.items():
                if week in result.week_results:
                    estimates = result.week_results[week]
                    comparisons[week][method] = self._find_lowest(estimates)
        return comparisons

    def _find_lowest(self, estimates: Dict[str, FanVoteEstimate]) -> str:
        """找到最低点估计的选手"""
        if not estimates:
            return "N/A"
        return min(estimates.items(), key=lambda x: x[1].point_estimate)[0]

    def get_reversal_rate(self) -> float:
        """计算不同方法产生不同淘汰的比例"""
        if len(self.results) < 2:
            return 0.0

        comparisons = self.compare_eliminations()
        reversals = 0
        total_weeks = 0

        for week_comp in comparisons.values():
            if len(week_comp) >= 2:
                total_weeks += 1
                if len(set(week_comp.values())) > 1:
                    reversals += 1

        return reversals / total_weeks if total_weeks > 0 else 0.0

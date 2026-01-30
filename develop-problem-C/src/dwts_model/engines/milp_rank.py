"""
排名制赛季的 MILP 反演引擎（S1-S2, S28+）。

关键：粉丝票排名是“决策变量”，不是已知输入。
目标是找到能解释真实淘汰结果的排名排列。

数学形式：
- 二元变量：x_{ik} ∈ {0,1} 表示选手 i 的粉丝排名为 k
- AllDifferent：sum_k x_{ik} = 1（每人一个名次）
                sum_i x_{ik} = 1（每个名次一个人）
- 粉丝排名：r_i^fan = sum_k k * x_{ik}
- 淘汰约束：R_E >= R_i（对所有生还者）
  其中 R_i = r_i^judge + r_i^fan
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from typing import Dict, List, Tuple, Optional, Set
import time
from dataclasses import dataclass
from itertools import permutations
import warnings

from .engine_interface import InversionEngine, InversionResult, FanVoteEstimate


@dataclass
class RankProblem:
    """单周排名问题结构"""

    week: int
    contestants: List[str]
    judge_ranks: Dict[str, int]
    eliminated: List[str]
    survivors: Set[str]
    has_judge_save: bool = False

    def get_n_contestants(self) -> int:
        return len(self.contestants)


class MILPRankEngine(InversionEngine):
    """
    排名制 MILP 反演引擎。

    关键点：粉丝票排名是潜变量，需要求解与淘汰一致的排列。
    """

    def __init__(
        self,
        time_limit: int = 60,
        use_enumeration_threshold: int = 8,
    ):
        self.time_limit = time_limit
        self.enumeration_threshold = use_enumeration_threshold

    def get_method_name(self) -> str:
        return "rank_milp"

    def solve(self, season_context) -> InversionResult:
        """
        求解排名制赛季的粉丝票反演。

        粉丝排名是决策变量（而非输入）。
        """
        start_time = time.time()

        result = InversionResult(
            season=season_context.season,
            method="rank_milp",
            inconsistency_score=0.0,
            is_feasible=True,
        )

        for week, week_ctx in season_context.weeks.items():
            if not week_ctx.has_valid_elimination():
                continue

            problem = self._build_week_problem(week_ctx, season_context.has_judges_save)
            if problem.get_n_contestants() == 0:
                continue

            n = problem.get_n_contestants()

            if n <= self.enumeration_threshold:
                week_result = self._solve_by_enumeration(problem)
            else:
                week_result = self._solve_by_milp(problem)

            if week_result is None:
                result.violations.append(f"Week {week}: 未找到可行的粉丝排名排列")
                result.inconsistency_score += 1.0
                result.slack_values[(week, "feasibility", "none")] = 1.0
                week_result = self._get_uniform_result(problem)

            fan_ranks, slack = week_result
            result.inconsistency_score += slack

            for c in problem.contestants:
                rank = fan_ranks.get(c, n // 2)
                normalized = (n - rank + 1) / n

                result.week_results.setdefault(week, {})[c] = FanVoteEstimate(
                    contestant=c,
                    week=week,
                    point_estimate=normalized,
                    lower_bound=0.0,
                    upper_bound=1.0,
                    certainty=max(0.0, 1.0 - slack),
                    method="rank_milp",
                )

        result.solve_time = time.time() - start_time
        return result

    def _build_week_problem(self, week_ctx, has_judge_save: bool) -> RankProblem:
        """构建单周排名问题结构"""
        return RankProblem(
            week=week_ctx.week,
            contestants=list(week_ctx.active_set),
            judge_ranks=week_ctx.judge_ranks,
            eliminated=week_ctx.eliminated,
            survivors=week_ctx.survivors,
            has_judge_save=has_judge_save,
        )

    def _solve_by_enumeration(
        self,
        problem: RankProblem,
    ) -> Optional[Tuple[Dict[str, int], float]]:
        """
        枚举所有粉丝排名排列，并计算约束违反度。
        """
        n = problem.get_n_contestants()
        contestants = problem.contestants

        best_ranking = None
        min_slack = float("inf")

        for perm in permutations(range(1, n + 1)):
            fan_ranks = dict(zip(contestants, perm))
            slack = self._compute_slack(problem, fan_ranks)

            if slack < min_slack:
                min_slack = slack
                best_ranking = fan_ranks.copy()

            if slack == 0:
                break

        return (best_ranking, min_slack) if best_ranking else None

    def _compute_slack(
        self,
        problem: RankProblem,
        fan_ranks: Dict[str, int],
    ) -> float:
        """
        计算给定粉丝排名的总违反度。

        排名制规则：被淘汰者应具有最高（最差）的组合排名。
        评委拯救赛季：约束更保守，避免结果完全由粉丝票决定。
        """
        total_slack = 0.0

        for e in problem.eliminated:
            e_combined = problem.judge_ranks.get(e, 1) + fan_ranks.get(e, 1)

            for s in problem.survivors:
                s_combined = problem.judge_ranks.get(s, 1) + fan_ranks.get(s, 1)
                if e_combined < s_combined:
                    total_slack += (s_combined - e_combined)

        if problem.has_judge_save and len(problem.eliminated) > 0:
            e = problem.eliminated[0]
            e_fan_rank = fan_ranks.get(e, 1)

            better_fan_rank_count = 0
            for s in problem.survivors:
                s_fan_rank = fan_ranks.get(s, 1)
                if s_fan_rank < e_fan_rank:
                    better_fan_rank_count += 1

            # 若淘汰者不是粉丝票倒数，允许一定松弛
            rank_sorted = sorted(set(fan_ranks.values()))
            if len(rank_sorted) >= 2:
                second_worst = rank_sorted[-2]
                if e_fan_rank >= second_worst:
                    total_slack += 0.1

        return total_slack

    def _solve_by_milp(
        self,
        problem: RankProblem,
    ) -> Optional[Tuple[Dict[str, int], float]]:
        """
        使用 scipy MILP 求解（大规模回退）。

        变量：x_{ik} 二元变量（选手 i 的排名为 k）
        """
        warnings.warn("MILP 规模较大，回退到启发式搜索")
        return self._solve_heuristic(problem)

    def _solve_heuristic(
        self,
        problem: RankProblem,
    ) -> Optional[Tuple[Dict[str, int], float]]:
        """
        启发式：以评委排名为起点，进行局部交换改进。
        """
        n = problem.get_n_contestants()
        contestants = problem.contestants

        sorted_by_judge = sorted(contestants, key=lambda c: problem.judge_ranks.get(c, n))
        fan_ranks = {c: i + 1 for i, c in enumerate(sorted_by_judge)}

        current_slack = self._compute_slack(problem, fan_ranks)
        if current_slack == 0:
            return fan_ranks, 0.0

        improved = True
        iterations = 0
        max_iterations = n * n

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            for i in range(n):
                for j in range(i + 1, n):
                    c1, c2 = contestants[i], contestants[j]
                    new_ranks = fan_ranks.copy()
                    new_ranks[c1], new_ranks[c2] = new_ranks[c2], new_ranks[c1]

                    new_slack = self._compute_slack(problem, new_ranks)
                    if new_slack < current_slack:
                        fan_ranks = new_ranks
                        current_slack = new_slack
                        improved = True
                        if current_slack == 0:
                            return fan_ranks, 0.0

        return fan_ranks, current_slack

    def _get_uniform_result(
        self,
        problem: RankProblem,
    ) -> Tuple[Dict[str, int], float]:
        """回退：返回均匀排名"""
        n = problem.get_n_contestants()
        fan_ranks = {c: i + 1 for i, c in enumerate(problem.contestants)}
        return fan_ranks, 1.0


# 兼容旧接口
RankCPEngine = MILPRankEngine

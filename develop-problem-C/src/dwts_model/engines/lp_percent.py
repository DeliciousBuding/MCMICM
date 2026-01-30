"""
百分制赛季的两阶段鲁棒 LP 反演引擎（S3-S27）。

关键：评委百分比应为 J_i / sum(J_k)，而非 J_i / mean(J_k)。

阶段 1：最小化松弛（L1），求最小不一致度 S*
阶段 2：锁定 S*，求每位选手粉丝票的硬区间 [L_i, U_i]

数学形式：
- 变量：v_i（选手 i 的粉丝票百分比）、松弛变量 s_j
- 评委百分比：J_i^% = J_i / sum(J_k)
- 合成分数：C_i = J_i^% + v_i
- 淘汰约束：C_E <= C_i（对所有生还者）
- 单纯形约束：sum(v_i) = 1
- 变量边界：epsilon <= v_i <= 1 - (n-1)*epsilon

阶段 1 目标：min sum(|s_j|)
阶段 2 目标：min/max v_i 且 sum(|s_j|) <= S* + delta
"""
import numpy as np
from scipy.optimize import linprog, OptimizeResult
from typing import Dict, List, Tuple, Optional, Set
import time
from dataclasses import dataclass

from .engine_interface import InversionEngine, InversionResult, FanVoteEstimate


@dataclass
class LPProblem:
    """单周 LP 问题结构"""

    week: int
    contestants: List[str]
    judge_percentages: Dict[str, float]
    constraints: List[Tuple[str, str]]

    def get_n_vars(self) -> int:
        """粉丝票变量数"""
        return len(self.contestants)

    def get_n_constraints(self) -> int:
        """成对约束数量"""
        return len(self.constraints)

    def get_contestant_index(self, name: str) -> int:
        """获取选手在变量向量中的索引"""
        return self.contestants.index(name)


class PercentLPEngine(InversionEngine):
    """
    百分制赛季 LP 反演引擎。

    两阶段流程：
    1) 最小化不一致度（松弛）
    2) 在最小松弛下求每位选手的硬区间
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        slack_weight: float = 1.0,
        min_vote_floor: float = 0.01,
        regularization: float = 0.1,
    ):
        """
        Args:
            tolerance: 约束数值容差
            slack_weight: 松弛项权重
            min_vote_floor: 粉丝票最小占比下限（默认 1%）
            regularization: 向均匀分布收缩的正则权重
        """
        self.tolerance = tolerance
        self.slack_weight = slack_weight
        self.min_vote_floor = min_vote_floor
        self.regularization = regularization

    def get_method_name(self) -> str:
        return "percent"

    def solve(self, season_context) -> InversionResult:
        """
        求解百分制赛季的粉丝票反演。

        Args:
            season_context: ActiveSetManager 生成的 SeasonContext

        Returns:
            InversionResult（包含点估计与区间）
        """
        start_time = time.time()

        result = InversionResult(
            season=season_context.season,
            method="percent",
            inconsistency_score=0.0,
            is_feasible=True,
        )

        for week, week_ctx in season_context.weeks.items():
            if not week_ctx.has_valid_elimination():
                continue

            lp_problem = self._build_week_problem(week_ctx)

            if lp_problem.get_n_constraints() == 0:
                # 无有效约束时给出均匀分布
                for c in lp_problem.contestants:
                    result.week_results.setdefault(week, {})[c] = FanVoteEstimate(
                        contestant=c,
                        week=week,
                        point_estimate=1.0 / len(lp_problem.contestants),
                        lower_bound=0.0,
                        upper_bound=1.0,
                        certainty=0.0,
                        method="percent",
                    )
                continue

            phase1_result = self._solve_phase1(lp_problem)
            if phase1_result is None:
                result.violations.append(f"Week {week}: 阶段 1 失败")
                result.is_feasible = False
                continue

            fan_votes, slack_sum, slacks = phase1_result
            result.inconsistency_score += slack_sum
            result.slack_values[(week, "__total__", "__total__")] = slack_sum

            for i, (e, s) in enumerate(lp_problem.constraints):
                result.slack_values[(week, e, s)] = slacks[i]

            bounds = self._solve_phase2(lp_problem, slack_sum)

            for c in lp_problem.contestants:
                idx = lp_problem.get_contestant_index(c)
                point_est = fan_votes[idx]
                lower, upper = bounds.get(c, (0.0, 1.0))

                interval_width = upper - lower
                certainty = max(0.0, 1.0 - interval_width)

                result.week_results.setdefault(week, {})[c] = FanVoteEstimate(
                    contestant=c,
                    week=week,
                    point_estimate=point_est,
                    lower_bound=lower,
                    upper_bound=upper,
                    certainty=certainty,
                    method="percent",
                )

        result.solve_time = time.time() - start_time
        return result

    def _build_week_problem(self, week_ctx) -> LPProblem:
        """构建单周 LP 问题结构"""
        contestants = list(week_ctx.active_set)
        constraints = week_ctx.get_pairwise_constraints()

        return LPProblem(
            week=week_ctx.week,
            contestants=contestants,
            judge_percentages=week_ctx.judge_percentages,
            constraints=constraints,
        )

    def _solve_phase1(
        self,
        problem: LPProblem,
    ) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
        """
        阶段 1：最小化总松弛。

        变量：[F_1, ..., F_n, s_1^+, s_1^-, ..., s_m^+, s_m^-]
        Returns: (fan_votes, total_slack, slack_values) 或 None
        """
        n_contestants = problem.get_n_vars()
        n_constraints = problem.get_n_constraints()

        if n_constraints == 0:
            return (
                np.ones(n_contestants) / n_contestants,
                0.0,
                np.array([]),
            )

        n_vars = n_contestants + 2 * n_constraints

        uniform = 1.0 / n_contestants
        c = np.zeros(n_vars)
        c[n_contestants:] = self.slack_weight

        # 正则项：偏离均匀分布的惩罚
        for i in range(n_contestants):
            c[i] = self.regularization * abs(1.0 - uniform * n_contestants)

        A_ub = []
        b_ub = []

        for i, (e, s) in enumerate(problem.constraints):
            row = np.zeros(n_vars)
            e_idx = problem.get_contestant_index(e)
            s_idx = problem.get_contestant_index(s)

            row[e_idx] = 1.0
            row[s_idx] = -1.0

            slack_plus_idx = n_contestants + 2 * i
            slack_minus_idx = n_contestants + 2 * i + 1
            row[slack_plus_idx] = -1.0
            row[slack_minus_idx] = 1.0

            J_e = problem.judge_percentages.get(e, 0)
            J_s = problem.judge_percentages.get(s, 0)

            A_ub.append(row)
            b_ub.append(J_s - J_e - self.tolerance)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        A_eq = np.zeros((1, n_vars))
        A_eq[0, :n_contestants] = 1.0
        b_eq = np.array([1.0])

        bounds = []
        max_vote = 1.0 - (n_contestants - 1) * self.min_vote_floor
        for _ in range(n_contestants):
            bounds.append((self.min_vote_floor, max_vote))
        for _ in range(2 * n_constraints):
            bounds.append((0.0, None))

        try:
            res = linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
            )

            if res.success:
                fan_votes = res.x[:n_contestants]
                slack_plus = res.x[n_contestants : n_contestants + n_constraints]
                slack_minus = res.x[n_contestants + n_constraints :]
                total_slack = np.sum(slack_plus) + np.sum(slack_minus)
                slack_values = slack_plus + slack_minus
                return fan_votes, total_slack, slack_values
            return None
        except Exception as exc:
            print(f"LP 求解失败: {exc}")
            return None

    def _solve_phase2(
        self,
        problem: LPProblem,
        max_slack: float,
    ) -> Dict[str, Tuple[float, float]]:
        """
        阶段 2：在最小松弛下求每位选手的区间。

        对每个选手分别求：
        - min F_i 且 total_slack <= max_slack + epsilon
        - max F_i 且 total_slack <= max_slack + epsilon

        Returns: {contestant: (lower_bound, upper_bound)}
        """
        n_contestants = problem.get_n_vars()
        n_constraints = problem.get_n_constraints()

        bounds_result = {}
        epsilon = self.tolerance

        n_vars = n_contestants + 2 * n_constraints

        A_ub_base = []
        b_ub_base = []

        for i, (e, s) in enumerate(problem.constraints):
            row = np.zeros(n_vars)
            e_idx = problem.get_contestant_index(e)
            s_idx = problem.get_contestant_index(s)

            row[e_idx] = 1.0
            row[s_idx] = -1.0
            row[n_contestants + 2 * i] = -1.0
            row[n_contestants + 2 * i + 1] = 1.0

            J_e = problem.judge_percentages.get(e, 0)
            J_s = problem.judge_percentages.get(s, 0)

            A_ub_base.append(row)
            b_ub_base.append(J_s - J_e - self.tolerance)

        slack_budget_row = np.zeros(n_vars)
        slack_budget_row[n_contestants:] = 1.0
        A_ub_base.append(slack_budget_row)
        b_ub_base.append(max_slack + epsilon)

        A_ub = np.array(A_ub_base)
        b_ub = np.array(b_ub_base)

        A_eq = np.zeros((1, n_vars))
        A_eq[0, :n_contestants] = 1.0
        b_eq = np.array([1.0])

        var_bounds = []
        max_vote = 1.0 - (n_contestants - 1) * self.min_vote_floor
        for _ in range(n_contestants):
            var_bounds.append((self.min_vote_floor, max_vote))
        for _ in range(2 * n_constraints):
            var_bounds.append((0.0, None))

        for contestant in problem.contestants:
            idx = problem.get_contestant_index(contestant)

            c_min = np.zeros(n_vars)
            c_min[idx] = 1.0
            try:
                res_min = linprog(
                    c=c_min,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    A_eq=A_eq,
                    b_eq=b_eq,
                    bounds=var_bounds,
                    method="highs",
                )
                lower = res_min.x[idx] if res_min.success else 0.0
            except Exception:
                lower = 0.0

            c_max = np.zeros(n_vars)
            c_max[idx] = -1.0
            try:
                res_max = linprog(
                    c=c_max,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    A_eq=A_eq,
                    b_eq=b_eq,
                    bounds=var_bounds,
                    method="highs",
                )
                upper = res_max.x[idx] if res_max.success else 1.0
            except Exception:
                upper = 1.0

            bounds_result[contestant] = (
                max(0.0, lower - epsilon),
                min(1.0, upper + epsilon),
            )

        return bounds_result

    def analyze_sensitivity(
        self,
        problem: LPProblem,
        base_result: Tuple[np.ndarray, float, np.ndarray],
    ) -> Dict[str, float]:
        """
        评委分数扰动敏感性分析。

        Returns: {contestant: sensitivity_score}
        """
        fan_votes, _, _ = base_result
        sensitivities = {}
        delta = 0.01

        for contestant in problem.contestants:
            idx = problem.get_contestant_index(contestant)

            perturbed_pcts = problem.judge_percentages.copy()
            original = perturbed_pcts[contestant]

            perturbed_pcts[contestant] = min(1.0, original + delta)
            total = sum(perturbed_pcts.values())
            perturbed_pcts = {k: v / total for k, v in perturbed_pcts.items()}

            perturbed_problem = LPProblem(
                week=problem.week,
                contestants=problem.contestants,
                judge_percentages=perturbed_pcts,
                constraints=problem.constraints,
            )

            perturbed_result = self._solve_phase1(perturbed_problem)
            if perturbed_result:
                new_votes, _, _ = perturbed_result
                change = abs(new_votes[idx] - fan_votes[idx])
                sensitivities[contestant] = change / delta
            else:
                sensitivities[contestant] = float("inf")

        return sensitivities


class PercentMethodSimulator:
    """使用百分制模拟淘汰结果"""

    def __init__(self):
        pass

    def simulate_elimination(
        self,
        fan_percentages: Dict[str, float],
        judge_percentages: Dict[str, float],
    ) -> str:
        """
        模拟百分制下的淘汰选手。

        Returns: 被淘汰者
        """
        combined = {}
        for contestant in fan_percentages:
            fan_pct = fan_percentages.get(contestant, 0)
            judge_pct = judge_percentages.get(contestant, 0)
            combined[contestant] = fan_pct + judge_pct

        return min(combined.items(), key=lambda x: x[1])[0]

    def compare_with_rank(
        self,
        fan_percentages: Dict[str, float],
        judge_percentages: Dict[str, float],
    ) -> Tuple[str, str]:
        """
        对比百分制与排名制的淘汰结果。

        Returns: (percent_eliminated, rank_eliminated)
        """
        percent_elim = self.simulate_elimination(fan_percentages, judge_percentages)

        fan_sorted = sorted(fan_percentages.items(), key=lambda x: x[1], reverse=True)
        fan_ranks = {c: r + 1 for r, (c, _) in enumerate(fan_sorted)}

        judge_sorted = sorted(judge_percentages.items(), key=lambda x: x[1], reverse=True)
        judge_ranks = {c: r + 1 for r, (c, _) in enumerate(judge_sorted)}

        combined_ranks = {c: fan_ranks.get(c, 0) + judge_ranks.get(c, 0) for c in fan_percentages}
        rank_elim = max(combined_ranks.items(), key=lambda x: x[1])[0]

        return percent_elim, rank_elim

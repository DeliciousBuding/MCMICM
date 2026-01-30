"""
排名制赛季的约束规划反演引擎（S1-S2, S28+）。

使用离散约束满足求解排名制投票。
若 CP 不可用，则回退到枚举或连续松弛。

数学形式：
- 变量：R_fan_i ∈ {1, ..., N}（选手 i 的粉丝票排名）
- 约束：AllDifferent(R_fan)（粉丝排名不允许并列）
- 淘汰约束：R_fan_e + R_judge_e > R_fan_s + R_judge_s
"""
import numpy as np
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
    judge_ranks: Dict[str, int]  # 1 = 最好
    constraints: List[Tuple[str, str]]  # (被淘汰者, 生还者) 约束对

    def get_n_contestants(self) -> int:
        return len(self.contestants)


class RankCPEngine(InversionEngine):
    """
    排名制反演引擎（CP 优先）。

    求解路径：
    1) 尝试 OR-Tools CP-SAT（若可用）
    2) 小规模回退到枚举
    3) 大规模回退到 LP 松弛
    """

    def __init__(
        self,
        time_limit: int = 60,
        use_enumeration_threshold: int = 8,
        use_lp_fallback: bool = True,
    ):
        """
        Args:
            time_limit: CP 求解时间上限（秒）
            use_enumeration_threshold: 选手数不超过该值时启用枚举
            use_lp_fallback: CP 失败时是否回退 LP
        """
        self.time_limit = time_limit
        self.enumeration_threshold = use_enumeration_threshold
        self.use_lp_fallback = use_lp_fallback

        # 检查求解器可用性
        self.has_ortools = self._check_ortools()

    def _check_ortools(self) -> bool:
        """检查 OR-Tools 是否可用"""
        try:
            from ortools.sat.python import cp_model
            return True
        except ImportError:
            warnings.warn("OR-Tools 不可用，将使用回退求解方案")
            return False

    def get_method_name(self) -> str:
        return "rank"

    def solve(self, season_context) -> InversionResult:
        """求解排名制赛季的粉丝票反演"""
        start_time = time.time()

        result = InversionResult(
            season=season_context.season,
            method="rank",
            inconsistency_score=0.0,
            is_feasible=True,
        )

        for week, week_ctx in season_context.weeks.items():
            if not week_ctx.has_valid_elimination():
                continue

            problem = self._build_week_problem(week_ctx)
            if problem.get_n_contestants() == 0:
                continue

            # 选择求解方式
            n = problem.get_n_contestants()
            if n <= self.enumeration_threshold:
                week_result = self._solve_by_enumeration(problem)
            elif self.has_ortools:
                week_result = self._solve_by_cp(problem)
            elif self.use_lp_fallback:
                week_result = self._solve_by_lp_relaxation(problem)
            else:
                week_result = None

            if week_result is None:
                result.violations.append(f"Week {week}: 未找到可行解")
                result.inconsistency_score += 1.0
                week_result = self._get_uniform_result(problem)

            fan_ranks, violations = week_result
            result.inconsistency_score += violations
            result.slack_values[(week, "__total__", "__total__")] = violations

            for c in problem.contestants:
                rank = fan_ranks.get(c, n // 2)
                # 统一尺度：将排名映射为“比例型”数值（排名越小越好）
                normalized = (n - rank + 1) / n

                result.week_results.setdefault(week, {})[c] = FanVoteEstimate(
                    contestant=c,
                    week=week,
                    point_estimate=normalized,
                    lower_bound=0.0,
                    upper_bound=1.0,
                    certainty=1.0 - violations,
                    method="rank",
                )

        result.solve_time = time.time() - start_time
        return result

    def _build_week_problem(self, week_ctx) -> RankProblem:
        """构建单周排名问题结构"""
        contestants = list(week_ctx.active_set)

        return RankProblem(
            week=week_ctx.week,
            contestants=contestants,
            judge_ranks=week_ctx.judge_ranks,
            constraints=week_ctx.get_pairwise_constraints(),
        )

    def _solve_by_enumeration(
        self,
        problem: RankProblem,
    ) -> Optional[Tuple[Dict[str, int], float]]:
        """
        枚举所有可能的粉丝排名排列。

        Returns: (最佳排名, 违反度) 或 None
        """
        n = problem.get_n_contestants()
        best_ranking = None
        min_violations = float("inf")

        for perm in permutations(range(1, n + 1)):
            fan_ranks = dict(zip(problem.contestants, perm))
            violations = self._count_violations(problem, fan_ranks)

            if violations < min_violations:
                min_violations = violations
                best_ranking = fan_ranks.copy()

            if violations == 0:
                break

        return (best_ranking, min_violations) if best_ranking else None

    def _solve_by_cp(
        self,
        problem: RankProblem,
    ) -> Optional[Tuple[Dict[str, int], float]]:
        """使用 OR-Tools CP-SAT 求解"""
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            return None

        n = problem.get_n_contestants()
        model = cp_model.CpModel()

        # 变量：每位选手的粉丝排名（1..总人数）
        fan_vars = {c: model.NewIntVar(1, n, f"fan_rank_{c}") for c in problem.contestants}

        # 全异约束
        model.AddAllDifferent(list(fan_vars.values()))

        # 软约束松弛变量
        slack_vars = []
        for i, (e, s) in enumerate(problem.constraints):
            slack = model.NewIntVar(0, 2 * n, f"slack_{i}")
            slack_vars.append(slack)

            # 组合排名约束：被淘汰者应更差（排名更高）
            e_judge = problem.judge_ranks.get(e, n)
            s_judge = problem.judge_ranks.get(s, 1)
            model.Add(
                fan_vars[e] + e_judge >= fan_vars[s] + s_judge + 1 - slack
            )

        model.Minimize(sum(slack_vars))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        status = solver.Solve(model)

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            fan_ranks = {c: solver.Value(v) for c, v in fan_vars.items()}
            violations = sum(solver.Value(s) for s in slack_vars) / n
            return fan_ranks, violations

        return None

    def _solve_by_lp_relaxation(
        self,
        problem: RankProblem,
    ) -> Optional[Tuple[Dict[str, int], float]]:
        """使用 LP 松弛求解（连续排名后再四舍五入）"""
        from scipy.optimize import linprog

        n = problem.get_n_contestants()
        m = len(problem.constraints)

        if m == 0:
            return {c: i + 1 for i, c in enumerate(problem.contestants)}, 0.0

        # 变量向量：排名与松弛项
        n_vars = n + m

        # 目标：最小化松弛和
        c = np.zeros(n_vars)
        c[n:] = 1.0

        A_ub = []
        b_ub = []

        for i, (e, s) in enumerate(problem.constraints):
            row = np.zeros(n_vars)
            e_idx = problem.contestants.index(e)
            s_idx = problem.contestants.index(s)

            e_judge = problem.judge_ranks.get(e, n)
            s_judge = problem.judge_ranks.get(s, 1)

            # 被淘汰者组合排名应更差（含松弛）
            row[e_idx] = -1.0
            row[s_idx] = 1.0
            row[n + i] = 1.0

            A_ub.append(-row)
            b_ub.append(-(s_judge - e_judge + 1))

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        bounds = [(1.0, float(n)) for _ in range(n)]
        bounds += [(0.0, None) for _ in range(m)]

        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            if res.success:
                raw_ranks = res.x[:n]
                sorted_indices = np.argsort(raw_ranks)
                fan_ranks = {problem.contestants[idx]: rank for rank, idx in enumerate(sorted_indices, 1)}
                violations = np.sum(res.x[n:]) / n
                return fan_ranks, violations
        except Exception:
            pass

        return None

    def _count_violations(
        self,
        problem: RankProblem,
        fan_ranks: Dict[str, int],
    ) -> int:
        """
        统计给定粉丝排名的约束违反数。

        对评委拯救赛季：若结果“过于明显”，可视为保留一定不确定性。
        """
        violations = 0

        for e, s in problem.constraints:
            e_total = fan_ranks.get(e, 0) + problem.judge_ranks.get(e, 0)
            s_total = fan_ranks.get(s, 0) + problem.judge_ranks.get(s, 0)

            # 被淘汰者应当具有更高（更差）的组合排名
            if e_total <= s_total:
                violations += 1

        # 评委拯救赛季的启发式说明：
        # 若底部两名差距过小，则视为存在裁量空间
        contestants = problem.contestants
        n = len(contestants)

        combined_ranks = []
        for c in contestants:
            c_total = fan_ranks.get(c, 0) + problem.judge_ranks.get(c, 0)
            combined_ranks.append((c, c_total))

        combined_ranks.sort(key=lambda x: x[1], reverse=True)

        if len(combined_ranks) >= 2:
            worst_rank = combined_ranks[0][1]
            second_worst_rank = combined_ranks[1][1]
            if worst_rank - second_worst_rank <= 2:
                # 接近时认为存在裁量，不额外惩罚
                pass
            elif worst_rank - second_worst_rank >= 3:
                # 间隔过大也不额外惩罚
                pass

        # 更保守的启发式：淘汰者位于粉丝票倒数两名时不施加惩罚
        fan_ranks_only = sorted([fan_ranks.get(c, n) for c in contestants])
        eliminated_list = [e for e, _ in problem.constraints if True]
        for e in eliminated_list:
            e_fan_rank = fan_ranks.get(e, n)
            if e_fan_rank >= fan_ranks_only[-2]:
                pass

        return violations

    def _get_uniform_result(
        self,
        problem: RankProblem,
    ) -> Tuple[Dict[str, int], float]:
        """回退：返回均匀排名"""
        n = problem.get_n_contestants()
        ranks = {c: i + 1 for i, c in enumerate(problem.contestants)}
        return ranks, 1.0


class RankMethodSimulator:
    """使用排名法模拟淘汰结果"""

    def simulate_elimination(
        self,
        fan_ranks: Dict[str, int],
        judge_ranks: Dict[str, int],
    ) -> str:
        """
        模拟排名制下的淘汰选手。

        Returns: 被淘汰者（组合排名最高/最差）
        """
        combined = {}
        for contestant in fan_ranks:
            fan_r = fan_ranks.get(contestant, len(fan_ranks))
            judge_r = judge_ranks.get(contestant, len(judge_ranks))
            combined[contestant] = fan_r + judge_r

        return max(combined.items(), key=lambda x: x[1])[0]

    def find_feasible_fan_ranks(
        self,
        judge_ranks: Dict[str, int],
        eliminated: str,
        survivors: Set[str],
    ) -> List[Dict[str, int]]:
        """
        枚举所有可行的粉丝排名。

        Returns: 满足淘汰一致性的粉丝排名列表
        """
        contestants = [eliminated] + list(survivors)
        n = len(contestants)

        valid_rankings = []
        for perm in permutations(range(1, n + 1)):
            fan_ranks = dict(zip(contestants, perm))
            e_total = fan_ranks[eliminated] + judge_ranks.get(eliminated, n)

            is_valid = True
            for s in survivors:
                s_total = fan_ranks[s] + judge_ranks.get(s, 1)
                if e_total <= s_total:
                    is_valid = False
                    break

            if is_valid:
                valid_rankings.append(fan_ranks)

        return valid_rankings

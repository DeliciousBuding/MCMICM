"""
Two-Phase Robust LP Engine for Percent Seasons (S3-S27)

Phase 1: Minimize slack (L1-norm) to find minimum inconsistency score S*
Phase 2: With S* locked, find hard bounds [L_i, U_i] for each fan vote

Mathematical formulation:
- Variables: F_i (fan vote percentage for contestant i), slack variables s_j
- Constraint: F_e + J_e - s_j <= F_s + J_s + s_j for each (eliminated, survivor) pair
- Simplex: sum(F_i) = 1 for each week
- Bounds: 0 <= F_i <= 1

Phase 1 Objective: min sum(|s_j|)
Phase 2 Objective: min/max F_i subject to sum(|s_j|) <= S* + epsilon
"""
import numpy as np
from scipy.optimize import linprog, OptimizeResult
from typing import Dict, List, Tuple, Optional, Set
import time
from dataclasses import dataclass

from .engine_interface import InversionEngine, InversionResult, FanVoteEstimate


@dataclass
class LPProblem:
    """LP problem structure for a single week"""
    week: int
    contestants: List[str]           # Ordered list of contestants
    judge_percentages: Dict[str, float]
    constraints: List[Tuple[str, str]]  # (eliminated, survivor) pairs
    
    def get_n_vars(self) -> int:
        """Number of fan vote variables"""
        return len(self.contestants)
    
    def get_n_constraints(self) -> int:
        """Number of pairwise constraints"""
        return len(self.constraints)
    
    def get_contestant_index(self, name: str) -> int:
        """Get index of contestant in variable array"""
        return self.contestants.index(name)


class PercentLPEngine(InversionEngine):
    """
    LP-based inversion engine for percent seasons.
    
    Uses two-phase approach:
    1. Find minimum violation (inconsistency score)
    2. Find hard bounds for each fan vote given minimum violation
    """
    
    def __init__(self, tolerance: float = 1e-6, slack_weight: float = 1.0):
        """
        Args:
            tolerance: Numerical tolerance for constraints
            slack_weight: Weight for slack variables in objective
        """
        self.tolerance = tolerance
        self.slack_weight = slack_weight
        
    def get_method_name(self) -> str:
        return "percent"
    
    def solve(self, season_context) -> InversionResult:
        """
        Solve fan vote inversion for a percent-based season.
        
        Args:
            season_context: SeasonContext from ActiveSetManager
            
        Returns:
            InversionResult with estimated fan votes and bounds
        """
        start_time = time.time()
        
        result = InversionResult(
            season=season_context.season,
            method='percent',
            inconsistency_score=0.0,
            is_feasible=True
        )
        
        # Process each week independently
        for week, week_ctx in season_context.weeks.items():
            if not week_ctx.has_valid_elimination():
                continue
            
            # Build LP problem for this week
            lp_problem = self._build_week_problem(week_ctx)
            
            if lp_problem.get_n_constraints() == 0:
                # No constraints - uniform distribution
                for c in lp_problem.contestants:
                    result.week_results.setdefault(week, {})[c] = FanVoteEstimate(
                        contestant=c,
                        week=week,
                        point_estimate=1.0 / len(lp_problem.contestants),
                        lower_bound=0.0,
                        upper_bound=1.0,
                        certainty=0.0,  # No certainty without constraints
                        method='percent'
                    )
                continue
            
            # Phase 1: Find minimum inconsistency
            phase1_result = self._solve_phase1(lp_problem)
            
            if phase1_result is None:
                result.violations.append(f"Week {week}: Phase 1 failed")
                result.is_feasible = False
                continue
            
            fan_votes, slack_sum, slacks = phase1_result
            result.inconsistency_score += slack_sum
            
            # Store slack values
            for i, (e, s) in enumerate(lp_problem.constraints):
                result.slack_values[(week, e, s)] = slacks[i]
            
            # Phase 2: Find bounds for each contestant
            bounds = self._solve_phase2(lp_problem, slack_sum)
            
            # Create estimates
            for c in lp_problem.contestants:
                idx = lp_problem.get_contestant_index(c)
                point_est = fan_votes[idx]
                lower, upper = bounds.get(c, (0.0, 1.0))
                
                # Calculate certainty based on interval width
                interval_width = upper - lower
                # Certainty: 1 if interval is 0, 0 if interval is 1
                certainty = max(0.0, 1.0 - interval_width)
                
                result.week_results.setdefault(week, {})[c] = FanVoteEstimate(
                    contestant=c,
                    week=week,
                    point_estimate=point_est,
                    lower_bound=lower,
                    upper_bound=upper,
                    certainty=certainty,
                    method='percent'
                )
        
        result.solve_time = time.time() - start_time
        return result
    
    def _build_week_problem(self, week_ctx) -> LPProblem:
        """Build LP problem structure for a week"""
        contestants = list(week_ctx.active_set)
        constraints = week_ctx.get_pairwise_constraints()
        
        return LPProblem(
            week=week_ctx.week,
            contestants=contestants,
            judge_percentages=week_ctx.judge_percentages,
            constraints=constraints
        )
    
    def _solve_phase1(
        self, 
        problem: LPProblem
    ) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
        """
        Phase 1: Minimize total slack.
        
        Variables: [F_1, ..., F_n, s_1^+, s_1^-, ..., s_m^+, s_m^-]
        where s_j^+, s_j^- are positive/negative slack for constraint j
        
        Returns: (fan_votes, total_slack, slack_values) or None if infeasible
        """
        n_contestants = problem.get_n_vars()
        n_constraints = problem.get_n_constraints()
        
        if n_constraints == 0:
            # No constraints - return uniform
            return (
                np.ones(n_contestants) / n_contestants,
                0.0,
                np.array([])
            )
        
        # Variables: F (n) + s+ (m) + s- (m)
        n_vars = n_contestants + 2 * n_constraints
        
        # Objective: minimize sum of slack (L1 norm)
        c = np.zeros(n_vars)
        c[n_contestants:] = self.slack_weight  # All slack vars have weight
        
        # Inequality constraints: A_ub @ x <= b_ub
        # For each (e, s) pair: F_e + J_e - (s+ - s-) <= F_s + J_s
        # Rearranged: F_e - F_s - s+ + s- <= J_s - J_e
        A_ub = []
        b_ub = []
        
        for i, (e, s) in enumerate(problem.constraints):
            row = np.zeros(n_vars)
            e_idx = problem.get_contestant_index(e)
            s_idx = problem.get_contestant_index(s)
            
            row[e_idx] = 1.0   # F_e
            row[s_idx] = -1.0  # -F_s
            
            # Slack variables for this constraint
            slack_plus_idx = n_contestants + 2 * i
            slack_minus_idx = n_contestants + 2 * i + 1
            row[slack_plus_idx] = -1.0   # -s+
            row[slack_minus_idx] = 1.0   # +s-
            
            # RHS: J_s - J_e (survivor's judge pct minus eliminated's)
            J_e = problem.judge_percentages.get(e, 0)
            J_s = problem.judge_percentages.get(s, 0)
            
            A_ub.append(row)
            b_ub.append(J_s - J_e - self.tolerance)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Equality constraint: sum(F_i) = 1
        A_eq = np.zeros((1, n_vars))
        A_eq[0, :n_contestants] = 1.0
        b_eq = np.array([1.0])
        
        # Bounds: 0 <= F_i <= 1, 0 <= s+, s- 
        bounds = []
        for i in range(n_contestants):
            bounds.append((0.0, 1.0))  # Fan votes
        for i in range(2 * n_constraints):
            bounds.append((0.0, None))  # Slack variables
        
        # Solve LP
        try:
            res = linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method='highs'
            )
            
            if res.success:
                fan_votes = res.x[:n_contestants]
                slack_plus = res.x[n_contestants:n_contestants + n_constraints]
                slack_minus = res.x[n_contestants + n_constraints:]
                total_slack = np.sum(slack_plus) + np.sum(slack_minus)
                slack_values = slack_plus + slack_minus  # Combined slack per constraint
                
                return fan_votes, total_slack, slack_values
            else:
                return None
                
        except Exception as e:
            print(f"LP solver error: {e}")
            return None
    
    def _solve_phase2(
        self, 
        problem: LPProblem, 
        max_slack: float
    ) -> Dict[str, Tuple[float, float]]:
        """
        Phase 2: Find bounds for each fan vote.
        
        For each contestant, solve two LPs:
        - min F_i subject to total slack <= max_slack + epsilon
        - max F_i subject to total slack <= max_slack + epsilon
        
        Returns: {contestant: (lower_bound, upper_bound)}
        """
        n_contestants = problem.get_n_vars()
        n_constraints = problem.get_n_constraints()
        
        bounds_result = {}
        epsilon = self.tolerance
        
        # Build common constraint matrix
        n_vars = n_contestants + 2 * n_constraints
        
        # Original constraints from Phase 1
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
        
        # Add slack budget constraint: sum(s+ + s-) <= max_slack + epsilon
        slack_budget_row = np.zeros(n_vars)
        slack_budget_row[n_contestants:] = 1.0
        A_ub_base.append(slack_budget_row)
        b_ub_base.append(max_slack + epsilon)
        
        A_ub = np.array(A_ub_base)
        b_ub = np.array(b_ub_base)
        
        # Equality constraint: sum(F_i) = 1
        A_eq = np.zeros((1, n_vars))
        A_eq[0, :n_contestants] = 1.0
        b_eq = np.array([1.0])
        
        # Bounds
        var_bounds = []
        for i in range(n_contestants):
            var_bounds.append((0.0, 1.0))
        for i in range(2 * n_constraints):
            var_bounds.append((0.0, None))
        
        # For each contestant, find min and max
        for contestant in problem.contestants:
            idx = problem.get_contestant_index(contestant)
            
            # Minimize F_i
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
                    method='highs'
                )
                lower = res_min.x[idx] if res_min.success else 0.0
            except:
                lower = 0.0
            
            # Maximize F_i (minimize -F_i)
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
                    method='highs'
                )
                upper = res_max.x[idx] if res_max.success else 1.0
            except:
                upper = 1.0
            
            bounds_result[contestant] = (
                max(0.0, lower - epsilon),
                min(1.0, upper + epsilon)
            )
        
        return bounds_result
    
    def analyze_sensitivity(
        self, 
        problem: LPProblem, 
        base_result: Tuple[np.ndarray, float, np.ndarray]
    ) -> Dict[str, float]:
        """
        Analyze sensitivity of solution to judge score changes.
        
        Returns: {contestant: sensitivity_score}
        Higher sensitivity = fan vote estimate changes more with small judge score changes
        """
        fan_votes, _, _ = base_result
        sensitivities = {}
        delta = 0.01  # 1% perturbation
        
        for contestant in problem.contestants:
            idx = problem.get_contestant_index(contestant)
            
            # Perturb judge score up
            perturbed_pcts = problem.judge_percentages.copy()
            original = perturbed_pcts[contestant]
            
            # Normalize perturbation
            perturbed_pcts[contestant] = min(1.0, original + delta)
            total = sum(perturbed_pcts.values())
            perturbed_pcts = {k: v/total for k, v in perturbed_pcts.items()}
            
            # Re-solve with perturbed values
            perturbed_problem = LPProblem(
                week=problem.week,
                contestants=problem.contestants,
                judge_percentages=perturbed_pcts,
                constraints=problem.constraints
            )
            
            perturbed_result = self._solve_phase1(perturbed_problem)
            
            if perturbed_result:
                new_votes, _, _ = perturbed_result
                change = abs(new_votes[idx] - fan_votes[idx])
                sensitivities[contestant] = change / delta
            else:
                sensitivities[contestant] = float('inf')
        
        return sensitivities


class PercentMethodSimulator:
    """
    Simulate elimination outcomes using percent method.
    Given fan votes and judge scores, determine who would be eliminated.
    """
    
    def __init__(self):
        pass
    
    def simulate_elimination(
        self,
        fan_percentages: Dict[str, float],
        judge_percentages: Dict[str, float]
    ) -> str:
        """
        Simulate who would be eliminated under percent method.
        
        Returns: Name of eliminated contestant
        """
        combined = {}
        for contestant in fan_percentages:
            fan_pct = fan_percentages.get(contestant, 0)
            judge_pct = judge_percentages.get(contestant, 0)
            combined[contestant] = fan_pct + judge_pct
        
        # Lowest combined score is eliminated
        return min(combined.items(), key=lambda x: x[1])[0]
    
    def compare_with_rank(
        self,
        fan_percentages: Dict[str, float],
        judge_percentages: Dict[str, float]
    ) -> Tuple[str, str]:
        """
        Compare elimination outcomes between percent and rank methods.
        
        Returns: (percent_eliminated, rank_eliminated)
        """
        # Percent method
        percent_elim = self.simulate_elimination(fan_percentages, judge_percentages)
        
        # Rank method
        # Convert to ranks
        fan_sorted = sorted(fan_percentages.items(), key=lambda x: x[1], reverse=True)
        fan_ranks = {c: r+1 for r, (c, _) in enumerate(fan_sorted)}
        
        judge_sorted = sorted(judge_percentages.items(), key=lambda x: x[1], reverse=True)
        judge_ranks = {c: r+1 for r, (c, _) in enumerate(judge_sorted)}
        
        combined_ranks = {}
        for c in fan_percentages:
            combined_ranks[c] = fan_ranks.get(c, 0) + judge_ranks.get(c, 0)
        
        rank_elim = max(combined_ranks.items(), key=lambda x: x[1])[0]
        
        return percent_elim, rank_elim

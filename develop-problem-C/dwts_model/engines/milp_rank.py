"""
Mixed-Integer Linear Programming Engine for Rank Seasons (S1-S2, S28+)

CRITICAL FIX: Fan vote ranks are DECISION VARIABLES, not inputs!

The problem faced: what fan vote ranking permutation could explain
the observed elimination? Fan ranks are LATENT.

Mathematical formulation:
- Binary variables: x_{ik} ∈ {0,1} = contestant i has fan rank k
- AllDifferent via: Σ_k x_{ik} = 1 (each person one rank)
                    Σ_i x_{ik} = 1 (each rank one person)
- Fan rank: r_i^fan = Σ_k k * x_{ik}
- Elimination constraint: R_E >= R_i for all survivors
  where R_i = r_i^judge + r_i^fan (combined rank)
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
    """Rank-based problem structure for a week"""
    week: int
    contestants: List[str]
    judge_ranks: Dict[str, int]  # 1 = best (observed)
    eliminated: List[str]        # Who was eliminated
    survivors: Set[str]          # Who survived
    has_judge_save: bool = False # S28+ rule
    
    def get_n_contestants(self) -> int:
        return len(self.contestants)


class MILPRankEngine(InversionEngine):
    """
    MILP-based inversion engine for rank seasons.
    
    Key insight: Fan vote ranks are LATENT - we solve for which
    permutation of fan ranks is consistent with elimination.
    """
    
    def __init__(
        self, 
        time_limit: int = 60,
        use_enumeration_threshold: int = 8
    ):
        self.time_limit = time_limit
        self.enumeration_threshold = use_enumeration_threshold
        
    def get_method_name(self) -> str:
        return "rank_milp"
    
    def solve(self, season_context) -> InversionResult:
        """
        Solve fan vote inversion for a rank-based season.
        
        Fan ranks are DECISION VARIABLES, not inputs!
        """
        start_time = time.time()
        
        result = InversionResult(
            season=season_context.season,
            method='rank_milp',
            inconsistency_score=0.0,
            is_feasible=True
        )
        
        for week, week_ctx in season_context.weeks.items():
            if not week_ctx.has_valid_elimination():
                continue
            
            problem = self._build_week_problem(week_ctx, season_context.has_judges_save)
            
            if problem.get_n_contestants() == 0:
                continue
            
            n = problem.get_n_contestants()
            
            # Choose solving method
            if n <= self.enumeration_threshold:
                week_result = self._solve_by_enumeration(problem)
            else:
                week_result = self._solve_by_milp(problem)
            
            if week_result is None:
                result.violations.append(f"Week {week}: No feasible fan rank permutation")
                result.inconsistency_score += 1.0
                # Record which constraints failed
                result.slack_values[(week, 'feasibility', 'none')] = 1.0
                week_result = self._get_uniform_result(problem)
            
            fan_ranks, slack = week_result
            result.inconsistency_score += slack
            
            # Store results
            for c in problem.contestants:
                rank = fan_ranks.get(c, n // 2)
                normalized = (n - rank + 1) / n  # Convert to 0-1 scale
                
                result.week_results.setdefault(week, {})[c] = FanVoteEstimate(
                    contestant=c,
                    week=week,
                    point_estimate=normalized,
                    lower_bound=0.0,
                    upper_bound=1.0,
                    certainty=max(0.0, 1.0 - slack),
                    method='rank_milp'
                )
        
        result.solve_time = time.time() - start_time
        return result
    
    def _build_week_problem(self, week_ctx, has_judge_save: bool) -> RankProblem:
        """Build rank problem structure"""
        return RankProblem(
            week=week_ctx.week,
            contestants=list(week_ctx.active_set),
            judge_ranks=week_ctx.judge_ranks,
            eliminated=week_ctx.eliminated,
            survivors=week_ctx.survivors,
            has_judge_save=has_judge_save
        )
    
    def _solve_by_enumeration(
        self, 
        problem: RankProblem
    ) -> Optional[Tuple[Dict[str, int], float]]:
        """
        Brute force: enumerate all fan rank permutations.
        
        For each permutation, check if it's consistent with elimination.
        """
        n = problem.get_n_contestants()
        contestants = problem.contestants
        
        feasible_count = 0
        best_ranking = None
        min_slack = float('inf')
        
        # All possible fan rankings
        for perm in permutations(range(1, n + 1)):
            fan_ranks = dict(zip(contestants, perm))
            slack = self._compute_slack(problem, fan_ranks)
            
            if slack < min_slack:
                min_slack = slack
                best_ranking = fan_ranks.copy()
            
            if slack == 0:
                feasible_count += 1
        
        return (best_ranking, min_slack) if best_ranking else None
    
    def _compute_slack(
        self, 
        problem: RankProblem, 
        fan_ranks: Dict[str, int]
    ) -> float:
        """
        Compute total constraint violation for given fan ranking.
        
        Under rank rules: eliminated contestant should have
        highest combined rank (worst).
        """
        total_slack = 0.0
        
        for e in problem.eliminated:
            e_combined = problem.judge_ranks.get(e, 1) + fan_ranks.get(e, 1)
            
            for s in problem.survivors:
                s_combined = problem.judge_ranks.get(s, 1) + fan_ranks.get(s, 1)
                
                # Eliminated should have HIGHER combined rank (worse)
                # R_E >= R_s (eliminated is at least as bad as survivor)
                if e_combined < s_combined:
                    # Constraint violated
                    total_slack += (s_combined - e_combined)
        
        return total_slack
    
    def _solve_by_milp(
        self, 
        problem: RankProblem
    ) -> Optional[Tuple[Dict[str, int], float]]:
        """
        Solve using scipy MILP (for larger problems).
        
        Variables: x_{ik} binary (contestant i has rank k)
        """
        n = problem.get_n_contestants()
        contestants = problem.contestants
        
        # Variables: x_{ik} for i in [0,n), k in [0,n)
        # Total: n^2 binary variables
        # Plus slack variables for soft constraints
        
        n_vars = n * n  # x_{ik}
        n_elim_constraints = len(problem.eliminated) * len(problem.survivors)
        
        # For simplicity with scipy, fall back to enumeration-like search
        # (scipy.milp doesn't easily handle AllDifferent)
        # TODO: Use OR-Tools or Gurobi for true MILP
        
        warnings.warn("Large MILP problem - falling back to heuristic search")
        return self._solve_heuristic(problem)
    
    def _solve_heuristic(
        self, 
        problem: RankProblem
    ) -> Optional[Tuple[Dict[str, int], float]]:
        """
        Heuristic: start with judge-aligned fan ranks, then perturb.
        """
        n = problem.get_n_contestants()
        contestants = problem.contestants
        
        # Start with fan ranks = judge ranks (most likely alignment)
        sorted_by_judge = sorted(contestants, key=lambda c: problem.judge_ranks.get(c, n))
        fan_ranks = {c: i + 1 for i, c in enumerate(sorted_by_judge)}
        
        current_slack = self._compute_slack(problem, fan_ranks)
        
        if current_slack == 0:
            return fan_ranks, 0.0
        
        # Try swapping pairs to reduce slack
        improved = True
        iterations = 0
        max_iterations = n * n
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            for i in range(n):
                for j in range(i + 1, n):
                    c1, c2 = contestants[i], contestants[j]
                    
                    # Swap
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
        problem: RankProblem
    ) -> Tuple[Dict[str, int], float]:
        """Return uniform ranking as fallback"""
        n = problem.get_n_contestants()
        fan_ranks = {c: i + 1 for i, c in enumerate(problem.contestants)}
        return fan_ranks, 1.0


# Keep backward compatibility
RankCPEngine = MILPRankEngine  # Alias

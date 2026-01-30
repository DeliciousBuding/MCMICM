"""
Judges' Save Handler for S28+ Seasons

Handles the special rule where:
1. Bottom two contestants identified by combined scores
2. Judges vote to save one of them

This creates mathematical ambiguity since the eliminated contestant
may NOT have the absolute lowest combined score.

Approach: Union of Polytopes
- For each potential "saved" contestant, create a separate LP
- Union all feasible regions
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import warnings

from .engine_interface import InversionEngine, InversionResult, FanVoteEstimate
from .lp_percent import PercentLPEngine, LPProblem
from .cp_rank import RankCPEngine


@dataclass
class JudgesSaveScenario:
    """One possible scenario for judges' save"""
    eliminated: str        # Who was actually eliminated
    potentially_saved: str # Who might have been in bottom 2 and saved
    lp_result: Optional[Tuple[np.ndarray, float, np.ndarray]] = None
    is_feasible: bool = False
    
    
@dataclass
class UnionPolytopeResult:
    """Result from union of polytopes approach"""
    scenarios: List[JudgesSaveScenario]
    combined_bounds: Dict[str, Tuple[float, float]]  # Union of all scenario bounds
    inconsistency_score: float
    

class JudgesSaveHandler:
    """
    Handle S28+ seasons with judges' save rule.
    
    Algorithm:
    1. For each survivor s_k (except eliminated e):
       - Hypothesize that s_k was in bottom 2 with e
       - Constraints: {e, s_k} combined scores < all others
       - Solve LP under this hypothesis
    
    2. Keep all feasible hypotheses
    
    3. Fan vote bounds = union of bounds from all valid scenarios
    """
    
    def __init__(
        self, 
        base_engine: Optional[InversionEngine] = None,
        tolerance: float = 1e-6
    ):
        """
        Args:
            base_engine: Engine to use for sub-problems (default: PercentLPEngine)
            tolerance: Numerical tolerance
        """
        self.base_engine = base_engine or PercentLPEngine(tolerance=tolerance)
        self.tolerance = tolerance
        
    def solve_week_with_save(
        self,
        week_ctx,
        voting_method: str = 'rank'
    ) -> UnionPolytopeResult:
        """
        Solve a week with potential judges' save.
        
        Args:
            week_ctx: WeekContext from ActiveSetManager
            voting_method: 'rank' or 'percent'
            
        Returns:
            UnionPolytopeResult with all valid scenarios
        """
        eliminated = week_ctx.eliminated
        survivors = week_ctx.survivors
        active = week_ctx.active_set
        
        if len(eliminated) != 1:
            warnings.warn(f"Expected 1 eliminated, got {len(eliminated)}")
            return self._default_result(week_ctx)
        
        e = eliminated[0]
        scenarios = []
        
        # Scenario 0: No save needed (eliminated was truly bottom)
        base_scenario = self._try_scenario(
            week_ctx, e, None, voting_method
        )
        scenarios.append(base_scenario)
        
        # For each survivor, try as the "saved" one
        for s_k in survivors:
            scenario = self._try_scenario(
                week_ctx, e, s_k, voting_method
            )
            scenarios.append(scenario)
        
        # Combine results
        valid_scenarios = [s for s in scenarios if s.is_feasible]
        
        if not valid_scenarios:
            # No valid scenario - use default
            return self._default_result(week_ctx)
        
        # Compute union of bounds
        combined_bounds = self._compute_union_bounds(
            valid_scenarios, 
            week_ctx
        )
        
        # Inconsistency = minimum across valid scenarios
        min_inconsistency = min(
            s.lp_result[1] if s.lp_result else float('inf')
            for s in valid_scenarios
        )
        
        return UnionPolytopeResult(
            scenarios=valid_scenarios,
            combined_bounds=combined_bounds,
            inconsistency_score=min_inconsistency
        )
    
    def _try_scenario(
        self,
        week_ctx,
        eliminated: str,
        potentially_saved: Optional[str],
        voting_method: str
    ) -> JudgesSaveScenario:
        """
        Try one judges' save scenario.
        
        If potentially_saved is None, assume normal elimination (no save).
        Otherwise, assume {eliminated, potentially_saved} were bottom 2.
        """
        scenario = JudgesSaveScenario(
            eliminated=eliminated,
            potentially_saved=potentially_saved or ""
        )
        
        contestants = list(week_ctx.active_set)
        
        if potentially_saved:
            # Modified constraints: 
            # {e, s_k} must have lower combined than all others
            modified_constraints = self._build_save_constraints(
                contestants, eliminated, potentially_saved
            )
        else:
            # Normal constraints: e lower than all survivors
            modified_constraints = week_ctx.get_pairwise_constraints()
        
        # Build LP problem
        problem = LPProblem(
            week=week_ctx.week,
            contestants=contestants,
            judge_percentages=week_ctx.judge_percentages,
            constraints=modified_constraints
        )
        
        # Solve
        from .lp_percent import PercentLPEngine
        engine = PercentLPEngine(tolerance=self.tolerance)
        result = engine._solve_phase1(problem)
        
        if result:
            fan_votes, slack, slacks = result
            scenario.lp_result = result
            scenario.is_feasible = slack < 0.5  # Threshold for feasibility
        
        return scenario
    
    def _build_save_constraints(
        self,
        contestants: List[str],
        eliminated: str,
        saved: str
    ) -> List[Tuple[str, str]]:
        """
        Build constraints for judges' save scenario.
        
        {eliminated, saved} are bottom 2, so their combined scores
        must be lower than all others.
        """
        constraints = []
        bottom_two = {eliminated, saved}
        others = [c for c in contestants if c not in bottom_two]
        
        for bottom in bottom_two:
            for other in others:
                # bottom's combined < other's combined
                constraints.append((bottom, other))
        
        return constraints
    
    def _compute_union_bounds(
        self,
        scenarios: List[JudgesSaveScenario],
        week_ctx
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute union of bounds from all valid scenarios.
        """
        contestants = list(week_ctx.active_set)
        combined_bounds = {}
        
        for c in contestants:
            lower = float('inf')
            upper = float('-inf')
            
            for scenario in scenarios:
                if not scenario.lp_result:
                    continue
                
                fan_votes, _, _ = scenario.lp_result
                idx = contestants.index(c)
                val = fan_votes[idx]
                
                # Expand bounds
                lower = min(lower, val - 0.1)  # Add margin
                upper = max(upper, val + 0.1)
            
            # Clip to valid range
            combined_bounds[c] = (
                max(0.0, lower),
                min(1.0, upper)
            )
        
        return combined_bounds
    
    def _default_result(self, week_ctx) -> UnionPolytopeResult:
        """Return default result when no scenario works"""
        contestants = list(week_ctx.active_set)
        n = len(contestants)
        
        return UnionPolytopeResult(
            scenarios=[],
            combined_bounds={c: (0.0, 1.0) for c in contestants},
            inconsistency_score=1.0
        )


class JudgesSaveAnalyzer:
    """
    Analyze impact of judges' save rule under different scenarios.
    """
    
    def __init__(self):
        self.handler = JudgesSaveHandler()
    
    def analyze_save_impact(
        self,
        week_ctx,
        fan_votes: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Analyze how judges' save affects outcomes.
        
        Three scenarios:
        1. Technocratic: Judges save whoever has better judge score
        2. Populist: Judges save whoever has more fan votes
        3. Random: 50/50
        
        Returns dict with survival probabilities under each scenario.
        """
        eliminated = week_ctx.eliminated[0] if week_ctx.eliminated else None
        if not eliminated:
            return {}
        
        # Find bottom 2 by combined score
        combined_scores = {}
        for c in week_ctx.active_set:
            j_pct = week_ctx.judge_percentages.get(c, 0)
            f_pct = fan_votes.get(c, 0)
            combined_scores[c] = j_pct + f_pct
        
        sorted_scores = sorted(combined_scores.items(), key=lambda x: x[1])
        bottom_two = [c for c, _ in sorted_scores[:2]]
        
        if eliminated not in bottom_two:
            # Eliminated wasn't in bottom 2 by pure score - anomaly
            return {
                'anomaly': True,
                'eliminated': eliminated,
                'computed_bottom_two': bottom_two,
                'message': "Eliminated contestant wasn't in computed bottom 2"
            }
        
        other_bottom = [c for c in bottom_two if c != eliminated][0]
        
        # Scenario analysis
        results = {
            'bottom_two': bottom_two,
            'eliminated': eliminated,
            'other_in_bottom': other_bottom,
            'scenarios': {}
        }
        
        # Technocratic: save by judge score
        e_judge = week_ctx.judge_scores.get(eliminated, 0)
        o_judge = week_ctx.judge_scores.get(other_bottom, 0)
        
        if e_judge > o_judge:
            tech_saved = eliminated
        else:
            tech_saved = other_bottom
        
        results['scenarios']['technocratic'] = {
            'saved': tech_saved,
            'eliminated_same': tech_saved != eliminated
        }
        
        # Populist: save by fan vote
        e_fan = fan_votes.get(eliminated, 0)
        o_fan = fan_votes.get(other_bottom, 0)
        
        if e_fan > o_fan:
            pop_saved = eliminated
        else:
            pop_saved = other_bottom
        
        results['scenarios']['populist'] = {
            'saved': pop_saved,
            'eliminated_same': pop_saved != eliminated
        }
        
        # Random
        results['scenarios']['random'] = {
            'p_eliminated_saved': 0.5,
            'p_other_saved': 0.5
        }
        
        return results
    
    def simulate_season_without_save(
        self,
        season_context,
        inversion_result: InversionResult
    ) -> Dict[str, any]:
        """
        Simulate what would have happened without judges' save.
        
        Returns comparison of actual vs. simulated outcomes.
        """
        comparisons = {}
        
        for week, week_ctx in season_context.weeks.items():
            if not week_ctx.has_valid_elimination():
                continue
            
            estimates = inversion_result.week_results.get(week, {})
            if not estimates:
                continue
            
            # Compute combined scores
            combined = {}
            for c in week_ctx.active_set:
                j_pct = week_ctx.judge_percentages.get(c, 0)
                f_pct = estimates.get(c, FanVoteEstimate(
                    c, week, 0, 0, 1, 0, 'percent'
                )).point_estimate
                combined[c] = j_pct + f_pct
            
            # Who would be eliminated by pure score (no save)?
            pure_elim = min(combined.items(), key=lambda x: x[1])[0]
            
            # Actual eliminated
            actual_elim = week_ctx.eliminated[0] if week_ctx.eliminated else None
            
            comparisons[week] = {
                'actual_eliminated': actual_elim,
                'pure_score_eliminated': pure_elim,
                'save_affected_outcome': actual_elim != pure_elim
            }
        
        # Summary statistics
        total_weeks = len(comparisons)
        save_affected = sum(
            1 for c in comparisons.values() 
            if c['save_affected_outcome']
        )
        
        return {
            'week_comparisons': comparisons,
            'total_weeks': total_weeks,
            'weeks_affected_by_save': save_affected,
            'save_impact_rate': save_affected / total_weeks if total_weeks > 0 else 0
        }

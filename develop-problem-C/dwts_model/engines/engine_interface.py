"""
Common Interface for Inversion Engines
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class FanVoteEstimate:
    """Estimated fan vote for a single contestant-week"""
    contestant: str
    week: int
    point_estimate: float      # Best estimate
    lower_bound: float         # 95% CI lower
    upper_bound: float         # 95% CI upper
    certainty: float           # 0-1, how certain are we?
    method: str                # 'percent' or 'rank'
    
    def get_interval_width(self) -> float:
        """Get width of confidence interval"""
        return self.upper_bound - self.lower_bound
    
    def contains_value(self, value: float) -> bool:
        """Check if a value is within the CI"""
        return self.lower_bound <= value <= self.upper_bound


@dataclass
class InversionResult:
    """Complete result from an inversion engine"""
    season: int
    method: str  # 'percent', 'rank', 'judges_save'
    
    # Overall metrics
    inconsistency_score: float     # S* from Phase 1 (lower = better)
    is_feasible: bool              # Did we find a solution?
    
    # Per-week results
    week_results: Dict[int, Dict[str, FanVoteEstimate]] = field(default_factory=dict)
    
    # Slack variables (for debugging)
    slack_values: Dict[Tuple[int, str, str], float] = field(default_factory=dict)
    
    # Constraint violations
    violations: List[str] = field(default_factory=list)
    
    # Computation metadata
    solve_time: float = 0.0
    iterations: int = 0
    
    def get_contestant_trajectory(self, contestant: str) -> List[FanVoteEstimate]:
        """Get all estimates for a contestant across weeks"""
        trajectory = []
        for week in sorted(self.week_results.keys()):
            if contestant in self.week_results[week]:
                trajectory.append(self.week_results[week][contestant])
        return trajectory
    
    def get_week_estimates(self, week: int) -> Dict[str, FanVoteEstimate]:
        """Get all estimates for a specific week"""
        return self.week_results.get(week, {})
    
    def get_point_estimates_matrix(self) -> Dict[int, Dict[str, float]]:
        """Get just the point estimates as a simple dict"""
        result = {}
        for week, estimates in self.week_results.items():
            result[week] = {
                c: e.point_estimate for c, e in estimates.items()
            }
        return result
    
    def get_uncertainty_matrix(self) -> Dict[int, Dict[str, float]]:
        """Get certainty values for all estimates"""
        result = {}
        for week, estimates in self.week_results.items():
            result[week] = {
                c: e.certainty for c, e in estimates.items()
            }
        return result
    
    def compute_overall_certainty(self) -> float:
        """Compute average certainty across all estimates"""
        all_certainties = []
        for estimates in self.week_results.values():
            for e in estimates.values():
                all_certainties.append(e.certainty)
        return np.mean(all_certainties) if all_certainties else 0.0


class InversionEngine(ABC):
    """Abstract base class for inversion engines"""
    
    @abstractmethod
    def solve(self, season_context) -> InversionResult:
        """
        Solve the fan vote inversion problem.
        
        Args:
            season_context: SeasonContext from ActiveSetManager
            
        Returns:
            InversionResult with estimated fan votes
        """
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Return the name of the inversion method"""
        pass
    
    def validate_result(self, result: InversionResult, season_context) -> List[str]:
        """
        Validate that results are consistent with elimination outcomes.
        
        Returns list of validation errors.
        """
        errors = []
        
        for week, week_ctx in season_context.weeks.items():
            if week not in result.week_results:
                continue
            
            estimates = result.week_results[week]
            eliminated = week_ctx.eliminated
            survivors = week_ctx.survivors
            
            # Check pairwise constraints
            for e in eliminated:
                for s in survivors:
                    if e not in estimates or s not in estimates:
                        continue
                    
                    e_total = self._compute_total(
                        estimates[e].point_estimate,
                        week_ctx.judge_percentages.get(e, 0),
                        week_ctx.judge_ranks.get(e, 0),
                        result.method
                    )
                    s_total = self._compute_total(
                        estimates[s].point_estimate,
                        week_ctx.judge_percentages.get(s, 0),
                        week_ctx.judge_ranks.get(s, 0),
                        result.method
                    )
                    
                    # For both methods, eliminated should have higher total
                    # (higher = worse in rank, lower = worse in percent)
                    if result.method == 'rank':
                        if e_total < s_total:  # Lower rank is better
                            errors.append(
                                f"Week {week}: {e} (rank {e_total}) beats "
                                f"{s} (rank {s_total}) but was eliminated"
                            )
                    else:  # percent
                        if e_total > s_total:  # Higher percent is better
                            errors.append(
                                f"Week {week}: {e} ({e_total:.3f}) beats "
                                f"{s} ({s_total:.3f}) but was eliminated"
                            )
        
        return errors
    
    def _compute_total(
        self, 
        fan_value: float, 
        judge_pct: float,
        judge_rank: int,
        method: str
    ) -> float:
        """Compute combined total based on method"""
        if method == 'rank':
            # Assume fan_value is fan rank
            return fan_value + judge_rank
        else:
            # Percent: fan_value is fan percentage
            return fan_value + judge_pct


@dataclass
class EngineComparison:
    """Compare results from different engines/methods"""
    season: int
    results: Dict[str, InversionResult] = field(default_factory=dict)
    
    def add_result(self, method: str, result: InversionResult):
        self.results[method] = result
    
    def compare_eliminations(self) -> Dict[int, Dict[str, str]]:
        """
        Compare which contestant would be eliminated under each method.
        
        Returns: {week: {method: eliminated_contestant}}
        """
        comparisons = {}
        
        # Get all weeks from any result
        all_weeks = set()
        for result in self.results.values():
            all_weeks.update(result.week_results.keys())
        
        for week in sorted(all_weeks):
            comparisons[week] = {}
            for method, result in self.results.items():
                if week in result.week_results:
                    # Find contestant with lowest combined score
                    estimates = result.week_results[week]
                    # This is a simplified comparison
                    comparisons[week][method] = self._find_lowest(estimates)
        
        return comparisons
    
    def _find_lowest(self, estimates: Dict[str, FanVoteEstimate]) -> str:
        """Find contestant with lowest point estimate"""
        if not estimates:
            return "N/A"
        return min(estimates.items(), key=lambda x: x[1].point_estimate)[0]
    
    def get_reversal_rate(self) -> float:
        """
        Calculate rate at which different methods produce different eliminations.
        """
        if len(self.results) < 2:
            return 0.0
        
        comparisons = self.compare_eliminations()
        reversals = 0
        total_weeks = 0
        
        methods = list(self.results.keys())
        for week, week_comp in comparisons.items():
            if len(week_comp) >= 2:
                total_weeks += 1
                # Check if any two methods disagree
                values = list(week_comp.values())
                if len(set(values)) > 1:
                    reversals += 1
        
        return reversals / total_weeks if total_weeks > 0 else 0.0

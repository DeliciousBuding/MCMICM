"""
Monte Carlo Robustness Analysis for Fan Vote Intervals

Uses MCMC sampling within LP-derived bounds to compute probabilistic
robustness metrics instead of binary interval classifications.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from .dirichlet_sampler import DirichletHitAndRunSampler, SamplerConfig
from .counterfactual import CounterfactualSimulator


@dataclass
class MCRobustnessResult:
    """Results from Monte Carlo robustness analysis"""
    contestant: str
    week: int
    season: int
    
    # Probabilistic metrics
    p_wrongful: float  # Probability of wrongful elimination
    p_correct: float   # Probability elimination was correct
    
    # Sample statistics
    n_samples: int
    wrongful_count: int
    correct_count: int
    
    # Confidence intervals
    ci_lower: float = 0.0
    ci_upper: float = 1.0
    
    # Original bounds
    fan_vote_lower: float = 0.0
    fan_vote_upper: float = 1.0
    
    # Additional metrics
    mean_fan_vote: float = 0.0
    median_fan_vote: float = 0.0
    
    def get_classification(self, threshold: float = 0.05) -> str:
        """
        Classify based on probability threshold
        
        Args:
            threshold: Probability threshold (default 5%)
            
        Returns:
            'Definite-Wrongful' if p_wrongful > 1-threshold
            'Definite-Correct' if p_wrongful < threshold
            'Uncertain' otherwise
        """
        if self.p_wrongful > 1 - threshold:
            return 'Definite-Wrongful'
        elif self.p_wrongful < threshold:
            return 'Definite-Correct'
        else:
            return 'Uncertain'


class MonteCarloRobustnessAnalyzer:
    """
    Monte Carlo-based robustness analysis for elimination fairness.
    
    Key improvements over interval-robust classification:
    1. Probabilistic rather than binary (3-class) classification
    2. Quantifies uncertainty with confidence intervals
    3. Can incorporate non-uniform priors (e.g., historical fan distributions)
    4. Sensitivity analysis across different sampling distributions
    """
    
    def __init__(
        self,
        n_samples: int = 10000,
        burnin: int = 1000,
        thin: int = 5,
        confidence_level: float = 0.95
    ):
        """
        Initialize analyzer
        
        Args:
            n_samples: Number of MCMC samples to draw
            burnin: Burn-in period for MCMC
            thin: Thinning interval
            confidence_level: Confidence level for intervals (default 95%)
        """
        self.n_samples = n_samples
        self.burnin = burnin
        self.thin = thin
        self.confidence_level = confidence_level
        
        # Initialize sampler and simulator
        sampler_config = SamplerConfig(
            n_samples=n_samples,
            burnin=burnin,
            thin=thin
        )
        self.sampler = DirichletHitAndRunSampler(sampler_config)
        self.simulator = CounterfactualSimulator()
    
    def analyze_elimination(
        self,
        season: int,
        week: int,
        eliminated: str,
        week_context,
        interval_bounds: Dict[str, Tuple[float, float]],
        voting_method: str = 'percent'
    ) -> MCRobustnessResult:
        """
        Analyze robustness of a single elimination using Monte Carlo sampling.
        
        Args:
            season: Season number
            week: Week number
            eliminated: Name of eliminated contestant
            week_context: WeekContext with judge scores etc
            interval_bounds: LP-derived bounds {contestant: (lower, upper)}
            voting_method: 'percent' or 'rank'
            
        Returns:
            MCRobustnessResult with probabilistic metrics
        """
        contestants = list(week_context.active_set)
        
        # Sample fan vote distributions within bounds
        samples = self._sample_fan_votes_bounded(
            contestants=contestants,
            judge_percentages=week_context.judge_percentages,
            interval_bounds=interval_bounds,
            eliminated=eliminated
        )
        
        # Simulate elimination for each sample
        wrongful_count = 0
        correct_count = 0
        fan_vote_samples = []
        
        for sample in samples:
            # Simulate who would be eliminated under this fan vote
            simulated_eliminated = self._simulate_single_week(
                contestants=contestants,
                judge_percentages=week_context.judge_percentages,
                judge_ranks=week_context.judge_ranks,
                fan_votes=sample,
                method=voting_method
            )
            
            # Check if actual elimination matches simulated
            if simulated_eliminated == eliminated:
                correct_count += 1
            else:
                wrongful_count += 1
            
            # Track fan vote for eliminated contestant
            fan_vote_samples.append(sample.get(eliminated, 0.0))
        
        # Compute probabilities
        total = wrongful_count + correct_count
        p_wrongful = wrongful_count / total if total > 0 else 0.0
        p_correct = correct_count / total if total > 0 else 0.0
        
        # Confidence interval for p_wrongful (Wilson score interval)
        ci_lower, ci_upper = self._wilson_confidence_interval(
            wrongful_count, total, self.confidence_level
        )
        
        # Fan vote statistics
        fan_vote_samples = np.array(fan_vote_samples)
        mean_fan = np.mean(fan_vote_samples)
        median_fan = np.median(fan_vote_samples)
        
        return MCRobustnessResult(
            contestant=eliminated,
            week=week,
            season=season,
            p_wrongful=p_wrongful,
            p_correct=p_correct,
            n_samples=total,
            wrongful_count=wrongful_count,
            correct_count=correct_count,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            fan_vote_lower=interval_bounds.get(eliminated, (0, 1))[0],
            fan_vote_upper=interval_bounds.get(eliminated, (0, 1))[1],
            mean_fan_vote=mean_fan,
            median_fan_vote=median_fan
        )
    
    def _sample_fan_votes_bounded(
        self,
        contestants: List[str],
        judge_percentages: Dict[str, float],
        interval_bounds: Dict[str, Tuple[float, float]],
        eliminated: str
    ) -> List[Dict[str, float]]:
        """
        Sample fan votes respecting LP interval bounds.
        
        Strategy: Use rejection sampling or constrained MCMC
        """
        samples = []
        max_attempts = self.n_samples * 10  # Prevent infinite loop
        attempts = 0
        
        while len(samples) < self.n_samples and attempts < max_attempts:
            attempts += 1
            
            # Sample from Dirichlet (using judge scores as prior)
            alpha = [judge_percentages.get(c, 1.0) for c in contestants]
            alpha = np.array(alpha) * 10 + 1  # Scale for stronger prior
            
            sample_array = np.random.dirichlet(alpha)
            sample_dict = {c: sample_array[i] for i, c in enumerate(contestants)}
            
            # Check if sample respects bounds
            valid = True
            for c, (lower, upper) in interval_bounds.items():
                if c in sample_dict:
                    if not (lower <= sample_dict[c] <= upper):
                        valid = False
                        break
            
            if valid:
                samples.append(sample_dict)
        
        # If rejection sampling failed, fall back to grid sampling
        if len(samples) < self.n_samples // 2:
            samples = self._grid_sample_within_bounds(
                contestants, interval_bounds, self.n_samples
            )
        
        return samples[:self.n_samples]
    
    def _grid_sample_within_bounds(
        self,
        contestants: List[str],
        interval_bounds: Dict[str, Tuple[float, float]],
        n_samples: int
    ) -> List[Dict[str, float]]:
        """Fallback: uniform grid sampling within bounds"""
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            remaining = 1.0
            
            for i, c in enumerate(contestants[:-1]):
                lower, upper = interval_bounds.get(c, (0.01, remaining))
                lower = max(lower, 0.01)
                upper = min(upper, remaining - 0.01 * (len(contestants) - i - 1))
                
                if lower < upper:
                    val = np.random.uniform(lower, upper)
                else:
                    val = lower
                
                sample[c] = val
                remaining -= val
            
            # Last contestant gets remainder
            sample[contestants[-1]] = max(0.01, remaining)
            
            # Normalize
            total = sum(sample.values())
            sample = {k: v/total for k, v in sample.items()}
            
            samples.append(sample)
        
        return samples
    
    def _simulate_single_week(
        self,
        contestants: List[str],
        judge_percentages: Dict[str, float],
        judge_ranks: Dict[str, int],
        fan_votes: Dict[str, float],
        method: str = 'percent'
    ) -> str:
        """Simulate who gets eliminated under given fan votes"""
        combined = {}
        
        for c in contestants:
            if method == 'percent':
                # 50/50 combined score
                j_pct = judge_percentages.get(c, 0.0)
                f_pct = fan_votes.get(c, 0.0)
                combined[c] = 0.5 * j_pct + 0.5 * f_pct
            else:  # rank
                j_rank = judge_ranks.get(c, len(contestants))
                # Approximate fan rank from percentages
                sorted_fans = sorted(
                    contestants,
                    key=lambda x: fan_votes.get(x, 0.0),
                    reverse=True
                )
                f_rank = sorted_fans.index(c) + 1
                combined[c] = j_rank + f_rank
        
        # Lowest percent or highest rank = eliminated
        if method == 'percent':
            return min(combined.items(), key=lambda x: x[1])[0]
        else:
            return max(combined.items(), key=lambda x: x[1])[0]
    
    def _wilson_confidence_interval(
        self,
        successes: int,
        total: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Wilson score confidence interval for binomial proportion.
        
        More robust than normal approximation for small samples.
        """
        if total == 0:
            return 0.0, 1.0
        
        p = successes / total
        z = 1.96 if confidence == 0.95 else 2.576  # z-score
        
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
        
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        
        return lower, upper
    
    def analyze_season(
        self,
        season_context,
        inversion_result,
        voting_method: str = 'percent'
    ) -> List[MCRobustnessResult]:
        """
        Analyze all eliminations in a season.
        
        Args:
            season_context: SeasonContext from ActiveSetManager
            inversion_result: InversionResult with bounds
            voting_method: 'percent' or 'rank'
            
        Returns:
            List of MCRobustnessResult for each elimination
        """
        results = []
        
        for week, week_ctx in season_context.weeks.items():
            if not week_ctx.has_valid_elimination():
                continue
            
            eliminated = week_ctx.eliminated
            if not eliminated:
                continue
            
            # Get interval bounds for this week
            week_bounds = inversion_result.get_week_bounds(week)
            
            result = self.analyze_elimination(
                season=season_context.season,
                week=week,
                eliminated=eliminated,
                week_context=week_ctx,
                interval_bounds=week_bounds,
                voting_method=voting_method
            )
            
            results.append(result)
        
        return results

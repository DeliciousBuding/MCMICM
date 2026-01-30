"""
Monte Carlo Robustness Analysis for Fan Vote Intervals

Uses MCMC sampling within LP-derived bounds to compute probabilistic
robustness metrics instead of binary interval classifications.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from .dirichlet_sampler import DirichletHitAndRunSampler, SamplerConfig, SimplexProjection
from .counterfactual import CounterfactualSimulator


@dataclass
class MCRobustnessResult:
    """Results from Monte Carlo robustness analysis"""
    contestant: str
    week: int
    season: int
    
    # Probabilistic metrics
    p_wrongful: float  # Probability of popularity dissonance
    p_correct: float   # Probability of popularity alignment
    
    # Sample statistics
    n_samples: int
    wrongful_count: int
    correct_count: int

    # Sampling diagnostics
    attempts: int = 0
    acceptance_rate: float = 0.0
    
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
        if self.n_samples == 0:
            return 'Uncertain'
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
        confidence_level: float = 0.95,
        max_attempts_multiplier: int = 25
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
        self.max_attempts_multiplier = max_attempts_multiplier
        self.rng = np.random.default_rng()
        
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
        voting_method: str = 'percent',
        has_judges_save: bool = False
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
        
        # Rejection sampling: enforce rule-consistent outcomes first
        samples, attempts = self._sample_fan_votes_consistent(
            contestants=contestants,
            judge_percentages=week_context.judge_percentages,
            judge_ranks=week_context.judge_ranks,
            interval_bounds=interval_bounds,
            eliminated=eliminated,
            week_context=week_context,
            voting_method=voting_method,
            has_judges_save=has_judges_save
        )
        
        # Popularity dissonance counts (conditional on rule consistency)
        wrongful_count = 0
        correct_count = 0
        fan_vote_samples = []
        
        for sample in samples:
            min_vote = min(sample.values()) if sample else 0.0
            eliminated_vote = sample.get(eliminated, 0.0)
            
            if eliminated_vote > min_vote + 1e-12:
                wrongful_count += 1
            else:
                correct_count += 1
            
            fan_vote_samples.append(eliminated_vote)
        
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
        
        acceptance_rate = (len(samples) / attempts) if attempts > 0 else 0.0
        
        return MCRobustnessResult(
            contestant=eliminated,
            week=week,
            season=season,
            p_wrongful=p_wrongful,
            p_correct=p_correct,
            n_samples=total,
            wrongful_count=wrongful_count,
            correct_count=correct_count,
            attempts=attempts,
            acceptance_rate=acceptance_rate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            fan_vote_lower=interval_bounds.get(eliminated, (0, 1))[0],
            fan_vote_upper=interval_bounds.get(eliminated, (0, 1))[1],
            mean_fan_vote=mean_fan,
            median_fan_vote=median_fan
        )
    
    def _sample_fan_votes_consistent(
        self,
        contestants: List[str],
        judge_percentages: Dict[str, float],
        judge_ranks: Dict[str, int],
        interval_bounds: Dict[str, Tuple[float, float]],
        eliminated: str,
        week_context,
        voting_method: str,
        has_judges_save: bool
    ) -> Tuple[List[Dict[str, float]], int]:
        """
        Sample fan votes respecting interval bounds and observed outcomes.
        
        Strategy: Proposal sampling within bounds + rejection on rule consistency.
        """
        samples = []
        max_attempts = self.n_samples * self.max_attempts_multiplier  # Prevent infinite loop
        attempts = 0
        observed_fan_ranks = self._extract_observed_fan_ranks(week_context)
        
        while len(samples) < self.n_samples and attempts < max_attempts:
            attempts += 1
            
            sample_dict = self._propose_fan_votes(contestants, interval_bounds)
            if sample_dict is None:
                continue
            
            if not self._is_rule_consistent(
                contestants=contestants,
                judge_percentages=judge_percentages,
                judge_ranks=judge_ranks,
                fan_votes=sample_dict,
                eliminated=eliminated,
                week_context=week_context,
                voting_method=voting_method,
                has_judges_save=has_judges_save
            ):
                continue
            
            if observed_fan_ranks and not self._matches_observed_fan_ranks(
                contestants=contestants,
                fan_votes=sample_dict,
                observed_fan_ranks=observed_fan_ranks
            ):
                continue
            
            samples.append(sample_dict)
        
        # If acceptance is low, fall back to grid sampling + rule filter
        if len(samples) < max(1, self.n_samples // 2):
            grid_samples = self._grid_sample_within_bounds(
                contestants, interval_bounds, self.n_samples * 2
            )
            for sample_dict in grid_samples:
                if len(samples) >= self.n_samples:
                    break
                if not self._is_rule_consistent(
                    contestants=contestants,
                    judge_percentages=judge_percentages,
                    judge_ranks=judge_ranks,
                    fan_votes=sample_dict,
                    eliminated=eliminated,
                    week_context=week_context,
                    voting_method=voting_method,
                    has_judges_save=has_judges_save
                ):
                    continue
                if observed_fan_ranks and not self._matches_observed_fan_ranks(
                    contestants=contestants,
                    fan_votes=sample_dict,
                    observed_fan_ranks=observed_fan_ranks
                ):
                    continue
                samples.append(sample_dict)
        
        return samples[:self.n_samples], attempts

    def _propose_fan_votes(
        self,
        contestants: List[str],
        interval_bounds: Dict[str, Tuple[float, float]]
    ) -> Optional[Dict[str, float]]:
        """
        Propose a fan vote vector within bounds using uniform simplex sampling.
        """
        sample_array = SimplexProjection.sample_uniform_simplex(len(contestants), rng=self.rng)
        sample_dict = {c: sample_array[i] for i, c in enumerate(contestants)}
        
        if self._within_bounds(sample_dict, interval_bounds):
            return sample_dict
        
        return None

    def _within_bounds(
        self,
        sample_dict: Dict[str, float],
        interval_bounds: Dict[str, Tuple[float, float]],
        tol: float = 1e-12
    ) -> bool:
        """Check if a sample respects interval bounds."""
        for c, (lower, upper) in interval_bounds.items():
            if c in sample_dict:
                if sample_dict[c] < lower - tol or sample_dict[c] > upper + tol:
                    return False
        return True

    def _is_rule_consistent(
        self,
        contestants: List[str],
        judge_percentages: Dict[str, float],
        judge_ranks: Dict[str, int],
        fan_votes: Dict[str, float],
        eliminated: str,
        week_context,
        voting_method: str,
        has_judges_save: bool
    ) -> bool:
        """
        Check if a fan vote sample reproduces the observed elimination.
        """
        combined = self._compute_combined_scores(
            contestants=contestants,
            judge_percentages=judge_percentages,
            judge_ranks=judge_ranks,
            fan_votes=fan_votes,
            method=voting_method
        )
        
        elimination_slots = max(1, len(week_context.eliminated))
        
        if voting_method == 'percent':
            bottom_set = self._bottom_set(combined, elimination_slots, worst_high=False)
        else:
            required_slots = max(elimination_slots, 2) if has_judges_save else elimination_slots
            bottom_set = self._bottom_set(combined, required_slots, worst_high=True)
        
        return eliminated in bottom_set

    def _bottom_set(
        self,
        combined: Dict[str, float],
        k: int,
        worst_high: bool = True,
        tol: float = 1e-12
    ) -> set:
        """
        Return the bottom-k set under a scoring rule.
        worst_high=True means higher score is worse (rank method).
        """
        if not combined:
            return set()
        
        k = max(1, min(k, len(combined)))
        sorted_items = sorted(combined.items(), key=lambda x: x[1], reverse=worst_high)
        if len(sorted_items) <= k:
            return {c for c, _ in sorted_items}
        
        threshold = sorted_items[k - 1][1]
        if worst_high:
            return {c for c, v in combined.items() if v >= threshold - tol}
        return {c for c, v in combined.items() if v <= threshold + tol}

    def _extract_observed_fan_ranks(self, week_context) -> Optional[Dict[str, int]]:
        """
        Optional: return observed fan ranks if available in context.
        """
        for attr in ("observed_fan_ranks", "fan_ranks_observed", "fan_vote_ranks"):
            if hasattr(week_context, attr):
                value = getattr(week_context, attr)
                if isinstance(value, dict) and value:
                    return value
        return None

    def _matches_observed_fan_ranks(
        self,
        contestants: List[str],
        fan_votes: Dict[str, float],
        observed_fan_ranks: Dict[str, int]
    ) -> bool:
        """
        Check if simulated fan ranks match observed ranks (when available).
        """
        fan_sorted = sorted(
            contestants,
            key=lambda x: fan_votes.get(x, 0.0),
            reverse=True
        )
        simulated_ranks = {c: r + 1 for r, c in enumerate(fan_sorted)}
        for c, observed_rank in observed_fan_ranks.items():
            if c in simulated_ranks and simulated_ranks[c] != observed_rank:
                return False
        return True
    
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
    
    def _compute_combined_scores(
        self,
        contestants: List[str],
        judge_percentages: Dict[str, float],
        judge_ranks: Dict[str, int],
        fan_votes: Dict[str, float],
        method: str = 'percent'
    ) -> Dict[str, float]:
        """Compute combined scores under given fan votes"""
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
        
        return combined
    
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
                voting_method=voting_method,
                has_judges_save=season_context.has_judges_save
            )
            
            results.append(result)
        
        return results

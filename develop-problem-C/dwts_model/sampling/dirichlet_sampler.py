"""
Dirichlet Hit-and-Run Sampler for Fan Vote Distribution

Samples from the feasible region of fan votes while respecting:
1. Simplex constraint: sum(F_i) = 1
2. Elimination constraints from LP
3. Dirichlet prior with judge-correlated alpha

Algorithm: Hit-and-Run MCMC on constrained simplex
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import warnings


@dataclass
class SamplerConfig:
    """Configuration for MCMC sampler"""
    n_samples: int = 10000
    burnin: int = 2000
    thin: int = 10
    
    # Dirichlet prior parameters
    alpha_base: float = 1.0           # Base alpha (1 = uniform)
    judge_correlation: float = 0.3    # Lambda: alpha = 1 + lambda * J_normalized
    
    # Hit-and-run parameters
    step_scale: float = 0.1           # Initial step size
    adapt_interval: int = 100         # Adapt step size every N iterations
    target_acceptance: float = 0.4    # Target acceptance rate
    
    # Constraints
    feasibility_tolerance: float = 1e-6


@dataclass
class MCMCSample:
    """Single MCMC sample"""
    fan_votes: Dict[str, float]
    log_prior: float
    log_likelihood: float
    accepted: bool


@dataclass
class MCMCResult:
    """Complete MCMC result"""
    samples: List[Dict[str, float]]
    acceptance_rate: float
    effective_sample_size: Dict[str, float]
    
    # Posterior summaries
    posterior_mean: Dict[str, float] = field(default_factory=dict)
    posterior_std: Dict[str, float] = field(default_factory=dict)
    credible_interval_95: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    def compute_summaries(self):
        """Compute posterior summaries from samples"""
        if not self.samples:
            return
        
        contestants = list(self.samples[0].keys())
        
        for c in contestants:
            values = [s[c] for s in self.samples]
            self.posterior_mean[c] = np.mean(values)
            self.posterior_std[c] = np.std(values)
            self.credible_interval_95[c] = (
                np.percentile(values, 2.5),
                np.percentile(values, 97.5)
            )


class DirichletHitAndRunSampler:
    """
    MCMC sampler for fan votes using Hit-and-Run algorithm
    on a constrained simplex with Dirichlet prior.
    
    Key features:
    1. Respects simplex constraint (sum = 1)
    2. Incorporates elimination constraints as soft/hard boundaries
    3. Uses Dirichlet prior with judge-correlated alpha
    4. Adaptive step size for efficient sampling
    """
    
    def __init__(self, config: Optional[SamplerConfig] = None):
        self.config = config or SamplerConfig()
        self.rng = np.random.default_rng()
        
    def sample(
        self,
        contestants: List[str],
        judge_percentages: Dict[str, float],
        constraints: List[Tuple[str, str]],
        initial_point: Optional[Dict[str, float]] = None
    ) -> MCMCResult:
        """
        Sample from posterior distribution of fan votes.
        
        Args:
            contestants: List of contestant names
            judge_percentages: Judge score percentages
            constraints: List of (eliminated, survivor) pairs
            initial_point: Starting point (default: uniform)
            
        Returns:
            MCMCResult with samples and summaries
        """
        n = len(contestants)
        
        # Compute Dirichlet alpha based on judge scores
        alpha = self._compute_alpha(contestants, judge_percentages)
        
        # Initialize
        if initial_point:
            current = np.array([initial_point[c] for c in contestants])
        else:
            # Start with Dirichlet draw
            current = self.rng.dirichlet(alpha)
        
        # Ensure feasibility
        current = self._project_to_feasible(current, contestants, constraints, judge_percentages)
        
        samples = []
        accepted = 0
        total = 0
        step_scale = self.config.step_scale
        
        total_iterations = self.config.burnin + self.config.n_samples * self.config.thin
        
        for iteration in range(total_iterations):
            # Propose new point using hit-and-run
            proposal = self._hit_and_run_proposal(current, step_scale)
            
            # Check feasibility
            if not self._is_feasible(proposal, contestants, constraints, judge_percentages):
                # Reject infeasible proposals
                total += 1
                continue
            
            # Compute acceptance probability (Metropolis-Hastings)
            log_ratio = (
                self._log_dirichlet_pdf(proposal, alpha) -
                self._log_dirichlet_pdf(current, alpha)
            )
            
            # Accept/reject
            if np.log(self.rng.random()) < log_ratio:
                current = proposal
                accepted += 1
            
            total += 1
            
            # Save sample (after burnin, with thinning)
            if iteration >= self.config.burnin:
                if (iteration - self.config.burnin) % self.config.thin == 0:
                    sample_dict = {c: current[i] for i, c in enumerate(contestants)}
                    samples.append(sample_dict)
            
            # Adapt step size
            if iteration > 0 and iteration % self.config.adapt_interval == 0:
                acceptance_rate = accepted / max(1, total)
                if acceptance_rate < self.config.target_acceptance - 0.1:
                    step_scale *= 0.9
                elif acceptance_rate > self.config.target_acceptance + 0.1:
                    step_scale *= 1.1
                step_scale = np.clip(step_scale, 0.01, 1.0)
        
        # Compute results
        result = MCMCResult(
            samples=samples,
            acceptance_rate=accepted / max(1, total),
            effective_sample_size=self._compute_ess(samples, contestants)
        )
        result.compute_summaries()
        
        return result
    
    def _compute_alpha(
        self,
        contestants: List[str],
        judge_percentages: Dict[str, float]
    ) -> np.ndarray:
        """
        Compute Dirichlet alpha parameters.
        
        alpha_i = alpha_base + lambda * J_i_normalized
        
        This encodes the prior belief that fan votes are
        weakly correlated with judge scores.
        """
        alpha = []
        
        # Normalize judge scores to [0, 1] range
        j_values = [judge_percentages.get(c, 0) for c in contestants]
        j_min, j_max = min(j_values), max(j_values)
        j_range = j_max - j_min if j_max > j_min else 1.0
        
        for c in contestants:
            j_normalized = (judge_percentages.get(c, 0) - j_min) / j_range
            a = self.config.alpha_base + self.config.judge_correlation * j_normalized
            alpha.append(max(0.1, a))  # Ensure positive
        
        return np.array(alpha)
    
    def _hit_and_run_proposal(
        self,
        current: np.ndarray,
        step_scale: float
    ) -> np.ndarray:
        """
        Generate proposal using hit-and-run on simplex.
        
        1. Choose random direction in simplex tangent space
        2. Move along direction by random step
        3. Project back to simplex
        """
        n = len(current)
        
        # Random direction (sum to 0 for tangent to simplex)
        direction = self.rng.standard_normal(n)
        direction = direction - np.mean(direction)  # Project to sum=0
        
        # Normalize
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
        
        # Random step size
        step = self.rng.normal(0, step_scale)
        
        # Propose
        proposal = current + step * direction
        
        # Project to simplex (non-negative, sum to 1)
        proposal = np.maximum(proposal, 0)
        proposal = proposal / np.sum(proposal)
        
        return proposal
    
    def _project_to_feasible(
        self,
        point: np.ndarray,
        contestants: List[str],
        constraints: List[Tuple[str, str]],
        judge_percentages: Dict[str, float]
    ) -> np.ndarray:
        """
        Project point to feasible region.
        Uses iterative adjustment.
        """
        max_iterations = 100
        
        for _ in range(max_iterations):
            if self._is_feasible(point, contestants, constraints, judge_percentages):
                return point
            
            # Find most violated constraint and adjust
            worst_violation = 0
            worst_idx_e = -1
            worst_idx_s = -1
            
            for e, s in constraints:
                e_idx = contestants.index(e)
                s_idx = contestants.index(s)
                
                J_e = judge_percentages.get(e, 0)
                J_s = judge_percentages.get(s, 0)
                
                # Violation: F_e + J_e should be < F_s + J_s
                violation = (point[e_idx] + J_e) - (point[s_idx] + J_s)
                
                if violation > worst_violation:
                    worst_violation = violation
                    worst_idx_e = e_idx
                    worst_idx_s = s_idx
            
            if worst_idx_e >= 0:
                # Adjust: decrease F_e, increase F_s
                delta = worst_violation / 2 + self.config.feasibility_tolerance
                point[worst_idx_e] = max(0, point[worst_idx_e] - delta)
                point[worst_idx_s] = point[worst_idx_s] + delta
                
                # Renormalize
                point = point / np.sum(point)
            else:
                break
        
        return point
    
    def _is_feasible(
        self,
        point: np.ndarray,
        contestants: List[str],
        constraints: List[Tuple[str, str]],
        judge_percentages: Dict[str, float]
    ) -> bool:
        """Check if point satisfies all constraints"""
        tol = self.config.feasibility_tolerance
        
        # Non-negativity
        if np.any(point < -tol):
            return False
        
        # Simplex
        if abs(np.sum(point) - 1.0) > tol:
            return False
        
        # Elimination constraints
        for e, s in constraints:
            e_idx = contestants.index(e)
            s_idx = contestants.index(s)
            
            J_e = judge_percentages.get(e, 0)
            J_s = judge_percentages.get(s, 0)
            
            # F_e + J_e < F_s + J_s (with tolerance)
            if point[e_idx] + J_e > point[s_idx] + J_s + tol:
                return False
        
        return True
    
    def _log_dirichlet_pdf(
        self,
        x: np.ndarray,
        alpha: np.ndarray
    ) -> float:
        """Compute log of Dirichlet PDF (up to normalization constant)"""
        # log p(x) ∝ sum((alpha_i - 1) * log(x_i))
        x_safe = np.maximum(x, 1e-10)
        return np.sum((alpha - 1) * np.log(x_safe))
    
    def _compute_ess(
        self,
        samples: List[Dict[str, float]],
        contestants: List[str]
    ) -> Dict[str, float]:
        """
        Compute effective sample size using autocorrelation.
        """
        ess = {}
        
        for c in contestants:
            values = np.array([s[c] for s in samples])
            
            if len(values) < 10:
                ess[c] = len(values)
                continue
            
            # Simple autocorrelation-based ESS
            n = len(values)
            mean = np.mean(values)
            var = np.var(values)
            
            if var < 1e-10:
                ess[c] = n
                continue
            
            # Compute autocorrelation up to lag 100
            max_lag = min(100, n // 2)
            rho_sum = 0
            
            for lag in range(1, max_lag):
                cov = np.mean((values[:-lag] - mean) * (values[lag:] - mean))
                rho = cov / var
                
                if rho < 0.05:  # Cut off at small autocorrelation
                    break
                rho_sum += rho
            
            # ESS = n / (1 + 2 * sum of autocorrelations)
            ess[c] = n / (1 + 2 * rho_sum)
        
        return ess


class SimplexProjection:
    """Utility class for simplex operations"""
    
    @staticmethod
    def project_to_simplex(v: np.ndarray) -> np.ndarray:
        """
        Project vector onto probability simplex.
        Uses Duchi et al. algorithm.
        """
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
        theta = (cssv[rho] - 1) / (rho + 1)
        return np.maximum(v - theta, 0)
    
    @staticmethod
    def sample_uniform_simplex(n: int, rng=None) -> np.ndarray:
        """Sample uniformly from n-simplex"""
        if rng is None:
            rng = np.random.default_rng()
        
        # Exponential method
        e = rng.exponential(scale=1.0, size=n)
        return e / np.sum(e)

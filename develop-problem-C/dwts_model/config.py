"""
Global Configuration for DWTS Model
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
CACHE_DIR = OUTPUT_DIR / "cache"

# Create directories
for d in [OUTPUT_DIR, FIGURE_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

@dataclass
class SeasonConfig:
    """Season-specific configurations based on known rule changes"""
    
    # Rule periods (based on problem statement)
    RANK_SEASONS_EARLY: Tuple[int, ...] = (1, 2)  # Original rank-based
    PERCENT_SEASONS: Tuple[int, ...] = tuple(range(3, 28))  # S3-S27: Percent-based
    RANK_SEASONS_LATE: Tuple[int, ...] = tuple(range(28, 35))  # S28-S34: Rank + Judges' Save
    
    # Judges' Save introduced in S28
    JUDGES_SAVE_START: int = 28
    
    # All-Stars season (special handling)
    ALLSTAR_SEASON: int = 15
    
    # Controversial cases for analysis
    CONTROVERSY_CASES: Dict[int, List[str]] = field(default_factory=lambda: {
        2: ["Jerry Rice"],      # Runner up despite lowest judge scores
        4: ["Billy Ray Cyrus"], # 5th despite last place judge scores 6 weeks
        11: ["Bristol Palin"],  # 3rd with lowest judge scores 12 times
        27: ["Bobby Bones"],    # Won despite consistently low scores
    })
    
    def get_voting_method(self, season: int) -> str:
        """Determine voting method for a season"""
        if season in self.RANK_SEASONS_EARLY or season in self.RANK_SEASONS_LATE:
            return "rank"
        return "percent"
    
    def has_judges_save(self, season: int) -> bool:
        """Check if season has judges' save rule"""
        return season >= self.JUDGES_SAVE_START

@dataclass
class ModelConfig:
    """Model hyperparameters"""
    
    # LP Solver settings
    LP_SLACK_WEIGHT: float = 1.0  # L1 penalty weight for slack variables
    LP_TOLERANCE: float = 1e-6   # Numerical tolerance
    
    # CP-SAT settings
    CP_TIME_LIMIT: int = 60      # Seconds
    CP_NUM_WORKERS: int = 4      # Parallel workers
    
    # MCMC settings
    MCMC_SAMPLES: int = 10000    # Number of samples
    MCMC_BURNIN: int = 2000      # Burn-in period
    MCMC_THIN: int = 10          # Thinning interval
    
    # Dirichlet prior
    DIRICHLET_ALPHA_BASE: float = 1.0  # Base alpha (uniform)
    DIRICHLET_JUDGE_CORRELATION: float = 0.3  # Lambda in alpha = 1 + lambda * J
    
    # Bootstrap settings
    BOOTSTRAP_SAMPLES: int = 100  # For Cox model CI
    
    # Interval estimation
    INTERVAL_CONFIDENCE: float = 0.95

# Global instances
SEASON_CONFIG = SeasonConfig()
MODEL_CONFIG = ModelConfig()

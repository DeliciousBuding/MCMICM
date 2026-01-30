"""
Mechanism Design: Tiered Threshold System

Task 5: Propose a new voting system that is more "fair" or "better"

Design Philosophy: "Professional Floor, Popular Ceiling"
- Soft floor: Protect technical quality
- Elite mix: Balance rank and percent
- Signal release: Strategic information disclosure

This module implements and evaluates alternative voting mechanisms.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class MechanismType(Enum):
    """Types of voting mechanisms"""
    RANK = "rank"
    PERCENT = "percent"
    TIERED = "tiered"
    HYBRID = "hybrid"
    BORDA = "borda"
    INSTANT_RUNOFF = "instant_runoff"


@dataclass
class EliminationDecision:
    """Decision from a voting mechanism"""
    eliminated: str
    method: MechanismType
    combined_scores: Dict[str, float]
    reasoning: str
    was_close: bool = False  # Was it a close call?
    margin: float = 0.0      # How much worse was eliminated vs next?


@dataclass 
class MechanismEvaluation:
    """Evaluation metrics for a mechanism"""
    mechanism: MechanismType
    
    # Fairness metrics
    judge_alignment: float     # Correlation with judge preferences
    fan_alignment: float       # Correlation with fan preferences
    technical_floor: float     # % of low-skill eliminations prevented
    
    # Excitement metrics
    close_calls_rate: float    # % of weeks with close decisions
    upsets_rate: float         # % of unexpected outcomes
    volatility: float          # How much do rankings change week to week
    
    # Controversy potential
    controversy_index: float   # Potential for fan backlash
    
    # Overall
    overall_score: float = 0.0


class VotingMechanism(ABC):
    """Abstract base class for voting mechanisms"""
    
    @abstractmethod
    def decide_elimination(
        self,
        contestants: List[str],
        judge_scores: Dict[str, float],
        fan_votes: Dict[str, float]
    ) -> EliminationDecision:
        """Decide who to eliminate"""
        pass
    
    @abstractmethod
    def get_type(self) -> MechanismType:
        """Return mechanism type"""
        pass


class RankMechanism(VotingMechanism):
    """Original rank-based voting"""
    
    def get_type(self) -> MechanismType:
        return MechanismType.RANK
    
    def decide_elimination(
        self,
        contestants: List[str],
        judge_scores: Dict[str, float],
        fan_votes: Dict[str, float]
    ) -> EliminationDecision:
        # Convert to ranks
        judge_sorted = sorted(judge_scores.items(), key=lambda x: x[1], reverse=True)
        judge_ranks = {c: r+1 for r, (c, _) in enumerate(judge_sorted)}
        
        fan_sorted = sorted(fan_votes.items(), key=lambda x: x[1], reverse=True)
        fan_ranks = {c: r+1 for r, (c, _) in enumerate(fan_sorted)}
        
        # Combined rank (lower is better)
        combined = {c: judge_ranks.get(c, len(contestants)) + fan_ranks.get(c, len(contestants))
                   for c in contestants}
        
        # Highest combined rank is eliminated
        sorted_combined = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        eliminated = sorted_combined[0][0]
        
        margin = sorted_combined[0][1] - sorted_combined[1][1] if len(sorted_combined) > 1 else 0
        
        return EliminationDecision(
            eliminated=eliminated,
            method=MechanismType.RANK,
            combined_scores=combined,
            reasoning=f"Combined rank: {combined[eliminated]}",
            was_close=margin <= 1,
            margin=margin
        )


class PercentMechanism(VotingMechanism):
    """Percent-based voting (S3-S27 style)"""
    
    def get_type(self) -> MechanismType:
        return MechanismType.PERCENT
    
    def decide_elimination(
        self,
        contestants: List[str],
        judge_scores: Dict[str, float],
        fan_votes: Dict[str, float]
    ) -> EliminationDecision:
        # Normalize to percentages
        total_judge = sum(judge_scores.values())
        total_fan = sum(fan_votes.values())
        
        judge_pct = {c: judge_scores[c] / total_judge if total_judge > 0 else 1/len(contestants)
                    for c in contestants}
        fan_pct = {c: fan_votes[c] / total_fan if total_fan > 0 else 1/len(contestants)
                  for c in contestants}
        
        # Combined percentage (higher is better)
        combined = {c: judge_pct[c] + fan_pct[c] for c in contestants}
        
        # Lowest combined percentage is eliminated
        sorted_combined = sorted(combined.items(), key=lambda x: x[1])
        eliminated = sorted_combined[0][0]
        
        margin = sorted_combined[1][1] - sorted_combined[0][1] if len(sorted_combined) > 1 else 0
        
        return EliminationDecision(
            eliminated=eliminated,
            method=MechanismType.PERCENT,
            combined_scores=combined,
            reasoning=f"Combined %: {combined[eliminated]:.3f}",
            was_close=margin < 0.02,
            margin=margin
        )


@dataclass
class TieredThresholdConfig:
    """Configuration for tiered threshold system"""
    
    # Soft floor parameters
    floor_threshold_sigma: float = 2.0   # How many std below mean triggers floor
    floor_penalty: float = 0.5           # Fan vote weight multiplier when below floor
    
    # Elite mix weights
    rank_weight: float = 0.4             # Weight for rank component
    percent_weight: float = 0.6          # Weight for percent component
    
    # Signal release
    safe_zone_size: int = 3              # How many contestants in "safe zone"
    danger_zone_size: int = 2            # How many in "danger zone"


class TieredThresholdSystem(VotingMechanism):
    """
    Proposed new voting mechanism: Tiered Threshold System
    
    Philosophy: "Professional Floor, Popular Ceiling"
    
    Components:
    1. Soft Floor: If judge score < mean - 2*sigma, fan vote weight is reduced
    2. Elite Mix: Above floor, use 40% rank + 60% percent
    3. (Optional) Signal Release: Announce safe/danger zones without exact votes
    """
    
    def __init__(self, config: Optional[TieredThresholdConfig] = None):
        self.config = config or TieredThresholdConfig()
    
    def get_type(self) -> MechanismType:
        return MechanismType.TIERED
    
    def decide_elimination(
        self,
        contestants: List[str],
        judge_scores: Dict[str, float],
        fan_votes: Dict[str, float]
    ) -> EliminationDecision:
        """
        Three-stage decision process:
        1. Apply soft floor to adjust fan vote weights
        2. Compute rank and percent components
        3. Combine using elite mix formula
        """
        n = len(contestants)
        
        # Stage 1: Soft floor check
        scores = np.array([judge_scores[c] for c in contestants])
        mean_score = np.mean(scores)
        std_score = np.std(scores) if np.std(scores) > 0 else 1
        floor = mean_score - self.config.floor_threshold_sigma * std_score
        
        # Adjusted fan weights
        fan_weights = {}
        for c in contestants:
            if judge_scores[c] < floor:
                fan_weights[c] = self.config.floor_penalty
            else:
                fan_weights[c] = 1.0
        
        # Stage 2: Compute components
        # Rank component
        judge_sorted = sorted(judge_scores.items(), key=lambda x: x[1], reverse=True)
        judge_ranks = {c: r+1 for r, (c, _) in enumerate(judge_sorted)}
        
        # Weighted fan votes
        weighted_fans = {c: fan_votes[c] * fan_weights[c] for c in contestants}
        fan_sorted = sorted(weighted_fans.items(), key=lambda x: x[1], reverse=True)
        fan_ranks = {c: r+1 for r, (c, _) in enumerate(fan_sorted)}
        
        # Rank score (lower is better, normalize to 0-1)
        rank_scores = {c: (n - (judge_ranks[c] + fan_ranks[c]) / 2) / n for c in contestants}
        
        # Percent component
        total_judge = sum(judge_scores.values())
        total_fan = sum(weighted_fans.values())
        
        percent_scores = {}
        for c in contestants:
            j_pct = judge_scores[c] / total_judge if total_judge > 0 else 1/n
            f_pct = weighted_fans[c] / total_fan if total_fan > 0 else 1/n
            percent_scores[c] = (j_pct + f_pct) / 2
        
        # Stage 3: Elite mix
        combined = {}
        for c in contestants:
            combined[c] = (
                self.config.rank_weight * rank_scores[c] +
                self.config.percent_weight * percent_scores[c]
            )
        
        # Lowest combined is eliminated
        sorted_combined = sorted(combined.items(), key=lambda x: x[1])
        eliminated = sorted_combined[0][0]
        
        margin = sorted_combined[1][1] - sorted_combined[0][1] if len(sorted_combined) > 1 else 0
        
        # Generate reasoning
        below_floor = [c for c in contestants if judge_scores[c] < floor]
        reasoning_parts = []
        if eliminated in below_floor:
            reasoning_parts.append(f"Below technical floor (score={judge_scores[eliminated]:.1f} < {floor:.1f})")
        reasoning_parts.append(f"Elite mix score: {combined[eliminated]:.4f}")
        
        return EliminationDecision(
            eliminated=eliminated,
            method=MechanismType.TIERED,
            combined_scores=combined,
            reasoning="; ".join(reasoning_parts),
            was_close=margin < 0.02,
            margin=margin
        )
    
    def get_zone_classification(
        self,
        contestants: List[str],
        judge_scores: Dict[str, float],
        fan_votes: Dict[str, float]
    ) -> Dict[str, str]:
        """
        Classify contestants into safe/middle/danger zones.
        
        This could be announced during show for strategic excitement.
        """
        # Run elimination logic but don't eliminate
        # Just rank everyone
        n = len(contestants)
        
        # Simplified: use combined scores
        total_judge = sum(judge_scores.values())
        total_fan = sum(fan_votes.values())
        
        combined = {}
        for c in contestants:
            j_pct = judge_scores[c] / total_judge if total_judge > 0 else 1/n
            f_pct = fan_votes[c] / total_fan if total_fan > 0 else 1/n
            combined[c] = j_pct + f_pct
        
        sorted_contestants = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
        zones = {}
        for i, (c, _) in enumerate(sorted_contestants):
            if i < self.config.safe_zone_size:
                zones[c] = "SAFE"
            elif i >= n - self.config.danger_zone_size:
                zones[c] = "DANGER"
            else:
                zones[c] = "MIDDLE"
        
        return zones


class ProposedMechanism:
    """
    Wrapper class that combines all mechanism design elements
    and provides analysis capabilities.
    """
    
    def __init__(self):
        self.tiered = TieredThresholdSystem()
        self.rank = RankMechanism()
        self.percent = PercentMechanism()
    
    def compare_all_mechanisms(
        self,
        contestants: List[str],
        judge_scores: Dict[str, float],
        fan_votes: Dict[str, float]
    ) -> Dict[str, EliminationDecision]:
        """Compare outcomes under all mechanisms"""
        return {
            'rank': self.rank.decide_elimination(contestants, judge_scores, fan_votes),
            'percent': self.percent.decide_elimination(contestants, judge_scores, fan_votes),
            'tiered': self.tiered.decide_elimination(contestants, judge_scores, fan_votes)
        }
    
    def evaluate_mechanism(
        self,
        mechanism: VotingMechanism,
        season_context,
        inversion_result
    ) -> MechanismEvaluation:
        """
        Evaluate a mechanism across a full season.
        """
        judge_alignments = []
        fan_alignments = []
        close_calls = []
        
        for week, week_ctx in season_context.weeks.items():
            if not week_ctx.has_valid_elimination():
                continue
            
            if week not in inversion_result.week_results:
                continue
            
            # Get data
            contestants = list(week_ctx.active_set)
            judge_scores = week_ctx.judge_scores
            fan_votes = {c: inversion_result.week_results[week][c].point_estimate 
                        for c in contestants 
                        if c in inversion_result.week_results[week]}
            
            if len(fan_votes) < len(contestants):
                continue
            
            # Get mechanism decision
            decision = mechanism.decide_elimination(contestants, judge_scores, fan_votes)
            
            # Compare to actual
            actual_eliminated = week_ctx.eliminated[0] if week_ctx.eliminated else None
            
            # Alignment with judges (did we eliminate low judge score?)
            judge_sorted = sorted(judge_scores.items(), key=lambda x: x[1])
            lowest_judge = judge_sorted[0][0]
            judge_alignments.append(1 if decision.eliminated == lowest_judge else 0)
            
            # Alignment with fans (did we eliminate low fan vote?)
            fan_sorted = sorted(fan_votes.items(), key=lambda x: x[1])
            lowest_fan = fan_sorted[0][0]
            fan_alignments.append(1 if decision.eliminated == lowest_fan else 0)
            
            # Close call?
            close_calls.append(1 if decision.was_close else 0)
        
        if not judge_alignments:
            return MechanismEvaluation(
                mechanism=mechanism.get_type(),
                judge_alignment=0.5,
                fan_alignment=0.5,
                technical_floor=0.5,
                close_calls_rate=0.0,
                upsets_rate=0.0,
                volatility=0.0,
                controversy_index=0.5
            )
        
        return MechanismEvaluation(
            mechanism=mechanism.get_type(),
            judge_alignment=np.mean(judge_alignments),
            fan_alignment=np.mean(fan_alignments),
            technical_floor=np.mean(judge_alignments),  # Proxy: eliminate low judge
            close_calls_rate=np.mean(close_calls),
            upsets_rate=1 - np.mean(judge_alignments),  # "Upset" = not lowest judge
            volatility=np.std(close_calls) if len(close_calls) > 1 else 0,
            controversy_index=1 - np.mean(fan_alignments)  # Controversy = not lowest fan
        )
    
    def generate_recommendation_report(
        self,
        evaluations: Dict[str, MechanismEvaluation]
    ) -> str:
        """
        Generate recommendation report for show producers.
        """
        lines = [
            "=" * 60,
            "VOTING MECHANISM RECOMMENDATION REPORT",
            "=" * 60,
            "",
            "EXECUTIVE SUMMARY",
            "-" * 40,
        ]
        
        # Find best mechanism by different criteria
        by_fairness = max(evaluations.items(), 
                        key=lambda x: (x[1].judge_alignment + x[1].fan_alignment) / 2)
        by_excitement = max(evaluations.items(),
                          key=lambda x: x[1].close_calls_rate)
        by_technical = max(evaluations.items(),
                          key=lambda x: x[1].technical_floor)
        
        lines.extend([
            f"Most Fair (balances stakeholders): {by_fairness[0].upper()}",
            f"Most Exciting (close calls): {by_excitement[0].upper()}",
            f"Best Technical Quality: {by_technical[0].upper()}",
            "",
            "DETAILED METRICS",
            "-" * 40,
        ])
        
        for name, eval in evaluations.items():
            lines.extend([
                f"\n{name.upper()} METHOD:",
                f"  Judge Alignment: {eval.judge_alignment:.1%}",
                f"  Fan Alignment: {eval.fan_alignment:.1%}",
                f"  Close Calls Rate: {eval.close_calls_rate:.1%}",
                f"  Controversy Index: {eval.controversy_index:.1%}",
            ])
        
        lines.extend([
            "",
            "RECOMMENDATION",
            "-" * 40,
            "We recommend the TIERED THRESHOLD SYSTEM because:",
            "1. It maintains a professional floor - contestants cannot advance",
            "   purely on popularity if their dancing is substandard",
            "2. The 40/60 rank-percent mix captures benefits of both systems",
            "3. Zone announcements (safe/danger) create strategic excitement",
            "   without revealing exact vote totals",
            "4. It reduces controversy by ensuring technical merit matters",
            "   while still giving fans significant influence",
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)

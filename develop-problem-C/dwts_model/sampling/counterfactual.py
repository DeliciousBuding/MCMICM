"""
Counterfactual Simulation Module

Simulate alternative outcomes under different voting rules.
Key question: What would have happened if a different rule was used?

Experiments:
1. Rank vs Percent comparison
2. With vs Without judges' save
3. Different fan vote distributions
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class EliminationEvent:
    """Single elimination event"""
    week: int
    actual_eliminated: str
    method: str
    combined_scores: Dict[str, float]


@dataclass
class SimulationOutcome:
    """Outcome of one simulation run"""
    method: str
    eliminations: List[EliminationEvent]
    final_standings: List[str]  # 1st place first
    
    def matches_actual(self, actual_eliminations: Dict[int, str]) -> Dict[int, bool]:
        """Check which weeks match actual outcomes"""
        matches = {}
        for event in self.eliminations:
            actual = actual_eliminations.get(event.week)
            matches[event.week] = (event.actual_eliminated == actual)
        return matches


@dataclass
class CounterfactualResult:
    """Result of counterfactual comparison"""
    season: int
    rank_outcome: SimulationOutcome
    percent_outcome: SimulationOutcome
    
    # Comparison metrics
    reversal_weeks: List[int] = field(default_factory=list)
    reversal_rate: float = 0.0
    
    # Which method favors fans?
    fan_shielding_rank: Dict[str, int] = field(default_factory=dict)
    fan_shielding_percent: Dict[str, int] = field(default_factory=dict)


class CounterfactualSimulator:
    """
    Simulate DWTS competition under different voting rules.
    
    Core capability: Given estimated fan votes, simulate
    what would have happened under different rules.
    """
    
    def __init__(self):
        pass
    
    def simulate_season(
        self,
        season_context,
        fan_votes: Dict[int, Dict[str, float]],  # week -> contestant -> vote %
        method: str = 'percent'
    ) -> SimulationOutcome:
        """
        Simulate a full season under specified voting method.
        
        Args:
            season_context: SeasonContext from ActiveSetManager
            fan_votes: Estimated fan vote percentages per week
            method: 'rank' or 'percent'
            
        Returns:
            SimulationOutcome with simulated eliminations
        """
        eliminations = []
        remaining = set()
        
        # Initialize with all contestants
        for week_ctx in season_context.weeks.values():
            remaining.update(week_ctx.active_set)
            break
        
        for week in sorted(season_context.weeks.keys()):
            week_ctx = season_context.weeks[week]
            
            if not week_ctx.has_valid_elimination():
                continue
            
            # Get current active contestants
            active = week_ctx.active_set & remaining
            
            if len(active) <= 3:  # Finals
                break
            
            # Get scores
            week_fan = fan_votes.get(week, {})
            
            # Compute combined scores based on method
            combined = self._compute_combined_scores(
                active, 
                week_ctx.judge_percentages,
                week_ctx.judge_ranks,
                week_fan,
                method
            )
            
            # Find eliminated (lowest/highest depending on method)
            if method == 'rank':
                # Highest combined rank = worst
                eliminated = max(combined.items(), key=lambda x: x[1])[0]
            else:
                # Lowest combined percent = worst
                eliminated = min(combined.items(), key=lambda x: x[1])[0]
            
            eliminations.append(EliminationEvent(
                week=week,
                actual_eliminated=eliminated,
                method=method,
                combined_scores=combined
            ))
            
            remaining.discard(eliminated)
        
        # Final standings (remaining after eliminations)
        final_standings = list(remaining)
        
        return SimulationOutcome(
            method=method,
            eliminations=eliminations,
            final_standings=final_standings
        )
    
    def _compute_combined_scores(
        self,
        contestants: Set[str],
        judge_pct: Dict[str, float],
        judge_rank: Dict[str, int],
        fan_pct: Dict[str, float],
        method: str
    ) -> Dict[str, float]:
        """
        Compute combined scores based on voting method.
        """
        combined = {}
        
        if method == 'percent':
            # Normalize fan votes to sum to 1
            total_fan = sum(fan_pct.get(c, 0) for c in contestants)
            if total_fan > 0:
                fan_normalized = {c: fan_pct.get(c, 0) / total_fan for c in contestants}
            else:
                fan_normalized = {c: 1/len(contestants) for c in contestants}
            
            for c in contestants:
                combined[c] = judge_pct.get(c, 0) + fan_normalized.get(c, 0)
                
        else:  # rank
            # Convert fan percentages to ranks
            fan_sorted = sorted(
                [(c, fan_pct.get(c, 0)) for c in contestants],
                key=lambda x: x[1],
                reverse=True
            )
            fan_rank = {c: r+1 for r, (c, _) in enumerate(fan_sorted)}
            
            for c in contestants:
                combined[c] = judge_rank.get(c, len(contestants)) + fan_rank.get(c, len(contestants))
        
        return combined
    
    def compare_methods(
        self,
        season_context,
        fan_votes: Dict[int, Dict[str, float]]
    ) -> CounterfactualResult:
        """
        Compare rank and percent methods for a season.
        """
        rank_outcome = self.simulate_season(season_context, fan_votes, 'rank')
        percent_outcome = self.simulate_season(season_context, fan_votes, 'percent')
        
        # Find reversal weeks
        reversal_weeks = []
        for r_event, p_event in zip(rank_outcome.eliminations, percent_outcome.eliminations):
            if r_event.actual_eliminated != p_event.actual_eliminated:
                reversal_weeks.append(r_event.week)
        
        total_weeks = max(len(rank_outcome.eliminations), len(percent_outcome.eliminations))
        reversal_rate = len(reversal_weeks) / total_weeks if total_weeks > 0 else 0
        
        return CounterfactualResult(
            season=season_context.season,
            rank_outcome=rank_outcome,
            percent_outcome=percent_outcome,
            reversal_weeks=reversal_weeks,
            reversal_rate=reversal_rate
        )
    
    def analyze_fan_shielding(
        self,
        season_context,
        fan_votes: Dict[int, Dict[str, float]]
    ) -> Dict[str, any]:
        """
        Analyze which method better protects high-fan-vote contestants.
        
        "Fan shielding" = ability of popular contestants to survive
        despite low judge scores.
        """
        result = self.compare_methods(season_context, fan_votes)
        
        # For each reversal week, analyze who benefited
        beneficiaries = {'rank': [], 'percent': []}
        
        for week in result.reversal_weeks:
            week_fan = fan_votes.get(week, {})
            week_ctx = season_context.weeks[week]
            
            # Who was eliminated under each method?
            for event in result.rank_outcome.eliminations:
                if event.week == week:
                    rank_elim = event.actual_eliminated
            
            for event in result.percent_outcome.eliminations:
                if event.week == week:
                    percent_elim = event.actual_eliminated
            
            # Compare fan votes of eliminated contestants
            rank_elim_fans = week_fan.get(rank_elim, 0)
            percent_elim_fans = week_fan.get(percent_elim, 0)
            
            if rank_elim_fans > percent_elim_fans:
                # Rank method eliminated someone more popular
                beneficiaries['percent'].append({
                    'week': week,
                    'saved': rank_elim,
                    'fan_votes': rank_elim_fans
                })
            else:
                beneficiaries['rank'].append({
                    'week': week,
                    'saved': percent_elim,
                    'fan_votes': percent_elim_fans
                })
        
        return {
            'reversal_weeks': result.reversal_weeks,
            'reversal_rate': result.reversal_rate,
            'rank_beneficiaries': beneficiaries['rank'],
            'percent_beneficiaries': beneficiaries['percent'],
            'fan_favoring_method': 'percent' if len(beneficiaries['percent']) > len(beneficiaries['rank']) else 'rank'
        }
    
    def simulate_judges_save_scenarios(
        self,
        week_ctx,
        fan_votes: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Simulate different judges' save strategies.
        
        Scenarios:
        1. Technocratic: Save higher judge score
        2. Populist: Save higher fan vote
        3. Random: 50/50
        """
        # Find bottom 2 by combined score
        combined = {}
        for c in week_ctx.active_set:
            combined[c] = week_ctx.judge_percentages.get(c, 0) + fan_votes.get(c, 0)
        
        sorted_combined = sorted(combined.items(), key=lambda x: x[1])
        bottom_two = [c for c, _ in sorted_combined[:2]]
        
        if len(bottom_two) < 2:
            return {'error': 'Not enough contestants for bottom two'}
        
        b1, b2 = bottom_two
        
        scenarios = {}
        
        # Technocratic
        if week_ctx.judge_scores.get(b1, 0) > week_ctx.judge_scores.get(b2, 0):
            scenarios['technocratic'] = {'saved': b1, 'eliminated': b2}
        else:
            scenarios['technocratic'] = {'saved': b2, 'eliminated': b1}
        
        # Populist
        if fan_votes.get(b1, 0) > fan_votes.get(b2, 0):
            scenarios['populist'] = {'saved': b1, 'eliminated': b2}
        else:
            scenarios['populist'] = {'saved': b2, 'eliminated': b1}
        
        # Random
        scenarios['random'] = {
            'p_b1_eliminated': 0.5,
            'p_b2_eliminated': 0.5
        }
        
        return {
            'bottom_two': bottom_two,
            'scenarios': scenarios,
            'actual_eliminated': week_ctx.eliminated[0] if week_ctx.eliminated else None
        }


class ControversyCaseStudy:
    """
    Deep-dive analysis of controversial cases mentioned in problem.
    
    - Season 2: Jerry Rice (runner up with low judge scores)
    - Season 4: Billy Ray Cyrus (5th with 6 weeks of last place judges)
    - Season 11: Bristol Palin (3rd with 12 times lowest judge score)
    - Season 27: Bobby Bones (won with consistently low scores)
    """
    
    CASES = {
        2: {'name': 'Jerry Rice', 'placement': 2},
        4: {'name': 'Billy Ray Cyrus', 'placement': 5},
        11: {'name': 'Bristol Palin', 'placement': 3},
        27: {'name': 'Bobby Bones', 'placement': 1}
    }
    
    def __init__(self, simulator: CounterfactualSimulator):
        self.simulator = simulator
    
    def analyze_case(
        self,
        season_context,
        fan_votes: Dict[int, Dict[str, float]],
        controversy: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Analyze a single controversy case.
        """
        name = controversy['name']
        expected_placement = controversy['placement']
        
        # Simulate under both methods
        comparison = self.simulator.compare_methods(season_context, fan_votes)
        
        # Track the controversial contestant
        rank_survival_weeks = 0
        percent_survival_weeks = 0
        
        for event in comparison.rank_outcome.eliminations:
            if event.actual_eliminated == name:
                break
            rank_survival_weeks += 1
        
        for event in comparison.percent_outcome.eliminations:
            if event.actual_eliminated == name:
                break
            percent_survival_weeks += 1
        
        # Calculate fan vote advantage
        fan_advantage_weeks = []
        judge_disadvantage_weeks = []
        
        for week, week_ctx in season_context.weeks.items():
            if name not in week_ctx.active_set:
                continue
            
            week_fan = fan_votes.get(week, {})
            
            # Compare ranks
            fan_sorted = sorted(
                [(c, week_fan.get(c, 0)) for c in week_ctx.active_set],
                key=lambda x: x[1],
                reverse=True
            )
            fan_rank = {c: r+1 for r, (c, _) in enumerate(fan_sorted)}
            
            judge_rank = week_ctx.judge_ranks.get(name, len(week_ctx.active_set))
            contestant_fan_rank = fan_rank.get(name, len(week_ctx.active_set))
            
            if contestant_fan_rank < judge_rank:  # Better fan rank
                fan_advantage_weeks.append(week)
            if judge_rank == len(week_ctx.active_set):  # Worst judge score
                judge_disadvantage_weeks.append(week)
        
        return {
            'contestant': name,
            'expected_placement': expected_placement,
            'rank_method_survival_weeks': rank_survival_weeks,
            'percent_method_survival_weeks': percent_survival_weeks,
            'fan_advantage_weeks': len(fan_advantage_weeks),
            'worst_judge_score_weeks': len(judge_disadvantage_weeks),
            'would_survive_longer_under': 'percent' if percent_survival_weeks > rank_survival_weeks else 'rank',
            'summary': self._generate_summary(
                name, rank_survival_weeks, percent_survival_weeks,
                len(judge_disadvantage_weeks)
            )
        }
    
    def _generate_summary(
        self,
        name: str,
        rank_weeks: int,
        percent_weeks: int,
        low_judge_weeks: int
    ) -> str:
        """Generate human-readable summary"""
        if percent_weeks > rank_weeks:
            return (
                f"{name} survived {percent_weeks - rank_weeks} more weeks under "
                f"percent method despite {low_judge_weeks} weeks with lowest judge scores. "
                "Percent method favors fan-popular contestants."
            )
        elif rank_weeks > percent_weeks:
            return (
                f"{name} survived {rank_weeks - percent_weeks} more weeks under "
                f"rank method. Rank method's discrete nature can protect contestants "
                "whose poor judge scores don't translate to proportionally worse ranks."
            )
        else:
            return (
                f"{name}'s outcome was the same under both methods. "
                f"Survived {rank_weeks} weeks despite {low_judge_weeks} weeks "
                "with lowest judge scores."
            )

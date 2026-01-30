"""
Active Set Manager
Manages the active contestant sets across all seasons and weeks.
Provides the foundation for constraint generation in the inversion engines.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field

from .fsm import ContestantFSM, ContestantState, WeekType
from .data_loader import DWTSDataLoader


@dataclass
class WeekContext:
    """Complete context for a single week's constraint generation"""
    season: int
    week: int
    week_type: WeekType
    active_set: Set[str]           # Contestants in denominator
    eliminated: List[str]          # Contestants eliminated this week
    survivors: Set[str]            # Contestants who survive
    judge_scores: Dict[str, float] # Contestant -> total judge score
    judge_percentages: Dict[str, float]  # Contestant -> judge score %
    judge_ranks: Dict[str, int]    # Contestant -> judge score rank (1=best)
    
    def get_pairwise_constraints(self) -> List[Tuple[str, str]]:
        """Get (eliminated, survivor) pairs"""
        pairs = []
        for e in self.eliminated:
            for s in self.survivors:
                pairs.append((e, s))
        return pairs
    
    def has_valid_elimination(self) -> bool:
        """Check if this week has valid elimination constraints"""
        return len(self.eliminated) > 0 and len(self.survivors) > 0


@dataclass 
class SeasonContext:
    """Complete context for a season"""
    season: int
    voting_method: str  # 'rank' or 'percent'
    has_judges_save: bool
    num_weeks: int
    num_contestants: int
    weeks: Dict[int, WeekContext] = field(default_factory=dict)
    fsm: Optional[ContestantFSM] = None
    
    def get_valid_weeks(self) -> List[int]:
        """Get weeks with valid elimination constraints"""
        return [w for w, ctx in self.weeks.items() if ctx.has_valid_elimination()]
    
    def get_all_constraints(self) -> List[Tuple[int, str, str]]:
        """Get all constraints as (week, eliminated, survivor) tuples"""
        constraints = []
        for week, ctx in self.weeks.items():
            for e, s in ctx.get_pairwise_constraints():
                constraints.append((week, e, s))
        return constraints


class ActiveSetManager:
    """
    Central manager for active sets and constraint generation.
    
    This is the bridge between ETL and the inversion engines.
    """
    
    def __init__(self, loader: DWTSDataLoader):
        """
        Args:
            loader: Initialized DWTSDataLoader with loaded data
        """
        self.loader = loader
        self.season_contexts: Dict[int, SeasonContext] = {}
        self.fsm_cache: Dict[int, ContestantFSM] = {}
        
        # Import config
        from ..config import SEASON_CONFIG
        self.config = SEASON_CONFIG
        
    def build_season_context(self, season: int) -> SeasonContext:
        """Build complete context for a season"""
        if season in self.season_contexts:
            return self.season_contexts[season]
        
        # Get season data
        season_data = self.loader.get_season_data(season)
        score_matrix = self.loader.score_matrix[
            self.loader.score_matrix['season'] == season
        ]
        
        # Build FSM
        fsm = ContestantFSM(season_data, score_matrix)
        self.fsm_cache[season] = fsm
        
        # Create season context
        context = SeasonContext(
            season=season,
            voting_method=self.config.get_voting_method(season),
            has_judges_save=self.config.has_judges_save(season),
            num_weeks=season_data['num_weeks'],
            num_contestants=season_data['num_contestants'],
            fsm=fsm
        )
        
        # Build week contexts
        for week in range(1, season_data['num_weeks'] + 1):
            week_ctx = self._build_week_context(season, week, fsm, score_matrix)
            if week_ctx:
                context.weeks[week] = week_ctx
        
        self.season_contexts[season] = context
        return context
    
    def _build_week_context(
        self, 
        season: int, 
        week: int, 
        fsm: ContestantFSM,
        score_matrix: pd.DataFrame
    ) -> Optional[WeekContext]:
        """Build context for a single week"""
        
        week_type = fsm.get_week_type(week)
        
        # Skip weeks that didn't happen
        if week_type == WeekType.SKIPPED:
            return None
        
        active_set = fsm.get_active_set(week)
        eliminated = fsm.get_eliminated_this_week(week)
        survivors = fsm.get_survivors(week)
        
        # Get judge scores for active contestants
        week_scores = score_matrix[score_matrix['week'] == week]
        
        judge_scores = {}
        for contestant in active_set:
            contestant_score = week_scores[week_scores['contestant'] == contestant]
            if len(contestant_score) > 0:
                total = contestant_score.iloc[0]['total_score']
                judge_scores[contestant] = total if pd.notna(total) else 0.0
            else:
                judge_scores[contestant] = 0.0
        
        # Calculate percentages
        total_sum = sum(judge_scores.values())
        judge_percentages = {}
        if total_sum > 0:
            for c, s in judge_scores.items():
                judge_percentages[c] = s / total_sum
        else:
            # Equal distribution if no scores (edge case)
            n = len(active_set)
            for c in active_set:
                judge_percentages[c] = 1.0 / n if n > 0 else 0.0
        
        # Calculate ranks (1 = highest score = best)
        sorted_contestants = sorted(
            judge_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        judge_ranks = {}
        for rank, (contestant, _) in enumerate(sorted_contestants, 1):
            judge_ranks[contestant] = rank
        
        return WeekContext(
            season=season,
            week=week,
            week_type=week_type,
            active_set=active_set,
            eliminated=eliminated,
            survivors=survivors,
            judge_scores=judge_scores,
            judge_percentages=judge_percentages,
            judge_ranks=judge_ranks
        )
    
    def get_season_context(self, season: int) -> SeasonContext:
        """Get or build season context"""
        if season not in self.season_contexts:
            return self.build_season_context(season)
        return self.season_contexts[season]
    
    def get_all_seasons(self) -> List[int]:
        """Get list of all seasons in data"""
        return sorted(self.loader.processed_df['season'].unique())
    
    def build_all_contexts(self) -> Dict[int, SeasonContext]:
        """Build contexts for all seasons"""
        for season in self.get_all_seasons():
            self.build_season_context(season)
        return self.season_contexts
    
    def get_constraint_matrix(self, season: int) -> pd.DataFrame:
        """
        Get constraint matrix for a season.
        
        Returns DataFrame with columns:
        - week: Week number
        - eliminated: Eliminated contestant
        - survivor: Surviving contestant
        - elim_judge_score: Eliminated's judge score
        - surv_judge_score: Survivor's judge score
        - elim_judge_pct: Eliminated's judge %
        - surv_judge_pct: Survivor's judge %
        - elim_judge_rank: Eliminated's judge rank
        - surv_judge_rank: Survivor's judge rank
        - judge_score_diff: Survivor - Eliminated judge score
        - min_fan_diff_needed: Minimum fan vote diff needed for valid elimination
        """
        context = self.get_season_context(season)
        
        records = []
        for week, ctx in context.weeks.items():
            for e, s in ctx.get_pairwise_constraints():
                e_score = ctx.judge_scores.get(e, 0)
                s_score = ctx.judge_scores.get(s, 0)
                e_pct = ctx.judge_percentages.get(e, 0)
                s_pct = ctx.judge_percentages.get(s, 0)
                e_rank = ctx.judge_ranks.get(e, 0)
                s_rank = ctx.judge_ranks.get(s, 0)
                
                # For percent method: F_e + J_e < F_s + J_s
                # => F_e - F_s < J_s - J_e
                judge_score_diff = s_score - e_score
                
                # For percent: min fan pct diff needed
                min_fan_diff = e_pct - s_pct  # Eliminated needs to lose by at least this
                
                records.append({
                    'week': week,
                    'eliminated': e,
                    'survivor': s,
                    'elim_judge_score': e_score,
                    'surv_judge_score': s_score,
                    'elim_judge_pct': e_pct,
                    'surv_judge_pct': s_pct,
                    'elim_judge_rank': e_rank,
                    'surv_judge_rank': s_rank,
                    'judge_score_diff': judge_score_diff,
                    'min_fan_diff_needed': min_fan_diff
                })
        
        return pd.DataFrame(records)
    
    def get_active_matrix(self, season: int) -> pd.DataFrame:
        """
        Get active matrix for a season.
        
        Returns DataFrame with (contestant, week) -> is_active mapping.
        This is the fundamental denominator matrix.
        """
        context = self.get_season_context(season)
        fsm = context.fsm
        
        return fsm.to_dataframe()
    
    def generate_summary_report(self) -> pd.DataFrame:
        """Generate summary statistics for all seasons"""
        records = []
        
        for season in self.get_all_seasons():
            ctx = self.get_season_context(season)
            
            total_constraints = len(ctx.get_all_constraints())
            valid_weeks = len(ctx.get_valid_weeks())
            
            # Count special weeks
            no_elim_weeks = sum(
                1 for w in ctx.weeks.values() 
                if w.week_type == WeekType.NO_ELIM
            )
            multi_elim_weeks = sum(
                1 for w in ctx.weeks.values() 
                if w.week_type in [WeekType.MULTI_ELIM, WeekType.DOUBLE_ELIM]
            )
            
            records.append({
                'season': season,
                'voting_method': ctx.voting_method,
                'has_judges_save': ctx.has_judges_save,
                'num_contestants': ctx.num_contestants,
                'num_weeks': ctx.num_weeks,
                'valid_weeks': valid_weeks,
                'no_elim_weeks': no_elim_weeks,
                'multi_elim_weeks': multi_elim_weeks,
                'total_constraints': total_constraints
            })
        
        return pd.DataFrame(records)
    
    def export_all(self, output_dir: str):
        """Export all data structures to files"""
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Summary
        summary = self.generate_summary_report()
        summary.to_csv(output_path / 'season_summary.csv', index=False)
        
        # Per-season data
        for season in self.get_all_seasons():
            # Constraints
            constraints = self.get_constraint_matrix(season)
            constraints.to_csv(
                output_path / f'constraints_s{season}.csv', 
                index=False
            )
            
            # Active matrix
            active = self.get_active_matrix(season)
            active.to_csv(
                output_path / f'active_matrix_s{season}.csv',
                index=False
            )

"""
Contestant Finite State Machine (FSM)
Core defensive engineering for tracking contestant lifecycle

States:
- Active: Competing this week
- Eliminated_This_Week: Being eliminated at end of this week
- Withdrew: Withdrew from competition
- Finalist: Made it to finals
- Inactive: No longer competing
"""
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np


class ContestantState(Enum):
    """Contestant lifecycle states"""
    ACTIVE = auto()              # Currently competing
    ELIMINATED_THIS_WEEK = auto() # Will be eliminated at end of week
    WITHDREW = auto()            # Withdrew (special handling)
    FINALIST = auto()            # Made finals
    INACTIVE = auto()            # No longer in competition


class WeekType(Enum):
    """Week classification for special handling"""
    NORMAL = auto()         # Standard elimination (1 person)
    NO_ELIM = auto()        # No elimination this week
    MULTI_ELIM = auto()     # Multiple eliminations
    DOUBLE_ELIM = auto()    # Exactly 2 eliminated
    SKIPPED = auto()        # Week didn't happen (all N/A)


@dataclass
class ContestantLifecycle:
    """Track a single contestant's journey through the season"""
    name: str
    season: int
    final_status: str          # From results column
    final_placement: int
    elimination_week: Optional[int] = None  # Week eliminated (if applicable)
    withdrew_week: Optional[int] = None     # Inferred withdrawal week
    state_history: Dict[int, ContestantState] = field(default_factory=dict)
    
    def get_state(self, week: int) -> ContestantState:
        """Get state at a specific week"""
        return self.state_history.get(week, ContestantState.INACTIVE)
    
    def was_active(self, week: int) -> bool:
        """Check if contestant was active (in denominator) for week"""
        state = self.get_state(week)
        return state in [ContestantState.ACTIVE, ContestantState.ELIMINATED_THIS_WEEK,
                        ContestantState.FINALIST]
    
    def get_active_weeks(self) -> List[int]:
        """Get list of weeks contestant was active"""
        return [w for w, s in self.state_history.items() 
                if s in [ContestantState.ACTIVE, ContestantState.ELIMINATED_THIS_WEEK, 
                        ContestantState.FINALIST]]


class ContestantFSM:
    """
    Finite State Machine for contestant lifecycle management.
    
    This is the CORE DEFENSIVE component that ensures correct denominator
    calculation for vote inversion.
    
    Key logic:
    1. Parse result column to determine final fate
    2. For withdrew: infer withdrawal week from last positive score
    3. Build state history week by week
    4. Handle edge cases (0 scores while active, N/A weeks, etc.)
    """
    
    def __init__(self, season_data: Dict, score_matrix: pd.DataFrame):
        """
        Args:
            season_data: From DWTSDataLoader.get_season_data()
            score_matrix: Score data from loader
        """
        self.contestants_df = season_data['contestants']
        self.scores = score_matrix
        self.num_weeks = season_data['num_weeks']
        self.season = self.contestants_df['season'].iloc[0]
        
        self.lifecycles: Dict[str, ContestantLifecycle] = {}
        self.week_types: Dict[int, WeekType] = {}
        self.week_events: Dict[int, List[str]] = {}  # Elimination events per week
        
        self._build_lifecycles()
        self._classify_weeks()
        
    def _build_lifecycles(self):
        """Build lifecycle for each contestant"""
        for _, row in self.contestants_df.iterrows():
            name = row['celebrity_name']
            status = row['status']
            elim_week = row['elimination_week']
            placement = row['placement']
            
            lifecycle = ContestantLifecycle(
                name=name,
                season=self.season,
                final_status=status,
                final_placement=placement,
                elimination_week=elim_week
            )
            
            # Infer withdrawal week if needed
            if status == 'withdrew':
                lifecycle.withdrew_week = self._infer_withdrew_week(name)
            
            # Build state history
            self._build_state_history(lifecycle)
            
            self.lifecycles[name] = lifecycle
            
    def _infer_withdrew_week(self, contestant: str) -> int:
        """
        Infer withdrawal week from score data.
        
        Logic: Find last week with positive score, withdrawal is next week.
        """
        contestant_scores = self.scores[self.scores['contestant'] == contestant]
        
        last_positive_week = 0
        for week in range(1, self.num_weeks + 1):
            week_data = contestant_scores[contestant_scores['week'] == week]
            if len(week_data) > 0:
                total = week_data.iloc[0]['total_score']
                if pd.notna(total) and total > 0:
                    last_positive_week = week
        
        # Withdrawal happens after last positive score
        return last_positive_week + 1 if last_positive_week > 0 else 1
    
    def _build_state_history(self, lifecycle: ContestantLifecycle):
        """Build week-by-week state history for a contestant"""
        name = lifecycle.name
        status = lifecycle.final_status
        elim_week = lifecycle.elimination_week
        withdrew_week = lifecycle.withdrew_week
        
        for week in range(1, self.num_weeks + 1):
            # Determine state for this week
            if status == 'withdrew':
                if withdrew_week and week >= withdrew_week:
                    state = ContestantState.INACTIVE
                else:
                    state = ContestantState.ACTIVE
                    
            elif status == 'eliminated':
                if week < elim_week:
                    state = ContestantState.ACTIVE
                elif week == elim_week:
                    state = ContestantState.ELIMINATED_THIS_WEEK
                else:
                    state = ContestantState.INACTIVE
                    
            elif status == 'winner':
                # Winner is active all weeks
                state = ContestantState.FINALIST
                
            elif status == 'finalist':
                # Finalist active all weeks
                state = ContestantState.FINALIST
                
            else:
                # Unknown status - check scores
                contestant_scores = self.scores[
                    (self.scores['contestant'] == name) &
                    (self.scores['week'] == week)
                ]
                if len(contestant_scores) > 0:
                    total = contestant_scores.iloc[0]['total_score']
                    if pd.notna(total) and total > 0:
                        state = ContestantState.ACTIVE
                    else:
                        state = ContestantState.INACTIVE
                else:
                    state = ContestantState.INACTIVE
            
            lifecycle.state_history[week] = state
            
    def _classify_weeks(self):
        """Classify each week by elimination pattern"""
        for week in range(1, self.num_weeks + 1):
            eliminated = []
            
            for name, lifecycle in self.lifecycles.items():
                state = lifecycle.get_state(week)
                if state == ContestantState.ELIMINATED_THIS_WEEK:
                    eliminated.append(name)
            
            self.week_events[week] = eliminated
            
            # Check if week is skipped (all N/A)
            week_scores = self.scores[self.scores['week'] == week]
            if week_scores['all_na'].all():
                self.week_types[week] = WeekType.SKIPPED
            elif len(eliminated) == 0:
                self.week_types[week] = WeekType.NO_ELIM
            elif len(eliminated) == 1:
                self.week_types[week] = WeekType.NORMAL
            elif len(eliminated) == 2:
                self.week_types[week] = WeekType.DOUBLE_ELIM
            else:
                self.week_types[week] = WeekType.MULTI_ELIM
                
    def get_active_set(self, week: int) -> Set[str]:
        """
        Get set of active contestants for a week.
        This is the DENOMINATOR for vote calculations.
        
        CRITICAL: Includes contestants being eliminated this week.
        """
        active = set()
        for name, lifecycle in self.lifecycles.items():
            if lifecycle.was_active(week):
                active.add(name)
        return active
    
    def get_eliminated_this_week(self, week: int) -> List[str]:
        """Get contestants eliminated at end of this week"""
        return self.week_events.get(week, [])
    
    def get_survivors(self, week: int) -> Set[str]:
        """Get contestants who survive this week (active but not eliminated)"""
        active = self.get_active_set(week)
        eliminated = set(self.get_eliminated_this_week(week))
        return active - eliminated
    
    def get_week_type(self, week: int) -> WeekType:
        """Get classification of week"""
        return self.week_types.get(week, WeekType.SKIPPED)
    
    def get_pairwise_constraints(self, week: int) -> List[Tuple[str, str]]:
        """
        Generate pairwise constraints for elimination.
        
        Returns list of (eliminated, survivor) pairs.
        Each eliminated contestant must have lower combined score
        than each survivor.
        """
        eliminated = self.get_eliminated_this_week(week)
        survivors = self.get_survivors(week)
        
        constraints = []
        for e in eliminated:
            for s in survivors:
                constraints.append((e, s))
        
        return constraints
    
    def generate_event_log(self) -> List[str]:
        """Generate human-readable event log"""
        log = []
        log.append(f"=== Season {self.season} Event Log ===")
        log.append(f"Total contestants: {len(self.lifecycles)}")
        log.append(f"Total weeks: {self.num_weeks}")
        log.append("")
        
        for week in range(1, self.num_weeks + 1):
            week_type = self.get_week_type(week)
            active = self.get_active_set(week)
            eliminated = self.get_eliminated_this_week(week)
            
            log.append(f"Week {week}: {week_type.name}")
            log.append(f"  Active: {len(active)} contestants")
            
            if eliminated:
                log.append(f"  Eliminated: {', '.join(eliminated)}")
            
            # Check for special events
            for name, lifecycle in self.lifecycles.items():
                if lifecycle.withdrew_week == week:
                    log.append(f"  WITHDREW: {name}")
            
            log.append("")
        
        return log
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export lifecycle data to DataFrame"""
        records = []
        for name, lifecycle in self.lifecycles.items():
            for week in range(1, self.num_weeks + 1):
                state = lifecycle.get_state(week)
                records.append({
                    'season': self.season,
                    'contestant': name,
                    'week': week,
                    'state': state.name,
                    'is_active': lifecycle.was_active(week),
                    'final_placement': lifecycle.final_placement
                })
        return pd.DataFrame(records)

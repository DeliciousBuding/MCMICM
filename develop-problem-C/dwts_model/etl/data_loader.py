"""
Data Loader with Defensive Parsing
Handles: N/A values, score normalization, multi-judge aggregation
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import re
import warnings

class DWTSDataLoader:
    """
    Load and preprocess DWTS data with defensive engineering.
    
    Key defensive measures:
    1. N/A vs 0 distinction (N/A = no data, 0 = eliminated)
    2. Handle decimal scores (averaged multi-dance weeks)
    3. Handle bonus points spread across judges
    4. Detect and flag data anomalies
    """
    
    # Maximum number of weeks across all seasons
    MAX_WEEKS = 11
    MAX_JUDGES = 4
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.raw_df = None
        self.processed_df = None
        self.score_matrix = None  # (season, week, contestant) -> judge scores
        self.anomaly_log = []
        
    def load(self) -> pd.DataFrame:
        """Load raw data with proper NA handling"""
        self.raw_df = pd.read_csv(
            self.data_path,
            na_values=['N/A', 'NA', ''],
            keep_default_na=True
        )
        self._validate_columns()
        self._process_data()
        return self.processed_df
    
    def _validate_columns(self):
        """Validate expected columns exist"""
        required_cols = [
            'celebrity_name', 'ballroom_partner', 'celebrity_industry',
            'celebrity_age_during_season', 'season', 'results', 'placement'
        ]
        for col in required_cols:
            if col not in self.raw_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Check for score columns
        score_cols = [c for c in self.raw_df.columns if 'judge' in c.lower()]
        if len(score_cols) == 0:
            raise ValueError("No judge score columns found")
    
    def _process_data(self):
        """Main processing pipeline"""
        df = self.raw_df.copy()
        
        # Parse results to extract elimination week and status
        df['elimination_week'], df['status'] = zip(*df['results'].apply(self._parse_result))
        
        # Create unique contestant ID
        df['contestant_id'] = df['season'].astype(str) + '_' + df['celebrity_name']
        
        # Extract and reshape score data
        self.processed_df = df
        self._extract_scores()
        self._detect_anomalies()
        
    def _parse_result(self, result: str) -> Tuple[Optional[int], str]:
        """
        Parse result string to extract elimination week and status.
        
        Returns:
            (elimination_week, status)
            - elimination_week: Week number when eliminated (None if winner/withdrew)
            - status: 'winner', 'eliminated', 'withdrew', 'finalist'
        """
        result = str(result).lower().strip()
        
        # Check for placement (1st, 2nd, etc.)
        place_match = re.match(r'(\d+)(?:st|nd|rd|th)\s*place', result)
        if place_match:
            place = int(place_match.group(1))
            if place == 1:
                return None, 'winner'
            else:
                return None, 'finalist'
        
        # Check for elimination week
        elim_match = re.match(r'eliminated\s*week\s*(\d+)', result)
        if elim_match:
            return int(elim_match.group(1)), 'eliminated'
        
        # Check for withdrawal
        if 'withdrew' in result:
            return None, 'withdrew'
        
        # Unknown - log anomaly
        self.anomaly_log.append(f"Unknown result format: {result}")
        return None, 'unknown'
    
    def _extract_scores(self):
        """Extract judge scores into structured format"""
        df = self.processed_df
        
        # Initialize score storage
        score_data = []
        
        for _, row in df.iterrows():
            season = row['season']
            contestant = row['celebrity_name']
            
            for week in range(1, self.MAX_WEEKS + 1):
                week_scores = []
                has_any_score = False
                all_na = True
                
                for judge in range(1, self.MAX_JUDGES + 1):
                    col = f'week{week}_judge{judge}_score'
                    if col in df.columns:
                        val = row[col]
                        if pd.notna(val):
                            all_na = False
                            if val != 0:
                                has_any_score = True
                            week_scores.append(float(val))
                        else:
                            week_scores.append(np.nan)
                    else:
                        week_scores.append(np.nan)
                
                # Compute total (ignoring NaN)
                valid_scores = [s for s in week_scores if pd.notna(s)]
                total_score = sum(valid_scores) if valid_scores else np.nan
                
                score_data.append({
                    'season': season,
                    'contestant': contestant,
                    'week': week,
                    'judge1': week_scores[0] if len(week_scores) > 0 else np.nan,
                    'judge2': week_scores[1] if len(week_scores) > 1 else np.nan,
                    'judge3': week_scores[2] if len(week_scores) > 2 else np.nan,
                    'judge4': week_scores[3] if len(week_scores) > 3 else np.nan,
                    'total_score': total_score,
                    'num_judges': len(valid_scores),
                    'all_na': all_na,
                    'has_score': has_any_score
                })
        
        self.score_matrix = pd.DataFrame(score_data)
        
    def _detect_anomalies(self):
        """Detect and log data anomalies"""
        df = self.processed_df
        scores = self.score_matrix
        
        for season in df['season'].unique():
            season_df = df[df['season'] == season]
            season_scores = scores[scores['season'] == season]
            
            for _, row in season_df.iterrows():
                contestant = row['celebrity_name']
                status = row['status']
                elim_week = row['elimination_week']
                
                contestant_scores = season_scores[season_scores['contestant'] == contestant]
                
                # Check for score after elimination (should be 0)
                if status == 'eliminated' and pd.notna(elim_week):
                    for week in range(int(elim_week) + 1, self.MAX_WEEKS + 1):
                        week_data = contestant_scores[contestant_scores['week'] == week]
                        if len(week_data) > 0 and not week_data.iloc[0]['all_na']:
                            if week_data.iloc[0]['total_score'] > 0:
                                self.anomaly_log.append(
                                    f"S{season} {contestant}: Non-zero score after elimination "
                                    f"(Week {week})"
                                )
                
                # Check for 0 score before elimination (potential withdrew)
                if status == 'eliminated' and pd.notna(elim_week):
                    for week in range(1, int(elim_week)):
                        week_data = contestant_scores[contestant_scores['week'] == week]
                        if len(week_data) > 0 and not week_data.iloc[0]['all_na']:
                            if week_data.iloc[0]['total_score'] == 0:
                                self.anomaly_log.append(
                                    f"S{season} {contestant}: Zero score before elimination "
                                    f"(Week {week}) - possible data error or special event"
                                )
    
    def get_season_data(self, season: int) -> Dict:
        """Get all data for a specific season"""
        df = self.processed_df[self.processed_df['season'] == season].copy()
        scores = self.score_matrix[self.score_matrix['season'] == season].copy()
        
        # Determine number of weeks in season
        max_week = 1
        for week in range(1, self.MAX_WEEKS + 1):
            week_scores = scores[scores['week'] == week]
            if not week_scores['all_na'].all():
                max_week = week
        
        return {
            'contestants': df,
            'scores': scores,
            'num_weeks': max_week,
            'num_contestants': len(df)
        }
    
    def get_contestant_trajectory(self, season: int, contestant: str) -> pd.DataFrame:
        """Get week-by-week trajectory for a contestant"""
        scores = self.score_matrix[
            (self.score_matrix['season'] == season) &
            (self.score_matrix['contestant'] == contestant)
        ].copy()
        return scores
    
    def get_week_standings(self, season: int, week: int) -> pd.DataFrame:
        """Get all active contestants' scores for a specific week"""
        df = self.processed_df[self.processed_df['season'] == season]
        scores = self.score_matrix[
            (self.score_matrix['season'] == season) &
            (self.score_matrix['week'] == week)
        ].copy()
        
        # Merge with contestant info
        scores = scores.merge(
            df[['celebrity_name', 'elimination_week', 'status', 'placement']],
            left_on='contestant',
            right_on='celebrity_name',
            how='left'
        )
        
        return scores
    
    def print_anomaly_report(self):
        """Print all detected anomalies"""
        if not self.anomaly_log:
            print("No anomalies detected!")
        else:
            print(f"=== Anomaly Report ({len(self.anomaly_log)} issues) ===")
            for i, anomaly in enumerate(self.anomaly_log, 1):
                print(f"{i}. {anomaly}")


def load_dwts_data(data_path: str = None) -> DWTSDataLoader:
    """Convenience function to load data"""
    if data_path is None:
        from ..config import DATA_DIR
        data_path = DATA_DIR / "2026_MCM_Problem_C_Data.csv"
    
    loader = DWTSDataLoader(data_path)
    loader.load()
    return loader

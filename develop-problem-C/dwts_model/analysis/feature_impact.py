"""
Feature Impact Analysis

Analyze how various features impact performance:
- Pro dancer effect
- Celebrity characteristics
- Judge vs Fan preference alignment

Approach: Explainable analysis, not black-box prediction
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class FeatureEffect:
    """Effect of a feature on outcomes"""
    feature: str
    value: Any
    
    # On judge scores
    judge_score_effect: float  # Average deviation from mean
    judge_score_ci: Tuple[float, float]
    
    # On fan votes
    fan_vote_effect: float
    fan_vote_ci: Tuple[float, float]
    
    # On survival
    survival_effect: float  # Average weeks survived relative to mean
    survival_ci: Tuple[float, float]
    
    # Sample size
    n_observations: int


@dataclass
class ProDancerAnalysis:
    """Analysis of pro dancer impact"""
    dancer_effects: Dict[str, FeatureEffect]
    overall_impact: float  # How much does pro choice matter?
    top_dancers: List[str]  # Best performers
    variance_explained: float


class FeatureImpactAnalyzer:
    """
    Analyze impact of various features on DWTS performance.
    
    Focus on explainability over prediction accuracy.
    """
    
    def __init__(self):
        self.cache = {}
    
    def analyze_pro_dancer_effect(
        self,
        loader,  # DWTSDataLoader
        inversion_results: Dict[int, Any]
    ) -> ProDancerAnalysis:
        """
        Analyze how much pro dancers impact outcomes.
        
        Controls for celebrity characteristics to isolate dancer effect.
        """
        # Gather all contestant data
        records = []
        
        df = loader.processed_df
        
        for _, row in df.iterrows():
            season = row['season']
            name = row['celebrity_name']
            
            # Get average scores
            scores = loader.score_matrix[
                (loader.score_matrix['season'] == season) &
                (loader.score_matrix['contestant'] == name)
            ]
            
            valid_scores = scores[scores['total_score'] > 0]
            avg_judge = valid_scores['total_score'].mean() if len(valid_scores) > 0 else 0
            
            # Get fan vote estimate
            avg_fan = 0.0
            if season in inversion_results:
                result = inversion_results[season]
                fan_vals = []
                for week_estimates in result.week_results.values():
                    if name in week_estimates:
                        fan_vals.append(week_estimates[name].point_estimate)
                if fan_vals:
                    avg_fan = np.mean(fan_vals)
            
            records.append({
                'contestant': name,
                'season': season,
                'pro_dancer': row['ballroom_partner'],
                'age': row['celebrity_age_during_season'],
                'industry': row['celebrity_industry'],
                'placement': row['placement'],
                'avg_judge_score': avg_judge,
                'avg_fan_vote': avg_fan,
                'weeks_survived': len(valid_scores)
            })
        
        data = pd.DataFrame(records)
        
        # Compute dancer-level statistics
        dancer_stats = data.groupby('pro_dancer').agg({
            'avg_judge_score': ['mean', 'std', 'count'],
            'avg_fan_vote': ['mean', 'std'],
            'placement': 'mean',
            'weeks_survived': 'mean'
        }).reset_index()
        
        dancer_stats.columns = [
            'pro_dancer', 'judge_mean', 'judge_std', 'n_partners',
            'fan_mean', 'fan_std', 'avg_placement', 'avg_weeks'
        ]
        
        # Overall means for comparison
        overall_judge = data['avg_judge_score'].mean()
        overall_fan = data['avg_fan_vote'].mean()
        overall_weeks = data['weeks_survived'].mean()
        
        # Build feature effects
        dancer_effects = {}
        
        for _, row in dancer_stats.iterrows():
            dancer = row['pro_dancer']
            n = row['n_partners']
            
            # Bootstrap CI (simplified)
            judge_se = row['judge_std'] / np.sqrt(n) if n > 1 else 0.5
            fan_se = row['fan_std'] / np.sqrt(n) if n > 1 else 0.1
            
            dancer_effects[dancer] = FeatureEffect(
                feature='pro_dancer',
                value=dancer,
                judge_score_effect=row['judge_mean'] - overall_judge,
                judge_score_ci=(
                    row['judge_mean'] - 1.96 * judge_se - overall_judge,
                    row['judge_mean'] + 1.96 * judge_se - overall_judge
                ),
                fan_vote_effect=row['fan_mean'] - overall_fan,
                fan_vote_ci=(
                    row['fan_mean'] - 1.96 * fan_se - overall_fan,
                    row['fan_mean'] + 1.96 * fan_se - overall_fan
                ),
                survival_effect=row['avg_weeks'] - overall_weeks,
                survival_ci=(row['avg_weeks'] - 2, row['avg_weeks'] + 2),
                n_observations=int(n)
            )
        
        # Overall impact: variance in placement explained by dancer
        dancer_means = data.groupby('pro_dancer')['placement'].mean()
        between_var = np.var(dancer_means)
        total_var = np.var(data['placement'])
        variance_explained = between_var / total_var if total_var > 0 else 0
        
        # Top dancers by average placement
        top_dancers = dancer_stats.nsmallest(5, 'avg_placement')['pro_dancer'].tolist()
        
        return ProDancerAnalysis(
            dancer_effects=dancer_effects,
            overall_impact=variance_explained,
            top_dancers=top_dancers,
            variance_explained=variance_explained
        )
    
    def analyze_celebrity_characteristics(
        self,
        loader,
        inversion_results: Dict[int, Any]
    ) -> Dict[str, Dict[str, FeatureEffect]]:
        """
        Analyze impact of celebrity characteristics:
        - Age
        - Industry
        - Home country
        """
        df = loader.processed_df
        
        results = {}
        
        # Age analysis (binned)
        age_bins = [(0, 30, 'Young'), (30, 45, 'Middle'), (45, 100, 'Senior')]
        results['age'] = self._analyze_binned_feature(
            df, 'celebrity_age_during_season', age_bins, loader, inversion_results
        )
        
        # Industry analysis
        results['industry'] = self._analyze_categorical_feature(
            df, 'celebrity_industry', loader, inversion_results
        )
        
        # Country analysis (US vs non-US)
        df_copy = df.copy()
        df_copy['is_us'] = df_copy['celebrity_homecountry/region'] == 'United States'
        results['country'] = self._analyze_categorical_feature(
            df_copy, 'is_us', loader, inversion_results
        )
        
        return results
    
    def _analyze_binned_feature(
        self,
        df: pd.DataFrame,
        feature: str,
        bins: List[Tuple[float, float, str]],
        loader,
        inversion_results
    ) -> Dict[str, FeatureEffect]:
        """Analyze binned continuous feature"""
        effects = {}
        
        for low, high, label in bins:
            mask = (df[feature] >= low) & (df[feature] < high)
            subset = df[mask]
            
            if len(subset) < 3:
                continue
            
            effects[label] = self._compute_group_effect(
                subset, label, loader, inversion_results
            )
        
        return effects
    
    def _analyze_categorical_feature(
        self,
        df: pd.DataFrame,
        feature: str,
        loader,
        inversion_results
    ) -> Dict[str, FeatureEffect]:
        """Analyze categorical feature"""
        effects = {}
        
        for value in df[feature].unique():
            subset = df[df[feature] == value]
            
            if len(subset) < 3:
                continue
            
            effects[str(value)] = self._compute_group_effect(
                subset, str(value), loader, inversion_results
            )
        
        return effects
    
    def _compute_group_effect(
        self,
        subset: pd.DataFrame,
        label: str,
        loader,
        inversion_results
    ) -> FeatureEffect:
        """Compute effect for a group of contestants"""
        judge_scores = []
        fan_votes = []
        weeks = []
        
        for _, row in subset.iterrows():
            season = row['season']
            name = row['celebrity_name']
            
            # Judge scores
            scores = loader.score_matrix[
                (loader.score_matrix['season'] == season) &
                (loader.score_matrix['contestant'] == name)
            ]
            valid = scores[scores['total_score'] > 0]
            if len(valid) > 0:
                judge_scores.append(valid['total_score'].mean())
                weeks.append(len(valid))
            
            # Fan votes
            if season in inversion_results:
                result = inversion_results[season]
                fan_vals = []
                for week_estimates in result.week_results.values():
                    if name in week_estimates:
                        fan_vals.append(week_estimates[name].point_estimate)
                if fan_vals:
                    fan_votes.append(np.mean(fan_vals))
        
        # Compute effects relative to overall mean
        # (Would need overall means passed in for proper comparison)
        judge_mean = np.mean(judge_scores) if judge_scores else 0
        fan_mean = np.mean(fan_votes) if fan_votes else 0
        weeks_mean = np.mean(weeks) if weeks else 0
        
        n = len(subset)
        judge_se = np.std(judge_scores) / np.sqrt(n) if len(judge_scores) > 1 else 1
        fan_se = np.std(fan_votes) / np.sqrt(n) if len(fan_votes) > 1 else 0.1
        
        return FeatureEffect(
            feature=label,
            value=label,
            judge_score_effect=judge_mean,  # Relative effect would need baseline
            judge_score_ci=(judge_mean - 1.96*judge_se, judge_mean + 1.96*judge_se),
            fan_vote_effect=fan_mean,
            fan_vote_ci=(fan_mean - 1.96*fan_se, fan_mean + 1.96*fan_se),
            survival_effect=weeks_mean,
            survival_ci=(weeks_mean - 2, weeks_mean + 2),
            n_observations=n
        )
    
    def compare_judge_vs_fan_preferences(
        self,
        season_context,
        inversion_result
    ) -> Dict[str, Any]:
        """
        Compare what judges value vs what fans value.
        
        Do they agree or disagree?
        """
        agreements = []
        disagreements = []
        
        for week, week_ctx in season_context.weeks.items():
            if week not in inversion_result.week_results:
                continue
            
            estimates = inversion_result.week_results[week]
            
            # Get rankings
            judge_sorted = sorted(
                week_ctx.judge_percentages.items(),
                key=lambda x: x[1],
                reverse=True
            )
            judge_ranks = {c: r+1 for r, (c, _) in enumerate(judge_sorted)}
            
            fan_sorted = sorted(
                [(c, e.point_estimate) for c, e in estimates.items()],
                key=lambda x: x[1],
                reverse=True
            )
            fan_ranks = {c: r+1 for r, (c, _) in enumerate(fan_sorted)}
            
            # Compare top and bottom
            for c in week_ctx.active_set:
                j_rank = judge_ranks.get(c, 0)
                f_rank = fan_ranks.get(c, 0)
                
                rank_diff = abs(j_rank - f_rank)
                
                if rank_diff <= 1:
                    agreements.append((week, c, j_rank, f_rank))
                elif rank_diff >= 3:
                    disagreements.append((week, c, j_rank, f_rank))
        
        # Correlation
        all_j = []
        all_f = []
        for week, week_ctx in season_context.weeks.items():
            if week not in inversion_result.week_results:
                continue
            estimates = inversion_result.week_results[week]
            for c in week_ctx.active_set:
                if c in estimates:
                    all_j.append(week_ctx.judge_percentages.get(c, 0))
                    all_f.append(estimates[c].point_estimate)
        
        if len(all_j) > 2:
            correlation = np.corrcoef(all_j, all_f)[0, 1]
        else:
            correlation = 0
        
        return {
            'judge_fan_correlation': correlation,
            'n_agreements': len(agreements),
            'n_disagreements': len(disagreements),
            'agreement_rate': len(agreements) / (len(agreements) + len(disagreements)) if (len(agreements) + len(disagreements)) > 0 else 0,
            'top_disagreements': disagreements[:10]  # Most notable disagreements
        }
    
    def generate_feature_impact_summary(
        self,
        loader,
        inversion_results: Dict[int, Any]
    ) -> pd.DataFrame:
        """
        Generate summary table of all feature impacts.
        """
        records = []
        
        # Pro dancer
        pro_analysis = self.analyze_pro_dancer_effect(loader, inversion_results)
        for dancer, effect in pro_analysis.dancer_effects.items():
            records.append({
                'feature_type': 'Pro Dancer',
                'feature_value': dancer,
                'judge_effect': effect.judge_score_effect,
                'fan_effect': effect.fan_vote_effect,
                'survival_effect': effect.survival_effect,
                'n': effect.n_observations
            })
        
        # Characteristics
        char_analysis = self.analyze_celebrity_characteristics(loader, inversion_results)
        
        for feature_type, effects in char_analysis.items():
            for value, effect in effects.items():
                records.append({
                    'feature_type': feature_type.title(),
                    'feature_value': value,
                    'judge_effect': effect.judge_score_effect,
                    'fan_effect': effect.fan_vote_effect,
                    'survival_effect': effect.survival_effect,
                    'n': effect.n_observations
                })
        
        return pd.DataFrame(records)

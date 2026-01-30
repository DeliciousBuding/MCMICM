"""
Money Plots: Key visualizations for the MCM paper

Required plots:
1. Ghost in the Data - Fan vote interval by season
2. Inconsistency Spectrum - S* by season 
3. Hazard Ratio Forest - Feature effects
4. Reversal Rate Heatmap - Method comparison
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


class DWTSVisualizer:
    """
    Generate all required visualizations for MCM paper.
    
    All methods return data structures suitable for plotting.
    Actual plotting can be done with matplotlib/seaborn.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_ghost_data_plot(
        self,
        inversion_results: Dict[int, Any]
    ) -> pd.DataFrame:
        """
        Prepare data for "Ghost in the Data" plot.
        
        Shows fan vote estimation intervals across seasons.
        Wider intervals = more uncertainty = "ghostly" data.
        
        X-axis: Season
        Y-axis: Fan vote % interval width (boxplot)
        
        Story: Mark rule change years. If S28+ boxplots are wider,
        it shows judges' save introduces ambiguity.
        """
        records = []
        
        for season, result in inversion_results.items():
            for week, week_estimates in result.week_results.items():
                for contestant, estimate in week_estimates.items():
                    interval_width = estimate.upper_bound - estimate.lower_bound
                    records.append({
                        'season': season,
                        'week': week,
                        'contestant': contestant,
                        'interval_width': interval_width,
                        'certainty': estimate.certainty,
                        'point_estimate': estimate.point_estimate
                    })
        
        df = pd.DataFrame(records)
        
        # Add metadata
        df['voting_method'] = df['season'].apply(
            lambda s: 'rank' if s in [1, 2] or s >= 28 else 'percent'
        )
        df['has_judges_save'] = df['season'] >= 28
        df['rule_era'] = df['season'].apply(self._get_rule_era)
        
        return df
    
    def _get_rule_era(self, season: int) -> str:
        """Classify season by rule era"""
        if season <= 2:
            return "Early Rank (S1-2)"
        elif season <= 27:
            return "Percent Era (S3-27)"
        else:
            return "Modern Rank + Save (S28+)"
    
    def prepare_inconsistency_spectrum(
        self,
        inversion_results: Dict[int, Any]
    ) -> pd.DataFrame:
        """
        Prepare data for "Inconsistency Spectrum" plot.
        
        Shows S* (minimum slack / inconsistency score) by season.
        
        Story: S28+ should have higher S* because judges' save
        introduces mathematical inconsistency.
        """
        records = []
        
        for season, result in inversion_results.items():
            records.append({
                'season': season,
                'inconsistency_score': result.inconsistency_score,
                'is_feasible': result.is_feasible,
                'num_violations': len(result.violations),
                'method': result.method,
                'has_judges_save': season >= 28
            })
        
        return pd.DataFrame(records)
    
    def prepare_hazard_ratio_forest(
        self,
        cox_result  # CoxModelResult or BootstrapCoxResult
    ) -> pd.DataFrame:
        """
        Prepare data for "Hazard Ratio Forest" plot.
        
        Shows hazard ratios with confidence intervals for each feature.
        
        Features:
        - Pro dancer
        - Age
        - Judge score
        - Fan vote
        - Industry
        
        Story: HR < 1 means protective; HR > 1 means increases elimination risk
        """
        records = []
        
        for var, hr in cox_result.hazard_ratios.items():
            records.append({
                'variable': var,
                'hr': hr.estimate,
                'lower_95': hr.lower_95,
                'upper_95': hr.upper_95,
                'p_value': hr.p_value,
                'significant': hr.significant,
                'log_hr': np.log(hr.estimate) if hr.estimate > 0 else 0
            })
        
        df = pd.DataFrame(records)
        
        # Sort by effect size
        df = df.sort_values('hr', ascending=False)
        
        return df
    
    def prepare_reversal_heatmap(
        self,
        counterfactual_results: Dict[int, Any]  # season -> CounterfactualResult
    ) -> pd.DataFrame:
        """
        Prepare data for "Reversal Rate Heatmap".
        
        Shows how often different methods produce different eliminations.
        
        Rows: Seasons
        Columns: Comparison pairs (Rank vs Percent, etc.)
        Values: Reversal rate
        """
        records = []
        
        for season, result in counterfactual_results.items():
            records.append({
                'season': season,
                'rank_vs_percent_reversal': result.reversal_rate,
                'n_reversal_weeks': len(result.reversal_weeks),
                'total_weeks': len(result.rank_outcome.eliminations),
                'rule_era': self._get_rule_era(season)
            })
        
        return pd.DataFrame(records)
    
    def prepare_controversy_case_plots(
        self,
        case_analyses: Dict[str, Dict]
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for controversy case study visualizations.
        
        For each controversial contestant:
        - Week-by-week judge vs fan rank comparison
        - Survival trajectory under different methods
        """
        plots = {}
        
        for contestant, analysis in case_analyses.items():
            # Create trajectory data
            # This would need the full trajectory data from analysis
            plots[contestant] = pd.DataFrame([analysis])
        
        return plots
    
    def prepare_mechanism_comparison(
        self,
        evaluations: Dict[str, Any]  # mechanism -> MechanismEvaluation
    ) -> pd.DataFrame:
        """
        Prepare data for mechanism comparison radar/bar chart.
        """
        records = []
        
        for name, eval in evaluations.items():
            records.append({
                'mechanism': name,
                'judge_alignment': eval.judge_alignment,
                'fan_alignment': eval.fan_alignment,
                'technical_floor': eval.technical_floor,
                'close_calls_rate': eval.close_calls_rate,
                'controversy_index': eval.controversy_index
            })
        
        return pd.DataFrame(records)
    
    def generate_matplotlib_code(self, plot_name: str) -> str:
        """
        Generate matplotlib code for a specific plot.
        
        This can be copied into a notebook or script.
        """
        code_templates = {
            'ghost_data': '''
import matplotlib.pyplot as plt
import seaborn as sns

# Ghost in the Data Plot
fig, ax = plt.subplots(figsize=(14, 6))

# Create boxplot of interval widths by season
sns.boxplot(data=df, x='season', y='interval_width', hue='rule_era', ax=ax)

# Add vertical lines at rule change points
ax.axvline(x=2.5, color='red', linestyle='--', alpha=0.5, label='Percent Era Start')
ax.axvline(x=27.5, color='blue', linestyle='--', alpha=0.5, label='Judges Save Start')

ax.set_xlabel('Season')
ax.set_ylabel('Fan Vote Interval Width')
ax.set_title('Ghost in the Data: Fan Vote Estimation Uncertainty by Season')
ax.legend()
plt.tight_layout()
plt.savefig('ghost_data.pdf', dpi=300, bbox_inches='tight')
''',
            'inconsistency_spectrum': '''
import matplotlib.pyplot as plt

# Inconsistency Spectrum Plot
fig, ax = plt.subplots(figsize=(12, 5))

colors = ['green' if s < 28 else 'orange' for s in df['season']]
ax.bar(df['season'], df['inconsistency_score'], color=colors)

ax.axvline(x=27.5, color='red', linestyle='--', label='Judges Save Introduced')
ax.set_xlabel('Season')
ax.set_ylabel('Inconsistency Score (S*)')
ax.set_title('Inconsistency Spectrum: Model Fit Quality by Season')
ax.legend()
plt.tight_layout()
plt.savefig('inconsistency_spectrum.pdf', dpi=300, bbox_inches='tight')
''',
            'hazard_forest': '''
import matplotlib.pyplot as plt

# Hazard Ratio Forest Plot
fig, ax = plt.subplots(figsize=(10, 6))

y_pos = range(len(df))
ax.errorbar(df['hr'], y_pos, 
            xerr=[df['hr'] - df['lower_95'], df['upper_95'] - df['hr']],
            fmt='o', capsize=5, capthick=2)

ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(df['variable'])
ax.set_xlabel('Hazard Ratio (95% CI)')
ax.set_title('Impact of Features on Elimination Risk')

# Mark significant effects
for i, (_, row) in enumerate(df.iterrows()):
    if row['significant']:
        ax.scatter(row['hr'], i, marker='*', s=200, c='red', zorder=5)

plt.tight_layout()
plt.savefig('hazard_forest.pdf', dpi=300, bbox_inches='tight')
''',
            'reversal_heatmap': '''
import matplotlib.pyplot as plt
import seaborn as sns

# Reversal Rate Heatmap
pivot = df.pivot_table(index='rule_era', values='rank_vs_percent_reversal', aggfunc='mean')

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn_r', ax=ax)
ax.set_title('Method Reversal Rates by Era')
plt.tight_layout()
plt.savefig('reversal_heatmap.pdf', dpi=300, bbox_inches='tight')
'''
        }
        
        return code_templates.get(plot_name, "# Plot template not found")
    
    def save_plot_data(self, df: pd.DataFrame, name: str):
        """Save plot data to CSV for reproducibility"""
        path = self.output_dir / f"{name}_data.csv"
        df.to_csv(path, index=False)
        return path


class PlotConfig:
    """Standard configuration for MCM paper plots"""
    
    # Figure sizes (inches)
    SINGLE_COLUMN = (6, 4)
    DOUBLE_COLUMN = (12, 4)
    SQUARE = (6, 6)
    
    # Colors
    RANK_COLOR = '#1f77b4'      # Blue
    PERCENT_COLOR = '#ff7f0e'   # Orange
    TIERED_COLOR = '#2ca02c'    # Green
    
    RULE_ERA_COLORS = {
        'Early Rank (S1-2)': '#1f77b4',
        'Percent Era (S3-27)': '#ff7f0e',
        'Modern Rank + Save (S28+)': '#2ca02c'
    }
    
    # Font sizes
    TITLE_SIZE = 14
    LABEL_SIZE = 12
    TICK_SIZE = 10
    
    @staticmethod
    def get_style_code() -> str:
        """Return matplotlib style setup code"""
        return '''
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set up publication-quality defaults
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['figure.dpi'] = 300
'''

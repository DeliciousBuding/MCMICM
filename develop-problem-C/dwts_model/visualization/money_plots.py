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
        Uses the unified DWTS Paper Palette.
        """
        code_templates = {
            'ghost_data': '''
import matplotlib.pyplot as plt
import seaborn as sns
from dwts_model.paper_palette import PALETTE, apply_paper_style

# Ghost in the Data Plot
fig, ax = plt.subplots(figsize=(14, 6))

# Create boxplot of interval widths by season using unified palette
palette = {
    'Early Rank (S1-2)': PALETTE['baseline'],
    'Percent Era (S3-27)': PALETTE['proposed'],
    'Modern Rank + Save (S28+)': PALETTE['aux']
}
sns.boxplot(data=df, x='season', y='interval_width', hue='rule_era', 
            palette=palette, ax=ax)

# Add vertical lines at rule change points
ax.axvline(x=2.5, color=PALETTE['warning'], linestyle='--', alpha=0.7, label='Percent Era Start')
ax.axvline(x=27.5, color=PALETTE['warning2'], linestyle='--', alpha=0.7, label='Judges Save Start')

ax.set_xlabel('Season')
ax.set_ylabel('Fan Vote Interval Width')
ax.set_title('Ghost in the Data: Fan Vote Estimation Uncertainty by Season')
ax.legend()
apply_paper_style(ax)
plt.tight_layout()
plt.savefig('ghost_data.pdf', dpi=300, bbox_inches='tight')
''',
            'inconsistency_spectrum': '''
import matplotlib.pyplot as plt
from dwts_model.paper_palette import PALETTE, get_season_colors, apply_paper_style

# Inconsistency Spectrum Plot
fig, ax = plt.subplots(figsize=(12, 5))

colors = get_season_colors(df['season'].tolist())
ax.bar(df['season'], df['inconsistency_score'], color=colors, edgecolor=PALETTE['aux'])

ax.axvline(x=27.5, color=PALETTE['warning'], linestyle='--', linewidth=2, label='Judges Save Introduced')
ax.set_xlabel('Season')
ax.set_ylabel('Inconsistency Score (S*)')
ax.set_title('Inconsistency Spectrum: Model Fit Quality by Season')
ax.legend()
apply_paper_style(ax)
plt.tight_layout()
plt.savefig('inconsistency_spectrum.pdf', dpi=300, bbox_inches='tight')
''',
            'hazard_forest': '''
import matplotlib.pyplot as plt
from dwts_model.paper_palette import PALETTE, apply_paper_style

# Hazard Ratio Forest Plot
fig, ax = plt.subplots(figsize=(10, 6))

y_pos = range(len(df))
ax.errorbar(df['hr'], y_pos, 
            xerr=[df['hr'] - df['lower_95'], df['upper_95'] - df['hr']],
            fmt='o', capsize=5, capthick=2, color=PALETTE['proposed'])

ax.axvline(x=1, color=PALETTE['aux'], linestyle='--', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(df['variable'])
ax.set_xlabel('Hazard Ratio (95% CI)')
ax.set_title('Impact of Features on Elimination Risk')

# Mark significant effects with warning color
for i, (_, row) in enumerate(df.iterrows()):
    if row['significant']:
        ax.scatter(row['hr'], i, marker='*', s=200, c=PALETTE['warning'], zorder=5)

apply_paper_style(ax)
plt.tight_layout()
plt.savefig('hazard_forest.pdf', dpi=300, bbox_inches='tight')
''',
            'reversal_heatmap': '''
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from dwts_model.paper_palette import PALETTE, apply_paper_style

# Reversal Rate Heatmap - Custom colormap using palette
colors = [PALETTE['fill'], PALETTE['warning']]
cmap = LinearSegmentedColormap.from_list('dwts_reversal', colors)

pivot = df.pivot_table(index='rule_era', values='rank_vs_percent_reversal', aggfunc='mean')

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt='.1%', cmap=cmap, ax=ax)
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
    """Standard configuration for MCM paper plots - 使用统一调色板"""
    
    # 导入统一调色板
    try:
        from dwts_model.paper_palette import (
            PALETTE, VOTING_METHODS, MECHANISMS, 
            get_season_color, apply_paper_style
        )
        _palette_available = True
    except ImportError:
        _palette_available = False
        PALETTE = {
            "proposed": "#219EBC",
            "baseline": "#02304A", 
            "warning":  "#FA8600",
            "warning2": "#FF9E02",
            "fill":     "#90C9E7",
            "accent":   "#FEB705",
            "aux":      "#136783",
        }
    
    # Figure sizes (inches)
    SINGLE_COLUMN = (6, 4)
    DOUBLE_COLUMN = (12, 4)
    SQUARE = (6, 6)
    
    # Colors - 使用统一调色板
    RANK_COLOR = PALETTE['baseline'] if _palette_available else '#02304A'      # 藏蓝 - 排名制
    PERCENT_COLOR = PALETTE['proposed'] if _palette_available else '#219EBC'   # 青蓝 - 百分比制
    WARNING_COLOR = PALETTE['warning'] if _palette_available else '#FA8600'    # 深橙 - 警示
    
    RULE_ERA_COLORS = {
        'Early Rank (S1-2)': PALETTE['baseline'] if _palette_available else '#02304A',
        'Percent Era (S3-27)': PALETTE['proposed'] if _palette_available else '#219EBC',
        'Modern Rank + Save (S28+)': PALETTE['aux'] if _palette_available else '#136783'
    }
    
    # Font sizes
    TITLE_SIZE = 14
    LABEL_SIZE = 12
    TICK_SIZE = 10
    
    @staticmethod
    def get_style_code() -> str:
        """Return matplotlib style setup code with unified palette"""
        return '''
import matplotlib.pyplot as plt
import matplotlib as mpl

# DWTS Paper Palette - 统一配色方案
PALETTE = {
    "proposed": "#219EBC",   # 青蓝 - 新机制
    "baseline": "#02304A",   # 藏蓝 - 基准
    "warning":  "#FA8600",   # 深橙 - 警示
    "warning2": "#FF9E02",   # 亮橙 - 次级警示
    "fill":     "#90C9E7",   # 浅蓝 - 填充
    "accent":   "#FEB705",   # 黄色 - 标注
    "aux":      "#136783",   # 深青 - 辅助
}

# Set up publication-quality defaults
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[
    PALETTE["proposed"], PALETTE["baseline"], PALETTE["warning"],
    PALETTE["aux"], PALETTE["warning2"]
])
'''

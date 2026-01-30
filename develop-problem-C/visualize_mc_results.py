"""
Monte Carlo Results Visualization
Generates comprehensive PDF visualizations for probabilistic robustness analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

OUTPUT_DIR = Path('outputs')
FIGURES_DIR = Path('figures')
FIGURES_DIR.mkdir(exist_ok=True)

# Paper-quality settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def load_results(csv_path: str = 'outputs/mc_robustness_results.csv') -> pd.DataFrame:
    """Load Monte Carlo results"""
    df = pd.read_csv(csv_path)
    return df


def plot_probability_distribution(df: pd.DataFrame, output_path: str = 'figures/mc_probability_distribution.pdf'):
    """
    Plot distribution of P(Wrongful) across all eliminations
    Shows histogram + KDE to visualize the overall fairness landscape
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Left: Histogram with KDE
    ax1 = axes[0]
    ax1.hist(df['p_wrongful'], bins=30, density=True, alpha=0.6, 
             color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Add KDE
    from scipy import stats
    kde = stats.gaussian_kde(df['p_wrongful'])
    x_range = np.linspace(0, 1, 200)
    ax1.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    # Add reference lines
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold (50%)')
    mean_val = df['p_wrongful'].mean()
    ax1.axvline(x=mean_val, color='darkred', linestyle='-', linewidth=2, 
                label=f'Mean ({mean_val:.3f})')
    
    ax1.set_xlabel('P(Wrongful Elimination)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Distribution of Wrongful Elimination Probabilities\n(Monte Carlo Analysis)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Cumulative distribution
    ax2 = axes[1]
    sorted_probs = np.sort(df['p_wrongful'])
    cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
    ax2.plot(sorted_probs, cumulative, linewidth=2, color='steelblue')
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add key percentiles
    percentiles = [0.25, 0.5, 0.75]
    for p in percentiles:
        val = np.percentile(df['p_wrongful'], p * 100)
        ax2.plot(val, p, 'ro', markersize=8)
        ax2.text(val + 0.02, p, f'{val:.2f}', fontsize=9, verticalalignment='center')
    
    ax2.set_xlabel('P(Wrongful Elimination)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution Function')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_season_evolution(df: pd.DataFrame, output_path: str = 'figures/mc_season_evolution.pdf'):
    """
    Plot evolution of P(Wrongful) across seasons
    Shows how fairness changed over time
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Compute season statistics
    season_stats = df.groupby('season').agg({
        'p_wrongful': ['mean', 'median', 'std', 'count']
    }).reset_index()
    season_stats.columns = ['season', 'mean', 'median', 'std', 'count']
    
    # Top: Mean ± 95% CI
    ax1 = axes[0]
    seasons = season_stats['season']
    means = season_stats['mean']
    stds = season_stats['std']
    counts = season_stats['count']
    
    # Calculate 95% CI
    ci = 1.96 * stds / np.sqrt(counts)
    
    ax1.plot(seasons, means, 'o-', linewidth=2, markersize=6, 
             color='steelblue', label='Mean P(Wrongful)')
    ax1.fill_between(seasons, means - ci, means + ci, 
                     alpha=0.3, color='steelblue', label='95% CI')
    
    # Add reference line
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold (50%)')
    
    # Highlight rule changes
    rule_changes = {
        3: 'Percent\nRule Start',
        28: 'Judge\nSave',
        32: 'Model\nMismatch'
    }
    for season, label in rule_changes.items():
        if season in seasons.values:
            ax1.axvline(x=season, color='red', linestyle=':', alpha=0.4)
            ax1.text(season, ax1.get_ylim()[1] * 0.95, label, 
                    fontsize=8, ha='center', color='red')
    
    ax1.set_xlabel('Season')
    ax1.set_ylabel('Mean P(Wrongful)')
    ax1.set_title('Evolution of Wrongful Elimination Probability Across Seasons')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Sample counts per season
    ax2 = axes[1]
    ax2.bar(seasons, counts, color='lightcoral', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Season')
    ax2.set_ylabel('Number of Eliminations')
    ax2.set_title('Sample Size per Season')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_confidence_intervals(df: pd.DataFrame, top_n: int = 20, 
                              output_path: str = 'figures/mc_confidence_intervals.pdf'):
    """
    Plot confidence intervals for top cases
    Shows uncertainty quantification
    """
    # Get top N by P(Wrongful)
    top_cases = df.nlargest(top_n, 'p_wrongful').copy()
    top_cases['label'] = (top_cases['contestant'].str[:15] + 
                          ' (S' + top_cases['season'].astype(str) + 
                          'W' + top_cases['week'].astype(str) + ')')
    
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
    
    y_pos = np.arange(len(top_cases))
    
    # Plot error bars
    for i, (idx, row) in enumerate(top_cases.iterrows()):
        ci_width = row['ci_upper'] - row['ci_lower']
        color = 'darkred' if row['p_wrongful'] > 0.9 else 'steelblue'
        
        # Error bar
        ax.errorbar(row['p_wrongful'], i, 
                   xerr=[[row['p_wrongful'] - row['ci_lower']], 
                         [row['ci_upper'] - row['p_wrongful']]],
                   fmt='o', markersize=8, capsize=5, capthick=2,
                   color=color, ecolor=color, alpha=0.7)
        
        # Add numeric label
        ax.text(row['ci_upper'] + 0.02, i, 
               f"{row['p_wrongful']:.3f}", 
               fontsize=8, verticalalignment='center')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_cases['label'], fontsize=8)
    ax.set_xlabel('P(Wrongful Elimination)')
    ax.set_title(f'Top {top_n} Most Likely Wrongful Eliminations\n(with 95% Confidence Intervals)')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_voting_method_comparison(df: pd.DataFrame, 
                                  output_path: str = 'figures/mc_voting_method_comparison.pdf'):
    """
    Compare P(Wrongful) between percent and rank seasons
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Box plot comparison
    ax1 = axes[0]
    df_plot = df[df['voting_method'].isin(['percent', 'rank'])]
    
    box_data = [df_plot[df_plot['voting_method'] == 'percent']['p_wrongful'],
                df_plot[df_plot['voting_method'] == 'rank']['p_wrongful']]
    
    bp = ax1.boxplot(box_data, labels=['Percent Rule\n(S3-S27)', 'Rank Rule\n(S1-S2, S28+)'],
                     patch_artist=True, widths=0.6)
    
    # Color boxes
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('P(Wrongful Elimination)')
    ax1.set_title('P(Wrongful) by Voting Method')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Right: Violin plot
    ax2 = axes[1]
    parts = ax2.violinplot(box_data, positions=[0, 1], 
                          showmeans=True, showmedians=True, widths=0.7)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Percent Rule\n(S3-S27)', 'Rank Rule\n(S1-S2, S28+)'])
    ax2.set_ylabel('P(Wrongful Elimination)')
    ax2.set_title('Distribution Density by Voting Method')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add statistics
    from scipy import stats as sp_stats
    percent_data = df_plot[df_plot['voting_method'] == 'percent']['p_wrongful']
    rank_data = df_plot[df_plot['voting_method'] == 'rank']['p_wrongful']
    
    if len(percent_data) > 0 and len(rank_data) > 0:
        t_stat, p_val = sp_stats.ttest_ind(percent_data, rank_data)
        ax2.text(0.5, ax2.get_ylim()[0] + 0.05, 
                f't-test: p={p_val:.4f}', 
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_classification_breakdown(df: pd.DataFrame, 
                                  output_path: str = 'figures/mc_classification_breakdown.pdf'):
    """
    Breakdown of classifications (Definite-Wrongful, Uncertain, Definite-Correct)
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Pie chart
    ax1 = axes[0]
    class_counts = df['classification'].value_counts()
    colors = {'Definite-Wrongful': '#d62728', 
              'Uncertain': '#ff7f0e', 
              'Definite-Correct': '#2ca02c'}
    
    wedges, texts, autotexts = ax1.pie(class_counts.values, 
                                       labels=class_counts.index,
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       colors=[colors.get(c, 'gray') for c in class_counts.index])
    
    # Enhance text
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_weight('bold')
    
    ax1.set_title('Classification Distribution\n(Threshold: 95% for Definite)')
    
    # Right: Stacked bar by season
    ax2 = axes[1]
    
    # Get classification counts by season
    pivot = pd.crosstab(df['season'], df['classification'], normalize='index') * 100
    
    # Reorder columns
    col_order = ['Definite-Correct', 'Uncertain', 'Definite-Wrongful']
    pivot = pivot[[c for c in col_order if c in pivot.columns]]
    
    pivot.plot(kind='bar', stacked=True, ax=ax2, 
              color=[colors.get(c, 'gray') for c in pivot.columns],
              width=0.8, edgecolor='black', linewidth=0.3)
    
    ax2.set_xlabel('Season')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Classification Breakdown by Season')
    ax2.legend(title='Classification', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_interval_width_analysis(df: pd.DataFrame,
                                 output_path: str = 'figures/mc_interval_width_analysis.pdf'):
    """
    Analyze relationship between interval width and certainty
    Addresses the "解空间体积" question
    """
    df['interval_width'] = df['fan_vote_upper'] - df['fan_vote_lower']
    df['ci_width'] = df['ci_upper'] - df['ci_lower']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top-left: Interval width vs P(Wrongful)
    ax1 = axes[0, 0]
    scatter = ax1.scatter(df['interval_width'], df['p_wrongful'], 
                         c=df['season'], cmap='viridis', 
                         alpha=0.6, s=30, edgecolors='black', linewidth=0.3)
    ax1.set_xlabel('LP Interval Width (Upper - Lower)')
    ax1.set_ylabel('P(Wrongful)')
    ax1.set_title('Interval Width vs Wrongful Probability')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Season')
    
    # Add correlation
    from scipy.stats import pearsonr
    if len(df) > 2:
        corr, p_val = pearsonr(df['interval_width'], df['p_wrongful'])
        ax1.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.4f}',
                transform=ax1.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Top-right: CI width distribution
    ax2 = axes[0, 1]
    ax2.hist(df['ci_width'], bins=30, color='steelblue', 
            edgecolor='black', linewidth=0.5, alpha=0.7)
    ax2.set_xlabel('Confidence Interval Width')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of CI Widths\n(Uncertainty Quantification)')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axvline(df['ci_width'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {df["ci_width"].mean():.3f}')
    ax2.legend()
    
    # Bottom-left: Interval width by season
    ax3 = axes[1, 0]
    season_widths = df.groupby('season')['interval_width'].mean()
    ax3.bar(season_widths.index, season_widths.values, 
           color='lightcoral', edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Season')
    ax3.set_ylabel('Mean Interval Width')
    ax3.set_title('LP Interval Width Evolution\n(Proxy for "解空间体积")')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Highlight S28 (Judge Save introduction)
    if 28 in season_widths.index:
        ax3.axvline(x=28, color='red', linestyle=':', alpha=0.5)
        ax3.text(28, ax3.get_ylim()[1] * 0.9, 'Judge\nSave', 
                fontsize=8, ha='center', color='red')
    
    # Bottom-right: Scatter matrix style
    ax4 = axes[1, 1]
    
    # Create bins for interval width
    df['width_bin'] = pd.qcut(df['interval_width'], q=4, 
                              labels=['Narrow', 'Medium-Narrow', 'Medium-Wide', 'Wide'])
    
    box_data = [df[df['width_bin'] == cat]['p_wrongful'] 
                for cat in ['Narrow', 'Medium-Narrow', 'Medium-Wide', 'Wide']]
    
    bp = ax4.boxplot(box_data, labels=['Narrow', 'Med-Narrow', 'Med-Wide', 'Wide'],
                    patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax4.set_xlabel('LP Interval Width Category')
    ax4.set_ylabel('P(Wrongful)')
    ax4.set_title('P(Wrongful) by Interval Width Category')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def generate_summary_table(df: pd.DataFrame, 
                           output_path: str = 'outputs/mc_summary_statistics.tex'):
    """
    Generate LaTeX table with summary statistics
    """
    summary_data = []
    
    # Overall statistics
    summary_data.append({
        'Category': 'Overall',
        'N': len(df),
        'Mean P(W)': f"{df['p_wrongful'].mean():.3f}",
        'Median P(W)': f"{df['p_wrongful'].median():.3f}",
        'Std P(W)': f"{df['p_wrongful'].std():.3f}",
        'Mean CI Width': f"{(df['ci_upper'] - df['ci_lower']).mean():.3f}"
    })
    
    # By voting method
    for method in df['voting_method'].unique():
        subset = df[df['voting_method'] == method]
        summary_data.append({
            'Category': f'Method: {method}',
            'N': len(subset),
            'Mean P(W)': f"{subset['p_wrongful'].mean():.3f}",
            'Median P(W)': f"{subset['p_wrongful'].median():.3f}",
            'Std P(W)': f"{subset['p_wrongful'].std():.3f}",
            'Mean CI Width': f"{(subset['ci_upper'] - subset['ci_lower']).mean():.3f}"
        })
    
    # By classification
    for cls in df['classification'].unique():
        subset = df[df['classification'] == cls]
        summary_data.append({
            'Category': f'Class: {cls}',
            'N': len(subset),
            'Mean P(W)': f"{subset['p_wrongful'].mean():.3f}",
            'Median P(W)': f"{subset['p_wrongful'].median():.3f}",
            'Std P(W)': f"{subset['p_wrongful'].std():.3f}",
            'Mean CI Width': f"{(subset['ci_upper'] - subset['ci_lower']).mean():.3f}"
        })
    
    # Convert to DataFrame and save as LaTeX
    summary_df = pd.DataFrame(summary_data)
    
    latex_table = summary_df.to_latex(
        index=False,
        caption='Monte Carlo Robustness Analysis Summary Statistics',
        label='tab:mc_summary',
        column_format='lrrrrrr',
        escape=False
    )
    
    with open(output_path, 'w') as f:
        f.write(latex_table)
    
    print(f"✓ Saved: {output_path}")
    
    return summary_df


def main():
    """Generate all visualizations"""
    print("\n" + "=" * 60)
    print("MONTE CARLO RESULTS VISUALIZATION")
    print("=" * 60 + "\n")
    
    # Load results
    print("Loading results...")
    df = load_results()
    print(f"Loaded {len(df)} eliminations from {df['season'].nunique()} seasons\n")
    
    # Generate all plots
    print("Generating visualizations...\n")
    
    plot_probability_distribution(df)
    plot_season_evolution(df)
    plot_confidence_intervals(df, top_n=20)
    plot_voting_method_comparison(df)
    plot_classification_breakdown(df)
    plot_interval_width_analysis(df)
    
    # Generate summary table
    print("\nGenerating summary statistics table...")
    summary_df = generate_summary_table(df)
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("✓ All visualizations complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Figures: {FIGURES_DIR}/")
    print(f"  - Summary: outputs/mc_summary_statistics.tex")
    print(f"  - Raw data: outputs/mc_robustness_results.csv")


if __name__ == '__main__':
    main()

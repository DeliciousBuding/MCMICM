"""
Run Monte Carlo Robustness Analysis

Generates probabilistic fairness metrics instead of binary classifications.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dwts_model.etl import DWTSDataLoader, ActiveSetManager
from dwts_model.engines import PercentLPEngine, RankCPEngine
from dwts_model.sampling import MonteCarloRobustnessAnalyzer
from dwts_model.config import OUTPUT_DIR


def run_mc_robustness_analysis(
    seasons=None,
    n_samples=10000,
    output_file='mc_robustness_results.csv'
):
    """
    Run Monte Carlo robustness analysis for all eliminations.
    
    Args:
        seasons: List of seasons to analyze (None = all)
        n_samples: Number of Monte Carlo samples per elimination
        output_file: Output CSV filename
    """
    print("=" * 60)
    print("MONTE CARLO ROBUSTNESS ANALYSIS")
    print("=" * 60)
    print(f"Samples per elimination: {n_samples}")
    print()
    
    # Load data
    print("Loading data...")
    loader = DWTSDataLoader('2026_MCM_Problem_C_Data.csv')
    loader.load()
    
    manager = ActiveSetManager(loader)
    manager.build_all_contexts()
    
    # Initialize engines and analyzer
    lp_engine = PercentLPEngine()
    cp_engine = RankCPEngine()
    mc_analyzer = MonteCarloRobustnessAnalyzer(
        n_samples=n_samples,
        burnin=1000,
        thin=5
    )
    
    # Analyze all seasons
    all_results = []
    
    season_list = seasons if seasons else manager.get_all_seasons()
    
    for season in tqdm(season_list, desc="Analyzing seasons"):
        context = manager.get_season_context(season)
        
        # Run inversion to get bounds
        if context.voting_method == 'percent':
            inversion_result = lp_engine.solve(context)
            method = 'percent'
        else:
            inversion_result = cp_engine.solve(context)
            method = 'rank'
        
        # Get interval bounds for each week
        for week, week_ctx in context.weeks.items():
            if not week_ctx.has_valid_elimination():
                continue
            
            eliminated_list = week_ctx.eliminated
            if not eliminated_list:
                continue
            
            # Handle potential double eliminations - analyze each separately
            for eliminated in (eliminated_list if isinstance(eliminated_list, list) else [eliminated_list]):
                
                # Get bounds from inversion result
                week_estimates = inversion_result.week_results.get(week, {})
                interval_bounds = {}
                
                for contestant in week_ctx.active_set:
                    est = week_estimates.get(contestant)
                    if est:
                        interval_bounds[contestant] = (
                            est.lower_bound,
                            est.upper_bound
                        )
                    else:
                        interval_bounds[contestant] = (0.01, 0.99)
                
                # Run Monte Carlo analysis
                try:
                    mc_result = mc_analyzer.analyze_elimination(
                        season=season,
                        week=week,
                        eliminated=eliminated,
                        week_context=week_ctx,
                        interval_bounds=interval_bounds,
                        voting_method=method
                    )
                    
                    all_results.append({
                        'season': season,
                        'week': week,
                        'contestant': eliminated,
                        'voting_method': method,
                        'p_wrongful': mc_result.p_wrongful,
                        'p_correct': mc_result.p_correct,
                        'ci_lower': mc_result.ci_lower,
                        'ci_upper': mc_result.ci_upper,
                        'n_samples': mc_result.n_samples,
                        'wrongful_count': mc_result.wrongful_count,
                        'correct_count': mc_result.correct_count,
                        'classification': mc_result.get_classification(threshold=0.05),
                        'fan_vote_lower': mc_result.fan_vote_lower,
                        'fan_vote_upper': mc_result.fan_vote_upper,
                        'fan_vote_mean': mc_result.mean_fan_vote,
                        'fan_vote_median': mc_result.median_fan_vote
                    })
                    
                except Exception as e:
                    print(f"  [Warning] S{season} Wk{week} {eliminated}: {str(e)}")
                    continue
    
    # Save results
    df = pd.DataFrame(all_results)
    output_path = OUTPUT_DIR / output_file
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    print(f"\nTotal eliminations analyzed: {len(df)}")
    print(f"Mean P(Wrongful): {df['p_wrongful'].mean():.3f}")
    print(f"Median P(Wrongful): {df['p_wrongful'].median():.3f}")
    
    print("\nClassification Distribution:")
    print(df['classification'].value_counts())
    
    print("\nTop 10 Most Likely Wrongful Eliminations:")
    top_wrongful = df.nlargest(10, 'p_wrongful')[
        ['season', 'week', 'contestant', 'p_wrongful', 'ci_lower', 'ci_upper']
    ]
    print(top_wrongful.to_string(index=False))
    
    print("\nBy Season (Mean P(Wrongful)):")
    season_summary = df.groupby('season')['p_wrongful'].agg(['mean', 'median', 'count'])
    season_summary = season_summary.sort_values('mean', ascending=False).head(10)
    print(season_summary)
    
    return df


def compare_with_interval_robust():
    """
    Compare Monte Carlo results with original interval-robust classification.
    """
    print("\n" + "=" * 60)
    print("COMPARISON: Monte Carlo vs Interval-Robust")
    print("=" * 60)
    
    # This would compare the probabilistic results with the original
    # Definite/Possible/Safe classifications
    
    # Load MC results
    mc_df = pd.read_csv(OUTPUT_DIR / 'mc_robustness_results.csv')
    
    # Count by classification
    mc_counts = mc_df['classification'].value_counts()
    
    print("\nMonte Carlo Classification (threshold=5%):")
    for cls, count in mc_counts.items():
        pct = count / len(mc_df) * 100
        print(f"  {cls}: {count} ({pct:.1f}%)")
    
    print("\nOriginal Interval-Robust would give:")
    print("  Definite-Wrongful: ~40 (17.4%)")
    print("  Possible-Wrongful: ~37 (16.1%)")
    print("  Definite-Safe: ~190 (66.5%)")
    
    print("\nKey Insights:")
    print("  • Monte Carlo provides continuous probabilities")
    print("  • Can adjust threshold to trade precision/recall")
    print("  • Confidence intervals quantify uncertainty")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Monte Carlo Robustness Analysis")
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=5000,
        help='Number of Monte Carlo samples (default: 5000)'
    )
    parser.add_argument(
        '--seasons', '-s',
        type=str,
        default=None,
        help='Seasons to analyze (e.g., "1-10" or "32,33")'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare with interval-robust method'
    )
    
    args = parser.parse_args()
    
    # Parse seasons
    seasons = None
    if args.seasons:
        if '-' in args.seasons:
            start, end = map(int, args.seasons.split('-'))
            seasons = list(range(start, end + 1))
        else:
            seasons = [int(s) for s in args.seasons.split(',')]
    
    # Run analysis
    results_df = run_mc_robustness_analysis(
        seasons=seasons,
        n_samples=args.samples
    )
    
    if args.compare:
        compare_with_interval_robust()
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

"""
DWTS Fan Vote Inversion System - Main Entry Point

MCM 2026 Problem C: Data With The Stars

This script orchestrates the complete analysis pipeline:
1. Load and preprocess data
2. Run fan vote inversion for all seasons
3. Perform counterfactual analysis
4. Analyze feature impacts
5. Evaluate mechanism designs
6. Generate visualizations and reports

Usage:
    python run_analysis.py [--seasons 1-34] [--output outputs/]
"""
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json
import warnings

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from dwts_model.config import SEASON_CONFIG, MODEL_CONFIG, OUTPUT_DIR, DATA_DIR
from dwts_model.etl import DWTSDataLoader, ActiveSetManager
from dwts_model.engines import PercentLPEngine, RankCPEngine, JudgesSaveHandler
from dwts_model.sampling import DirichletHitAndRunSampler, CounterfactualSimulator
from dwts_model.analysis import CoxSurvivalAnalyzer, FeatureImpactAnalyzer
from dwts_model.mechanism import TieredThresholdSystem, ProposedMechanism
from dwts_model.visualization import DWTSVisualizer, ReportGenerator


class DWTSAnalysisPipeline:
    """
    Main analysis pipeline for DWTS voting system analysis.
    """
    
    def __init__(
        self, 
        data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        verbose: bool = True
    ):
        self.data_path = data_path or str(DATA_DIR / "2026_MCM_Problem_C_Data.csv")
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        # Initialize components
        self.loader = None
        self.active_manager = None
        self.inversion_results = {}
        self.counterfactual_results = {}
        
    def log(self, message: str):
        """Print message if verbose"""
        if self.verbose:
            print(f"[DWTS] {message}")
    
    def run_full_pipeline(
        self,
        seasons: Optional[List[int]] = None,
        skip_mcmc: bool = False
    ):
        """
        Run complete analysis pipeline.
        
        Args:
            seasons: List of seasons to analyze (default: all)
            skip_mcmc: Skip MCMC sampling (faster, less uncertainty quantification)
        """
        self.log("=" * 60)
        self.log("DWTS Fan Vote Inversion Analysis Pipeline")
        self.log("=" * 60)
        
        # Step 1: Load data
        self.log("\n[1/6] Loading and preprocessing data...")
        self.load_data()
        
        # Step 2: Run inversion
        self.log("\n[2/6] Running fan vote inversion...")
        self.run_inversion(seasons)
        
        # Step 3: Counterfactual analysis
        self.log("\n[3/6] Running counterfactual analysis...")
        self.run_counterfactual_analysis(seasons)
        
        # Step 4: Feature impact analysis
        self.log("\n[4/6] Analyzing feature impacts...")
        feature_analysis = self.run_feature_analysis(seasons)
        
        # Step 5: Mechanism evaluation
        self.log("\n[5/6] Evaluating voting mechanisms...")
        mechanism_evaluation = self.run_mechanism_evaluation(seasons)
        
        # Step 6: Generate outputs
        self.log("\n[6/6] Generating reports and visualizations...")
        self.generate_outputs(feature_analysis, mechanism_evaluation)
        
        self.log("\n" + "=" * 60)
        self.log("Analysis complete! Results saved to: " + str(self.output_dir))
        self.log("=" * 60)
        
        return {
            'inversion_results': self.inversion_results,
            'counterfactual_results': self.counterfactual_results,
            'feature_analysis': feature_analysis,
            'mechanism_evaluation': mechanism_evaluation
        }
    
    def load_data(self):
        """Load and preprocess DWTS data"""
        self.loader = DWTSDataLoader(self.data_path)
        self.loader.load()
        
        # Print data summary
        n_seasons = self.loader.processed_df['season'].nunique()
        n_contestants = len(self.loader.processed_df)
        
        self.log(f"  Loaded {n_contestants} contestants across {n_seasons} seasons")
        
        # Print anomalies if any
        if self.loader.anomaly_log:
            self.log(f"  Found {len(self.loader.anomaly_log)} anomalies (see anomaly_log.txt)")
            with open(self.output_dir / 'anomaly_log.txt', 'w') as f:
                for anomaly in self.loader.anomaly_log:
                    f.write(anomaly + '\n')
        
        # Build active set manager
        self.active_manager = ActiveSetManager(self.loader)
        self.active_manager.build_all_contexts()
        
        # Save summary
        summary = self.active_manager.generate_summary_report()
        summary.to_csv(self.output_dir / 'season_summary.csv', index=False)
        self.log(f"  Season summary saved to season_summary.csv")
    
    def run_inversion(self, seasons: Optional[List[int]] = None):
        """Run fan vote inversion for specified seasons"""
        all_seasons = seasons or self.active_manager.get_all_seasons()
        
        # Initialize engines
        lp_engine = PercentLPEngine()
        cp_engine = RankCPEngine()
        judges_save = JudgesSaveHandler()
        
        for season in all_seasons:
            context = self.active_manager.get_season_context(season)
            
            # Choose engine based on voting method
            if context.voting_method == 'percent':
                result = lp_engine.solve(context)
            else:
                result = cp_engine.solve(context)
            
            self.inversion_results[season] = result
            
            # Log summary
            feasible = "[OK]" if result.is_feasible else "[X]"
            self.log(f"  Season {season}: {context.voting_method} method, "
                    f"S*={result.inconsistency_score:.3f} {feasible}")
        
        # Save results
        self._save_inversion_results()
    
    def _save_inversion_results(self):
        """Save inversion results to files"""
        # Summary CSV
        records = []
        for season, result in self.inversion_results.items():
            for week, estimates in result.week_results.items():
                for contestant, est in estimates.items():
                    records.append({
                        'season': season,
                        'week': week,
                        'contestant': contestant,
                        'fan_vote_estimate': est.point_estimate,
                        'lower_bound': est.lower_bound,
                        'upper_bound': est.upper_bound,
                        'certainty': est.certainty,
                        'method': est.method
                    })
        
        import pandas as pd
        df = pd.DataFrame(records)
        df.to_csv(self.output_dir / 'fan_vote_estimates.csv', index=False)
        self.log(f"  Fan vote estimates saved ({len(records)} estimates)")
    
    def run_counterfactual_analysis(self, seasons: Optional[List[int]] = None):
        """Run counterfactual comparison between methods"""
        all_seasons = seasons or list(self.inversion_results.keys())
        simulator = CounterfactualSimulator()
        
        for season in all_seasons:
            if season not in self.inversion_results:
                continue
            
            context = self.active_manager.get_season_context(season)
            result = self.inversion_results[season]
            
            # Convert inversion result to fan votes dict
            fan_votes = result.get_point_estimates_matrix()
            
            # Run comparison
            comparison = simulator.compare_methods(context, fan_votes)
            self.counterfactual_results[season] = comparison
            
            if comparison.reversal_rate > 0:
                self.log(f"  Season {season}: {comparison.reversal_rate:.0%} reversal rate "
                        f"({len(comparison.reversal_weeks)} weeks differ)")
        
        # Save results
        self._save_counterfactual_results()
    
    def _save_counterfactual_results(self):
        """Save counterfactual results"""
        import pandas as pd
        
        records = []
        for season, result in self.counterfactual_results.items():
            records.append({
                'season': season,
                'reversal_rate': result.reversal_rate,
                'n_reversal_weeks': len(result.reversal_weeks),
                'reversal_weeks': str(result.reversal_weeks)
            })
        
        df = pd.DataFrame(records)
        df.to_csv(self.output_dir / 'counterfactual_results.csv', index=False)
    
    def run_feature_analysis(
        self, 
        seasons: Optional[List[int]] = None
    ) -> Dict:
        """Analyze feature impacts on outcomes"""
        analyzer = FeatureImpactAnalyzer()
        
        # Pro dancer analysis
        pro_analysis = analyzer.analyze_pro_dancer_effect(
            self.loader, self.inversion_results
        )
        
        self.log(f"  Pro dancer variance explained: {pro_analysis.variance_explained:.1%}")
        self.log(f"  Top dancers: {', '.join(pro_analysis.top_dancers[:3])}")
        
        # Characteristic analysis
        char_analysis = analyzer.analyze_celebrity_characteristics(
            self.loader, self.inversion_results
        )
        
        # Generate summary table
        summary_df = analyzer.generate_feature_impact_summary(
            self.loader, self.inversion_results
        )
        summary_df.to_csv(self.output_dir / 'feature_impacts.csv', index=False)
        
        return {
            'pro_dancer': pro_analysis,
            'characteristics': char_analysis,
            'summary': summary_df
        }
    
    def run_mechanism_evaluation(
        self, 
        seasons: Optional[List[int]] = None
    ) -> Dict:
        """Evaluate different voting mechanisms"""
        proposed = ProposedMechanism()
        
        evaluations = {}
        all_seasons = seasons or list(self.inversion_results.keys())
        
        # Evaluate each mechanism
        for mechanism_name in ['rank', 'percent', 'tiered']:
            if mechanism_name == 'rank':
                mechanism = proposed.rank
            elif mechanism_name == 'percent':
                mechanism = proposed.percent
            else:
                mechanism = proposed.tiered
            
            # Aggregate evaluations across seasons
            all_evals = []
            for season in all_seasons[:10]:  # Sample of seasons
                if season not in self.inversion_results:
                    continue
                
                context = self.active_manager.get_season_context(season)
                result = self.inversion_results[season]
                
                eval_result = proposed.evaluate_mechanism(mechanism, context, result)
                all_evals.append(eval_result)
            
            # Average evaluations
            if all_evals:
                from dwts_model.mechanism.tiered_threshold import MechanismEvaluation, MechanismType
                
                evaluations[mechanism_name] = MechanismEvaluation(
                    mechanism=mechanism.get_type(),
                    judge_alignment=sum(e.judge_alignment for e in all_evals) / len(all_evals),
                    fan_alignment=sum(e.fan_alignment for e in all_evals) / len(all_evals),
                    technical_floor=sum(e.technical_floor for e in all_evals) / len(all_evals),
                    close_calls_rate=sum(e.close_calls_rate for e in all_evals) / len(all_evals),
                    upsets_rate=sum(e.upsets_rate for e in all_evals) / len(all_evals),
                    volatility=sum(e.volatility for e in all_evals) / len(all_evals),
                    controversy_index=sum(e.controversy_index for e in all_evals) / len(all_evals)
                )
        
        return evaluations
    
    def generate_outputs(self, feature_analysis: Dict, mechanism_evaluation: Dict):
        """Generate all output files"""
        visualizer = DWTSVisualizer(str(self.output_dir / 'figures'))
        reporter = ReportGenerator()
        
        # Visualization data
        ghost_data = visualizer.prepare_ghost_data_plot(self.inversion_results)
        ghost_data.to_csv(self.output_dir / 'figures' / 'ghost_data.csv', index=False)
        
        inconsistency_data = visualizer.prepare_inconsistency_spectrum(self.inversion_results)
        inconsistency_data.to_csv(self.output_dir / 'figures' / 'inconsistency_spectrum.csv', index=False)
        
        reversal_data = visualizer.prepare_reversal_heatmap(self.counterfactual_results)
        reversal_data.to_csv(self.output_dir / 'figures' / 'reversal_heatmap.csv', index=False)
        
        # Generate memo
        memo = reporter.generate_mechanism_recommendation_memo(mechanism_evaluation)
        with open(self.output_dir / 'producer_memo.txt', 'w', encoding='utf-8') as f:
            f.write(memo)
        
        # Generate LaTeX tables
        latex_table = reporter.generate_model_summary_latex(self.inversion_results)
        with open(self.output_dir / 'results_table.tex', 'w', encoding='utf-8') as f:
            f.write(latex_table)
        
        # Save plot generation code
        plot_code = visualizer.generate_matplotlib_code('ghost_data')
        with open(self.output_dir / 'figures' / 'plot_ghost_data.py', 'w', encoding='utf-8') as f:
            f.write("import pandas as pd\ndf = pd.read_csv('ghost_data.csv')\n" + plot_code)
        
        self.log("  Generated: producer_memo.txt, results_table.tex, plot data files")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="DWTS Fan Vote Inversion Analysis"
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=None,
        help='Path to data CSV file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory'
    )
    parser.add_argument(
        '--seasons', '-s',
        type=str,
        default=None,
        help='Seasons to analyze (e.g., "1-10" or "1,5,10")'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
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
    
    # Run pipeline
    pipeline = DWTSAnalysisPipeline(
        data_path=args.data,
        output_dir=args.output,
        verbose=not args.quiet
    )
    
    results = pipeline.run_full_pipeline(seasons=seasons)
    
    return results


if __name__ == "__main__":
    main()

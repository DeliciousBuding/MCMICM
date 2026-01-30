#!/usr/bin/env python3
"""
Test Plan A: Tightening intervals for rank seasons
"""
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from run_mc_analysis import run_mc_robustness_analysis

print("=" * 70)
print("PLAN A EVALUATION: Interval Tightening for Rank-Rule Seasons")
print("=" * 70)
print()

# Test on rank seasons (S28-S34) with 1000 samples
print("Testing on rank-rule seasons: S28-S34")
print(f"Samples per elimination: 1000")
print()

run_mc_robustness_analysis(
    seasons=[28, 29, 30, 31, 32, 33, 34],
    n_samples=1000,
    output_file='test_plan_a_results.csv',
    use_regularization=True,
    tightening_factor=0.12
)

# Load and analyze results
df = pd.read_csv(Path('outputs') / 'test_plan_a_results.csv')

print("\n" + "=" * 70)
print("PLAN A RESULTS (Rank Seasons with 12% Tightening)")
print("=" * 70)
print(f"\nMean P(Wrongful): {df['p_wrongful'].mean():.4f}")
print(f"Median P(Wrongful): {df['p_wrongful'].median():.4f}")
print(f"Std Dev: {df['p_wrongful'].std():.4f}")
print(f"\nDegenerate samples (width > 0.95): {(df['fan_vote_upper'] - df['fan_vote_lower'] > 0.95).sum()}")
print(f"Percentage: {(df['fan_vote_upper'] - df['fan_vote_lower'] > 0.95).sum() / len(df) * 100:.1f}%")
print(f"\nClassification distribution:")
print(df['classification'].value_counts())

print("\nOK Test complete! Results saved to: outputs/test_plan_a_results.csv")

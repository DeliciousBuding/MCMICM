#!/usr/bin/env python3
"""
Run MC analysis in batches to handle memory/timeout issues
"""
import pandas as pd
from pathlib import Path
from run_mc_analysis import run_mc_robustness_analysis

print("=" * 80)
print("PLAN A: Full MC Analysis (Batch Mode)")
print("=" * 80)

# Divide into batches to avoid timeout
batches = [
    (list(range(1, 9)), "Batch 1: S1-S8"),
    (list(range(9, 17)), "Batch 2: S9-S16"),
    (list(range(17, 25)), "Batch 3: S17-S24"),
    (list(range(25, 35)), "Batch 4: S25-S34"),
]

all_results = []

for season_list, batch_name in batches:
    print(f"\n{batch_name}")
    print("-" * 80)
    
    df = run_mc_robustness_analysis(
        seasons=season_list,
        n_samples=5000,
        output_file=f'temp_batch_{season_list[0]}_{season_list[-1]}.csv',
        use_regularization=True,
        tightening_factor=0.12
    )
    
    if df is not None:
        all_results.append(df)
        print(f"OK Batch complete ({len(df)} eliminations)")
    else:
        print(f"⚠ Batch failed")

# Combine all batches
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(Path('outputs') / 'mc_robustness_results.csv', index=False)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS (Plan A - All 34 Seasons)")
    print("=" * 80)
    print(f"\nTotal eliminations: {len(final_df)}")
    print(f"Mean P(Wrongful): {final_df['p_wrongful'].mean():.4f}")
    print(f"Median P(Wrongful): {final_df['p_wrongful'].median():.4f}")
    print(f"Std Dev: {final_df['p_wrongful'].std():.4f}")
    
    # Count degenerate samples
    degenerate = (final_df['fan_vote_upper'] - final_df['fan_vote_lower'] > 0.95).sum()
    print(f"\nDegenerate samples (width > 0.95): {degenerate}/{len(final_df)} ({degenerate/len(final_df)*100:.1f}%)")
    
    print(f"\nClassification distribution:")
    print(final_df['classification'].value_counts())
    
    print("\nOK Full analysis complete!")
else:
    print("\n✗ All batches failed")

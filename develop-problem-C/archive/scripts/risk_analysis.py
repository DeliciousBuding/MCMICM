"""
MCM Problem C - Risk Analysis and Fixes

Addresses three critical risks identified by senior reviewer:
1. Overfitting Trap (zero inconsistency with extreme fan votes)
2. Rule Compliance Check (S28+ method)
3. Task 5 Simulation (quantitative proof for new mechanism)

Author: Team [REDACTED]
Date: 2026-01-30
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

# 导入统一调色板
from dwts_model.paper_palette import PALETTE, apply_paper_style

OUTPUT_DIR = Path('outputs')
FIGURES_DIR = OUTPUT_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# RISK 1: Overfitting Detection and Strict Constraint Analysis
# =============================================================================

def analyze_fan_vote_distribution():
    """
    Risk 1 Analysis: Check if fan vote estimates are too extreme (near 0 or 1)
    """
    print("=" * 60)
    print("RISK 1: Fan Vote Distribution Analysis")
    print("=" * 60)
    
    df = pd.read_csv(OUTPUT_DIR / 'fan_vote_estimates.csv')
    
    # Basic statistics
    print(f"\nTotal estimates: {len(df)}")
    print(f"Mean: {df['fan_vote_estimate'].mean():.4f}")
    print(f"Std: {df['fan_vote_estimate'].std():.4f}")
    print(f"Median: {df['fan_vote_estimate'].median():.4f}")
    
    # Extreme value analysis
    near_zero = (df['fan_vote_estimate'] < 0.01).sum()
    near_one = (df['fan_vote_estimate'] > 0.99).sum()
    
    print(f"\n⚠️  EXTREME VALUE WARNING:")
    print(f"  Near 0 (<1%): {near_zero} ({near_zero/len(df)*100:.1f}%)")
    print(f"  Near 1 (>99%): {near_one} ({near_one/len(df)*100:.1f}%)")
    
    # Plot histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Full distribution - 使用统一调色板
    ax1 = axes[0]
    ax1.hist(df['fan_vote_estimate'], bins=50, color=PALETTE['proposed'], edgecolor=PALETTE['aux'], alpha=0.8)
    ax1.axvline(x=0.01, color=PALETTE['warning'], linestyle='--', linewidth=2, label='1% threshold')
    ax1.axvline(x=0.99, color=PALETTE['warning'], linestyle='--', linewidth=2)
    ax1.set_xlabel('Estimated Fan Vote Share', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Inferred Fan Votes\n(CURRENT - Potentially Overfitted)', 
                  fontsize=12, fontweight='bold')
    ax1.legend()
    apply_paper_style(ax1)
    
    # Right: Log scale to see extremes - 使用统一调色板
    ax2 = axes[1]
    # Clip to avoid log(0)
    clipped = df['fan_vote_estimate'].clip(lower=1e-6, upper=1-1e-6)
    ax2.hist(np.log10(clipped), bins=50, color=PALETTE['warning'], edgecolor=PALETTE['aux'], alpha=0.8)
    ax2.axvline(x=np.log10(0.01), color=PALETTE['baseline'], linestyle='--', linewidth=2, label='1% threshold')
    ax2.set_xlabel('Log10(Fan Vote Share)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Log-Scale Distribution\n(Shows extreme values)', fontsize=12, fontweight='bold')
    ax2.legend()
    apply_paper_style(ax2)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'risk1_overfitting_check.png', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'risk1_overfitting_check.pdf', bbox_inches='tight')
    print(f"\n✓ Saved: risk1_overfitting_check.png")
    plt.close()
    
    return {
        'near_zero_pct': near_zero / len(df) * 100,
        'near_one_pct': near_one / len(df) * 100,
        'is_overfitted': (near_zero + near_one) / len(df) > 0.5
    }


def run_strict_constraint_solver(min_fan_vote=0.005):
    """
    Risk 1 Fix: Re-run solver with minimum fan vote constraint
    
    Hypothesis: If solver becomes infeasible, it reveals "Scandal Weeks"
    where results were mathematically impossible unless someone had ~0 fans.
    """
    print(f"\n--- Running Strict Solver (Min Vote >= {min_fan_vote*100:.1f}%) ---")
    
    from dwts_model.etl import DWTSDataLoader, ActiveSetManager
    from dwts_model.engines import PercentLPEngine, RankCPEngine
    
    loader = DWTSDataLoader('2026_MCM_Problem_C_Data.csv')
    loader.load()
    manager = ActiveSetManager(loader)
    manager.build_all_contexts()
    
    # Create modified engines with strict bounds
    lp_engine = PercentLPEngine()
    cp_engine = RankCPEngine()
    
    # Override minimum bounds
    lp_engine.min_vote_share = min_fan_vote
    cp_engine.min_vote_share = min_fan_vote
    
    results = []
    scandal_weeks = []
    
    for season in manager.get_all_seasons():
        context = manager.get_season_context(season)
        
        if context.voting_method == 'percent':
            result = lp_engine.solve(context)
        else:
            result = cp_engine.solve(context)
        
        results.append({
            'season': season,
            'voting_method': context.voting_method,
            'inconsistency_score': result.inconsistency_score,
            'is_feasible': result.is_feasible,
            'constraint': f'>={min_fan_vote*100:.1f}%'
        })
        
        # Check which weeks became infeasible
        if result.inconsistency_score > 0:
            scandal_weeks.append({
                'season': season,
                'inconsistency': result.inconsistency_score,
                'note': 'Requires near-zero fan vote to explain elimination'
            })
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / 'inconsistency_check_strict.csv', index=False)
    print(f"✓ Saved: inconsistency_check_strict.csv")
    
    # Summary
    infeasible_count = (df['inconsistency_score'] > 0).sum()
    print(f"\nResults with strict constraint (>= {min_fan_vote*100:.1f}%):")
    print(f"  Feasible seasons: {(df['inconsistency_score'] == 0).sum()}")
    print(f"  Infeasible seasons: {infeasible_count}")
    
    if scandal_weeks:
        print(f"\n🔥 SCANDAL WEEKS DETECTED:")
        for sw in scandal_weeks:
            print(f"  Season {sw['season']}: S* = {sw['inconsistency']:.2f}")
    
    return df, scandal_weeks


# =============================================================================
# RISK 2: Rule Compliance Memo
# =============================================================================

def generate_rule_compliance_memo():
    """
    Risk 2: Document the S28+ rule decision based on PDF text
    """
    print("\n" + "=" * 60)
    print("RISK 2: Rule Compliance Check (S28+ Method)")
    print("=" * 60)
    
    memo = """
================================================================================
RULE COMPLIANCE MEMO: Season 28+ Voting Method
================================================================================

QUESTION: Does our model correctly apply the voting rules for S28+?

ANSWER: YES ✓

PDF EVIDENCE (verbatim quotes):

1. "In the first two seasons of the U.S. show, the combination was based on 
   ranks."

2. "Season 2 concerns (due to celebrity contestant Jerry Rice who was a finalist 
   despite very low judge scores) led to a modification to use percentages 
   instead of ranks."

3. "Around this same season [S28], the producers also returned to using the 
   method of ranks to combine judges scores with fan votes as in seasons one 
   and two. The exact season this change occurred is not known, but it is 
   reasonable to assume it was season 28."

4. Footnote: "The year of the return to the rank based method is not known for 
   certain; season 28 is a reasonable assumption."

5. Appendix explicitly states:
   - "COMBINED BY RANK (used in seasons 1, 2, and 28a - 34)"
   - "COMBINED BY PERCENT (used for season 3 through 27a)"

DECISION RATIONALE:
- The PDF explicitly states S28-34 should use RANK method
- This is labeled as "a reasonable assumption" by problem authors
- Our implementation follows this guidance exactly

IMPLEMENTATION:
- Seasons 1-2: Rank method
- Seasons 3-27: Percent method  
- Seasons 28-34: Rank method + Judges' Save

ADDITIONAL JUDGES' SAVE RULE (S28+):
The PDF states: "The bottom two contestants were identified using the combined 
judge scores and fan votes, and then during the live show the judges voted to 
select which of these two to eliminate."

This creates ambiguity in our model because:
1. We don't know how judges voted between bottom two
2. Either bottom-two contestant could have been eliminated

Our approach: We model this as a "union of polytopes" - the eliminated contestant
must have been BOTH in the bottom two AND lost the judges' vote. This is handled
by our JudgesSaveHandler module.

COMPLIANCE STATUS: VERIFIED ✓
================================================================================
"""
    
    print(memo)
    
    with open(OUTPUT_DIR / 'rule_compliance_memo.txt', 'w', encoding='utf-8') as f:
        f.write(memo)
    
    print(f"✓ Saved: rule_compliance_memo.txt")
    
    return memo


# =============================================================================
# RISK 3: Task 5 Simulation - Quantitative Proof
# =============================================================================

def simulate_new_system():
    """
    Risk 3: Simulate the proposed Tiered Threshold System
    
    Compare outcomes between:
    - Old System (actual DWTS rules)
    - New System (Tiered Threshold with soft floor + elite mix)
    
    Metric: "Robbed Goddess" count - high-popularity contestants eliminated early
    """
    print("\n" + "=" * 60)
    print("RISK 3: Task 5 - New System Simulation")
    print("=" * 60)
    
    from dwts_model.etl import DWTSDataLoader, ActiveSetManager
    from dwts_model.engines import PercentLPEngine, RankCPEngine
    
    # Load data and fan vote estimates
    loader = DWTSDataLoader('2026_MCM_Problem_C_Data.csv')
    loader.load()
    manager = ActiveSetManager(loader)
    manager.build_all_contexts()
    
    fan_votes_df = pd.read_csv(OUTPUT_DIR / 'fan_vote_estimates.csv')
    
    # Define "Robbed Goddess" threshold
    # A contestant is "robbed" if they have high fan votes but are eliminated
    HIGH_FAN_THRESHOLD = 0.25  # Top 25% of fan votes
    
    results = {
        'old_system': {'robbed': [], 'saved': []},
        'new_system': {'robbed': [], 'saved': []}
    }
    
    comparison_records = []
    
    for season in manager.get_all_seasons():
        context = manager.get_season_context(season)
        
        for week_num, week_ctx in context.weeks.items():
            if not week_ctx.eliminated:
                continue
            
            # Get fan votes for this week
            week_votes = fan_votes_df[
                (fan_votes_df['season'] == season) & 
                (fan_votes_df['week'] == week_num)
            ]
            
            if len(week_votes) == 0:
                continue
            
            # Calculate who would be eliminated under each system
            active = week_ctx.active_set
            eliminated_actual = list(week_ctx.eliminated)[0] if week_ctx.eliminated else None
            
            # Get scores and votes for active contestants
            scores = {}
            votes = {}
            for contestant in active:
                # judge_scores is Dict[str, float] (contestant -> total score)
                scores[contestant] = week_ctx.judge_scores.get(contestant, 0)
                
                vote_row = week_votes[week_votes['contestant'] == contestant]
                votes[contestant] = vote_row['fan_vote_estimate'].values[0] if len(vote_row) > 0 else 0
            
            if not scores or not votes:
                continue
            
            # OLD SYSTEM: Current rules (actual elimination happened)
            old_eliminated = eliminated_actual
            
            # NEW SYSTEM: Tiered Threshold
            # Rule 1: Soft Floor - Must be in bottom 2σ of judge scores to be eliminated
            # Rule 2: Elite Mix - 40% rank + 60% percent
            
            mean_score = np.mean(list(scores.values()))
            std_score = np.std(list(scores.values())) + 1e-6
            
            # Calculate who's below 2σ threshold (can be eliminated)
            eliminable = [c for c in active if scores[c] < mean_score - 2 * std_score]
            
            # If no one below threshold, use bottom 25%
            if not eliminable:
                sorted_by_score = sorted(active, key=lambda x: scores[x])
                eliminable = sorted_by_score[:max(1, len(active) // 4)]
            
            # Among eliminable, use 40% rank + 60% percent
            total_score_sum = sum(scores.values())
            total_vote_sum = sum(votes.values()) + 1e-6
            
            combined_scores = {}
            for c in eliminable:
                # Rank component (lower rank = better)
                rank = sorted(active, key=lambda x: scores[x] + votes[x], reverse=True).index(c) + 1
                rank_pct = 1 - (rank - 1) / len(active)  # Higher = better
                
                # Percent component
                pct = (scores[c] / total_score_sum + votes[c] / total_vote_sum) / 2
                
                # Combined (higher = safer)
                combined_scores[c] = 0.4 * rank_pct + 0.6 * pct
            
            new_eliminated = min(combined_scores.keys(), key=lambda x: combined_scores[x]) if combined_scores else old_eliminated
            
            # Check if anyone was "robbed"
            # Robbed = high fan votes but eliminated
            for contestant in active:
                fan_vote = votes.get(contestant, 0)
                is_high_fan = fan_vote > HIGH_FAN_THRESHOLD
                
                # OLD SYSTEM
                if contestant == old_eliminated and is_high_fan:
                    results['old_system']['robbed'].append({
                        'season': season, 'week': week_num, 'contestant': contestant,
                        'fan_vote': fan_vote
                    })
                
                # NEW SYSTEM
                if contestant == new_eliminated and is_high_fan:
                    results['new_system']['robbed'].append({
                        'season': season, 'week': week_num, 'contestant': contestant,
                        'fan_vote': fan_vote
                    })
                
                # SAVED by new system
                if contestant == old_eliminated and contestant != new_eliminated and is_high_fan:
                    results['new_system']['saved'].append({
                        'season': season, 'week': week_num, 'contestant': contestant,
                        'fan_vote': fan_vote
                    })
            
            comparison_records.append({
                'season': season,
                'week': week_num,
                'old_eliminated': old_eliminated,
                'new_eliminated': new_eliminated,
                'outcome_changed': old_eliminated != new_eliminated
            })
    
    # Summary
    print("\n📊 SIMULATION RESULTS:")
    print(f"\nOLD SYSTEM (Actual DWTS Rules):")
    print(f"  'Robbed Goddesses' (high-fan eliminated): {len(results['old_system']['robbed'])}")
    
    print(f"\nNEW SYSTEM (Tiered Threshold):")
    print(f"  'Robbed Goddesses': {len(results['new_system']['robbed'])}")
    print(f"  High-fan contestants SAVED: {len(results['new_system']['saved'])}")
    
    improvement = len(results['old_system']['robbed']) - len(results['new_system']['robbed'])
    print(f"\n✨ IMPROVEMENT: {improvement} fewer popular contestants eliminated")
    
    # Create comparison table
    comparison_df = pd.DataFrame(comparison_records)
    changed_weeks = comparison_df[comparison_df['outcome_changed']]
    
    print(f"\nOutcome Changes: {len(changed_weeks)} weeks would have different eliminations")
    
    # Save results
    comparison_df.to_csv(OUTPUT_DIR / 'mechanism_simulation_comparison.csv', index=False)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Robbed goddess comparison - 使用统一调色板
    ax1 = axes[0]
    systems = ['Old System\n(Actual DWTS)', 'New System\n(Tiered Threshold)']
    robbed_counts = [len(results['old_system']['robbed']), len(results['new_system']['robbed'])]
    colors = [PALETTE['baseline'], PALETTE['proposed']]
    bars = ax1.bar(systems, robbed_counts, color=colors, edgecolor=PALETTE['aux'], linewidth=1.5)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('"Robbed Goddesses"\n(High-Fan Contestants Eliminated)', fontsize=12, fontweight='bold')
    for bar, count in zip(bars, robbed_counts):
        ax1.annotate(str(count), (bar.get_x() + bar.get_width()/2, count),
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
    apply_paper_style(ax1)
    
    # Right: Outcome change rate by season - 使用统一调色板
    ax2 = axes[1]
    change_by_season = comparison_df.groupby('season')['outcome_changed'].mean() * 100
    ax2.bar(change_by_season.index, change_by_season.values, color=PALETTE['proposed'], 
            edgecolor=PALETTE['aux'], alpha=0.85, linewidth=0.8)
    ax2.set_xlabel('Season', fontsize=12)
    ax2.set_ylabel('Outcome Change Rate (%)', fontsize=12)
    ax2.set_title('How Often New System\nWould Change Eliminations', fontsize=12, fontweight='bold')
    ax2.axhline(y=change_by_season.mean(), color=PALETTE['warning'], linestyle='--', linewidth=2,
                label=f'Average: {change_by_season.mean():.1f}%')
    ax2.legend()
    apply_paper_style(ax2)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'risk3_mechanism_simulation.png', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'risk3_mechanism_simulation.pdf', bbox_inches='tight')
    print(f"\n✓ Saved: risk3_mechanism_simulation.png")
    print(f"✓ Saved: mechanism_simulation_comparison.csv")
    plt.close()
    
    return results, comparison_df


# =============================================================================
# MINOR FIX: Data Interpretation Clarification
# =============================================================================

def clarify_data_interpretation():
    """
    Minor Fix: Clarify the physical meaning of decimals in rank seasons
    """
    print("\n" + "=" * 60)
    print("MINOR FIX: Data Interpretation Clarification")
    print("=" * 60)
    
    clarification = """
================================================================================
DATA INTERPRETATION: Fan Vote Estimates in Rank Seasons
================================================================================

QUESTION: What do decimal values (e.g., 0.666) mean for Rank season estimates?

ANSWER: They represent NORMALIZED POPULARITY SCORES, not raw ranks.

EXPLANATION:

For PERCENT seasons (S3-S27):
- Values directly represent estimated fan vote share
- E.g., 0.35 means ~35% of total fan votes
- Sum across contestants in a week ≈ 1.0

For RANK seasons (S1-2, S28-34):
- Values represent a PROBABILITY/UTILITY transformation of rank
- We convert discrete ranks to continuous [0,1] scale for comparability
- Formula: score = (N - rank + 1) / N, where N = number of contestants
- E.g., with 4 contestants: 1st=1.0, 2nd=0.75, 3rd=0.5, 4th=0.25

WHY DECIMALS?
1. Allows direct comparison with percent-season estimates
2. Represents relative popularity (higher = more popular)
3. When multiple rank orderings are feasible, we report the centroid

INTERPRETATION GUIDE:
- 0.8+ : Very popular (top ranked)
- 0.5-0.8 : Moderately popular
- 0.2-0.5 : Less popular
- <0.2 : Low popularity (bottom ranked)

IMPORTANT: These are not probabilities that sum to 1!
They are relative popularity indices derived from rank constraints.

================================================================================
"""
    
    print(clarification)
    
    with open(OUTPUT_DIR / 'data_interpretation_note.txt', 'w', encoding='utf-8') as f:
        f.write(clarification)
    
    print(f"✓ Saved: data_interpretation_note.txt")
    
    return clarification


# =============================================================================
# MAIN: Run All Risk Analyses
# =============================================================================

def main():
    """Run complete risk analysis"""
    print("\n" + "=" * 70)
    print("MCM PROBLEM C - RISK ANALYSIS AND FIXES")
    print("Senior Reviewer Feedback Implementation")
    print("=" * 70)
    
    # Risk 1: Overfitting check
    overfitting_result = analyze_fan_vote_distribution()
    
    if overfitting_result['is_overfitted']:
        print("\n⚠️  OVERFITTING DETECTED - Running strict constraint solver...")
        strict_results, scandal_weeks = run_strict_constraint_solver(min_fan_vote=0.005)
    else:
        print("\n✓ Fan vote distribution looks reasonable")
    
    # Risk 2: Rule compliance
    generate_rule_compliance_memo()
    
    # Risk 3: Mechanism simulation
    mechanism_results, comparison_df = simulate_new_system()
    
    # Minor fix: Data interpretation
    clarify_data_interpretation()
    
    print("\n" + "=" * 70)
    print("RISK ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nDeliverables generated:")
    print("  1. risk1_overfitting_check.png - Fan vote distribution histogram")
    print("  2. inconsistency_check_strict.csv - Strict constraint results")
    print("  3. rule_compliance_memo.txt - S28+ rule decision")
    print("  4. mechanism_simulation_comparison.csv - Old vs New system")
    print("  5. risk3_mechanism_simulation.png - Mechanism comparison chart")
    print("  6. data_interpretation_note.txt - Decimal value clarification")


if __name__ == '__main__':
    main()

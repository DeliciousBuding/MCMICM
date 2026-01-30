"""
Report Generator for MCM Paper

Generates structured output for paper writing:
- Model results summaries
- Table data
- Memo drafts
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime


class ReportGenerator:
    """
    Generate reports and summaries for MCM paper.
    """
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    def generate_model_summary_latex(
        self,
        inversion_results: Dict[int, Any],
        method: str = 'percent'
    ) -> str:
        """
        Generate LaTeX table summarizing model results by season.
        """
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Fan Vote Inversion Model Results by Season}",
            r"\label{tab:inversion_results}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Season & Method & Inconsistency ($S^*$) & Avg Certainty & Feasible \\",
            r"\midrule"
        ]
        
        for season in sorted(inversion_results.keys()):
            result = inversion_results[season]
            
            # Calculate average certainty
            certainties = []
            for week_est in result.week_results.values():
                for est in week_est.values():
                    certainties.append(est.certainty)
            avg_cert = np.mean(certainties) if certainties else 0
            
            feasible = r"\checkmark" if result.is_feasible else r"$\times$"
            
            lines.append(
                f"{season} & {result.method} & {result.inconsistency_score:.3f} & "
                f"{avg_cert:.1%} & {feasible} \\\\"
            )
        
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}"
        ])
        
        return "\n".join(lines)
    
    def generate_controversy_analysis_latex(
        self,
        case_analyses: Dict[str, Dict]
    ) -> str:
        """
        Generate LaTeX section for controversy case studies.
        """
        lines = [
            r"\subsection{Controversy Case Studies}",
            ""
        ]
        
        for contestant, analysis in case_analyses.items():
            lines.extend([
                r"\subsubsection{" + contestant.replace("_", " ") + "}",
                "",
                f"Final placement: {analysis.get('expected_placement', 'N/A')}",
                "",
                f"Weeks with lowest judge score: {analysis.get('worst_judge_score_weeks', 'N/A')}",
                "",
                analysis.get('summary', ''),
                ""
            ])
        
        return "\n".join(lines)
    
    def generate_mechanism_recommendation_memo(
        self,
        evaluations: Dict[str, Any]
    ) -> str:
        """
        Generate 1-2 page memo for show producers.
        """
        memo = f"""
MEMORANDUM

TO: Dancing with the Stars Producers
FROM: Mathematical Modeling Team
DATE: {self.timestamp}
RE: Voting System Analysis and Recommendations

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXECUTIVE SUMMARY

We analyzed 34 seasons of DWTS data to understand how different voting 
systems affect competition outcomes. Our key findings:

1. The PERCENT method (S3-S27) tends to favor popular contestants, 
   potentially at the expense of technical quality.

2. The RANK method (S1-2, S28+) provides some protection for technically 
   skilled but less popular contestants.

3. The JUDGES' SAVE rule (S28+) introduces mathematical ambiguity that 
   makes fan vote estimation less certain.

4. We propose a new TIERED THRESHOLD SYSTEM that balances technical 
   merit with popular appeal.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY FINDINGS

Method Comparison:
"""
        
        for name, eval in evaluations.items():
            memo += f"""
{name.upper()} METHOD:
  • Judge Score Alignment: {eval.judge_alignment:.0%}
  • Fan Vote Alignment: {eval.fan_alignment:.0%}  
  • Close Call Rate: {eval.close_calls_rate:.0%}
  • Controversy Potential: {eval.controversy_index:.0%}
"""
        
        memo += """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RECOMMENDATION: TIERED THRESHOLD SYSTEM

We recommend adopting a new voting system with three components:

1. SOFT FLOOR (Technical Quality Protection)
   If a contestant's judge score falls below the week's average by more 
   than 2 standard deviations, their fan vote weight is reduced by 50%.
   
   Rationale: Prevents contestants with poor technical skills from 
   advancing purely on popularity.

2. ELITE MIX (Balanced Combination)
   Above the floor, combine scores using 40% rank-based + 60% percent-based.
   
   Rationale: Rank method prevents vote monopolization; percent method 
   rewards exceptional performances.

3. SIGNAL RELEASE (Strategic Excitement)
   Announce "Safe Zone" (top 3) and "Danger Zone" (bottom 2) during live 
   show, without revealing exact vote counts.
   
   Rationale: Creates strategic tension and encourages fan engagement 
   without compromising vote integrity.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXPECTED OUTCOMES

Implementing this system would:

✓ Reduce controversies like Bobby Bones (S27) where low-skill contestants 
  win despite consistently poor judge scores

✓ Maintain fan engagement by ensuring popular contestants remain competitive

✓ Create more exciting finales by keeping close calls in the competition

✓ Provide clearer narratives for the show ("Will X escape the danger zone?")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONCLUSION

The current voting system creates inherent tensions between technical 
merit and popular appeal. Our proposed Tiered Threshold System addresses 
these tensions while maintaining the excitement that makes DWTS compelling.

We are happy to discuss these findings in more detail at your convenience.

Respectfully submitted,
Mathematical Modeling Team
"""
        
        return memo
    
    def generate_results_json(
        self,
        inversion_results: Dict[int, Any],
        counterfactual_results: Dict[int, Any],
        feature_analysis: Dict[str, Any]
    ) -> Dict:
        """
        Generate comprehensive JSON results for archiving.
        """
        return {
            'metadata': {
                'generated': self.timestamp,
                'model_version': '1.0.0',
                'n_seasons': len(inversion_results)
            },
            'inversion_summary': {
                season: {
                    'method': r.method,
                    'inconsistency': r.inconsistency_score,
                    'feasible': r.is_feasible,
                    'n_weeks': len(r.week_results)
                }
                for season, r in inversion_results.items()
            },
            'counterfactual_summary': {
                season: {
                    'reversal_rate': r.reversal_rate,
                    'n_reversals': len(r.reversal_weeks)
                }
                for season, r in counterfactual_results.items()
            },
            'feature_analysis': feature_analysis
        }
    
    def generate_abstract(self) -> str:
        """
        Generate draft abstract for MCM paper.
        """
        return """
\\begin{abstract}

Dancing with the Stars (DWTS) combines expert judge scores with fan votes 
to determine weekly eliminations. We develop a mathematical framework to 
reverse-engineer fan voting patterns from observed elimination outcomes 
using constrained optimization. Our two-phase linear programming approach 
for percent-based seasons and constraint programming for rank-based seasons 
successfully reconstructs fan vote distributions with quantified uncertainty.

We analyze the impact of rule changes across 34 seasons, finding that the 
percent-based method (Seasons 3-27) favors popular contestants over 
technically skilled ones, while the rank-based method provides more 
balanced outcomes. The judges' save rule (Season 28+) introduces 
mathematical ambiguity that reduces estimation certainty.

Using survival analysis, we quantify the impact of pro dancer assignment, 
celebrity age, and industry on elimination risk. Pro dancer choice explains 
approximately X\\% of outcome variance, suggesting strategic pairing matters.

We propose a new Tiered Threshold System that maintains a "professional 
floor" while preserving fan influence, combining the benefits of both 
existing methods. Our counterfactual analysis demonstrates this system 
would reduce controversial outcomes while maintaining viewer engagement.

\\textbf{Keywords:} voting systems, constrained optimization, survival analysis, 
mechanism design, sports analytics

\\end{abstract}
"""

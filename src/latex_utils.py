"""
Utility functions to print LaTeX tables for paper results
Prints formatted LaTeX tables directly to terminal for easy copy-paste into paper
"""
import numpy as np
from scipy import stats


def print_sample_efficiency_table(results_dict):
    """
    Print LaTeX table for sample efficiency analysis
    
    results_dict format:
    {
        'CartPole-v1': {
            'pure_rl_episodes': [250, 245, 260, ...],  # episodes to threshold per seed
            'bc_rl_episodes': [85, 90, 80, ...]
        },
        'LunarLander-v2': {...}
    }
    """
    print("\n" + "="*80)
    print("SAMPLE EFFICIENCY TABLE (LaTeX)")
    print("="*80)
    
    print("""
\\begin{table}[h]
\\centering
\\caption{Episodes to Reach Target Performance}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Environment} & \\textbf{Pure RL} & \\textbf{BC→RL} & \\textbf{Speedup} \\\\
\\midrule""")
    
    for env_name, data in results_dict.items():
        pure_episodes = np.array(data['pure_rl_episodes'])
        bc_episodes = np.array(data['bc_rl_episodes'])
        
        pure_mean = np.mean(pure_episodes)
        pure_std = np.std(pure_episodes)
        bc_mean = np.mean(bc_episodes)
        bc_std = np.std(bc_episodes)
        
        speedup = pure_mean / bc_mean if bc_mean > 0 else 0
        
        print(f"{env_name} & {pure_mean:.0f} $\\pm$ {pure_std:.0f} & {bc_mean:.0f} $\\pm$ {bc_std:.0f} & {speedup:.2f}$\\times$ \\\\")
    
    print("""\\bottomrule
\\end{tabular}
\\label{tab:sample_efficiency}
\\end{table}
""")


def print_statistical_tests(results_dict):
    """Print statistical test results"""
    print("\n" + "="*80)
    print("STATISTICAL TESTS")
    print("="*80)
    print("""
\\textbf{Statistical Tests:}
\\begin{itemize}""")
    
    for env_name, data in results_dict.items():
        pure_episodes = np.array(data['pure_rl_episodes'])
        bc_episodes = np.array(data['bc_rl_episodes'])
        
        # T-test
        t_stat, p_value = stats.ttest_ind(pure_episodes, bc_episodes)
        
        # Cohen's d (effect size)
        pooled_std = np.sqrt((np.std(pure_episodes)**2 + np.std(bc_episodes)**2) / 2)
        cohens_d = (np.mean(pure_episodes) - np.mean(bc_episodes)) / pooled_std
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect = "negligible"
        elif abs(cohens_d) < 0.5:
            effect = "small"
        elif abs(cohens_d) < 0.8:
            effect = "medium"
        else:
            effect = "large"
        
        sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        
        print(f"    \\item {env_name}: BC→RL vs Pure RL, $t$={t_stat:.2f}, $p$={p_value:.4f} ({sig_marker})")
        print(f"    \\item Effect size (Cohen's $d$): {cohens_d:.2f} ({effect} effect)")
    
    print("""\\end{itemize}
""")


def print_final_performance_table(results_dict):
    """
    Print LaTeX table for final performance comparison
    
    results_dict format:
    {
        'CartPole-v1': {
            'pure_rl_rewards': [450, 445, 460, ...],  # final rewards per seed
            'bc_only_rewards': [485, 490, 480, ...],
            'bc_rl_rewards': [498, 500, 497, ...]
        },
        'LunarLander-v2': {...}
    }
    """
    print("\n" + "="*80)
    print("FINAL PERFORMANCE TABLE (LaTeX)")
    print("="*80)
    
    # Calculate success rates
    threshold = {'CartPole-v1': 195, 'LunarLander-v2': 200, 'LunarLander-v3': 200}
    
    print("""
\\begin{table}[h]
\\centering
\\caption{Asymptotic Performance (Final 100 Episodes)}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Method} & \\textbf{LunarLander} & \\textbf{CartPole} & \\textbf{Avg Success Rate} \\\\
\\midrule""")
    
    # Organize by method
    methods = ['Pure RL', 'BC-Only', 'BC→RL']
    method_keys = ['pure_rl_rewards', 'bc_only_rewards', 'bc_rl_rewards']
    
    for method, key in zip(methods, method_keys):
        row_data = []
        success_rates = []
        
        for env in ['LunarLander-v2', 'CartPole-v1']:
            if env in results_dict and key in results_dict[env]:
                rewards = np.array(results_dict[env][key])
                mean_reward = np.mean(rewards)
                std_reward = np.std(rewards)
                
                # Calculate success rate
                thresh = threshold.get(env, 195)
                success_rate = np.mean(rewards >= thresh) * 100
                success_rates.append(success_rate)
                
                row_data.append(f"{mean_reward:.1f} $\\pm$ {std_reward:.1f}")
            else:
                row_data.append("N/A")
        
        avg_success = np.mean(success_rates) if success_rates else 0
        
        # Format method name
        method_latex = method.replace('→', '$\\rightarrow$')
        
        if len(row_data) >= 2:
            print(f"{method_latex} & {row_data[0]} & {row_data[1]} & {avg_success:.1f}\\% \\\\")
        else:
            print(f"{method_latex} & N/A & N/A & {avg_success:.1f}\\% \\\\")
    
    print("""\\bottomrule
\\end{tabular}
\\label{tab:final_performance}
\\end{table}
""")


def print_all_latex_tables(sample_efficiency_data, final_performance_data):
    """
    Print all LaTeX tables in one go
    
    Usage:
        sample_efficiency_data = {
            'CartPole-v1': {
                'pure_rl_episodes': [250, 245, 260],
                'bc_rl_episodes': [85, 90, 80]
            }
        }
        
        final_performance_data = {
            'CartPole-v1': {
                'pure_rl_rewards': [450, 445, 460],
                'bc_only_rewards': [485, 490, 480],
                'bc_rl_rewards': [498, 500, 497]
            }
        }
        
        print_all_latex_tables(sample_efficiency_data, final_performance_data)
    """
    print("\n\n" + "="*80)
    print("LATEX TABLES FOR PAPER")
    print("Copy-paste these directly into your LaTeX document")
    print("="*80)
    
    print_sample_efficiency_table(sample_efficiency_data)
    print_statistical_tests(sample_efficiency_data)
    print_final_performance_table(final_performance_data)
    
    print("\n" + "="*80)
    print("END OF LATEX TABLES")
    print("="*80 + "\n")


def print_quick_summary(env_name, pure_rl_rewards, bc_rl_rewards, bc_only_rewards=None):
    """
    Quick summary for a single run (not multi-seed)
    Prints simple LaTeX-ready numbers
    """
    print("\n" + "="*60)
    print(f"QUICK LATEX SUMMARY - {env_name}")
    print("="*60)
    
    threshold = 195 if env_name == 'CartPole-v1' else 200
    
    # Sample efficiency (episodes to threshold)
    pure_episodes = next((i for i, r in enumerate(pure_rl_rewards) if r >= threshold), len(pure_rl_rewards))
    bc_episodes = next((i for i, r in enumerate(bc_rl_rewards) if r >= threshold), len(bc_rl_rewards))
    speedup = pure_episodes / bc_episodes if bc_episodes > 0 else 0
    
    # Final performance (last 100 episodes)
    pure_final = np.mean(pure_rl_rewards[-100:])
    pure_std = np.std(pure_rl_rewards[-100:])
    bc_final = np.mean(bc_rl_rewards[-100:])
    bc_std = np.std(bc_rl_rewards[-100:])
    
    print(f"\nSample Efficiency:")
    print(f"  Pure RL: {pure_episodes} episodes to threshold")
    print(f"  BC→RL: {bc_episodes} episodes to threshold")
    print(f"  Speedup: {speedup:.2f}×")
    print(f"  LaTeX: {pure_episodes} & {bc_episodes} & {speedup:.2f}$\\times$ \\\\")
    
    print(f"\nFinal Performance (last 100 episodes):")
    print(f"  Pure RL: {pure_final:.1f} ± {pure_std:.1f}")
    print(f"  BC→RL: {bc_final:.1f} ± {bc_std:.1f}")
    if bc_only_rewards:
        bc_only_final = np.mean(bc_only_rewards)
        bc_only_std = np.std(bc_only_rewards)
        print(f"  BC-only: {bc_only_final:.1f} ± {bc_only_std:.1f}")
    print(f"  LaTeX: {pure_final:.1f} $\\pm$ {pure_std:.1f} & {bc_final:.1f} $\\pm$ {bc_std:.1f} \\\\")
    
    # Success rate
    pure_success = np.mean(np.array(pure_rl_rewards[-100:]) >= threshold) * 100
    bc_success = np.mean(np.array(bc_rl_rewards[-100:]) >= threshold) * 100
    
    print(f"\nSuccess Rate (last 100 episodes):")
    print(f"  Pure RL: {pure_success:.1f}%")
    print(f"  BC→RL: {bc_success:.1f}%")
    
    print("="*60 + "\n")


# Example usage printed to terminal:
if __name__ == "__main__":
    print("""
USAGE EXAMPLE:
    
# In your train_rl.py, at the end of compare_methods():

from src.latex_utils import print_quick_summary

def compare_methods(env_name='CartPole-v1'):
    # ... existing training code ...
    
    # At the end, print LaTeX summary:
    print_quick_summary(env_name, pure_rl_rewards, bc_rl_rewards, bc_only_rewards)

# For multi-seed analysis:

from src.latex_utils import print_all_latex_tables

sample_data = {
    'CartPole-v1': {
        'pure_rl_episodes': [250, 245, 260, 255, 248],
        'bc_rl_episodes': [85, 90, 80, 88, 83]
    }
}

final_data = {
    'CartPole-v1': {
        'pure_rl_rewards': [450, 445, 460, 448, 452],
        'bc_only_rewards': [485, 490, 480, 488, 487],
        'bc_rl_rewards': [498, 500, 497, 499, 500]
    }
}

print_all_latex_tables(sample_data, final_data)
    """)


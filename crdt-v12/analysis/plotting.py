"""
Publication-Quality Plotting Tools.

Uses Matplotlib and Seaborn to generate academic-style graphs
for IEEE paper submission.  Includes non-parametric significance
testing (Mann–Whitney U, Kruskal–Wallis H) with results exported
to CSV for supplementary material.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from itertools import combinations

from scipy.stats import mannwhitneyu, kruskal


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
def setup_style():
    """Configure matplotlib for academic IEEE style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300,
        'lines.linewidth': 2,
    })


# ---------------------------------------------------------------------------
# Confidence-interval helper
# ---------------------------------------------------------------------------
def _ci95(series: pd.Series) -> float:
    """Return the half-width of a 95 % bootstrap-free CI (t-based)."""
    n = len(series)
    if n < 2:
        return 0.0
    from scipy.stats import t as t_dist
    se = series.std(ddof=1) / np.sqrt(n)
    return float(t_dist.ppf(0.975, df=n - 1) * se)


# ---------------------------------------------------------------------------
# Bar plot with CI annotations
# ---------------------------------------------------------------------------
def plot_bar_with_errors(df: pd.DataFrame, metric: str, y_label: str,
                         title: str, filename: str):
    """Bar plot with 95 % CI error bars and numeric CI annotations."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=df,
        x='scenario',
        y=metric,
        hue='forwarding',
        capsize=.1,
        edgecolor=".2",
        ax=ax,
    )

    # Annotate each bar with its mean ± CI
    grouped = df.groupby(['scenario', 'forwarding'])[metric]
    summary = grouped.agg(['mean', 'count', 'std']).reset_index()
    summary['ci95'] = summary.apply(
        lambda r: _ci95(
            df.loc[
                (df['scenario'] == r['scenario']) &
                (df['forwarding'] == r['forwarding']),
                metric
            ]
        ),
        axis=1,
    )

    for container in ax.containers:
        # Seaborn BarContainer – iterate patches
        if hasattr(container, 'datavalues'):
            for bar_val, patch in zip(container.datavalues, container.patches):
                x = patch.get_x() + patch.get_width() / 2
                y = patch.get_height()
                if y > 0:
                    ax.text(x, y, f'{y:.2f}', ha='center', va='bottom',
                            fontsize=8, color='0.3')

    ax.set_ylabel(y_label)
    ax.set_xlabel('Scenario')
    ax.set_title(title)
    ax.legend(title='Protocol')
    fig.tight_layout()

    os.makedirs('plots', exist_ok=True)
    fig.savefig(f"plots/{filename}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Statistical significance tests
# ---------------------------------------------------------------------------
_METRICS = ['pdr', 'latency_median', 'energy_mean', 'total_overhead']
_PAIRS = [('flood', 'heuristic'), ('flood', 'ai'), ('heuristic', 'ai'), ('aodv', 'ai')]


def run_significance_tests(df: pd.DataFrame,
                           output_csv: str = 'stats_results.csv'):
    """Run non-parametric significance tests across forwarding modes.

    For every metric in *_METRICS*:
      - Pairwise Mann–Whitney U tests for each pair in *_PAIRS*.
      - Kruskal–Wallis H test across all four forwarding modes.

    Results are printed in a formatted table and written to *output_csv*.
    """
    rows: list[dict] = []

    header = (
        f"\n{'='*78}\n"
        f"  Statistical Significance Tests  (non-parametric)\n"
        f"{'='*78}"
    )
    print(header)

    for metric in _METRICS:
        if metric not in df.columns:
            print(f"  [skip] column '{metric}' not found in data")
            continue

        print(f"\n  ── Metric: {metric} {'─'*(50 - len(metric))}")

        groups = {
            mode: df.loc[df['forwarding'] == mode, metric].dropna()
            for mode in ['flood', 'heuristic', 'ai', 'aodv']
        }

        # Pairwise Mann–Whitney U
        for a, b in _PAIRS:
            ga, gb = groups.get(a), groups.get(b)
            if ga is None or gb is None or len(ga) < 2 or len(gb) < 2:
                print(f"    {a} vs {b}: insufficient data")
                continue

            stat, p = mannwhitneyu(ga, gb, alternative='two-sided')
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else
                   ('*' if p < 0.05 else 'ns'))
            print(f"    Mann–Whitney  {a:>10s} vs {b:<10s}  "
                  f"U={stat:>10.1f}  p={p:.4e}  {sig}")

            rows.append({
                'metric': metric,
                'test': 'Mann-Whitney U',
                'group_a': a,
                'group_b': b,
                'statistic': stat,
                'p_value': p,
                'significance': sig,
            })

        # Kruskal–Wallis H across all three modes
        valid_groups = [g for g in groups.values()
                        if g is not None and len(g) >= 2]
        if len(valid_groups) >= 2:
            h_stat, h_p = kruskal(*valid_groups)
            h_sig = '***' if h_p < 0.001 else ('**' if h_p < 0.01 else
                     ('*' if h_p < 0.05 else 'ns'))
            print(f"    Kruskal–Wallis (all modes)          "
                  f"H={h_stat:>10.2f}  p={h_p:.4e}  {h_sig}")

            rows.append({
                'metric': metric,
                'test': 'Kruskal-Wallis H',
                'group_a': 'all',
                'group_b': 'all',
                'statistic': h_stat,
                'p_value': h_p,
                'significance': h_sig,
            })

    print(f"\n{'='*78}\n")

    # Export
    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_csv, index=False)
    print(f"  Results written to {output_csv}\n")

    return results_df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def generate_all_plots(csv_file: str):
    """Generate all figures from the CSV results."""
    if not os.path.exists(csv_file):
        print(f"Results file {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)

    # 1. PDR
    plot_bar_with_errors(
        df, 'pdr', 'Packet Delivery Ratio (%)',
        'Delivery Reliability across Scenarios', 'pdr_comparison.png'
    )

    # 2. Latency
    plot_bar_with_errors(
        df, 'latency_median', 'Median Latency (s)',
        'End-to-End Latency', 'latency_comparison.png'
    )

    # 3. Overhead
    plot_bar_with_errors(
        df, 'total_overhead', 'Overhead Ratio (Control+Dup / Data)',
        'Network Overhead', 'overhead_comparison.png'
    )

    # 4. Energy
    plot_bar_with_errors(
        df, 'energy_mean', 'Average Energy per Node (J)',
        'Energy Consumption', 'energy_comparison.png'
    )

    print("Plots generated in the 'plots/' directory.")

    # 5. Statistical significance tests
    run_significance_tests(df)

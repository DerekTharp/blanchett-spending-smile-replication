"""
07_twoway_clustering.py
=======================
Two-way clustering sensitivity analysis for panel models.

This script compares standard errors under:
1. One-way clustering (by household only) - the baseline approach
2. Two-way clustering (by household AND wave) - accounts for cross-sectional dependence

The motivation: With macro shocks (e.g., 2008 financial crisis, COVID pandemic),
errors may be correlated across households within the same wave. One-way clustering
by household alone does not address this. Two-way clustering provides a more
conservative inference that accounts for both within-household correlation and
cross-household correlation within waves.

Author: Derek Tharp
Date: 2026
"""

import pandas as pd
import numpy as np
import os
import warnings
from scipy import stats
from linearmodels.panel import PanelOLS

# Suppress dependency warnings (statsmodels, linearmodels) that clutter pipeline output.
# Analysis-level warnings are not suppressed.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # peer_review/
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # Blanchett Smile Replication/
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_TABLES = os.path.join(BASE_DIR, "tables")
os.makedirs(OUTPUT_TABLES, exist_ok=True)


def load_panel():
    """Load the extension panel dataset."""
    panel_path = os.path.join(DATA_PROCESSED, "panel_analysis_2001_2021.csv")
    if not os.path.exists(panel_path):
        raise FileNotFoundError(f"Panel data not found: {panel_path}")

    panel = pd.read_csv(panel_path)
    print(f"Loaded panel: {len(panel):,} observations, {panel['hhidpn'].nunique():,} households")
    return panel


def run_fe_comparison(panel, period_name="2001-2021"):
    """
    Run Fixed Effects model with both one-way and two-way clustering.

    Returns comparison of standard errors.
    """
    print(f"\n{'='*70}")
    print(f"TWO-WAY CLUSTERING SENSITIVITY: {period_name}")
    print("="*70)

    # Prepare data
    df = panel.dropna(subset=['pct_change_total', 'lag_real_total', 'age_int', 'wave']).copy()
    df = df[(df['age_int'] >= 60) & (df['age_int'] <= 90)]
    df['ln_spending'] = np.log(df['lag_real_total'])
    df['age_sq'] = df['age_int'] ** 2

    # Set panel index
    df = df.set_index(['hhidpn', 'wave'])

    print(f"\nSample: {len(df):,} observations, {len(df.index.get_level_values(0).unique()):,} households")
    print(f"Waves: {sorted(df.index.get_level_values(1).unique())}")

    # =========================================================================
    # Fixed Effects - One-way clustering (household only)
    # =========================================================================
    print("\n--- One-Way Clustering (Household Only) ---")

    mod = PanelOLS(
        df['pct_change_total'],
        df[['age_sq', 'age_int', 'ln_spending']],
        entity_effects=True,
        time_effects=False,
        check_rank=False
    )

    fe_oneway = mod.fit(cov_type='clustered', cluster_entity=True)

    print(f"  Age²:        {fe_oneway.params['age_sq']:.6f} (SE: {fe_oneway.std_errors['age_sq']:.6f})")
    print(f"  Age:         {fe_oneway.params['age_int']:.6f} (SE: {fe_oneway.std_errors['age_int']:.6f})")
    print(f"  ln(Spending): {fe_oneway.params['ln_spending']:.6f} (SE: {fe_oneway.std_errors['ln_spending']:.6f})")

    # =========================================================================
    # Fixed Effects - Two-way clustering (household + wave)
    # =========================================================================
    print("\n--- Two-Way Clustering (Household + Wave) ---")

    fe_twoway = mod.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)

    print(f"  Age²:        {fe_twoway.params['age_sq']:.6f} (SE: {fe_twoway.std_errors['age_sq']:.6f})")
    print(f"  Age:         {fe_twoway.params['age_int']:.6f} (SE: {fe_twoway.std_errors['age_int']:.6f})")
    print(f"  ln(Spending): {fe_twoway.params['ln_spending']:.6f} (SE: {fe_twoway.std_errors['ln_spending']:.6f})")

    # =========================================================================
    # Compute SE ratios and compare inference
    # =========================================================================
    print("\n--- Standard Error Comparison ---")

    variables = ['age_sq', 'age_int', 'ln_spending']
    var_labels = ['Age²', 'Age', 'ln(Spending)']

    results = []
    for var, label in zip(variables, var_labels):
        coef = fe_oneway.params[var]
        se_oneway = fe_oneway.std_errors[var]
        se_twoway = fe_twoway.std_errors[var]

        se_ratio = se_twoway / se_oneway

        # t-stats and p-values
        t_oneway = coef / se_oneway
        t_twoway = coef / se_twoway

        # Approximate p-values (two-sided, using normal approximation for large samples)
        p_oneway = 2 * (1 - stats.norm.cdf(abs(t_oneway)))
        p_twoway = 2 * (1 - stats.norm.cdf(abs(t_twoway)))

        # 95% CIs
        ci_oneway_lo = coef - 1.96 * se_oneway
        ci_oneway_hi = coef + 1.96 * se_oneway
        ci_twoway_lo = coef - 1.96 * se_twoway
        ci_twoway_hi = coef + 1.96 * se_twoway

        print(f"\n  {label}:")
        print(f"    Coefficient: {coef:.6f}")
        print(f"    One-way SE:  {se_oneway:.6f} (t={t_oneway:.2f}, p={p_oneway:.4f})")
        print(f"    Two-way SE:  {se_twoway:.6f} (t={t_twoway:.2f}, p={p_twoway:.4f})")
        print(f"    SE Ratio (two-way/one-way): {se_ratio:.3f}")

        # Check if inference changes
        sig_oneway = p_oneway < 0.05
        sig_twoway = p_twoway < 0.05
        if sig_oneway != sig_twoway:
            print(f"    ** INFERENCE CHANGES: {'' if sig_oneway else 'not '}significant -> {'' if sig_twoway else 'not '}significant")
        else:
            print(f"    Inference unchanged: {'significant' if sig_twoway else 'not significant'} at 5% level")

        results.append({
            'Variable': label,
            'Coefficient': coef,
            'SE_OneWay': se_oneway,
            'SE_TwoWay': se_twoway,
            'SE_Ratio': se_ratio,
            't_OneWay': t_oneway,
            't_TwoWay': t_twoway,
            'p_OneWay': p_oneway,
            'p_TwoWay': p_twoway,
            'CI95_OneWay': f"[{ci_oneway_lo:.6f}, {ci_oneway_hi:.6f}]",
            'CI95_TwoWay': f"[{ci_twoway_lo:.6f}, {ci_twoway_hi:.6f}]",
            'Inference_Changes': sig_oneway != sig_twoway
        })

    return pd.DataFrame(results)


def main():
    """Run two-way clustering sensitivity analysis."""
    print("="*70)
    print("TWO-WAY CLUSTERING SENSITIVITY ANALYSIS")
    print("="*70)

    # Load panel
    try:
        panel = load_panel()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return None

    # Run comparison
    results = run_fe_comparison(panel, "2001-2021")

    # Save results
    output_path = os.path.join(OUTPUT_TABLES, 'twoway_clustering_comparison.csv')
    results.to_csv(output_path, index=False)
    print(f"\n\nResults saved to: {output_path}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    any_changes = results['Inference_Changes'].any()
    if any_changes:
        print("\nWARNING: Statistical inference changes for some coefficients")
        print("  when using two-way clustering. Results should be interpreted")
        print("  with caution.")
        changed = results[results['Inference_Changes']]['Variable'].tolist()
        print(f"  Affected: {', '.join(changed)}")
    else:
        print("\nStatistical inference is unchanged under two-way clustering.")
        print("  The main findings are robust to cross-sectional dependence.")

    avg_ratio = results['SE_Ratio'].mean()
    print(f"\n  Average SE inflation (two-way vs one-way): {avg_ratio:.1%}")

    return results


if __name__ == "__main__":
    import sys
    results = main()
    sys.exit(0)

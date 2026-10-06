#!/usr/bin/env python3
"""
09_additional_analyses.py
=========================
Additional analyses: descriptive statistics, time-varying controls,
stratified models, and survivorship robustness.

Produces:
1. Descriptive statistics table
2. FE models with time-varying controls: health, marital status, household size
3. Stratified analysis by wealth tercile and self-rated health
4. Survivorship robustness check (progressive wave restrictions)
5. Spline age profile table and Figure 4
6. Retirement-filter sensitivity check
7. Specification decomposition: pooled vs. FE crossed with/without the lagged
   spending level, plus wave effects, subperiods, spending-timing asymmetry,
   and a consumption-measure (CCTOT) check

All outputs go to peer_review/tables/ (Figure 4 to peer_review/figures/).

Date: 2026
"""

import sys
import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
from linearmodels.panel.utility import AbsorbingEffectWarning
from patsy import dmatrix

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=AbsorbingEffectWarning)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_TABLES = os.path.join(BASE_DIR, "tables")
OUTPUT_FIGURES = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_TABLES, exist_ok=True)
os.makedirs(OUTPUT_FIGURES, exist_ok=True)


def load_panel():
    """Load the extension panel dataset."""
    panel_path = os.path.join(DATA_PROCESSED, "panel_analysis_2001_2021.csv")
    if not os.path.exists(panel_path):
        raise FileNotFoundError(f"Panel data not found: {panel_path}")
    panel = pd.read_csv(panel_path)
    print(f"Loaded panel: {len(panel):,} observations, {panel['hhidpn'].nunique():,} households")
    return panel


def prepare_fe_data(panel):
    """Prepare panel data for FE estimation with required variables."""
    df = panel.dropna(subset=['pct_change_total', 'lag_real_total', 'age_int']).copy()
    df = df[(df['age_int'] >= 60) & (df['age_int'] <= 90)]
    df['ln_spending'] = np.log(df['lag_real_total'])
    df['age_sq'] = df['age_int'] ** 2
    df = df.set_index(['hhidpn', 'wave'])
    return df


# =========================================================================
# 1. Descriptive Statistics
# =========================================================================

def descriptive_statistics(panel):
    """Generate descriptive statistics table for the analysis sample."""
    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 70)

    df = panel.dropna(subset=['pct_change_total', 'lag_real_total', 'age_int']).copy()
    df = df[(df['age_int'] >= 60) & (df['age_int'] <= 90)]

    n_obs = len(df)
    n_hh = df['hhidpn'].nunique()

    rows = []

    # Panel structure
    obs_per_hh = df.groupby('hhidpn').size()
    rows.append({'Variable': 'Panel Structure', 'Mean': '', 'SD': '', 'Median': '',
                 'Min': '', 'Max': '', 'N': ''})
    rows.append({'Variable': '  Households', 'Mean': '', 'SD': '', 'Median': '',
                 'Min': '', 'Max': '', 'N': f'{n_hh:,}'})
    rows.append({'Variable': '  Household-interval observations', 'Mean': '', 'SD': '',
                 'Median': '', 'Min': '', 'Max': '', 'N': f'{n_obs:,}'})
    rows.append({'Variable': '  Intervals per household', 'Mean': f'{obs_per_hh.mean():.1f}',
                 'SD': f'{obs_per_hh.std():.1f}', 'Median': f'{obs_per_hh.median():.0f}',
                 'Min': f'{obs_per_hh.min()}', 'Max': f'{obs_per_hh.max()}', 'N': ''})

    # Age
    rows.append({'Variable': '', 'Mean': '', 'SD': '', 'Median': '', 'Min': '', 'Max': '', 'N': ''})
    rows.append({'Variable': 'Demographics', 'Mean': '', 'SD': '', 'Median': '',
                 'Min': '', 'Max': '', 'N': ''})
    rows.append({'Variable': '  Age (household average)', 'Mean': f'{df["age_int"].mean():.1f}',
                 'SD': f'{df["age_int"].std():.1f}', 'Median': f'{df["age_int"].median():.0f}',
                 'Min': f'{df["age_int"].min()}', 'Max': f'{df["age_int"].max()}',
                 'N': f'{df["age_int"].notna().sum():,}'})

    # Partnered
    if 'partnered' in df.columns:
        pct_partnered = df['partnered'].mean() * 100
        rows.append({'Variable': '  Partnered (%)', 'Mean': f'{pct_partnered:.1f}',
                     'SD': '', 'Median': '', 'Min': '', 'Max': '',
                     'N': f'{df["partnered"].notna().sum():,}'})

    # Household size
    if 'hh_size' in df.columns:
        rows.append({'Variable': '  Household size', 'Mean': f'{df["hh_size"].mean():.1f}',
                     'SD': f'{df["hh_size"].std():.1f}', 'Median': f'{df["hh_size"].median():.0f}',
                     'Min': f'{df["hh_size"].min():.0f}', 'Max': f'{df["hh_size"].max():.0f}',
                     'N': f'{df["hh_size"].notna().sum():,}'})

    # Spending
    rows.append({'Variable': '', 'Mean': '', 'SD': '', 'Median': '', 'Min': '', 'Max': '', 'N': ''})
    rows.append({'Variable': 'Spending (2009 $)', 'Mean': '', 'SD': '', 'Median': '',
                 'Min': '', 'Max': '', 'N': ''})
    rows.append({'Variable': '  Real spending (current wave)',
                 'Mean': f'{df["real_total"].mean():,.0f}',
                 'SD': f'{df["real_total"].std():,.0f}',
                 'Median': f'{df["real_total"].median():,.0f}',
                 'Min': f'{df["real_total"].min():,.0f}',
                 'Max': f'{df["real_total"].max():,.0f}',
                 'N': f'{df["real_total"].notna().sum():,}'})
    rows.append({'Variable': '  Real spending (lagged)',
                 'Mean': f'{df["lag_real_total"].mean():,.0f}',
                 'SD': f'{df["lag_real_total"].std():,.0f}',
                 'Median': f'{df["lag_real_total"].median():,.0f}',
                 'Min': f'{df["lag_real_total"].min():,.0f}',
                 'Max': f'{df["lag_real_total"].max():,.0f}',
                 'N': f'{df["lag_real_total"].notna().sum():,}'})
    rows.append({'Variable': '  Annualized spending change (%)',
                 'Mean': f'{df["pct_change_total"].mean()*100:.1f}',
                 'SD': f'{df["pct_change_total"].std()*100:.1f}',
                 'Median': f'{df["pct_change_total"].median()*100:.1f}',
                 'Min': f'{df["pct_change_total"].min()*100:.1f}',
                 'Max': f'{df["pct_change_total"].max()*100:.1f}',
                 'N': f'{n_obs:,}'})

    # Wealth
    if 'real_wealth' in df.columns:
        rows.append({'Variable': '', 'Mean': '', 'SD': '', 'Median': '', 'Min': '', 'Max': '', 'N': ''})
        rows.append({'Variable': 'Wealth (2009 $)', 'Mean': '', 'SD': '', 'Median': '',
                     'Min': '', 'Max': '', 'N': ''})
        rows.append({'Variable': '  Total wealth',
                     'Mean': f'{df["real_wealth"].mean():,.0f}',
                     'SD': f'{df["real_wealth"].std():,.0f}',
                     'Median': f'{df["real_wealth"].median():,.0f}',
                     'Min': f'{df["real_wealth"].min():,.0f}',
                     'Max': f'{df["real_wealth"].max():,.0f}',
                     'N': f'{df["real_wealth"].notna().sum():,}'})

    # Health
    if 'srhealth' in df.columns:
        rows.append({'Variable': '', 'Mean': '', 'SD': '', 'Median': '', 'Min': '', 'Max': '', 'N': ''})
        rows.append({'Variable': 'Health', 'Mean': '', 'SD': '', 'Median': '',
                     'Min': '', 'Max': '', 'N': ''})
        rows.append({'Variable': '  Self-rated health (1=exc, 5=poor)',
                     'Mean': f'{df["srhealth"].mean():.2f}',
                     'SD': f'{df["srhealth"].std():.2f}',
                     'Median': f'{df["srhealth"].median():.0f}',
                     'Min': f'{df["srhealth"].min():.0f}',
                     'Max': f'{df["srhealth"].max():.0f}',
                     'N': f'{df["srhealth"].notna().sum():,}'})
        pct_poor = df['poor_health'].mean() * 100
        rows.append({'Variable': '  Fair/poor health (%)', 'Mean': f'{pct_poor:.1f}',
                     'SD': '', 'Median': '', 'Min': '', 'Max': '',
                     'N': f'{df["poor_health"].notna().sum():,}'})

    desc_df = pd.DataFrame(rows)
    output_path = os.path.join(OUTPUT_TABLES, 'descriptive_statistics.csv')
    desc_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    # Print summary
    for _, row in desc_df.iterrows():
        if row['Mean'] == '' and row['SD'] == '':
            print(f"\n{row['Variable']}")
        else:
            print(f"  {row['Variable']}: {row['Mean']}")

    return desc_df


# =========================================================================
# 2. FE Models with Time-Varying Controls
# =========================================================================

def fe_with_time_varying_controls(panel):
    """Run FE models with and without time-varying controls."""
    print("\n" + "=" * 70)
    print("FE MODELS WITH TIME-VARYING CONTROLS")
    print("=" * 70)

    df = prepare_fe_data(panel)

    results = []

    # Model 1: Baseline FE (existing specification)
    print("\nModel 1: Baseline FE...")
    mod1 = PanelOLS(
        df['pct_change_total'],
        df[['age_sq', 'age_int', 'ln_spending']],
        entity_effects=True
    )
    fe1 = mod1.fit(cov_type='clustered', cluster_entity=True)
    results.append({
        'Model': 'Baseline FE',
        'Age_sq': f'{fe1.params["age_sq"]:.6f}',
        'Age_sq_SE': f'{fe1.std_errors["age_sq"]:.6f}',
        'Age': f'{fe1.params["age_int"]:.4f}',
        'Age_SE': f'{fe1.std_errors["age_int"]:.4f}',
        'ln_Spending': f'{fe1.params["ln_spending"]:.4f}',
        'ln_Spending_SE': f'{fe1.std_errors["ln_spending"]:.4f}',
        'N_obs': len(df),
        'N_hh': df.index.get_level_values(0).nunique()
    })
    print(f"  Age_sq: {fe1.params['age_sq']:.6f} (SE: {fe1.std_errors['age_sq']:.6f})")

    # Model 2: FE + health controls
    if 'poor_health' in df.columns:
        df_health = df.dropna(subset=['poor_health']).copy()
        print(f"\nModel 2: FE + health ({len(df_health):,} obs)...")
        mod2 = PanelOLS(
            df_health['pct_change_total'],
            df_health[['age_sq', 'age_int', 'ln_spending', 'poor_health']],
            entity_effects=True
        )
        fe2 = mod2.fit(cov_type='clustered', cluster_entity=True)
        results.append({
            'Model': 'FE + Health',
            'Age_sq': f'{fe2.params["age_sq"]:.6f}',
            'Age_sq_SE': f'{fe2.std_errors["age_sq"]:.6f}',
            'Age': f'{fe2.params["age_int"]:.4f}',
            'Age_SE': f'{fe2.std_errors["age_int"]:.4f}',
            'ln_Spending': f'{fe2.params["ln_spending"]:.4f}',
            'ln_Spending_SE': f'{fe2.std_errors["ln_spending"]:.4f}',
            'poor_health': f'{fe2.params["poor_health"]:.4f}',
            'poor_health_SE': f'{fe2.std_errors["poor_health"]:.4f}',
            'N_obs': len(df_health),
            'N_hh': df_health.index.get_level_values(0).nunique()
        })
        print(f"  Age_sq: {fe2.params['age_sq']:.6f}")
        print(f"  poor_health: {fe2.params['poor_health']:.4f} (SE: {fe2.std_errors['poor_health']:.4f})")

    # Model 3: FE + marital status + household size
    if 'partnered' in df.columns and 'hh_size' in df.columns:
        df_demo = df.dropna(subset=['partnered', 'hh_size']).copy()
        print(f"\nModel 3: FE + marital/HH size ({len(df_demo):,} obs)...")
        mod3 = PanelOLS(
            df_demo['pct_change_total'],
            df_demo[['age_sq', 'age_int', 'ln_spending', 'partnered', 'hh_size']],
            entity_effects=True
        )
        fe3 = mod3.fit(cov_type='clustered', cluster_entity=True)
        results.append({
            'Model': 'FE + Marital/HH Size',
            'Age_sq': f'{fe3.params["age_sq"]:.6f}',
            'Age_sq_SE': f'{fe3.std_errors["age_sq"]:.6f}',
            'Age': f'{fe3.params["age_int"]:.4f}',
            'Age_SE': f'{fe3.std_errors["age_int"]:.4f}',
            'ln_Spending': f'{fe3.params["ln_spending"]:.4f}',
            'ln_Spending_SE': f'{fe3.std_errors["ln_spending"]:.4f}',
            'partnered': f'{fe3.params["partnered"]:.4f}',
            'partnered_SE': f'{fe3.std_errors["partnered"]:.4f}',
            'hh_size': f'{fe3.params["hh_size"]:.4f}',
            'hh_size_SE': f'{fe3.std_errors["hh_size"]:.4f}',
            'N_obs': len(df_demo),
            'N_hh': df_demo.index.get_level_values(0).nunique()
        })
        print(f"  Age_sq: {fe3.params['age_sq']:.6f}")
        print(f"  partnered: {fe3.params['partnered']:.4f} (SE: {fe3.std_errors['partnered']:.4f})")
        print(f"  hh_size: {fe3.params['hh_size']:.4f} (SE: {fe3.std_errors['hh_size']:.4f})")

    # Model 4: FE + all time-varying controls
    tv_cols = [c for c in ['poor_health', 'partnered', 'hh_size'] if c in df.columns]
    if tv_cols:
        df_full = df.dropna(subset=tv_cols).copy()
        print(f"\nModel 4: FE + all controls ({len(df_full):,} obs)...")
        mod4 = PanelOLS(
            df_full['pct_change_total'],
            df_full[['age_sq', 'age_int', 'ln_spending'] + tv_cols],
            entity_effects=True
        )
        fe4 = mod4.fit(cov_type='clustered', cluster_entity=True)
        row4 = {
            'Model': 'FE + All Controls',
            'Age_sq': f'{fe4.params["age_sq"]:.6f}',
            'Age_sq_SE': f'{fe4.std_errors["age_sq"]:.6f}',
            'Age': f'{fe4.params["age_int"]:.4f}',
            'Age_SE': f'{fe4.std_errors["age_int"]:.4f}',
            'ln_Spending': f'{fe4.params["ln_spending"]:.4f}',
            'ln_Spending_SE': f'{fe4.std_errors["ln_spending"]:.4f}',
            'N_obs': len(df_full),
            'N_hh': df_full.index.get_level_values(0).nunique()
        }
        for c in tv_cols:
            row4[c] = f'{fe4.params[c]:.4f}'
            row4[f'{c}_SE'] = f'{fe4.std_errors[c]:.4f}'
        results.append(row4)
        print(f"  Age_sq: {fe4.params['age_sq']:.6f}")
        for c in tv_cols:
            print(f"  {c}: {fe4.params[c]:.4f} (SE: {fe4.std_errors[c]:.4f})")

    results_df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_TABLES, 'fe_time_varying_controls.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    return results_df


# =========================================================================
# 3. Stratified Analysis by Wealth and Health
# =========================================================================

def stratified_analysis(panel):
    """Run FE models stratified by wealth tercile and health status."""
    print("\n" + "=" * 70)
    print("STRATIFIED ANALYSIS")
    print("=" * 70)

    df = prepare_fe_data(panel)
    results = []

    # --- Wealth terciles ---
    # Use baseline (first observed) wealth as a fixed household-level classification
    # (a stable grouping device, not a strict-exogeneity guarantee)
    if 'real_wealth' in panel.columns:
        baseline_wealth = panel.dropna(subset=['real_wealth']).groupby('hhidpn')['real_wealth'].first()
        df['baseline_wealth'] = df.index.get_level_values(0).map(baseline_wealth)

        # Compute tercile cutoffs at the household level (one value per household),
        # not on the interval-level panel, so households are split into equal thirds.
        tercile_cuts = df.groupby(level=0)['baseline_wealth'].first().quantile([1/3, 2/3])
        df['wealth_tercile'] = pd.cut(
            df['baseline_wealth'],
            bins=[-np.inf, tercile_cuts.iloc[0], tercile_cuts.iloc[1], np.inf],
            labels=['Bottom', 'Middle', 'Top']
        )

        print(f"\nWealth tercile cutoffs: ${tercile_cuts.iloc[0]:,.0f}, ${tercile_cuts.iloc[1]:,.0f}")

        for tercile in ['Bottom', 'Middle', 'Top']:
            subset = df[df['wealth_tercile'] == tercile]
            if len(subset) < 100:
                print(f"  Skipping {tercile} tercile: only {len(subset)} obs")
                continue

            print(f"\n  Wealth {tercile} tercile ({len(subset):,} obs, {subset.index.get_level_values(0).nunique():,} HH)...")
            mod = PanelOLS(
                subset['pct_change_total'],
                subset[['age_sq', 'age_int', 'ln_spending']],
                entity_effects=True
            )
            fe = mod.fit(cov_type='clustered', cluster_entity=True)
            results.append({
                'Stratification': 'Wealth',
                'Group': f'{tercile} Tercile',
                'Age_sq': f'{fe.params["age_sq"]:.6f}',
                'Age_sq_SE': f'{fe.std_errors["age_sq"]:.6f}',
                'Age': f'{fe.params["age_int"]:.4f}',
                'Age_SE': f'{fe.std_errors["age_int"]:.4f}',
                'ln_Spending': f'{fe.params["ln_spending"]:.4f}',
                'ln_Spending_SE': f'{fe.std_errors["ln_spending"]:.4f}',
                'N_obs': len(subset),
                'N_hh': subset.index.get_level_values(0).nunique()
            })
            print(f"    Age_sq: {fe.params['age_sq']:.6f} (SE: {fe.std_errors['age_sq']:.6f})")
            print(f"    ln_Spending: {fe.params['ln_spending']:.4f}")

    # --- Health status ---
    if 'poor_health' in panel.columns:
        # Use baseline health (first observed)
        baseline_health = panel.dropna(subset=['srhealth']).groupby('hhidpn')['srhealth'].first()
        df['baseline_health'] = df.index.get_level_values(0).map(baseline_health)

        for label, condition in [('Good+ Health (1-3)', df['baseline_health'] <= 3),
                                  ('Fair/Poor Health (4-5)', df['baseline_health'] >= 4)]:
            subset = df[condition].dropna(subset=['pct_change_total'])
            if len(subset) < 100:
                print(f"  Skipping {label}: only {len(subset)} obs")
                continue

            print(f"\n  {label} ({len(subset):,} obs, {subset.index.get_level_values(0).nunique():,} HH)...")
            mod = PanelOLS(
                subset['pct_change_total'],
                subset[['age_sq', 'age_int', 'ln_spending']],
                entity_effects=True
            )
            fe = mod.fit(cov_type='clustered', cluster_entity=True)
            results.append({
                'Stratification': 'Health',
                'Group': label,
                'Age_sq': f'{fe.params["age_sq"]:.6f}',
                'Age_sq_SE': f'{fe.std_errors["age_sq"]:.6f}',
                'Age': f'{fe.params["age_int"]:.4f}',
                'Age_SE': f'{fe.std_errors["age_int"]:.4f}',
                'ln_Spending': f'{fe.params["ln_spending"]:.4f}',
                'ln_Spending_SE': f'{fe.std_errors["ln_spending"]:.4f}',
                'N_obs': len(subset),
                'N_hh': subset.index.get_level_values(0).nunique()
            })
            print(f"    Age_sq: {fe.params['age_sq']:.6f} (SE: {fe.std_errors['age_sq']:.6f})")
            print(f"    ln_Spending: {fe.params['ln_spending']:.4f}")

    results_df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_TABLES, 'stratified_analysis.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    return results_df


# =========================================================================
# 4. Survivorship Robustness
# =========================================================================

def survivorship_robustness(panel):
    """Compare results using progressively restricted subsamples to check survivorship effects."""
    print("\n" + "=" * 70)
    print("SURVIVORSHIP ROBUSTNESS")
    print("=" * 70)

    df = prepare_fe_data(panel)
    results = []

    # Full sample (baseline)
    print(f"\nFull sample ({len(df):,} obs)...")
    mod_full = PanelOLS(
        df['pct_change_total'],
        df[['age_sq', 'age_int', 'ln_spending']],
        entity_effects=True
    )
    fe_full = mod_full.fit(cov_type='clustered', cluster_entity=True)
    results.append({
        'Sample': 'Full sample',
        'Age_sq': f'{fe_full.params["age_sq"]:.6f}',
        'Age_sq_SE': f'{fe_full.std_errors["age_sq"]:.6f}',
        'Age': f'{fe_full.params["age_int"]:.4f}',
        'Age_SE': f'{fe_full.std_errors["age_int"]:.4f}',
        'ln_Spending': f'{fe_full.params["ln_spending"]:.4f}',
        'ln_Spending_SE': f'{fe_full.std_errors["ln_spending"]:.4f}',
        'N_obs': len(df),
        'N_hh': df.index.get_level_values(0).nunique()
    })

    # Balanced subsamples: households with 3+, 4+, 5+ intervals
    for min_intervals in [3, 4, 5]:
        obs_per_hh = df.groupby(level=0).size()
        keep_hh = obs_per_hh[obs_per_hh >= min_intervals].index
        subset = df.loc[df.index.get_level_values(0).isin(keep_hh)]

        if len(subset) < 200:
            print(f"  Skipping {min_intervals}+ intervals: only {len(subset)} obs")
            continue

        n_hh = subset.index.get_level_values(0).nunique()
        print(f"\n  Households with {min_intervals}+ intervals: {n_hh:,} HH, {len(subset):,} obs...")
        mod = PanelOLS(
            subset['pct_change_total'],
            subset[['age_sq', 'age_int', 'ln_spending']],
            entity_effects=True
        )
        fe = mod.fit(cov_type='clustered', cluster_entity=True)
        results.append({
            'Sample': f'HH with {min_intervals}+ intervals',
            'Age_sq': f'{fe.params["age_sq"]:.6f}',
            'Age_sq_SE': f'{fe.std_errors["age_sq"]:.6f}',
            'Age': f'{fe.params["age_int"]:.4f}',
            'Age_SE': f'{fe.std_errors["age_int"]:.4f}',
            'ln_Spending': f'{fe.params["ln_spending"]:.4f}',
            'ln_Spending_SE': f'{fe.std_errors["ln_spending"]:.4f}',
            'N_obs': len(subset),
            'N_hh': n_hh
        })
        print(f"    Age_sq: {fe.params['age_sq']:.6f} (SE: {fe.std_errors['age_sq']:.6f})")

    results_df = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_TABLES, 'survivorship_robustness.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")

    return results_df


# =========================================================================
# 5. Flexible Age Specification (Spline)
# =========================================================================

def spline_age_profile(panel):
    """Compare quadratic vs. restricted cubic spline age profiles in both
    Blanchett-style (cross-sectional) and FE (within-household) specifications."""
    print("\n" + "=" * 70)
    print("FLEXIBLE AGE SPECIFICATION (SPLINE)")
    print("=" * 70)

    df = prepare_fe_data(panel)
    age_vals = df['age_int'].values.astype(float)
    age_grid = np.arange(60, 91)

    # Natural cubic spline: 3 interior knots + boundary knots at percentile locations
    knots = np.percentile(age_vals, [5, 27.5, 50, 72.5, 95])
    inner_knots = list(knots[1:-1])
    lower_bound = float(knots[0])
    upper_bound = float(knots[-1])
    print(f"  Interior knots at ages: {[f'{k:.0f}' for k in knots[1:-1]]}")
    print(f"  Boundary knots at ages: {lower_bound:.0f}, {upper_bound:.0f}")

    spline_formula = (
        f"cr(age_int, knots={inner_knots}, "
        f"lower_bound={lower_bound}, upper_bound={upper_bound}) - 1"
    )

    # Build spline basis for estimation sample
    spline_basis = dmatrix(spline_formula, {"age_int": age_vals}, return_type='dataframe')
    spline_cols = [f'spline_{i}' for i in range(spline_basis.shape[1])]
    spline_basis.columns = spline_cols
    for col in spline_cols:
        df[col] = spline_basis[col].values

    # Build spline basis for prediction grid
    spline_grid = dmatrix(spline_formula, {"age_int": age_grid.astype(float)}, return_type='dataframe')

    # ---- FE models ----
    print("\n  FE Quadratic...")
    mod_fe_quad = PanelOLS(
        df['pct_change_total'], df[['age_sq', 'age_int', 'ln_spending']],
        entity_effects=True
    )
    fe_quad = mod_fe_quad.fit(cov_type='clustered', cluster_entity=True)

    print(f"  FE Spline ({len(spline_cols)} basis functions)...")
    mod_fe_spline = PanelOLS(
        df['pct_change_total'], df[spline_cols + ['ln_spending']],
        entity_effects=True, drop_absorbed=True
    )
    fe_spline = mod_fe_spline.fit(cov_type='clustered', cluster_entity=True)

    # ---- Blanchett-style (cross-sectional age-cell means) ----
    print("  Blanchett-style age-cell means...")
    age_means = df.reset_index().groupby('age_int')['pct_change_total'].agg(['mean', 'count'])
    age_means = age_means[age_means['count'] >= 10]
    am_ages = age_means.index.values.astype(float)
    am_changes = age_means['mean'].values
    am_weights = np.sqrt(age_means['count'].values)

    # Blanchett-style quadratic (weighted)
    bs_quad_coeffs = np.polyfit(am_ages, am_changes, 2, w=am_weights)

    # Blanchett-style spline (weighted OLS on age-cell means)
    am_spline_basis = dmatrix(spline_formula, {"age_int": am_ages}, return_type='dataframe')
    from numpy.linalg import lstsq
    W = np.diag(am_weights)
    X_bs = am_spline_basis.values
    bs_spline_coeffs, _, _, _ = lstsq(W @ X_bs, W @ am_changes, rcond=None)

    # ---- Predictions (all normalized to 0 at age 60) ----
    # FE quadratic
    fe_quad_pred = fe_quad.params['age_sq'] * age_grid**2 + fe_quad.params['age_int'] * age_grid
    fe_quad_pred -= fe_quad_pred[0]

    # FE spline
    surviving = {k: v for k, v in fe_spline.params.items() if k in spline_cols}
    fe_spline_pred = np.zeros(len(age_grid))
    for i, col in enumerate(spline_cols):
        if col in surviving:
            fe_spline_pred += surviving[col] * spline_grid.iloc[:, i].values
    fe_spline_pred -= fe_spline_pred[0]

    # Blanchett-style quadratic
    bs_quad_pred = np.polyval(bs_quad_coeffs, age_grid)
    bs_quad_pred -= bs_quad_pred[0]

    # Blanchett-style spline
    bs_spline_pred = spline_grid.values @ bs_spline_coeffs
    bs_spline_pred -= bs_spline_pred[0]

    # Save predictions
    pred_df = pd.DataFrame({
        'age': age_grid,
        'blanchett_quadratic': bs_quad_pred,
        'blanchett_spline': bs_spline_pred,
        'fe_quadratic': fe_quad_pred,
        'fe_spline': fe_spline_pred,
    })
    output_csv = os.path.join(OUTPUT_TABLES, 'spline_age_profile.csv')
    pred_df.to_csv(output_csv, index=False)
    print(f"\n  Saved: {output_csv}")

    # ---- Figure 4 (file name retains its original numbering) ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Cross-sectional (Blanchett-style)
    ax.plot(age_grid, bs_quad_pred * 100, '-', color='#E74C5C', linewidth=2,
            label='Cross-sectional: Quadratic')
    ax.plot(age_grid, bs_spline_pred * 100, '--', color='#E74C5C', linewidth=1.5,
            alpha=0.7, label='Cross-sectional: Spline')

    # Within-household (FE)
    ax.plot(age_grid, fe_quad_pred * 100, '-', color='#3498DB', linewidth=2,
            label='Fixed Effects: Quadratic')
    ax.plot(age_grid, fe_spline_pred * 100, '--', color='#3498DB', linewidth=1.5,
            alpha=0.7, label='Fixed Effects: Spline')

    # Scatter the cross-sectional age-cell means (normalized to age 60), which the
    # cross-sectional curves are fit to; these do not bear on the FE curves.
    valid_ages = (am_ages >= 60) & (am_ages <= 90)
    scatter_changes = am_changes[valid_ages] - am_changes[am_ages == 60][0]
    ax.scatter(am_ages[valid_ages], scatter_changes * 100,
               color='gray', s=15, alpha=0.4, zorder=1,
               label='Cross-sectional age-cell means')

    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('Age', fontsize=11)
    ax.set_ylabel('Age effect on annual spending-change rate\n(pp, relative to age 60)', fontsize=11)
    ax.set_title('Age Profiles Under Alternative Specifications', fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax.set_xlim(60, 90)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_FIGURES, 'figure5_spline_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {fig_path}")

    # Print summary
    print(f"\n  FE Quadratic Age_sq: {fe_quad.params['age_sq']:.6f}")
    print(f"  FE ln_Spending: quad={fe_quad.params['ln_spending']:.4f}, spline={fe_spline.params['ln_spending']:.4f}")
    print(f"  BS Quadratic Age_sq: {bs_quad_coeffs[0]:.6f}")

    return pred_df


# =========================================================================
# 6. Retirement-Filter Sensitivity (extension sample)
# =========================================================================

def retirement_filter_sensitivity(panel):
    """Compare extension FE on the full sample vs. the retired-only subsample.

    The replication applies a strict retirement filter; the extension does not.
    This checks whether the FE attenuation depends on including non-retired
    households by restricting to observations where any member is retired.
    """
    print("\n" + "=" * 70)
    print("RETIREMENT-FILTER SENSITIVITY (EXTENSION)")
    print("=" * 70)

    df = prepare_fe_data(panel)
    results = []

    for label, subset in [
        ('Full extension sample', df),
        ('Retired-only (any_retired)', df[df['any_retired'] == True]),
    ]:
        n_hh = subset.index.get_level_values(0).nunique()
        mod = PanelOLS(
            subset['pct_change_total'],
            subset[['age_sq', 'age_int', 'ln_spending']],
            entity_effects=True,
        )
        fe = mod.fit(cov_type='clustered', cluster_entity=True)
        results.append({
            'Sample': label,
            'Age_sq': f'{fe.params["age_sq"]:.6f}',
            'Age_sq_SE': f'{fe.std_errors["age_sq"]:.6f}',
            'Age': f'{fe.params["age_int"]:.4f}',
            'Age_SE': f'{fe.std_errors["age_int"]:.4f}',
            'ln_Spending': f'{fe.params["ln_spending"]:.4f}',
            'ln_Spending_SE': f'{fe.std_errors["ln_spending"]:.4f}',
            'N_obs': len(subset),
            'N_hh': n_hh,
        })
        print(f"  {label}: Age_sq={fe.params['age_sq']:.6f}, ln_Spending={fe.params['ln_spending']:.4f} ({n_hh:,} HH)")

    out = os.path.join(OUTPUT_TABLES, 'retirement_filter_sensitivity.csv')
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\n  Saved: {out}")
    return pd.DataFrame(results)


# =========================================================================
# 7. Specification Decomposition
# =========================================================================

def specification_decomposition(panel):
    """Unbundle the Table 8 contrast into its two margins.

    The Blanchett-style vs. FE comparison changes two things at once: pooled
    vs. within-household variation, and whether the model conditions on the
    lagged spending level. This estimates the full cross (pooled/FE x
    with/without lagged ln(S)) plus diagnostics for interpreting the
    unconditional within-household curvature: wave (period) effects,
    subperiod stability, the spending-timing asymmetry (lagged vs. current
    ln(S)), and a consumption-measure (CCTOT) variant of the FE model.
    """
    print("\n" + "=" * 70)
    print("SPECIFICATION DECOMPOSITION (AGE CURVATURE VS. SPENDING LEVEL)")
    print("=" * 70)

    base = panel.dropna(subset=['pct_change_total', 'lag_real_total',
                                'real_total', 'age_int']).copy()
    base = base[(base['age_int'] >= 60) & (base['age_int'] <= 90)]
    base['ln_lag'] = np.log(base['lag_real_total'])
    base['ln_cur'] = np.log(base['real_total'])
    base['age_sq'] = base['age_int'] ** 2
    base['const'] = 1.0

    results = []

    def run(df, cols, label, period, entity_effects, time_effects=False):
        d = df.set_index(['hhidpn', 'wave'])
        mod = PanelOLS(d['pct_change_total'], d[cols],
                       entity_effects=entity_effects, time_effects=time_effects,
                       drop_absorbed=True)
        fit = mod.fit(cov_type='clustered', cluster_entity=True)
        p, s = fit.params, fit.std_errors
        ln_col = next((c for c in ('ln_lag', 'ln_cur') if c in p.index), None)
        # Age-profile slope at age 75 (within the sample range), with
        # delta-method SE from the coefficient covariance
        slope75 = slope75_se = None
        if 'age_int' in p.index and 'age_sq' in p.index:
            V = fit.cov
            slope75 = p['age_int'] + 150 * p['age_sq']
            slope75_se = np.sqrt(V.loc['age_int', 'age_int']
                                 + 150 ** 2 * V.loc['age_sq', 'age_sq']
                                 + 2 * 150 * V.loc['age_int', 'age_sq'])
        row = {
            'Model': label,
            'Period': period,
            'Age_sq': f'{p["age_sq"]:.10f}',
            'Age_sq_SE': f'{s["age_sq"]:.10f}',
            'Age_sq_t': f'{p["age_sq"] / s["age_sq"]:.3f}',
            'Age': f'{p["age_int"]:.6f}' if 'age_int' in p.index else '',
            'Age_SE': f'{s["age_int"]:.6f}' if 'age_int' in p.index else '',
            'ln_Spending': f'{p[ln_col]:.6f}' if ln_col else '',
            'ln_Spending_SE': f'{s[ln_col]:.6f}' if ln_col else '',
            'Slope_age75': f'{slope75:.6f}' if slope75 is not None else '',
            'Slope_age75_SE': f'{slope75_se:.6f}' if slope75_se is not None else '',
            'N_obs': fit.nobs,
            'N_hh': d.index.get_level_values(0).nunique(),
        }
        results.append(row)
        print(f"  {label:44s} [{period}]  Age_sq={row['Age_sq']} (t={row['Age_sq_t']})")

    # The 2x2: pooled/FE x with/without lagged ln(S), full period
    run(base, ['const', 'age_sq', 'age_int'], 'Pooled OLS, age terms only', '2001-2021', False)
    run(base, ['const', 'age_sq', 'age_int', 'ln_lag'], 'Pooled OLS, age + lagged ln(S)', '2001-2021', False)
    run(base, ['age_sq', 'age_int'], 'FE, age terms only', '2001-2021', True)
    run(base, ['age_sq', 'age_int', 'ln_lag'], 'FE, age + lagged ln(S)', '2001-2021', True)

    # Diagnostic: period controls (linear age term largely absorbed by
    # entity + wave effects; the quadratic is identified from entry-age x
    # time interactions)
    run(base, ['age_sq', 'age_int'], 'FE, age terms only + wave effects', '2001-2021', True, time_effects=True)

    # Diagnostic: spending-timing asymmetry (lagged = interval start, the
    # DV denominator; current = interval end, the DV numerator)
    run(base, ['age_sq', 'age_int', 'ln_cur'], 'FE, age + current ln(S)', '2001-2021', True)

    # Diagnostic: subperiod stability of the unconditional FE curvature
    run(base[base['year'] <= 2009], ['age_sq', 'age_int'], 'FE, age terms only', '2001-2009', True)
    run(base[base['lag_year'] >= 2005], ['age_sq', 'age_int'], 'FE, age terms only', '2005-2021', True)
    run(base[base['lag_year'] >= 2011], ['age_sq', 'age_int'], 'FE, age terms only', '2011-2021', True)
    run(base[base['lag_year'] >= 2005], ['age_sq', 'age_int'], 'FE, age terms only + wave effects', '2005-2021', True, time_effects=True)

    # Consumption-measure check: FE with lagged ln(consumption), CCTOT-based
    # DV constructed on the same intervals as the spending panel
    full_path = os.path.join(DATA_PROCESSED, 'full_panel_unfiltered.csv')
    cons = pd.read_csv(full_path, usecols=['hhidpn', 'wave', 'real_consumption'])
    ccur = cons.rename(columns={'real_consumption': 'cons_cur'})
    clag = cons.rename(columns={'wave': 'lag_wave', 'real_consumption': 'cons_lag'})
    cc = base.merge(ccur, on=['hhidpn', 'wave'], how='left')
    cc = cc.merge(clag, on=['hhidpn', 'lag_wave'], how='left')
    cc = cc.dropna(subset=['cons_cur', 'cons_lag'])
    cc = cc[(cc['cons_cur'] > 0) & (cc['cons_lag'] > 0)]
    cc['pct_change_total'] = (cc['cons_cur'] / cc['cons_lag']) ** (1.0 / cc['interval_years']) - 1
    cc['ln_lag'] = np.log(cc['cons_lag'])
    # CCTOT is not available for the 2021 CAMS wave, so consumption-based
    # intervals end in 2019; the sample is otherwise the spending-defined panel
    run(cc, ['age_sq', 'age_int', 'ln_lag'], 'FE, age + lagged ln(C) [CCTOT]', '2001-2019', True)

    out = os.path.join(OUTPUT_TABLES, 'specification_decomposition.csv')
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\n  Saved: {out}")
    return pd.DataFrame(results)


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 70)
    print("ADDITIONAL ANALYSES")
    print("=" * 70)

    try:
        panel = load_panel()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    descriptive_statistics(panel)
    fe_with_time_varying_controls(panel)
    stratified_analysis(panel)
    survivorship_robustness(panel)
    spline_age_profile(panel)
    retirement_filter_sensitivity(panel)
    specification_decomposition(panel)

    print("\n" + "=" * 70)
    print("ALL ADDITIONAL ANALYSES COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
    sys.exit(0)

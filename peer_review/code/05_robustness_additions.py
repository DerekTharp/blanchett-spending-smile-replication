"""
05_robustness_additions.py
==========================
Additional robustness analyses:

1. Log-change DV specification: Use Δln(S) = (1/interval)*(ln(S_t) - ln(S_{t-1}))
   instead of the ratio-based DV as a functional-form check on the mechanical
   coupling (the linkage changes form, not presence; see create_log_change_dv).

2. Panel support diagnostic: Distribution of within-household age spans
   and number of observations per household to assess FE power.

3. Outlier filter sensitivity: Compare results with/without 50% change filter.
"""

import pandas as pd
import numpy as np
import os
import warnings
from statsmodels.formula.api import ols

# For proper panel models
try:
    from linearmodels.panel import PanelOLS
    HAS_LINEARMODELS = True
except ImportError:
    warnings.warn("linearmodels not installed; panel FE/RE models will be skipped.")
    HAS_LINEARMODELS = False

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
OUTPUT_FIGURES = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_TABLES, exist_ok=True)
os.makedirs(OUTPUT_FIGURES, exist_ok=True)


def load_panel():
    """Load the extended panel dataset (2001-2021)."""
    panel_path = os.path.join(DATA_PROCESSED, "panel_analysis_2001_2021.csv")
    if not os.path.exists(panel_path):
        raise FileNotFoundError(f"Panel dataset not found at {panel_path}")
    panel = pd.read_csv(panel_path)
    print(f"Loaded panel: {len(panel)} observations, {panel['hhidpn'].nunique()} households")
    return panel


def create_log_change_dv(panel):
    """
    Create log-change dependent variable: Δln(S) = (1/interval) * (ln(S_t) - ln(S_{t-1}))

    This changes the functional form of the mechanical linkage rather than reducing
    it: the ratio-based DV puts S_{t-1} in the denominator, while the log-change DV
    removes that denominator but still contains -(1/interval)*ln(S_{t-1}) exactly,
    which remains coupled to an ln(S_{t-1}) regressor.

    The (1/interval) factor annualizes the inter-wave gap (typically 2 years for CAMS).
    """
    df = panel.copy()

    # Need both current and lagged spending
    df = df.dropna(subset=['real_total', 'lag_real_total', 'age_int'])
    df = df[(df['age_int'] >= 60) & (df['age_int'] <= 90)]

    # Create log-change DV (annualized using actual interval)
    if 'interval_years' in df.columns:
        df = df[df['interval_years'] > 0]
        df['ln_change'] = (1.0 / df['interval_years']) * (np.log(df['real_total']) - np.log(df['lag_real_total']))
    else:
        # Fallback for older panel files without interval_years.
        # Assumes biennial (2-year) interval, matching standard CAMS wave spacing.
        df['ln_change'] = 0.5 * (np.log(df['real_total']) - np.log(df['lag_real_total']))

    # Regressors
    df['ln_spending'] = np.log(df['lag_real_total'])
    df['age_sq'] = df['age_int'] ** 2

    print(f"Log-change sample: {len(df)} observations, {df['hhidpn'].nunique()} households")

    return df


def run_log_change_models(panel):
    """
    Run panel models using log-change DV for robustness.

    Compare to ratio-based DV results to assess mechanical coupling sensitivity.
    """
    print("\n" + "="*70)
    print("LOG-CHANGE DV ROBUSTNESS ANALYSIS")
    print("="*70)

    df = create_log_change_dv(panel)
    results = []

    # 1. Pooled OLS with log-change DV
    print("\n--- Pooled OLS (Log-Change DV) ---")
    model_pooled = ols('ln_change ~ age_sq + age_int + ln_spending', data=df).fit(
        cov_type='cluster',
        cov_kwds={'groups': df['hhidpn']}
    )

    results.append({
        'DV': 'Log-Change',
        'Method': 'Pooled OLS',
        'Age²': model_pooled.params['age_sq'],
        'Age² SE': model_pooled.bse['age_sq'],
        'Age': model_pooled.params['age_int'],
        'Age SE': model_pooled.bse['age_int'],
        'ln(Spending)': model_pooled.params['ln_spending'],
        'ln(Spending) SE': model_pooled.bse['ln_spending'],
        'R²': model_pooled.rsquared,
        'N obs': len(df),
        'N HH': df['hhidpn'].nunique()
    })

    print(f"  Age²: {model_pooled.params['age_sq']:.6f} (SE: {model_pooled.bse['age_sq']:.6f})")
    print(f"  Age: {model_pooled.params['age_int']:.4f} (SE: {model_pooled.bse['age_int']:.4f})")
    print(f"  ln(Spending): {model_pooled.params['ln_spending']:.4f} (SE: {model_pooled.bse['ln_spending']:.4f})")
    print(f"  R²: {model_pooled.rsquared:.3f}")

    # 2. Fixed Effects with log-change DV
    if HAS_LINEARMODELS:
        print("\n--- Fixed Effects (Log-Change DV) ---")
        df_panel = df.set_index(['hhidpn', 'wave'])

        mod_fe = PanelOLS(
            df_panel['ln_change'],
            df_panel[['age_sq', 'age_int', 'ln_spending']],
            entity_effects=True,
            time_effects=False
        )
        fe_results = mod_fe.fit(cov_type='clustered', cluster_entity=True)

        results.append({
            'DV': 'Log-Change',
            'Method': 'Fixed Effects',
            'Age²': fe_results.params['age_sq'],
            'Age² SE': fe_results.std_errors['age_sq'],
            'Age': fe_results.params['age_int'],
            'Age SE': fe_results.std_errors['age_int'],
            'ln(Spending)': fe_results.params['ln_spending'],
            'ln(Spending) SE': fe_results.std_errors['ln_spending'],
            'R² within': fe_results.rsquared_within,
            'N obs': fe_results.nobs,
            'N HH': len(df_panel.index.get_level_values(0).unique())
        })

        print(f"  Age²: {fe_results.params['age_sq']:.6f} (SE: {fe_results.std_errors['age_sq']:.6f})")
        print(f"  Age: {fe_results.params['age_int']:.4f} (SE: {fe_results.std_errors['age_int']:.4f})")
        print(f"  ln(Spending): {fe_results.params['ln_spending']:.4f} (SE: {fe_results.std_errors['ln_spending']:.4f})")
        print(f"  R² (within): {fe_results.rsquared_within:.3f}")

        # 3. FE + Time with log-change DV
        print("\n--- Fixed Effects + Time (Log-Change DV) ---")
        mod_fe_time = PanelOLS(
            df_panel['ln_change'],
            df_panel[['age_sq', 'age_int', 'ln_spending']],
            entity_effects=True,
            time_effects=True
        )
        fe_time_results = mod_fe_time.fit(cov_type='clustered', cluster_entity=True)

        # Curvature is identified under additive entity and time effects;
        # withhold the linear age term because age and period drift cannot be separated.
        results.append({
            'DV': 'Log-Change',
            'Method': 'FE + Time',
            'Age²': fe_time_results.params['age_sq'],
            'Age² SE': fe_time_results.std_errors['age_sq'],
            'Age': np.nan,
            'Age SE': np.nan,
            'ln(Spending)': fe_time_results.params['ln_spending'],
            'ln(Spending) SE': fe_time_results.std_errors['ln_spending'],
            'R² within': fe_time_results.rsquared_within,
            'N obs': fe_time_results.nobs,
            'N HH': len(df_panel.index.get_level_values(0).unique())
        })

        print(f"  Age²: {fe_time_results.params['age_sq']:.6f} (SE: {fe_time_results.std_errors['age_sq']:.6f})")
        print("  Age: [withheld] (linear age-period drift)")
        print(f"  ln(Spending): {fe_time_results.params['ln_spending']:.4f} (SE: {fe_time_results.std_errors['ln_spending']:.4f})")
        print(f"  R² (within): {fe_time_results.rsquared_within:.3f}")

    return pd.DataFrame(results)


def run_ratio_dv_models(panel):
    """
    Run same models with ratio-based DV for comparison.
    """
    print("\n" + "="*70)
    print("RATIO-BASED DV (ORIGINAL SPECIFICATION)")
    print("="*70)

    df = panel.dropna(subset=['pct_change_total', 'lag_real_total', 'age_int']).copy()
    df = df[(df['age_int'] >= 60) & (df['age_int'] <= 90)]
    df['ln_spending'] = np.log(df['lag_real_total'])
    df['age_sq'] = df['age_int'] ** 2

    results = []

    # 1. Pooled OLS
    print("\n--- Pooled OLS (Ratio DV) ---")
    model_pooled = ols('pct_change_total ~ age_sq + age_int + ln_spending', data=df).fit(
        cov_type='cluster',
        cov_kwds={'groups': df['hhidpn']}
    )

    results.append({
        'DV': 'Ratio',
        'Method': 'Pooled OLS',
        'Age²': model_pooled.params['age_sq'],
        'Age² SE': model_pooled.bse['age_sq'],
        'Age': model_pooled.params['age_int'],
        'Age SE': model_pooled.bse['age_int'],
        'ln(Spending)': model_pooled.params['ln_spending'],
        'ln(Spending) SE': model_pooled.bse['ln_spending'],
        'R²': model_pooled.rsquared,
        'N obs': len(df),
        'N HH': df['hhidpn'].nunique()
    })

    print(f"  Age²: {model_pooled.params['age_sq']:.6f} (SE: {model_pooled.bse['age_sq']:.6f})")
    print(f"  ln(Spending): {model_pooled.params['ln_spending']:.4f} (SE: {model_pooled.bse['ln_spending']:.4f})")

    if HAS_LINEARMODELS:
        # 2. Fixed Effects
        print("\n--- Fixed Effects (Ratio DV) ---")
        df_panel = df.set_index(['hhidpn', 'wave'])

        mod_fe = PanelOLS(
            df_panel['pct_change_total'],
            df_panel[['age_sq', 'age_int', 'ln_spending']],
            entity_effects=True,
            time_effects=False
        )
        fe_results = mod_fe.fit(cov_type='clustered', cluster_entity=True)

        results.append({
            'DV': 'Ratio',
            'Method': 'Fixed Effects',
            'Age²': fe_results.params['age_sq'],
            'Age² SE': fe_results.std_errors['age_sq'],
            'Age': fe_results.params['age_int'],
            'Age SE': fe_results.std_errors['age_int'],
            'ln(Spending)': fe_results.params['ln_spending'],
            'ln(Spending) SE': fe_results.std_errors['ln_spending'],
            'R² within': fe_results.rsquared_within,
            'N obs': fe_results.nobs,
            'N HH': len(df_panel.index.get_level_values(0).unique())
        })

        print(f"  Age²: {fe_results.params['age_sq']:.6f} (SE: {fe_results.std_errors['age_sq']:.6f})")
        print(f"  ln(Spending): {fe_results.params['ln_spending']:.4f} (SE: {fe_results.std_errors['ln_spending']:.4f})")

        # 3. FE + Time
        print("\n--- Fixed Effects + Time (Ratio DV) ---")
        mod_fe_time = PanelOLS(
            df_panel['pct_change_total'],
            df_panel[['age_sq', 'age_int', 'ln_spending']],
            entity_effects=True,
            time_effects=True
        )
        fe_time_results = mod_fe_time.fit(cov_type='clustered', cluster_entity=True)

        # Curvature is identified under additive entity and time effects;
        # withhold the linear age term because age and period drift cannot be separated.
        results.append({
            'DV': 'Ratio',
            'Method': 'FE + Time',
            'Age²': fe_time_results.params['age_sq'],
            'Age² SE': fe_time_results.std_errors['age_sq'],
            'Age': np.nan,
            'Age SE': np.nan,
            'ln(Spending)': fe_time_results.params['ln_spending'],
            'ln(Spending) SE': fe_time_results.std_errors['ln_spending'],
            'R² within': fe_time_results.rsquared_within,
            'N obs': fe_time_results.nobs,
            'N HH': len(df_panel.index.get_level_values(0).unique())
        })

        print(f"  Age²: {fe_time_results.params['age_sq']:.6f} (SE: {fe_time_results.std_errors['age_sq']:.6f})")
        print("  Age: [withheld] (linear age-period drift)")
        print(f"  ln(Spending): {fe_time_results.params['ln_spending']:.4f} (SE: {fe_time_results.std_errors['ln_spending']:.4f})")

    return pd.DataFrame(results)


def panel_support_diagnostic(panel):
    """
    Compute panel support diagnostics to assess FE power:

    1. Distribution of number of intervals (observations) per household
    2. Distribution of within-household age span (max age - min age observed)
    3. Whether FE could detect Blanchett-sized curvature given this support
    """
    print("\n" + "="*70)
    print("PANEL SUPPORT DIAGNOSTIC")
    print("="*70)

    df = panel.dropna(subset=['pct_change_total', 'age_int']).copy()
    df = df[(df['age_int'] >= 60) & (df['age_int'] <= 90)]

    # Calculate household-level statistics
    hh_stats = df.groupby('hhidpn').agg({
        'wave': 'count',  # Number of intervals
        'age_int': ['min', 'max', 'mean']
    }).reset_index()
    hh_stats.columns = ['hhidpn', 'n_intervals', 'min_age', 'max_age', 'mean_age']
    hh_stats['age_span'] = hh_stats['max_age'] - hh_stats['min_age']

    print(f"\nSample: {len(hh_stats)} households, {len(df)} total observations")

    # 1. Distribution of intervals per household
    print("\n--- Number of Intervals per Household ---")
    interval_dist = hh_stats['n_intervals'].value_counts().sort_index()
    print(interval_dist)
    print(f"\nMean intervals per HH: {hh_stats['n_intervals'].mean():.2f}")
    print(f"Median intervals per HH: {hh_stats['n_intervals'].median():.1f}")
    print(f"Max intervals per HH: {hh_stats['n_intervals'].max()}")

    # 2. Distribution of age span
    print("\n--- Within-Household Age Span ---")
    print(f"Mean age span: {hh_stats['age_span'].mean():.1f} years")
    print(f"Median age span: {hh_stats['age_span'].median():.1f} years")
    print(f"25th percentile: {hh_stats['age_span'].quantile(0.25):.1f} years")
    print(f"75th percentile: {hh_stats['age_span'].quantile(0.75):.1f} years")
    print(f"Max age span: {hh_stats['age_span'].max():.0f} years")

    # 3. HHs with enough span to detect curvature.
    # Blanchett's Age^2 = 0.00008 governs how far the age profile bends away from a
    # straight line. The within-household-identifiable curvature is the deviation of
    # the quadratic from its tangent: over a 10-year span (e.g., 65-75) that is
    # 0.00008 * 10^2 = 0.008 = 0.8 percentage points, and over a 4-year span
    # 0.00008 * 4^2 = 0.0013 = 0.13 percentage points. (The raw quadratic component
    # 0.00008 * (75^2 - 65^2) = 0.112 is NOT the curvature effect: it omits the
    # linear age term, which offsets it to a net age change of about -1.3 pp over
    # 65-75.) Longer within-household age spans give more power to detect curvature.

    print("\n--- Households by Age Span ---")
    span_bins = [0, 4, 8, 12, 16, 100]
    span_labels = ['0-4 years', '5-8 years', '9-12 years', '13-16 years', '17+ years']
    hh_stats['span_bin'] = pd.cut(hh_stats['age_span'], bins=span_bins, labels=span_labels,
                                  right=True, include_lowest=True)
    span_dist = hh_stats['span_bin'].value_counts().sort_index()
    assert span_dist.sum() == len(hh_stats), (
        f"Age-span bins cover {span_dist.sum()} of {len(hh_stats)} households"
    )
    print(span_dist)
    print(f"\nHouseholds with age span >= 8 years: {(hh_stats['age_span'] >= 8).sum()} ({(hh_stats['age_span'] >= 8).mean()*100:.1f}%)")
    print(f"Households with age span >= 12 years: {(hh_stats['age_span'] >= 12).sum()} ({(hh_stats['age_span'] >= 12).mean()*100:.1f}%)")

    # Summary statistics table
    summary = {
        'Statistic': [
            'Total households',
            'Total observations',
            'Mean intervals per HH',
            'Median intervals per HH',
            'Mean within-HH age span (years)',
            'Median within-HH age span (years)',
            'HH with age span >= 8 years',
            'HH with age span >= 12 years'
        ],
        'Value': [
            len(hh_stats),
            len(df),
            f"{hh_stats['n_intervals'].mean():.2f}",
            f"{hh_stats['n_intervals'].median():.1f}",
            f"{hh_stats['age_span'].mean():.1f}",
            f"{hh_stats['age_span'].median():.1f}",
            f"{(hh_stats['age_span'] >= 8).sum()} ({(hh_stats['age_span'] >= 8).mean()*100:.1f}%)",
            f"{(hh_stats['age_span'] >= 12).sum()} ({(hh_stats['age_span'] >= 12).mean()*100:.1f}%)"
        ]
    }
    summary_df = pd.DataFrame(summary)

    # Interpretation
    print("\n--- Interpretation ---")
    mean_span = hh_stats['age_span'].mean()
    if mean_span < 6:
        print("WARNING: Mean age span is short (<6 years).")
        print("FE may lack power to detect curvature patterns.")
    elif mean_span < 10:
        print("CAUTION: Mean age span is moderate (6-10 years).")
        print("FE has some power but curvature estimates may be noisy.")
    else:
        print("GOOD: Mean age span is adequate (>10 years).")
        print("FE should have reasonable power to detect curvature if present.")

    # Singleton diagnostic: households contributing one interval supply no
    # within-household variation, so the FE slope is identified off the rest.
    # Compare the extension FE estimates with and without singleton households.
    if HAS_LINEARMODELS:
        print("\n--- Singleton Households (one interval, no within variation) ---")
        fe_df = df.dropna(subset=['lag_real_total']).copy()
        fe_df['ln_spending'] = np.log(fe_df['lag_real_total'])
        fe_df['age_sq'] = fe_df['age_int'] ** 2
        counts = fe_df.groupby('hhidpn')['wave'].count()
        singleton_ids = counts[counts == 1].index
        n_hh, n_single = len(counts), len(singleton_ids)
        print(f"Households: {n_hh}; singletons: {n_single}; slope-identifying: {n_hh - n_single}")

        rows = []
        for label, sub in [
            ('Full sample', fe_df),
            ('Singletons dropped', fe_df[~fe_df['hhidpn'].isin(singleton_ids)]),
        ]:
            panel_idx = sub.set_index(['hhidpn', 'wave'])
            fit = PanelOLS(
                panel_idx['pct_change_total'],
                panel_idx[['age_sq', 'age_int', 'ln_spending']],
                entity_effects=True,
            ).fit(cov_type='clustered', cluster_entity=True)
            rows.append({
                'Sample': label,
                'N_households': len(panel_idx.index.get_level_values(0).unique()),
                'N_singleton_households': n_single if label == 'Full sample' else 0,
                'N_obs': int(fit.nobs),
                'Age_sq': fit.params['age_sq'],
                'Age_sq_SE': fit.std_errors['age_sq'],
                'Age': fit.params['age_int'],
                'Age_SE': fit.std_errors['age_int'],
                'ln_Spending': fit.params['ln_spending'],
                'ln_Spending_SE': fit.std_errors['ln_spending'],
            })
            print(f"  {label}: Age_sq={rows[-1]['Age_sq']:.7f} (SE {rows[-1]['Age_sq_SE']:.7f}), "
                  f"ln(S)={rows[-1]['ln_Spending']:.4f} (SE {rows[-1]['ln_Spending_SE']:.4f})")

        singleton_df = pd.DataFrame(rows)
        singleton_path = os.path.join(OUTPUT_TABLES, 'singleton_household_diagnostic.csv')
        singleton_df.to_csv(singleton_path, index=False)
        print(f"Saved: {singleton_path}")

    return summary_df, hh_stats


def outlier_filter_sensitivity(panel):
    """
    Test sensitivity of results to the outlier filter (|annualized change| < 50%).

    The main analysis does NOT apply an outlier filter, preserving the actual
    distribution of retiree spending dynamics. This robustness check compares
    results with and without a Blanchett-style outlier filter to show that
    the core findings are robust to this methodological choice.
    """
    print("\n" + "="*70)
    print("OUTLIER FILTER SENSITIVITY ANALYSIS")
    print("="*70)
    print("Main analysis: NO outlier filter (actual distribution)")
    print("Robustness check: WITH 50% outlier filter (Blanchett-style)")

    if not HAS_LINEARMODELS:
        print("Requires linearmodels package. Skipping.")
        return None

    # Load filtered panel (with outlier filter applied)
    filtered_path = os.path.join(DATA_PROCESSED, "panel_with_outlier_filter.csv")
    if not os.path.exists(filtered_path):
        print(f"Filtered panel not found at {filtered_path}")
        print("Run 01_build_panel.py to generate it.")
        return None

    panel_filtered = pd.read_csv(filtered_path)

    results = []

    # Note: panel (input) is the main analysis WITHOUT outlier filter
    for filter_label, df_source in [("Without outlier filter (main)", panel), ("With outlier filter (robustness)", panel_filtered)]:
        print(f"\n--- {filter_label} ---")

        # Prepare data
        df = df_source.dropna(subset=['pct_change_total', 'lag_real_total', 'age_int']).copy()

        # Apply spending filter (always required)
        df = df[(df['real_total'] > 10000) & (df['lag_real_total'] > 10000)]

        # Apply age filter
        df = df[(df['age_int'] >= 60) & (df['age_int'] <= 90)]

        df['ln_spending'] = np.log(df['lag_real_total'])
        df['age_sq'] = df['age_int'] ** 2

        print(f"  Observations: {len(df)}, Households: {df['hhidpn'].nunique()}")

        # Outlier statistics
        n_extreme = (abs(df['pct_change_total']) >= 0.5).sum()
        pct_extreme = n_extreme / len(df) * 100 if len(df) > 0 else 0
        print(f"  Observations with |change| >= 50%: {n_extreme} ({pct_extreme:.1f}%)")

        # Fixed Effects model
        df_panel = df.set_index(['hhidpn', 'wave'])

        mod_fe = PanelOLS(
            df_panel['pct_change_total'],
            df_panel[['age_sq', 'age_int', 'ln_spending']],
            entity_effects=True,
            time_effects=False
        )
        fe_results = mod_fe.fit(cov_type='clustered', cluster_entity=True)

        results.append({
            'Filter': filter_label,
            'N obs': fe_results.nobs,
            'N HH': len(df_panel.index.get_level_values(0).unique()),
            'Age²': fe_results.params['age_sq'],
            'Age² SE': fe_results.std_errors['age_sq'],
            'Age': fe_results.params['age_int'],
            'Age SE': fe_results.std_errors['age_int'],
            'ln(Spending)': fe_results.params['ln_spending'],
            'ln(Spending) SE': fe_results.std_errors['ln_spending'],
            'R² within': fe_results.rsquared_within
        })

        print(f"  Age²: {fe_results.params['age_sq']:.6f} (SE: {fe_results.std_errors['age_sq']:.6f})")
        print(f"  Age: {fe_results.params['age_int']:.4f} (SE: {fe_results.std_errors['age_int']:.4f})")
        print(f"  ln(Spending): {fe_results.params['ln_spending']:.4f} (SE: {fe_results.std_errors['ln_spending']:.4f})")

    results_df = pd.DataFrame(results)

    # Comparison
    print("\n--- Comparison ---")
    without_filter = results_df[results_df['Filter'].str.contains('Without')].iloc[0]
    with_filter = results_df[results_df['Filter'].str.contains('With outlier')].iloc[0]

    print(f"\nSample size difference: {without_filter['N obs'] - with_filter['N obs']} observations")
    print(f"  ({(without_filter['N obs'] - with_filter['N obs']) / without_filter['N obs'] * 100:.1f}% removed by filter)")

    print(f"\nAge² coefficient:")
    print(f"  Without filter (main): {without_filter['Age²']:.6f}")
    print(f"  With filter (robust):  {with_filter['Age²']:.6f}")

    print(f"\nln(Spending) coefficient:")
    print(f"  Without filter (main): {without_filter['ln(Spending)']:.4f}")
    print(f"  With filter (robust):  {with_filter['ln(Spending)']:.4f}")

    # Save results
    output_path = os.path.join(OUTPUT_TABLES, 'outlier_filter_sensitivity.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return results_df


def main():
    """Run robustness additions."""
    print("="*70)
    print("ROBUSTNESS ADDITIONS")
    print("="*70)

    try:
        panel = load_panel()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return None

    # 1. Log-change DV robustness
    log_change_results = run_log_change_models(panel)

    # 2. Ratio DV (for comparison)
    ratio_results = run_ratio_dv_models(panel)

    # Combine and save
    all_results = pd.concat([log_change_results, ratio_results], ignore_index=True)
    output_path = os.path.join(OUTPUT_TABLES, 'dv_robustness_comparison.csv')
    all_results.to_csv(output_path, index=False)
    print(f"\nDV robustness results saved to: {output_path}")

    # 3. Panel support diagnostic
    summary_df, hh_stats = panel_support_diagnostic(panel)
    summary_path = os.path.join(OUTPUT_TABLES, 'panel_support_diagnostic.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nPanel support diagnostic saved to: {summary_path}")

    # 4. Outlier filter sensitivity
    outlier_sensitivity = outlier_filter_sensitivity(panel)
    if outlier_sensitivity is not None:
        print("\nOutlier filter sensitivity analysis complete.")

    # Print comparison summary
    print("\n" + "="*70)
    print("DV SPECIFICATION COMPARISON SUMMARY")
    print("="*70)
    print("\nKey finding: Does log-change DV produce similar results to ratio DV?")
    print("\nln(Spending) coefficient comparison:")
    for method in ['Fixed Effects', 'FE + Time']:
        log_coef = log_change_results[log_change_results['Method'] == method]['ln(Spending)'].values
        ratio_coef = ratio_results[ratio_results['Method'] == method]['ln(Spending)'].values
        if len(log_coef) > 0 and len(ratio_coef) > 0:
            print(f"\n  {method}:")
            print(f"    Ratio DV:      {ratio_coef[0]:.4f}")
            print(f"    Log-Change DV: {log_coef[0]:.4f}")
            print(f"    Ratio:         {ratio_coef[0]/log_coef[0]:.2f}x")

    print("\nAge² coefficient comparison:")
    for method in ['Fixed Effects', 'FE + Time']:
        log_coef = log_change_results[log_change_results['Method'] == method]['Age²'].values
        ratio_coef = ratio_results[ratio_results['Method'] == method]['Age²'].values
        if len(log_coef) > 0 and len(ratio_coef) > 0:
            print(f"\n  {method}:")
            print(f"    Ratio DV:      {ratio_coef[0]:.6f}")
            print(f"    Log-Change DV: {log_coef[0]:.6f}")

    return all_results, summary_df


if __name__ == "__main__":
    import sys
    results = main()
    sys.exit(0)

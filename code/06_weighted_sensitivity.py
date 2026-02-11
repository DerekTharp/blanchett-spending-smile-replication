"""
06_weighted_sensitivity.py
==========================
Weighted analysis sensitivity check.

This script compares unweighted and weighted estimates to assess
whether incorporating HRS/CAMS sampling weights materially affects results.

HRS/CAMS uses a complex survey design with oversampling of certain populations.
The household weights (cwgthh) adjust for differential selection probabilities,
nonresponse, and post-stratification. This is a sensitivity analysis using
weights as WLS analytic weights; it does not implement full complex-survey
variance estimation (e.g., Taylor linearization or BRR).

Author: Derek Tharp
Date: 2026
"""

import pandas as pd
import numpy as np
import os
import warnings
from statsmodels.formula.api import wls, ols

# Suppress dependency warnings (statsmodels, pandas) that clutter pipeline output.
# Analysis-level warnings are not suppressed.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # peer_review/
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_TABLES = os.path.join(BASE_DIR, "tables")
os.makedirs(OUTPUT_TABLES, exist_ok=True)


def load_cams_with_weights():
    """Load CAMS data including household weights."""
    cams_path = os.path.join(DATA_RAW, "randcams_2001_2021v1", "randcams_2001_2021v1.dta")

    if not os.path.exists(cams_path):
        raise FileNotFoundError(f"CAMS data not found at {cams_path}")

    cams = pd.read_stata(cams_path)
    print(f"Loaded CAMS: {len(cams)} households")

    # Extract household weights for each wave
    # Wave mapping: h5=2001, h6=2003, h7=2005, h8=2007, h9=2009, h10=2011, h11=2013, h12=2015, h13=2017, h14=2019, h15=2021
    wave_map = {
        2001: 'h5cwgthh', 2003: 'h6cwgthh', 2005: 'h7cwgthh', 2007: 'h8cwgthh',
        2009: 'h9cwgthh', 2011: 'h10cwgthh', 2013: 'h11cwgthh', 2015: 'h12cwgthh',
        2017: 'h13cwgthh', 2019: 'h14cwgthh', 2021: 'h15cwgthh'
    }

    return cams, wave_map


def load_panel_and_merge_weights():
    """Load panel data and merge in household weights."""
    # Load existing panel
    panel_path = os.path.join(DATA_PROCESSED, "panel_analysis_2001_2021.csv")
    if not os.path.exists(panel_path):
        raise FileNotFoundError(f"Panel data not found at {panel_path}")

    panel = pd.read_csv(panel_path)
    print(f"Loaded panel: {len(panel)} observations")

    # Load CAMS with weights
    cams, wave_map = load_cams_with_weights()

    # Reshape CAMS weights from wide to long: (hhidpn, year) -> weight
    weight_frames = []
    for year, weight_col in wave_map.items():
        if weight_col in cams.columns:
            wf = cams[['hhidpn', weight_col]].copy()
            wf = wf.rename(columns={weight_col: 'hh_weight'})
            wf['year'] = year
            weight_frames.append(wf)
    weights_long = pd.concat(weight_frames, ignore_index=True)
    weights_long.loc[weights_long['hh_weight'] <= 0, 'hh_weight'] = np.nan

    # Merge weights into panel
    panel = panel.merge(weights_long, on=['hhidpn', 'year'], how='left')

    # Report coverage
    n_with_weights = panel['hh_weight'].notna().sum()
    print(f"Observations with valid weights: {n_with_weights} ({100*n_with_weights/len(panel):.1f}%)")

    return panel


def blanchett_style_weighted(panel, use_weights=True, period_name="Full"):
    """
    Estimate Blanchett-style coefficients with or without weights.

    For weighted analysis:
    - Use weighted means at each age
    - Use weighted polynomial fit
    - Use weighted regression for ln(spending)
    """
    label = "Weighted" if use_weights else "Unweighted"
    print(f"\n--- Blanchett-Style Estimation ({label}, {period_name}) ---")

    # Prepare data
    df = panel.dropna(subset=['pct_change_total', 'lag_real_total', 'age_int']).copy()
    df = df[(df['age_int'] >= 60) & (df['age_int'] <= 90)]
    df['ln_spending'] = np.log(df['lag_real_total'])

    if use_weights:
        df = df.dropna(subset=['hh_weight'])
        df = df[df['hh_weight'] > 0]

    print(f"  Observations: {len(df)}")
    print(f"  Households: {df['hhidpn'].nunique()}")

    # Step 1: Age-level means (weighted or unweighted)
    if use_weights:
        age_groups = df.groupby('age_int').apply(
            lambda x: pd.Series({
                'mean_change': np.average(x['pct_change_total'], weights=x['hh_weight']),
                'n': len(x),
                'sum_weights': x['hh_weight'].sum()
            })
        ).reset_index()
    else:
        age_groups = df.groupby('age_int').agg({
            'pct_change_total': ['mean', 'count']
        }).reset_index()
        age_groups.columns = ['age_int', 'mean_change', 'n']

    age_groups = age_groups[age_groups['n'] >= 10]

    # Step 2: Polynomial fit (weighted by sqrt(n) for efficiency)
    ages = age_groups['age_int'].values
    changes = age_groups['mean_change'].values
    poly_weights = np.sqrt(age_groups['n'].values)

    # Weighted polynomial fit
    coeffs = np.polyfit(ages, changes, 2, w=poly_weights)
    age_sq_coef = coeffs[0]
    age_coef = coeffs[1]

    # Step 3: Household-level regression for ln(spending) - weighted or unweighted
    hh_data = df.groupby('hhidpn').agg({
        'pct_change_total': 'mean',
        'age_int': 'mean',
        'ln_spending': 'mean',
        'hh_weight': 'mean' if use_weights else 'count'
    }).reset_index()
    hh_data['age_sq'] = hh_data['age_int'] ** 2

    if use_weights:
        # Weighted least squares
        model = wls(
            'pct_change_total ~ age_sq + age_int + ln_spending',
            data=hh_data,
            weights=hh_data['hh_weight']
        ).fit()
    else:
        model = ols('pct_change_total ~ age_sq + age_int + ln_spending', data=hh_data).fit()

    ln_spend_coef = model.params['ln_spending']

    # Calculate constant
    if use_weights:
        observed_mean = np.average(hh_data['pct_change_total'], weights=hh_data['hh_weight'])
        avg_age = np.average(hh_data['age_int'], weights=hh_data['hh_weight'])
        avg_age_sq = np.average(hh_data['age_sq'], weights=hh_data['hh_weight'])
        avg_ln_spend = np.average(hh_data['ln_spending'], weights=hh_data['hh_weight'])
    else:
        observed_mean = hh_data['pct_change_total'].mean()
        avg_age = hh_data['age_int'].mean()
        avg_age_sq = hh_data['age_sq'].mean()
        avg_ln_spend = hh_data['ln_spending'].mean()

    constant = observed_mean - (age_sq_coef * avg_age_sq + age_coef * avg_age + ln_spend_coef * avg_ln_spend)

    results = {
        'method': f'Blanchett-style ({label})',
        'weighted': use_weights,
        'period': period_name,
        'age_sq': age_sq_coef,
        'age': age_coef,
        'ln_exp': ln_spend_coef,
        'constant': constant,
        'n_obs': len(df),
        'n_households': df['hhidpn'].nunique()
    }

    print(f"  Age²: {age_sq_coef:.6f}")
    print(f"  Age: {age_coef:.4f}")
    print(f"  ln(Exp): {ln_spend_coef:.4f}")

    return results


def fe_weighted_approximation(panel, use_weights=True, period_name="Full"):
    """
    Approximate weighted fixed effects using weighted demeaning.

    True weighted FE is complex; this is an approximation that:
    1. Weights observations by household weight
    2. Demeans within household (weighted)
    3. Runs weighted regression on demeaned data

    Note: This is a sensitivity check, not a fully rigorous weighted FE estimator.
    For rigorous weighted panel models, see survey econometrics literature.
    """
    label = "Weighted" if use_weights else "Unweighted"
    print(f"\n--- Fixed Effects Approximation ({label}, {period_name}) ---")

    df = panel.dropna(subset=['pct_change_total', 'lag_real_total', 'age_int']).copy()
    df = df[(df['age_int'] >= 60) & (df['age_int'] <= 90)]
    df['ln_spending'] = np.log(df['lag_real_total'])
    df['age_sq'] = df['age_int'] ** 2

    if use_weights:
        df = df.dropna(subset=['hh_weight'])
        df = df[df['hh_weight'] > 0]

    print(f"  Observations: {len(df)}")
    print(f"  Households: {df['hhidpn'].nunique()}")

    # Demean within household
    vars_to_demean = ['pct_change_total', 'age_sq', 'age_int', 'ln_spending']

    if use_weights:
        # Weighted demeaning
        for var in vars_to_demean:
            hh_means = df.groupby('hhidpn').apply(
                lambda x: np.average(x[var], weights=x['hh_weight'])
            )
            df[f'{var}_mean'] = df['hhidpn'].map(hh_means)
            df[f'{var}_dm'] = df[var] - df[f'{var}_mean']
    else:
        # Unweighted demeaning
        for var in vars_to_demean:
            hh_means = df.groupby('hhidpn')[var].transform('mean')
            df[f'{var}_dm'] = df[var] - hh_means

    # Run regression on demeaned data
    if use_weights:
        model = wls(
            'pct_change_total_dm ~ age_sq_dm + age_int_dm + ln_spending_dm - 1',
            data=df,
            weights=df['hh_weight']
        ).fit(cov_type='cluster', cov_kwds={'groups': df['hhidpn']})
    else:
        model = ols(
            'pct_change_total_dm ~ age_sq_dm + age_int_dm + ln_spending_dm - 1',
            data=df
        ).fit(cov_type='cluster', cov_kwds={'groups': df['hhidpn']})

    results = {
        'method': f'Fixed Effects ({label})',
        'weighted': use_weights,
        'period': period_name,
        'age_sq': model.params['age_sq_dm'],
        'age': model.params['age_int_dm'],
        'ln_exp': model.params['ln_spending_dm'],
        'age_sq_se': model.bse['age_sq_dm'],
        'age_se': model.bse['age_int_dm'],
        'ln_exp_se': model.bse['ln_spending_dm'],
        'n_obs': len(df),
        'n_households': df['hhidpn'].nunique()
    }

    print(f"  Age²: {results['age_sq']:.6f} (SE: {results['age_sq_se']:.6f})")
    print(f"  Age: {results['age']:.4f} (SE: {results['age_se']:.4f})")
    print(f"  ln(Exp): {results['ln_exp']:.4f} (SE: {results['ln_exp_se']:.4f})")

    return results


def main():
    """Run weighted sensitivity analysis."""
    print("="*70)
    print("WEIGHTED SENSITIVITY ANALYSIS")
    print("="*70)

    # Load panel with weights
    try:
        panel = load_panel_and_merge_weights()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return None

    results = []

    # Filter to full period (2001-2021)
    print("\n" + "="*70)
    print("FULL PERIOD (2001-2021)")
    print("="*70)

    # Blanchett-style: Unweighted vs Weighted
    results.append(blanchett_style_weighted(panel, use_weights=False, period_name="2001-2021"))
    results.append(blanchett_style_weighted(panel, use_weights=True, period_name="2001-2021"))

    # Fixed Effects: Unweighted vs Weighted
    results.append(fe_weighted_approximation(panel, use_weights=False, period_name="2001-2021"))
    results.append(fe_weighted_approximation(panel, use_weights=True, period_name="2001-2021"))

    # Replication period (2001-2009)
    print("\n" + "="*70)
    print("REPLICATION PERIOD (2001-2009)")
    print("="*70)

    panel_rep = panel[panel['year'] <= 2009]
    results.append(blanchett_style_weighted(panel_rep, use_weights=False, period_name="2001-2009"))
    results.append(blanchett_style_weighted(panel_rep, use_weights=True, period_name="2001-2009"))

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)

    # Calculate percent differences
    print("\n" + "="*70)
    print("WEIGHTED VS UNWEIGHTED COMPARISON")
    print("="*70)

    for period in ['2001-2021', '2001-2009']:
        print(f"\n{period}:")
        period_results = results_df[results_df['period'] == period]

        for method_base in ['Blanchett-style', 'Fixed Effects']:
            unw = period_results[period_results['method'] == f'{method_base} (Unweighted)']
            wgt = period_results[period_results['method'] == f'{method_base} (Weighted)']

            if len(unw) > 0 and len(wgt) > 0:
                unw = unw.iloc[0]
                wgt = wgt.iloc[0]

                print(f"\n  {method_base}:")
                for coef in ['age_sq', 'age', 'ln_exp']:
                    unw_val = unw[coef]
                    wgt_val = wgt[coef]
                    if unw_val != 0:
                        pct_diff = 100 * (wgt_val - unw_val) / abs(unw_val)
                        print(f"    {coef}: Unweighted={unw_val:.6f}, Weighted={wgt_val:.6f}, Diff={pct_diff:+.1f}%")
                    else:
                        print(f"    {coef}: Unweighted={unw_val:.6f}, Weighted={wgt_val:.6f}")

    # Save results
    output_path = os.path.join(OUTPUT_TABLES, 'weighted_sensitivity.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary conclusion
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # Check if weighting materially changes results
    full_blanchett_unw = results_df[(results_df['period'] == '2001-2021') &
                                     (results_df['method'] == 'Blanchett-style (Unweighted)')]
    full_blanchett_wgt = results_df[(results_df['period'] == '2001-2021') &
                                     (results_df['method'] == 'Blanchett-style (Weighted)')]

    if len(full_blanchett_unw) > 0 and len(full_blanchett_wgt) > 0:
        age_sq_diff = abs(full_blanchett_wgt['age_sq'].values[0] - full_blanchett_unw['age_sq'].values[0])
        age_sq_pct = 100 * age_sq_diff / abs(full_blanchett_unw['age_sq'].values[0])

        if age_sq_pct < 20:
            print("Weighting produces SIMILAR results to unweighted analysis.")
            print(f"Age² coefficient differs by {age_sq_pct:.1f}%.")
            print("This supports the unweighted approach used in the main analysis.")
        else:
            print("Weighting produces DIFFERENT results from unweighted analysis.")
            print(f"Age² coefficient differs by {age_sq_pct:.1f}%.")
            print("Consider reporting weighted results alongside unweighted.")

    return results_df


if __name__ == "__main__":
    import sys
    results = main()
    sys.exit(0)

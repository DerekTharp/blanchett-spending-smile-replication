"""
03_robustness_grid.py
=====================
Comprehensive robustness analysis testing sensitivity to implementation choices.

This script tests how the "spending smile" coefficients vary based on:
1. Spending variable: cstot (spending) vs cctot (consumption)
2. Retirement filter: strict, baseline, permissive
3. Age assignment: start, mid, end of interval
4. Annualization: arithmetic vs geometric
5. Change filter: 50% only vs 50% + 25%
6. Min observations per age: 10 vs 30
7. Age range: 60-85 vs 60-90
8. ln(Spending) definition: baseline (first) vs mean vs lagged (important - affects sign)

Target: ~30 key specifications focused on most impactful parameters

Author: Derek Tharp
Date: 2026
"""

import pandas as pd
import numpy as np
import pyreadstat
import os
from statsmodels.formula.api import ols
import warnings

# Suppress dependency warnings (statsmodels, pandas) that clutter pipeline output.
# Analysis-level warnings are not suppressed.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Paths - peer_review/code/ is 2 levels deep from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # peer_review/
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # Blanchett Smile Replication/
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
# Output to peer_review/tables and peer_review/figures (matches package structure)
OUTPUT_TABLES = os.path.join(BASE_DIR, "tables")
OUTPUT_FIGURES = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_TABLES, exist_ok=True)
os.makedirs(OUTPUT_FIGURES, exist_ok=True)

# Wave to year mapping (2001-2009 only for Blanchett replication)
WAVE_YEAR = {5: 2001, 6: 2003, 7: 2005, 8: 2007, 9: 2009}

# CPI-U for inflation adjustment (base year 2009)
# Source: BLS CPI-U annual averages, all items, U.S. city average (retrieved January 2026)
CPI_U = {2001: 177.1, 2003: 184.0, 2005: 195.3, 2007: 207.342, 2009: 214.537}

# Blanchett's published coefficients (targets)
BLANCHETT = {
    'age_sq': 0.00008,
    'age': -0.0125,
    'ln_exp': -0.0066,
    'constant': 0.546,
    'r2_60_85': 0.33,
    'r2_65_75': 0.57,
    'mean_change': -0.0096,
    'n_households': 591
}


def load_data():
    """Load and merge CAMS and HRS data."""
    print("Loading data...")

    # Load CAMS
    cams_path = os.path.join(DATA_RAW, "randcams_2001_2021v1", "randcams_2001_2021v1.dta")
    cams_cols = ['hhidpn']
    for w in WAVE_YEAR.keys():
        cams_cols.extend([f'h{w}cstot', f'h{w}cctot', f'incamsc{w}'])

    if not os.path.exists(cams_path):
        raise FileNotFoundError(
            f"CAMS data not found at {cams_path}. "
            "Download from https://hrsdata.isr.umich.edu/"
        )

    cams_df, _ = pyreadstat.read_dta(cams_path, usecols=cams_cols)

    # Load HRS
    hrs_path = os.path.join(DATA_RAW, "randhrs1992_2022v1_STATA", "randhrs1992_2022v1.dta")
    hrs_cols = ['hhidpn']
    for w in WAVE_YEAR.keys():
        hrs_cols.extend([
            f'r{w}agey_e', f's{w}agey_e',  # Age
            f'r{w}sayret', f's{w}sayret',  # Retirement self-id
            f'r{w}lbrf', f's{w}lbrf',      # Labor force status
        ])

    if not os.path.exists(hrs_path):
        raise FileNotFoundError(
            f"HRS data not found at {hrs_path}. "
            "Download from https://hrsdata.isr.umich.edu/"
        )

    hrs_df, _ = pyreadstat.read_dta(hrs_path, usecols=hrs_cols)

    # Merge
    merged = pd.merge(cams_df, hrs_df, on='hhidpn', how='inner')
    print(f"  Merged: {len(merged)} households")

    return merged


def build_panel(df, spending_var='cstot'):
    """
    Build panel dataset with specified spending variable.

    Args:
        spending_var: 'cstot' (spending) or 'cctot' (consumption)
    """
    records = []

    for w in WAVE_YEAR.keys():
        year = WAVE_YEAR[w]

        wave_df = pd.DataFrame({
            'hhidpn': df['hhidpn'],
            'wave': w,
            'year': year
        })

        # Spending/consumption
        spend_col = f'h{w}{spending_var}'  # h5cstot or h5cctot
        wave_df['nominal_spending'] = df[spend_col] if spend_col in df.columns else np.nan

        # Age (respondent and spouse)
        wave_df['resp_age'] = pd.to_numeric(df[f'r{w}agey_e'], errors='coerce')
        wave_df['spouse_age'] = pd.to_numeric(df[f's{w}agey_e'], errors='coerce')

        # Retirement status
        r_sayret = df[f'r{w}sayret'] == 1
        s_sayret = df[f's{w}sayret'] == 1 if f's{w}sayret' in df.columns else False
        # NOTE: Use lbrf == 5 (completely retired) only, consistent with 02_replication.py
        # lbrf codes: 1=works FT, 2=works PT, 3=unemployed, 4=partly retired, 5=completely retired, 6=not in LF, 7=disabled
        r_lbrf_ret = (df[f'r{w}lbrf'] == 5) if f'r{w}lbrf' in df.columns else False
        s_lbrf_ret = (df[f's{w}lbrf'] == 5) if f's{w}lbrf' in df.columns else False
        r_not_retired = df[f'r{w}sayret'] == 0
        s_not_retired = df[f's{w}sayret'] == 0 if f's{w}sayret' in df.columns else False

        wave_df['r_retired'] = r_sayret | r_lbrf_ret
        wave_df['s_retired'] = s_sayret | s_lbrf_ret
        wave_df['r_not_retired'] = r_not_retired
        wave_df['s_not_retired'] = s_not_retired
        wave_df['any_retired'] = wave_df['r_retired'] | wave_df['s_retired']
        wave_df['any_not_retired'] = wave_df['r_not_retired'] | wave_df['s_not_retired']

        # CAMS participation
        wave_df['in_cams'] = df[f'incamsc{w}'] if f'incamsc{w}' in df.columns else 1

        records.append(wave_df)

    panel = pd.concat(records, ignore_index=True)

    # Calculate real spending (2009 dollars)
    cpi_2009 = CPI_U[2009]
    panel['cpi'] = panel['year'].map(CPI_U)
    panel['real_spending'] = panel['nominal_spending'] * (cpi_2009 / panel['cpi'])

    return panel


def calculate_changes(panel, annualization='geometric', age_assignment='end'):
    """
    Calculate spending changes with specified methodology.

    Args:
        annualization: 'geometric' or 'arithmetic'
        age_assignment: 'start', 'mid', or 'end' (which wave's age to use)
    """
    panel = panel.sort_values(['hhidpn', 'wave']).copy()

    # Lagged values
    panel['lag_spending'] = panel.groupby('hhidpn')['real_spending'].shift(1)
    panel['lag_wave'] = panel.groupby('hhidpn')['wave'].shift(1)
    panel['lag_resp_age'] = panel.groupby('hhidpn')['resp_age'].shift(1)
    panel['lag_spouse_age'] = panel.groupby('hhidpn')['spouse_age'].shift(1)

    # Years between (should be 2)
    panel['years'] = panel['wave'].map(WAVE_YEAR) - panel['lag_wave'].map(lambda x: WAVE_YEAR.get(x, np.nan))

    # Calculate change
    if annualization == 'geometric':
        # (C_t / C_{t-1})^(1/years) - 1
        panel['annual_change'] = np.where(
            (panel['lag_spending'] > 0) & (panel['years'] > 0),
            (panel['real_spending'] / panel['lag_spending']) ** (1 / panel['years']) - 1,
            np.nan
        )
    else:  # arithmetic
        # (C_t - C_{t-1}) / C_{t-1} / years
        panel['annual_change'] = np.where(
            (panel['lag_spending'] > 0) & (panel['years'] > 0),
            ((panel['real_spending'] - panel['lag_spending']) / panel['lag_spending']) / panel['years'],
            np.nan
        )

    # Age assignment
    if age_assignment == 'start':
        panel['age'] = panel[['lag_resp_age', 'lag_spouse_age']].mean(axis=1)
    elif age_assignment == 'mid':
        curr_age = panel[['resp_age', 'spouse_age']].mean(axis=1)
        prev_age = panel[['lag_resp_age', 'lag_spouse_age']].mean(axis=1)
        panel['age'] = (curr_age + prev_age) / 2
    else:  # end
        panel['age'] = panel[['resp_age', 'spouse_age']].mean(axis=1)

    panel['age_int'] = panel['age'].round().astype('Int64')

    return panel


def apply_filters(panel, retirement_filter='permissive', change_filter='50_only',
                  min_spending=10000, max_change_50=0.50, max_change_25=0.25):
    """
    Apply sample filters.

    Args:
        retirement_filter: 'strict', 'baseline', 'permissive'
            - strict: Exclude if ANY member says NOT retired in ANY wave
            - baseline: Retired in wave 5 (2001)
            - permissive: Any member retired in ANY wave
        change_filter: '50_only' or '50_and_25'
    """
    waves = list(WAVE_YEAR.keys())

    # Pivot for filter calculations
    spend_wide = panel.pivot(index='hhidpn', columns='wave', values='nominal_spending')
    real_spend_wide = panel.pivot(index='hhidpn', columns='wave', values='real_spending')

    # Filter 1: Spending data in all 5 waves
    has_all = spend_wide[waves].notna().all(axis=1)
    valid_hh = has_all[has_all].index

    # Filter 2: Spending > $10k in all waves
    above_min = (spend_wide[waves] > min_spending).all(axis=1)
    valid_hh = valid_hh.intersection(above_min[above_min].index)

    # Filter 3: Retirement filter
    if retirement_filter == 'strict':
        # Exclude if ANY member says NOT retired in ANY wave
        not_ret_wide = panel.pivot(index='hhidpn', columns='wave', values='any_not_retired')
        never_not_retired = ~not_ret_wide[waves].any(axis=1)
        valid_hh = valid_hh.intersection(never_not_retired[never_not_retired].index)
    elif retirement_filter == 'baseline':
        # Retired in wave 5 (2001)
        ret_w5 = panel[panel['wave'] == 5].set_index('hhidpn')['any_retired']
        retired_baseline = ret_w5[ret_w5 == True].index
        valid_hh = valid_hh.intersection(retired_baseline)
    else:  # permissive
        # Any member retired in ANY wave
        any_ret_wide = panel.pivot(index='hhidpn', columns='wave', values='any_retired')
        ever_retired = any_ret_wide[waves].any(axis=1)
        valid_hh = valid_hh.intersection(ever_retired[ever_retired].index)

    # Filter 4: Change < 50% (raw between-survey change, NOT annualized)
    # NOTE: This uses the raw 2-year change |(S_t - S_{t-1})/S_{t-1}| < 50%,
    # which is a stricter, literal reading of Blanchett (2014, p. 36).
    # The primary specification in 02_replication.py uses geometric annualized
    # change |(S_t/S_{t-1})^(1/interval) - 1| < 50%, which yields a larger sample.
    # This difference is intentional: the robustness grid explores sensitivity
    # to this ambiguity. See change_filter_comparison.csv for exact sample sizes.
    changes = pd.DataFrame(index=real_spend_wide.index)
    for w in [6, 7, 8, 9]:
        changes[f'ch_{w}'] = (real_spend_wide[w] - real_spend_wide[w-1]) / real_spend_wide[w-1]
    max_abs = changes.abs().max(axis=1)
    valid_change_50 = max_abs < max_change_50
    valid_hh = valid_hh.intersection(valid_change_50[valid_change_50].index)

    # Apply filters
    panel_filtered = panel[panel['hhidpn'].isin(valid_hh)].copy()

    # Filter 5 (optional): Additional 25% annualized filter
    if change_filter == '50_and_25':
        valid_obs = panel_filtered['annual_change'].isna() | (panel_filtered['annual_change'].abs() < max_change_25)
        panel_filtered = panel_filtered[valid_obs]

    return panel_filtered


def estimate_coefficients(panel, min_n_per_age=10, age_min=60, age_max=85,
                          ln_spending_def='baseline'):
    """
    Estimate Blanchett-style coefficients using polynomial fit to age means.

    Args:
        ln_spending_def: How to define household spending level for ln(Spending) coef
            - 'baseline': First observation (wave 5) spending - MATCHES BLANCHETT
            - 'mean': Mean spending across all observations
            - 'lagged': Start-of-interval (lagged) spending for each change

    Returns dict with coefficients and fit statistics.
    """
    # Pre-compute household spending anchors from the FULL panel (including baseline wave)
    # so "baseline" truly means wave 5 (2001) and "mean" includes all waves.
    baseline_spending = panel.loc[panel['wave'] == 5].set_index('hhidpn')['real_spending']
    mean_spending = panel.groupby('hhidpn')['real_spending'].mean()

    # Filter to valid observations (this excludes baseline wave which has no annual_change)
    df = panel.dropna(subset=['annual_change', 'real_spending', 'age_int']).copy()
    df = df[(df['age_int'] >= age_min) & (df['age_int'] <= age_max)]
    df['ln_spending'] = np.log(df['real_spending'])

    # Also compute lagged ln_spending for the 'lagged' option
    df['lag_ln_spending'] = np.log(df['lag_spending'].replace(0, np.nan))

    if len(df) < 50:
        return None

    # Age-level means
    age_means = df.groupby('age_int').agg({
        'annual_change': ['mean', 'count'],
        'real_spending': 'mean'
    }).reset_index()
    age_means.columns = ['age', 'mean_change', 'n', 'mean_spending']

    # Filter by min n
    age_means_filt = age_means[age_means['n'] >= min_n_per_age]

    if len(age_means_filt) < 5:
        return None

    ages = age_means_filt['age'].values
    changes = age_means_filt['mean_change'].values

    # Polynomial fit for age coefficients
    try:
        coeffs = np.polyfit(ages, changes, 2)
        age_sq_coef = coeffs[0]
        age_coef = coeffs[1]

        # Calculate R²
        predicted = np.polyval(coeffs, ages)
        ss_res = np.sum((changes - predicted) ** 2)
        ss_tot = np.sum((changes - np.mean(changes)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    except (np.linalg.LinAlgError, ValueError):
        # Polyfit can fail on singular matrices or insufficient data
        return None

    # Household-level regression for ln(spending) coefficient
    # Use the specified ln_spending definition
    if ln_spending_def == 'baseline':
        # Use wave 5 (2001) spending from pre-computed map (matches Blanchett's likely approach)
        hh_data = df.groupby('hhidpn').agg({
            'annual_change': 'mean',
            'age_int': 'mean',
        }).reset_index()
        hh_data['baseline_spending'] = hh_data['hhidpn'].map(baseline_spending)
        hh_data['ln_spending'] = np.log(hh_data['baseline_spending'])
    elif ln_spending_def == 'mean':
        # Use mean spending across ALL observations (including wave 5) from pre-computed map
        hh_data = df.groupby('hhidpn').agg({
            'annual_change': 'mean',
            'age_int': 'mean',
        }).reset_index()
        hh_data['mean_spending'] = hh_data['hhidpn'].map(mean_spending)
        hh_data['ln_spending'] = np.log(hh_data['mean_spending'])
    else:  # lagged
        # Use lagged (start-of-interval) spending
        hh_data = df.groupby('hhidpn').agg({
            'annual_change': 'mean',
            'age_int': 'mean',
            'lag_ln_spending': 'mean'
        }).reset_index()
        hh_data = hh_data.rename(columns={'lag_ln_spending': 'ln_spending'})

    hh_data['age_sq'] = hh_data['age_int'] ** 2
    hh_data = hh_data.dropna(subset=['ln_spending'])

    if len(hh_data) < 30:
        return None

    try:
        model = ols('annual_change ~ age_sq + age_int + ln_spending', data=hh_data).fit()
        ln_spend_coef = model.params['ln_spending']
        ln_spend_se = model.bse['ln_spending']
        ln_spend_pval = model.pvalues['ln_spending']
    except (KeyError, ValueError, np.linalg.LinAlgError):
        # OLS can fail on perfect collinearity, missing data, or singular design matrix
        return None

    # Calculate constant to match mean
    # Important: Use E[Age²], not (E[Age])² for mathematical correctness
    observed_mean = hh_data['annual_change'].mean()
    avg_age = hh_data['age_int'].mean()
    avg_age_sq = hh_data['age_sq'].mean()  # E[Age²], not (E[Age])²
    avg_ln_spend = hh_data['ln_spending'].mean()
    constant = observed_mean - (age_sq_coef * avg_age_sq + age_coef * avg_age + ln_spend_coef * avg_ln_spend)

    return {
        'age_sq': age_sq_coef,
        'age': age_coef,
        'ln_exp': ln_spend_coef,
        'ln_exp_se': ln_spend_se,
        'ln_exp_pval': ln_spend_pval,
        'constant': constant,
        'r2': r2,
        'mean_change': observed_mean,
        'n_households': hh_data['hhidpn'].nunique(),
        'n_ages': len(age_means_filt)
    }


def run_specification(df, spec):
    """Run a single specification and return results."""
    # Build panel with specified spending variable
    panel = build_panel(df, spending_var=spec['spending_var'])

    # Calculate changes
    panel = calculate_changes(panel,
                             annualization=spec['annualization'],
                             age_assignment=spec['age_assignment'])

    # Apply filters
    panel = apply_filters(panel,
                         retirement_filter=spec['retirement_filter'],
                         change_filter=spec['change_filter'])

    # Estimate coefficients with specified ln_spending definition
    results = estimate_coefficients(panel,
                                   min_n_per_age=spec['min_n_per_age'],
                                   age_min=spec['age_min'],
                                   age_max=spec['age_max'],
                                   ln_spending_def=spec['ln_spending_def'])

    if results is None:
        return None

    # Calculate ratios to Blanchett
    results['age_sq_ratio'] = results['age_sq'] / BLANCHETT['age_sq'] if BLANCHETT['age_sq'] != 0 else np.nan
    results['age_ratio'] = results['age'] / BLANCHETT['age'] if BLANCHETT['age'] != 0 else np.nan
    results['ln_exp_ratio'] = results['ln_exp'] / BLANCHETT['ln_exp'] if BLANCHETT['ln_exp'] != 0 else np.nan
    results['r2_ratio'] = results['r2'] / BLANCHETT['r2_60_85'] if BLANCHETT['r2_60_85'] != 0 else np.nan

    # Add spec info
    results.update(spec)

    return results


def main():
    """Run comprehensive robustness analysis."""
    print("="*70)
    print("ROBUSTNESS GRID ANALYSIS")
    print("="*70)

    # Load data once
    df = load_data()

    # Define specifications to test
    # Focus on most impactful parameters based on prior analysis

    specifications = []

    # PRIMARY SPECIFICATION (matches Blanchett methodology)
    primary = {
        'spending_var': 'cstot',
        'retirement_filter': 'permissive',
        'age_assignment': 'end',
        'annualization': 'geometric',
        'change_filter': '50_only',
        'min_n_per_age': 10,
        'age_min': 60,
        'age_max': 85,
        'ln_spending_def': 'baseline'  # Use first observation
    }
    specifications.append({**primary, 'spec_name': 'PRIMARY_baseline_ln'})

    # Show effect of ln_spending definition
    for ln_def in ['baseline', 'mean', 'lagged']:
        spec = primary.copy()
        spec['ln_spending_def'] = ln_def
        spec['spec_name'] = f'ln_def_{ln_def}'
        specifications.append(spec)

    # Test retirement filters with baseline ln_spending
    for filt in ['strict', 'baseline', 'permissive']:
        spec = primary.copy()
        spec['retirement_filter'] = filt
        spec['spec_name'] = f'retirement_{filt}'
        specifications.append(spec)

    # Test age assignment with baseline ln_spending
    for assign in ['start', 'mid', 'end']:
        spec = primary.copy()
        spec['age_assignment'] = assign
        spec['spec_name'] = f'age_assign_{assign}'
        specifications.append(spec)

    # Test annualization with baseline ln_spending
    for ann in ['geometric', 'arithmetic']:
        spec = primary.copy()
        spec['annualization'] = ann
        spec['spec_name'] = f'annualization_{ann}'
        specifications.append(spec)

    # Test spending variable with baseline ln_spending
    for var in ['cstot', 'cctot']:
        spec = primary.copy()
        spec['spending_var'] = var
        spec['spec_name'] = f'spending_{var}'
        specifications.append(spec)

    # Combined variations with mean ln_spending (shows instability)
    for ln_def in ['baseline', 'mean']:
        for filt in ['strict', 'permissive']:
            spec = primary.copy()
            spec['ln_spending_def'] = ln_def
            spec['retirement_filter'] = filt
            spec['spec_name'] = f'{ln_def}_{filt}'
            specifications.append(spec)

    # Faithful Blanchett attempt
    faithful = {
        'spending_var': 'cstot',
        'retirement_filter': 'strict',
        'age_assignment': 'end',
        'annualization': 'arithmetic',
        'change_filter': '50_only',
        'min_n_per_age': 30,
        'age_min': 60,
        'age_max': 85,
        'ln_spending_def': 'baseline'
    }
    specifications.append({**faithful, 'spec_name': 'faithful_blanchett'})

    # Remove duplicates by spec_name
    seen = set()
    unique_specs = []
    for spec in specifications:
        if spec['spec_name'] not in seen:
            seen.add(spec['spec_name'])
            unique_specs.append(spec)
    specifications = unique_specs

    print(f"\nRunning {len(specifications)} specifications...")

    # Run all specifications
    results = []
    for i, spec in enumerate(specifications):
        print(f"  [{i+1}/{len(specifications)}] {spec['spec_name']}...", end='')
        try:
            result = run_specification(df, spec)
            if result is not None:
                results.append(result)
                sign = "+" if result['ln_exp'] > 0 else ""
                print(f" n={result['n_households']}, ln(Exp)={sign}{result['ln_exp']:.4f}")
            else:
                print(" FAILED (insufficient data)")
        except Exception as e:
            print(f" ERROR: {e}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Reorder columns
    key_cols = ['spec_name', 'n_households', 'age_sq', 'age', 'ln_exp', 'ln_exp_se',
                'ln_exp_pval', 'constant', 'r2', 'mean_change',
                'age_sq_ratio', 'age_ratio', 'ln_exp_ratio', 'r2_ratio']
    param_cols = ['spending_var', 'retirement_filter', 'age_assignment', 'annualization',
                  'change_filter', 'min_n_per_age', 'age_min', 'age_max', 'ln_spending_def']
    results_df = results_df[key_cols + param_cols + ['n_ages']]

    # Save results
    output_path = os.path.join(OUTPUT_TABLES, 'robustness_grid.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nBlanchett (2014) targets:")
    print(f"  Age²: {BLANCHETT['age_sq']:.6f}")
    print(f"  Age: {BLANCHETT['age']:.4f}")
    print(f"  ln(Exp): {BLANCHETT['ln_exp']:.4f}")
    print(f"  R²: {BLANCHETT['r2_60_85']:.2f}")
    print(f"  N households: {BLANCHETT['n_households']}")

    print(f"\nCoefficient ranges across {len(results_df)} specifications:")
    print(f"  Age²: {results_df['age_sq'].min():.6f} to {results_df['age_sq'].max():.6f}")
    print(f"  Age: {results_df['age'].min():.4f} to {results_df['age'].max():.4f}")
    print(f"  ln(Exp): {results_df['ln_exp'].min():.4f} to {results_df['ln_exp'].max():.4f}")
    print(f"  R²: {results_df['r2'].min():.2f} to {results_df['r2'].max():.2f}")

    # ln_spending definition effect
    print("\n" + "="*70)
    print("ln(Spending) Definition Effect")
    print("="*70)
    ln_def_results = results_df[results_df['spec_name'].str.startswith('ln_def_')]
    if len(ln_def_results) > 0:
        for _, row in ln_def_results.iterrows():
            sign = "+" if row['ln_exp'] > 0 else ""
            match = "matches" if row['ln_exp'] < 0 else "opposite"
            print(f"  {row['ln_spending_def']:10s}: ln(Exp) = {sign}{row['ln_exp']:.4f} ({match} Blanchett sign)")

    print("\nThe ln(Spending) coefficient sign depends on definition:")
    if len(ln_def_results) > 0:
        for _, row in ln_def_results.iterrows():
            sign_desc = "negative (matches Blanchett)" if row['ln_exp'] < 0 else "positive (opposite to Blanchett)"
            print(f"  - '{row['ln_spending_def']}': {sign_desc}")
    print("  This methodological choice is not fully specified in the original paper.")

    # Find best match to Blanchett
    results_df['total_match'] = (
        abs(results_df['age_sq_ratio'] - 1) +
        abs(results_df['age_ratio'] - 1) +
        abs(results_df['ln_exp_ratio'] - 1)
    )
    best_match = results_df.loc[results_df['total_match'].idxmin()]

    print(f"\nBest overall match to Blanchett: {best_match['spec_name']}")
    print(f"  Age² ratio: {best_match['age_sq_ratio']:.2f}")
    print(f"  Age ratio: {best_match['age_ratio']:.2f}")
    print(f"  ln(Exp) ratio: {best_match['ln_exp_ratio']:.2f}")

    return results_df


if __name__ == "__main__":
    import sys
    results = main()
    sys.exit(0)

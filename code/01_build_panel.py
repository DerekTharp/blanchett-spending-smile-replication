#!/usr/bin/env python3
"""
01_build_panel.py - Build Panel Dataset for Extension Analysis

This script builds the 2001-2021 panel dataset from raw RAND HRS and CAMS files.
It is the prerequisite for the extension analysis (04_panel_extension.py).

The script:
1. Loads raw RAND HRS Longitudinal File and RAND CAMS data
2. Merges on hhidpn (household-person identifier)
3. Reshapes from wide to long (panel) format
4. Calculates spending changes between CAMS waves
5. Applies filters: spending > $10k, valid ages, valid spending change
6. Saves three outputs:
   - data/processed/panel_analysis_2001_2021.csv (main panel, no outlier filter)
   - data/processed/panel_with_outlier_filter.csv (|annualized change| < 50%)
   - data/processed/full_panel_unfiltered.csv (all observations before filtering)

Data Requirements:
- data/raw/randhrs1992_2022v1_STATA/randhrs1992_2022v1.dta
- data/raw/randcams_2001_2021v1/randcams_2001_2021v1.dta

Variable Dictionary (output columns):
- hhidpn: Household-person identifier (HRS respondent ID)
- wave: CAMS wave number (5-15)
- year: CAMS reference year (2001, 2003, ..., 2021)
- real_total: Real spending (CSTOT), CPI-adjusted to 2009 dollars
- lag_real_total: Previous-wave real spending
- pct_change_total: Annualized percentage spending change
- age_int: Average household age (integer)
- cwgthh: CAMS household weight

Author: Derek Tharp
Date: 2026
"""

import sys
import pandas as pd
import numpy as np
import pyreadstat
import os
import warnings
from datetime import datetime

# Suppress dependency warnings (statsmodels, pandas) that clutter pipeline output.
# Analysis-level warnings are not suppressed.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Paths - peer_review/code/ is 2 levels deep from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # peer_review/
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # Blanchett Smile Replication/
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")

os.makedirs(DATA_PROCESSED, exist_ok=True)

# CAMS wave to year mapping (all available waves)
# CAMS waves are keyed to preceding even-year HRS wave
WAVE_YEAR = {
    5: 2001, 6: 2003, 7: 2005, 8: 2007, 9: 2009,
    10: 2011, 11: 2013, 12: 2015, 13: 2017, 14: 2019, 15: 2021
}

# CPI-U for inflation adjustment (base year 2009)
# Source: BLS CPI-U annual averages, all items, U.S. city average (retrieved January 2026)
CPI_U = {
    2001: 177.1, 2003: 184.0, 2005: 195.3, 2007: 207.342, 2009: 214.537,
    2011: 224.939, 2013: 232.957, 2015: 237.017, 2017: 245.120,
    2019: 255.657, 2021: 270.970
}


def load_raw_data():
    """
    Load raw RAND HRS Longitudinal File and CAMS data.

    Returns merged dataset with CAMS spending and HRS demographics.
    """
    print("Loading raw RAND data...")
    print(f"  RAND HRS: randhrs1992_2022v1.dta")
    print(f"  RAND CAMS: randcams_2001_2021v1.dta")

    # =========================================================================
    # Load CAMS data
    # =========================================================================
    cams_path = os.path.join(DATA_RAW, "randcams_2001_2021v1", "randcams_2001_2021v1.dta")

    # Columns to extract from CAMS
    cams_cols = ['hhidpn']
    for w in WAVE_YEAR.keys():
        cams_cols.extend([
            f'h{w}cstot',    # Total spending
            f'h{w}cctot',    # Total consumption (includes imputed housing)
            f'incamsc{w}',   # CAMS participation indicator
        ])
        # Household weight (RAND uses h{w}cwgthh for all waves)
        cams_cols.append(f'h{w}cwgthh')

    if not os.path.exists(cams_path):
        raise FileNotFoundError(
            f"CAMS data not found at {cams_path}\n"
            "Download from https://hrsdata.isr.umich.edu/"
        )

    cams_df, _ = pyreadstat.read_dta(cams_path, usecols=cams_cols)
    print(f"  CAMS loaded: {len(cams_df):,} households")

    # Check for 2021 consumption variable availability (may be unavailable in some RAND versions)
    if 'h15cctot' not in cams_df.columns:
        print("  Note: h15cctot (2021 consumption) not available in this RAND CAMS version")

    # =========================================================================
    # Load HRS data
    # =========================================================================
    hrs_path = os.path.join(DATA_RAW, "randhrs1992_2022v1_STATA", "randhrs1992_2022v1.dta")

    # Columns to extract from HRS
    hrs_cols = ['hhidpn']
    for w in WAVE_YEAR.keys():
        hrs_cols.extend([
            f'r{w}agey_e', f's{w}agey_e',  # Age (respondent and spouse)
            f'r{w}sayret', f's{w}sayret',  # Retirement self-id
            f'r{w}lbrf', f's{w}lbrf',      # Labor force status
        ])

    if not os.path.exists(hrs_path):
        raise FileNotFoundError(
            f"HRS data not found at {hrs_path}\n"
            "Download from https://hrsdata.isr.umich.edu/"
        )

    hrs_df, _ = pyreadstat.read_dta(hrs_path, usecols=hrs_cols)
    print(f"  HRS loaded: {len(hrs_df):,} households")

    # =========================================================================
    # Merge CAMS and HRS
    # =========================================================================
    merged = pd.merge(cams_df, hrs_df, on='hhidpn', how='inner')
    print(f"  Merged: {len(merged):,} households")

    return merged


def reshape_to_panel(df):
    """
    Reshape wide data to long (panel) format.

    Each row becomes one household-wave observation.
    Uses vectorized column extraction per wave, then concatenates.
    """
    print("\nReshaping to panel format...")

    frames = []

    for wave, year in WAVE_YEAR.items():
        # Map wide column names → common long-format names
        col_map = {
            f'h{wave}cstot': 'nominal_spending',
            f'h{wave}cctot': 'nominal_consumption',
            f'h{wave}cwgthh': 'cwgthh',
            f'r{wave}agey_e': 'r_age',
            f's{wave}agey_e': 's_age',
            f'r{wave}sayret': 'r_sayret',
            f's{wave}sayret': 's_sayret',
            f'r{wave}lbrf': 'r_lbrf',
            f's{wave}lbrf': 's_lbrf',
        }

        # Select only columns that exist (handles e.g. missing h15cctot)
        available = {k: v for k, v in col_map.items() if k in df.columns}
        wf = df[['hhidpn'] + list(available.keys())].rename(columns=available).copy()

        wf['wave'] = wave
        wf['year'] = year

        # Default missing columns to NaN
        for col in col_map.values():
            if col not in wf.columns:
                wf[col] = np.nan

        frames.append(wf)

    panel = pd.concat(frames, ignore_index=True)

    # Ensure numeric types (RAND STATA files may have mixed encoding across waves)
    for col in ['r_age', 's_age', 'r_sayret', 's_sayret', 'r_lbrf', 's_lbrf',
                'nominal_spending', 'nominal_consumption', 'cwgthh']:
        panel[col] = pd.to_numeric(panel[col], errors='coerce')

    # Age: average of respondent and spouse (skipna handles single-person HH)
    # Per Blanchett (2014, p. 36): "average age of the two spouses"
    panel['age'] = panel[['r_age', 's_age']].mean(axis=1, skipna=True)

    # Retirement status: True if any household member is retired
    # RAND HRS codebook: sayret=1 → "says retired"; lbrf=5 → "retired"
    panel['any_retired'] = (
        (panel['r_sayret'] == 1) | (panel['s_sayret'] == 1) |
        (panel['r_lbrf'] == 5) | (panel['s_lbrf'] == 5)
    )

    # Inflate to 2009 dollars using CPI-U
    panel['cpi'] = panel['year'].map(CPI_U)
    panel['real_total'] = panel['nominal_spending'] * (CPI_U[2009] / panel['cpi'])
    panel['real_consumption'] = panel['nominal_consumption'] * (CPI_U[2009] / panel['cpi'])

    # Integer age (for grouping)
    panel['age_int'] = panel['age'].round().astype('Int64')

    print(f"  Panel created: {len(panel):,} household-wave observations")

    return panel


def calculate_spending_changes(panel):
    """
    Calculate annualized spending changes between consecutive CAMS waves.

    The dependent variable for panel analysis is:
        pct_change_total = (S_t / S_{t-1})^(1/interval) - 1

    where the interval is the number of years between consecutive waves
    (typically 2 for biennial CAMS).
    """
    print("\nCalculating spending changes...")

    panel = panel.sort_values(['hhidpn', 'wave'])

    # Lag spending (previous wave)
    panel['lag_real_total'] = panel.groupby('hhidpn')['real_total'].shift(1)
    panel['lag_wave'] = panel.groupby('hhidpn')['wave'].shift(1)
    panel['lag_year'] = panel.groupby('hhidpn')['year'].shift(1)

    # Compute actual inter-wave interval in years (typically 2 for CAMS)
    panel['interval_years'] = panel['year'] - panel['lag_year']

    # Calculate annualized change (geometric) using actual interval
    # Note: Requires lag_real_total > 0 to avoid division issues
    valid_lag = (panel['lag_real_total'] > 0) & panel['lag_real_total'].notna() & panel['interval_years'].notna() & (panel['interval_years'] > 0)
    panel.loc[valid_lag, 'pct_change_total'] = (
        panel.loc[valid_lag, 'real_total'] / panel.loc[valid_lag, 'lag_real_total']
    ) ** (1.0 / panel.loc[valid_lag, 'interval_years']) - 1

    # Also calculate raw (unannualized) change for robustness
    panel.loc[valid_lag, 'raw_change_total'] = (
        panel.loc[valid_lag, 'real_total'] - panel.loc[valid_lag, 'lag_real_total']
    ) / panel.loc[valid_lag, 'lag_real_total']

    # Log change (alternative DV for robustness), annualized using actual interval
    panel.loc[valid_lag, 'log_change_total'] = (1.0 / panel.loc[valid_lag, 'interval_years']) * (
        np.log(panel.loc[valid_lag, 'real_total']) -
        np.log(panel.loc[valid_lag, 'lag_real_total'])
    )

    # Observations with valid changes
    n_valid = panel['pct_change_total'].notna().sum()
    print(f"  Observations with valid spending change: {n_valid:,}")

    return panel


def apply_filters(panel, min_spending=10000, apply_outlier_filter=False, max_change=0.5):
    """
    Apply sample filters for the extension analysis.

    Filters:
    1. Spending > min_spending in both current and lagged wave
    2. Valid age (not missing)
    3. Valid spending change (not missing)
    4. (Optional) |Annualized change| < max_change (outlier removal)

    Note: No retirement filter is applied here (extension uses all households).
    The retirement filter is applied only for the replication sample.

    The outlier filter is OFF by default for the extension analysis to preserve
    the actual distribution of retiree spending dynamics. The filtered version
    is saved separately for robustness checks.
    """
    filter_desc = f"spending > ${min_spending:,}"
    if apply_outlier_filter:
        filter_desc += f", |change| < {max_change*100:.0f}%"
    print(f"\nApplying filters ({filter_desc})...")

    # Filter 1: Minimum spending in both waves
    spending_filter = (
        (panel['real_total'] > min_spending) &
        (panel['lag_real_total'] > min_spending)
    )

    # Filter 2: Valid age
    age_filter = panel['age_int'].notna()

    # Filter 3: Valid spending change
    change_filter = panel['pct_change_total'].notna()

    # Combine basic filters
    combined = spending_filter & age_filter & change_filter

    # Filter 4: Outlier removal (optional)
    if apply_outlier_filter:
        outlier_filter = abs(panel['pct_change_total']) < max_change
        combined = combined & outlier_filter
        n_outliers = (spending_filter & age_filter & change_filter & ~outlier_filter).sum()
    else:
        n_outliers = 0

    panel_filtered = panel[combined].copy()

    print(f"  Before filters: {len(panel):,} observations")
    print(f"  After spending + age + change filters: {combined.sum():,}")
    if apply_outlier_filter:
        print(f"  After outlier filter: {len(panel_filtered):,} observations")
        print(f"  Outliers removed (|change| >= {max_change*100:.0f}%): {n_outliers:,}")
    else:
        print(f"  Final sample: {len(panel_filtered):,} observations")
        # Report how many would be removed by outlier filter (for reference)
        would_remove = (combined & (abs(panel['pct_change_total']) >= max_change)).sum()
        print(f"  Note: {would_remove:,} observations have |change| >= {max_change*100:.0f}% (not filtered)")
    print(f"  Unique households: {panel_filtered['hhidpn'].nunique():,}")

    return panel_filtered


def main():
    """Build the extension panel dataset from raw RAND files."""
    print("=" * 70)
    print("BUILD EXTENSION PANEL DATASET")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load and merge raw data
    try:
        merged = load_raw_data()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nTo run this script, download the following from HRS:")
        print("  1. RAND HRS Longitudinal File (randhrs1992_2022v1.dta)")
        print("  2. RAND CAMS (randcams_2001_2021v1.dta)")
        print("\nPlace files in:")
        print(f"  {DATA_RAW}/randhrs1992_2022v1_STATA/")
        print(f"  {DATA_RAW}/randcams_2001_2021v1/")
        return 1

    # Reshape to panel
    panel = reshape_to_panel(merged)

    # Calculate spending changes
    panel = calculate_spending_changes(panel)

    # Apply filters WITHOUT outlier filter (primary analysis)
    # This preserves the actual distribution of retiree spending dynamics
    panel_main = apply_filters(panel, apply_outlier_filter=False)

    # Save main panel (for extension analysis - NO outlier filter)
    output_path = os.path.join(DATA_PROCESSED, "panel_analysis_2001_2021.csv")
    panel_main.to_csv(output_path, index=False)
    print(f"\nMain panel saved to: {output_path}")

    # Also save version WITH outlier filter (for robustness checks)
    panel_filtered = apply_filters(panel, apply_outlier_filter=True)
    filtered_path = os.path.join(DATA_PROCESSED, "panel_with_outlier_filter.csv")
    panel_filtered.to_csv(filtered_path, index=False)
    print(f"Filtered panel (with outlier filter) saved to: {filtered_path}")

    # Also save raw unfiltered panel (before spending filter)
    unfiltered_path = os.path.join(DATA_PROCESSED, "full_panel_unfiltered.csv")
    panel.to_csv(unfiltered_path, index=False)
    print(f"Full unfiltered panel saved to: {unfiltered_path}")

    # Use main panel for summary statistics
    panel_filtered = panel_main  # For compatibility with summary code below

    # Summary statistics
    print("\n" + "=" * 70)
    print("PANEL SUMMARY")
    print("=" * 70)
    print(f"Total households: {panel_filtered['hhidpn'].nunique():,}")
    print(f"Total observations: {len(panel_filtered):,}")
    print(f"Years covered: {panel_filtered['year'].min()}-{panel_filtered['year'].max()}")
    print(f"Waves covered: {sorted(panel_filtered['wave'].unique())}")

    # Observations per household
    obs_per_hh = panel_filtered.groupby('hhidpn').size()
    print(f"\nObservations per household:")
    print(f"  Mean: {obs_per_hh.mean():.1f}")
    print(f"  Median: {obs_per_hh.median():.1f}")
    print(f"  Min: {obs_per_hh.min()}, Max: {obs_per_hh.max()}")

    # Age distribution
    print(f"\nAge distribution:")
    print(f"  Mean: {panel_filtered['age_int'].mean():.1f}")
    print(f"  Range: {panel_filtered['age_int'].min()}-{panel_filtered['age_int'].max()}")

    # Spending change distribution
    print(f"\nAnnualized spending change:")
    print(f"  Mean: {panel_filtered['pct_change_total'].mean()*100:.2f}%")
    print(f"  Median: {panel_filtered['pct_change_total'].median()*100:.2f}%")
    print(f"  Std: {panel_filtered['pct_change_total'].std()*100:.2f}%")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

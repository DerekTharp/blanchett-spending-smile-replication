"""
04_panel_extension.py
=====================
Panel extension: FE/RE models for 2001-2021.

This script extends the Blanchett replication to 2001-2021 using:
1. Blanchett-style aggregation (for comparability)
2. Fixed effects panel model (within-household variation)
3. Random effects panel model (allows between-household variation)
4. Hausman test as a diagnostic for FE vs RE specification (not a formal test under clustering)

Compares curvature estimates from aggregation vs panel specifications.

Author: Derek Tharp
Date: 2026
"""

import pandas as pd
import numpy as np
import os
import warnings
from scipy import stats
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

# For proper panel models
try:
    from linearmodels.panel import PanelOLS, RandomEffects
    HAS_LINEARMODELS = True
except ImportError:
    warnings.warn("linearmodels not installed; using statsmodels approximation for panel models.")
    HAS_LINEARMODELS = False

# Suppress dependency warnings (statsmodels, linearmodels) that clutter pipeline output.
# Analysis-level warnings are not suppressed.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Paths - peer_review/code/ is 2 levels deep from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # peer_review/
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # Blanchett Smile Replication/
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
# Output to peer_review/tables and peer_review/figures (matches package structure)
OUTPUT_TABLES = os.path.join(BASE_DIR, "tables")
OUTPUT_FIGURES = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_TABLES, exist_ok=True)
os.makedirs(OUTPUT_FIGURES, exist_ok=True)

# Blanchett's coefficients for comparison
BLANCHETT = {
    'age_sq': 0.00008,
    'age': -0.0125,
    'ln_exp': -0.0066,
    'constant': 0.546
}


def load_panel():
    """Load the extended panel dataset (2001-2021)."""
    panel_path = os.path.join(DATA_PROCESSED, "panel_analysis_2001_2021.csv")
    if not os.path.exists(panel_path):
        raise FileNotFoundError(
            f"Panel dataset not found at {panel_path}. "
            "Run 01_build_panel.py first to build from raw RAND data."
        )
    panel = pd.read_csv(panel_path)
    print(f"Loaded panel: {len(panel)} observations, {panel['hhidpn'].nunique()} households")
    return panel


def blanchett_style_estimation(panel, period_name="Full Period"):
    """
    Estimate coefficients using Blanchett's aggregation approach.

    1. Calculate mean spending change at each integer age
    2. Fit polynomial to age-level means
    3. Use household-level regression for ln(spending) coefficient

    NOTE: Uses LAGGED ln(spending) to avoid mechanical endogeneity.
    The dependent variable (pct_change) is constructed from current spending,
    so we must use lagged spending as the regressor.
    """
    print(f"\n--- Blanchett-Style Estimation ({period_name}) ---")

    # Prepare data - use lag_real_total if available
    if 'lag_real_total' in panel.columns:
        df = panel.dropna(subset=['pct_change_total', 'lag_real_total', 'age_int']).copy()
        df = df[(df['age_int'] >= 60) & (df['age_int'] <= 90)]
        df['ln_spending'] = np.log(df['lag_real_total'])  # LAGGED to avoid endogeneity
    else:
        df = panel.dropna(subset=['pct_change_total', 'real_total', 'age_int']).copy()
        df = df[(df['age_int'] >= 60) & (df['age_int'] <= 90)]
        df['ln_spending'] = np.log(df['real_total'])

    print(f"  Observations: {len(df)}")
    print(f"  Households: {df['hhidpn'].nunique()}")

    # Step 1: Age-level means
    age_means = df.groupby('age_int').agg({
        'pct_change_total': ['mean', 'count']
    }).reset_index()
    age_means.columns = ['age', 'mean_change', 'n']
    age_means = age_means[age_means['n'] >= 10]

    # Step 2: Polynomial fit to age means
    ages = age_means['age'].values
    changes = age_means['mean_change'].values

    coeffs = np.polyfit(ages, changes, 2)
    age_sq_coef = coeffs[0]
    age_coef = coeffs[1]

    # R² of polynomial fit
    predicted = np.polyval(coeffs, ages)
    ss_res = np.sum((changes - predicted) ** 2)
    ss_tot = np.sum((changes - np.mean(changes)) ** 2)
    r2_poly = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Step 3: Household-level regression for ln(spending)
    hh_data = df.groupby('hhidpn').agg({
        'pct_change_total': 'mean',
        'age_int': 'mean',
        'ln_spending': 'mean'
    }).reset_index()
    hh_data['age_sq'] = hh_data['age_int'] ** 2

    model = ols('pct_change_total ~ age_sq + age_int + ln_spending', data=hh_data).fit()
    ln_spend_coef = model.params['ln_spending']

    # Calculate constant
    # Important: Use E[Age²], not (E[Age])² for mathematical correctness
    observed_mean = hh_data['pct_change_total'].mean()
    avg_age = hh_data['age_int'].mean()
    avg_age_sq = hh_data['age_sq'].mean()  # E[Age²], not (E[Age])²
    avg_ln_spend = hh_data['ln_spending'].mean()
    constant = observed_mean - (age_sq_coef * avg_age_sq + age_coef * avg_age + ln_spend_coef * avg_ln_spend)

    results = {
        'method': 'Blanchett-style',
        'period': period_name,
        'age_sq': age_sq_coef,
        'age': age_coef,
        'ln_exp': ln_spend_coef,
        'constant': constant,
        'r2_polynomial': r2_poly,
        'n_obs': len(df),
        'n_households': df['hhidpn'].nunique()
    }

    print(f"  Age²: {age_sq_coef:.6f} (Blanchett: {BLANCHETT['age_sq']:.6f})")
    print(f"  Age: {age_coef:.4f} (Blanchett: {BLANCHETT['age']:.4f})")
    print(f"  ln(Exp): {ln_spend_coef:.4f} (Blanchett: {BLANCHETT['ln_exp']:.4f})")
    print(f"  R² (polynomial): {r2_poly:.3f}")

    return results, age_means


def panel_ols_pooled(panel, period_name="Full Period"):
    """
    Pooled OLS regression (ignores panel structure).
    This is a baseline for comparison.

    NOTE: Uses LAGGED ln(spending) to avoid mechanical endogeneity.
    NOTE: Uses cluster-robust SEs (clustered by household) since observations
          from the same household are not independent.
    """
    print(f"\n--- Pooled OLS ({period_name}) ---")

    # Use lag_real_total if available
    if 'lag_real_total' in panel.columns:
        df = panel.dropna(subset=['pct_change_total', 'lag_real_total', 'age_int']).copy()
        df = df[(df['age_int'] >= 60) & (df['age_int'] <= 90)]
        df['ln_spending'] = np.log(df['lag_real_total'])  # LAGGED
    else:
        df = panel.dropna(subset=['pct_change_total', 'real_total', 'age_int']).copy()
        df = df[(df['age_int'] >= 60) & (df['age_int'] <= 90)]
        df['ln_spending'] = np.log(df['real_total'])
    df['age_sq'] = df['age_int'] ** 2

    # Fit model with cluster-robust standard errors
    model = ols('pct_change_total ~ age_sq + age_int + ln_spending', data=df).fit(
        cov_type='cluster',
        cov_kwds={'groups': df['hhidpn']}
    )

    results = {
        'method': 'Pooled OLS',
        'period': period_name,
        'age_sq': model.params['age_sq'],
        'age': model.params['age_int'],
        'ln_exp': model.params['ln_spending'],
        'constant': model.params['Intercept'],
        'r2': model.rsquared,
        'n_obs': len(df),
        'n_households': df['hhidpn'].nunique(),
        'age_sq_se': model.bse['age_sq'],
        'age_se': model.bse['age_int'],
        'ln_exp_se': model.bse['ln_spending']
    }

    print(f"  Age²: {results['age_sq']:.6f} (SE: {results['age_sq_se']:.6f})")
    print(f"  Age: {results['age']:.4f} (SE: {results['age_se']:.4f})")
    print(f"  ln(Exp): {results['ln_exp']:.4f} (SE: {results['ln_exp_se']:.4f})")
    print(f"  R²: {results['r2']:.3f}")

    return results


def panel_fixed_effects(panel, period_name="Full Period", time_effects=False):
    """
    Fixed effects panel model.
    Controls for time-invariant household heterogeneity.

    Args:
        panel: DataFrame with panel data
        period_name: Label for output
        time_effects: If True, include wave dummies. Set False for Hausman comparison.

    NOTE: Uses LAGGED ln(spending) to avoid mechanical endogeneity.
    """
    time_label = " (w/ time effects)" if time_effects else " (no time effects)"
    print(f"\n--- Fixed Effects Panel Model{time_label} ({period_name}) ---")

    # Use lag_real_total if available
    if 'lag_real_total' in panel.columns:
        df = panel.dropna(subset=['pct_change_total', 'lag_real_total', 'age_int', 'wave']).copy()
        df = df[(df['age_int'] >= 60) & (df['age_int'] <= 90)]
        df['ln_spending'] = np.log(df['lag_real_total'])  # LAGGED
    else:
        df = panel.dropna(subset=['pct_change_total', 'real_total', 'age_int', 'wave']).copy()
        df = df[(df['age_int'] >= 60) & (df['age_int'] <= 90)]
        df['ln_spending'] = np.log(df['real_total'])
    df['age_sq'] = df['age_int'] ** 2

    if HAS_LINEARMODELS:
        # Use linearmodels for proper FE estimation
        df = df.set_index(['hhidpn', 'wave'])

        # Fixed effects model
        # NOTE: time_effects must match RE model for valid Hausman test
        mod = PanelOLS(
            df['pct_change_total'],
            df[['age_sq', 'age_int', 'ln_spending']],
            entity_effects=True,
            time_effects=time_effects,  # Must match RE for Hausman test
            check_rank=False
        )
        fe_results = mod.fit(cov_type='clustered', cluster_entity=True)

        method_name = 'Fixed Effects (time)' if time_effects else 'Fixed Effects'

        # NOTE: With time_effects=True, age terms are not separately identified from
        # period effects (APC identification problem). We store NaN for age coefficients
        # in FE+Time to avoid reporting uninterpretable values.
        if time_effects:
            age_sq_val = np.nan  # Not identified
            age_val = np.nan     # Not identified
            age_sq_se_val = np.nan
            age_se_val = np.nan
        else:
            age_sq_val = fe_results.params['age_sq']
            age_val = fe_results.params['age_int']
            age_sq_se_val = fe_results.std_errors['age_sq']
            age_se_val = fe_results.std_errors['age_int']

        results = {
            'method': method_name,
            'period': period_name,
            'age_sq': age_sq_val,
            'age': age_val,
            'ln_exp': fe_results.params['ln_spending'],
            'r2_within': fe_results.rsquared_within,
            'n_obs': fe_results.nobs,
            'n_households': len(df.index.get_level_values(0).unique()),
            'age_sq_se': age_sq_se_val,
            'age_se': age_se_val,
            'ln_exp_se': fe_results.std_errors['ln_spending'],
            'model_obj': fe_results,
            'time_effects': time_effects
        }
    else:
        # Fallback: Use OLS with household dummies (memory-intensive)
        # This is an approximation
        df['hhidpn_str'] = df['hhidpn'].astype(str)

        # With time effects: add wave dummies
        if time_effects:
            df['wave_str'] = df['wave'].astype(str)
            model = ols('pct_change_total ~ age_sq + age_int + ln_spending + C(hhidpn_str) + C(wave_str)', data=df).fit()
            method_name = 'Fixed Effects + Time (approx)'
        else:
            model = ols('pct_change_total ~ age_sq + age_int + ln_spending + C(hhidpn_str)', data=df).fit()
            method_name = 'Fixed Effects (approx)'

        results = {
            'method': method_name,
            'period': period_name,
            'age_sq': model.params['age_sq'],
            'age': model.params['age_int'],
            'ln_exp': model.params['ln_spending'],
            'r2': model.rsquared,
            'n_obs': len(df),
            'n_households': df['hhidpn'].nunique(),
            'age_sq_se': model.bse['age_sq'],
            'age_se': model.bse['age_int'],
            'ln_exp_se': model.bse['ln_spending'],
            'model_obj': None
        }

    print(f"  Age²: {results['age_sq']:.6f} (SE: {results.get('age_sq_se', 'N/A')})")
    print(f"  Age: {results['age']:.4f} (SE: {results.get('age_se', 'N/A')})")
    print(f"  ln(Exp): {results['ln_exp']:.4f} (SE: {results.get('ln_exp_se', 'N/A')})")
    print(f"  R² (within): {results.get('r2_within', results.get('r2', 'N/A'))}")

    return results


def panel_random_effects(panel, period_name="Full Period"):
    """
    Random effects panel model.
    More efficient than FE if assumptions hold.

    NOTE: Uses LAGGED ln(spending) to avoid mechanical endogeneity.
    """
    print(f"\n--- Random Effects Panel Model ({period_name}) ---")

    # Use lag_real_total if available
    if 'lag_real_total' in panel.columns:
        df = panel.dropna(subset=['pct_change_total', 'lag_real_total', 'age_int', 'wave']).copy()
        df = df[(df['age_int'] >= 60) & (df['age_int'] <= 90)]
        df['ln_spending'] = np.log(df['lag_real_total'])  # LAGGED
    else:
        df = panel.dropna(subset=['pct_change_total', 'real_total', 'age_int', 'wave']).copy()
        df = df[(df['age_int'] >= 60) & (df['age_int'] <= 90)]
        df['ln_spending'] = np.log(df['real_total'])
    df['age_sq'] = df['age_int'] ** 2

    if HAS_LINEARMODELS:
        df = df.set_index(['hhidpn', 'wave'])

        # Random effects model
        # NOTE: RandomEffects in linearmodels automatically includes an intercept
        # when fitting, unlike PanelOLS. We explicitly add a constant column
        # for clarity and to match the model specification.
        import statsmodels.api as sm_const
        exog = sm_const.add_constant(df[['age_sq', 'age_int', 'ln_spending']])
        mod = RandomEffects(
            df['pct_change_total'],
            exog
        )
        re_results = mod.fit(cov_type='clustered', cluster_entity=True)

        results = {
            'method': 'Random Effects',
            'period': period_name,
            'age_sq': re_results.params['age_sq'],
            'age': re_results.params['age_int'],
            'ln_exp': re_results.params['ln_spending'],
            'r2_between': re_results.rsquared_between,
            'r2_within': re_results.rsquared_within,
            'n_obs': re_results.nobs,
            'n_households': len(df.index.get_level_values(0).unique()),
            'age_sq_se': re_results.std_errors['age_sq'],
            'age_se': re_results.std_errors['age_int'],
            'ln_exp_se': re_results.std_errors['ln_spending'],
            'model_obj': re_results
        }
    else:
        # Fallback to pooled OLS (not a true RE model)
        model = ols('pct_change_total ~ age_sq + age_int + ln_spending', data=df).fit()

        results = {
            'method': 'Random Effects (pooled approx)',
            'period': period_name,
            'age_sq': model.params['age_sq'],
            'age': model.params['age_int'],
            'ln_exp': model.params['ln_spending'],
            'constant': model.params['Intercept'],
            'r2': model.rsquared,
            'n_obs': len(df),
            'n_households': df['hhidpn'].nunique(),
            'age_sq_se': model.bse['age_sq'],
            'age_se': model.bse['age_int'],
            'ln_exp_se': model.bse['ln_spending'],
            'model_obj': None
        }

    print(f"  Age²: {results['age_sq']:.6f} (SE: {results.get('age_sq_se', 'N/A')})")
    print(f"  Age: {results['age']:.4f} (SE: {results.get('age_se', 'N/A')})")
    print(f"  ln(Exp): {results['ln_exp']:.4f} (SE: {results.get('ln_exp_se', 'N/A')})")

    return results


def hausman_test(fe_results, re_results, period_name="Full Period"):
    """
    Perform Hausman test to compare FE and RE specifications.

    H0: RE is consistent and efficient (use RE)
    H1: RE is inconsistent, use FE

    If p < 0.05, reject H0 and use Fixed Effects.
    """
    print("\n--- Hausman Test ---")

    if not HAS_LINEARMODELS:
        print("  Cannot perform Hausman test without linearmodels package.")
        return None

    fe_model = fe_results.get('model_obj')
    re_model = re_results.get('model_obj')

    if fe_model is None or re_model is None:
        print("  Cannot perform Hausman test: model objects not available.")
        return None

    # Align parameters: only compare coefficients that exist in both models
    # (RE may have a constant, FE does not)
    common_params = fe_model.params.index.intersection(re_model.params.index)
    if len(common_params) == 0:
        print("  Cannot perform Hausman test: no common parameters between FE and RE.")
        return None

    # Get coefficient differences for common parameters only
    coef_diff = fe_model.params[common_params] - re_model.params[common_params]

    # Get variance difference for common parameters only
    # V(b_FE - b_RE) = V(b_FE) - V(b_RE) (under H0)
    fe_cov = fe_model.cov.loc[common_params, common_params]
    re_cov = re_model.cov.loc[common_params, common_params]
    var_diff = fe_cov - re_cov

    try:
        # Hausman statistic with numerical stability
        # Use pseudo-inverse if variance matrix is near-singular
        try:
            var_diff_inv = np.linalg.inv(var_diff)
        except np.linalg.LinAlgError:
            print("  Warning: Variance difference matrix is singular, using pseudo-inverse")
            var_diff_inv = np.linalg.pinv(var_diff)

        # Check for negative eigenvalues (can happen due to estimation uncertainty)
        eigenvalues = np.linalg.eigvalsh(var_diff)
        if np.any(eigenvalues < -1e-10):
            print(f"  Warning: Variance difference has negative eigenvalues (min={eigenvalues.min():.6f})")
            print("  This can occur when FE and RE are both inconsistent or sample is small")
            # Use absolute value of eigenvalues for stability
            var_diff_abs = var_diff @ var_diff.T
            var_diff_inv = np.linalg.pinv(var_diff_abs)

        chi2 = float(coef_diff @ var_diff_inv @ coef_diff)

        # Ensure chi2 is non-negative (can be negative due to numerical issues)
        if chi2 < 0:
            print(f"  Warning: Negative chi² value ({chi2:.4f}), setting to 0")
            chi2 = 0.0

        df = len(coef_diff)
        p_value = 1 - stats.chi2.cdf(chi2, df)

        results = {
            'period': period_name,
            'chi2_statistic': chi2,
            'degrees_of_freedom': df,
            'p_value_nominal': p_value,
            'recommendation': 'Fixed Effects' if p_value < 0.05 else 'Random Effects',
            'note': 'Diagnostic only; p-value assumes homoskedastic errors and is not valid under clustering'
        }

        print(f"  Chi² statistic: {chi2:.3f}")
        print(f"  Degrees of freedom: {df}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Recommendation: {results['recommendation']}")

        # Save Hausman test results to CSV (for complete reporting)
        hausman_df = pd.DataFrame([results])
        hausman_file = os.path.join(OUTPUT_TABLES, 'hausman_test_results.csv')
        # Append if file exists (for multiple periods)
        if os.path.exists(hausman_file):
            existing = pd.read_csv(hausman_file)
            hausman_df = pd.concat([existing, hausman_df], ignore_index=True)
        hausman_df.to_csv(hausman_file, index=False)
        print(f"  Saved to: {hausman_file}")

        return results
    except Exception as e:
        print(f"  Hausman test failed: {e}")
        return None


def correlated_random_effects(panel, period_name="Full Period"):
    """
    Correlated Random Effects (Mundlak) model.

    This augments the RE model with household means of time-varying regressors,
    providing a formal test of the RE orthogonality assumption under clustering.

    The Mundlak approach adds \bar{X}_i to the RE model:
        Y_it = X_it'β + \bar{X}_i'γ + α_i + ε_it

    If γ ≠ 0, the RE assumption is violated (regressors are correlated with
    household effects). This is a cluster-robust alternative to the Hausman test.

    NOTE: Uses LAGGED ln(spending) to avoid mechanical endogeneity.
    """
    print(f"\n--- Correlated Random Effects / Mundlak ({period_name}) ---")

    # Use lag_real_total if available
    if 'lag_real_total' in panel.columns:
        df = panel.dropna(subset=['pct_change_total', 'lag_real_total', 'age_int', 'wave']).copy()
        df = df[(df['age_int'] >= 60) & (df['age_int'] <= 90)]
        df['ln_spending'] = np.log(df['lag_real_total'])  # LAGGED
    else:
        df = panel.dropna(subset=['pct_change_total', 'real_total', 'age_int', 'wave']).copy()
        df = df[(df['age_int'] >= 60) & (df['age_int'] <= 90)]
        df['ln_spending'] = np.log(df['real_total'])
    df['age_sq'] = df['age_int'] ** 2

    # Calculate household means of time-varying regressors (Mundlak terms)
    df['age_sq_mean'] = df.groupby('hhidpn')['age_sq'].transform('mean')
    df['age_int_mean'] = df.groupby('hhidpn')['age_int'].transform('mean')
    df['ln_spending_mean'] = df.groupby('hhidpn')['ln_spending'].transform('mean')

    if HAS_LINEARMODELS:
        df_indexed = df.set_index(['hhidpn', 'wave'])

        # Random effects model with Mundlak terms
        import statsmodels.api as sm_const
        exog_vars = ['age_sq', 'age_int', 'ln_spending',
                     'age_sq_mean', 'age_int_mean', 'ln_spending_mean']
        exog = sm_const.add_constant(df_indexed[exog_vars])
        mod = RandomEffects(
            df_indexed['pct_change_total'],
            exog
        )
        cre_results = mod.fit(cov_type='clustered', cluster_entity=True)

        results = {
            'method': 'Correlated RE (Mundlak)',
            'period': period_name,
            'age_sq': cre_results.params['age_sq'],
            'age': cre_results.params['age_int'],
            'ln_exp': cre_results.params['ln_spending'],
            'age_sq_mean': cre_results.params['age_sq_mean'],
            'age_mean': cre_results.params['age_int_mean'],
            'ln_exp_mean': cre_results.params['ln_spending_mean'],
            'r2_between': cre_results.rsquared_between,
            'r2_within': cre_results.rsquared_within,
            'n_obs': cre_results.nobs,
            'n_households': len(df_indexed.index.get_level_values(0).unique()),
            'age_sq_se': cre_results.std_errors['age_sq'],
            'age_se': cre_results.std_errors['age_int'],
            'ln_exp_se': cre_results.std_errors['ln_spending'],
            'age_sq_mean_se': cre_results.std_errors['age_sq_mean'],
            'age_mean_se': cre_results.std_errors['age_int_mean'],
            'ln_exp_mean_se': cre_results.std_errors['ln_spending_mean'],
            'model_obj': cre_results
        }

        print(f"  Within-household coefficients (β):")
        print(f"    Age²: {results['age_sq']:.6f} (SE: {results['age_sq_se']:.6f})")
        print(f"    Age: {results['age']:.4f} (SE: {results['age_se']:.4f})")
        print(f"    ln(Exp): {results['ln_exp']:.4f} (SE: {results['ln_exp_se']:.4f})")
        print(f"  Household-mean coefficients (γ - Mundlak terms):")
        print(f"    Age² mean: {results['age_sq_mean']:.6f} (SE: {results['age_sq_mean_se']:.6f})")
        print(f"    Age mean: {results['age_mean']:.4f} (SE: {results['age_mean_se']:.4f})")
        print(f"    ln(Exp) mean: {results['ln_exp_mean']:.4f} (SE: {results['ln_exp_mean_se']:.4f})")

        # Test if Mundlak terms are jointly significant (implies RE violated)
        mundlak_coefs = [results['age_sq_mean'], results['age_mean'], results['ln_exp_mean']]
        mundlak_nonzero = any(abs(c) > 2 * results.get(f'{name}_mean_se', 0.001)
                              for c, name in zip(mundlak_coefs, ['age_sq', 'age', 'ln_exp']))
        if mundlak_nonzero:
            print("  Note: Mundlak terms appear significant → RE assumption likely violated")
        else:
            print("  Note: Mundlak terms not clearly significant")

    else:
        print("  Cannot estimate CRE model without linearmodels package.")
        results = {
            'method': 'Correlated RE (Mundlak)',
            'period': period_name,
            'age_sq': np.nan,
            'age': np.nan,
            'ln_exp': np.nan,
            'n_obs': len(df),
            'n_households': df['hhidpn'].nunique(),
            'model_obj': None
        }

    return results


def period_analysis(panel):
    """
    Analyze spending patterns by period:
    - 2001-2009 (Blanchett's period)
    - 2005-2021 (Excluding early CAMS waves - robustness check)
    - 2011-2021 (Extension period)
    - Full 2001-2021

    For each period, we run:
    - Blanchett-style aggregation
    - Pooled OLS
    - Fixed Effects (without time effects - for Hausman comparison)
    - Fixed Effects (with time effects - preferred specification)
    - Random Effects
    - Correlated Random Effects (Mundlak) - for 2001-2021 only
    - Hausman test (FE no time vs RE, for valid comparison)
    """
    results = []

    # Define periods
    # NOTE: 2005-2021 excludes early CAMS waves (2001, 2003) which had fewer
    # spending categories and may not be strictly comparable to later waves.
    periods = {
        '2001-2009': panel[panel['year'] <= 2009],
        '2005-2021': panel[panel['year'] >= 2005],  # Robustness: drop early waves
        '2011-2021': panel[panel['year'] >= 2011],
        '2001-2021': panel
    }

    for period_name, period_data in periods.items():
        print(f"\n{'='*70}")
        print(f"PERIOD: {period_name}")
        print(f"{'='*70}")

        if len(period_data) < 100:
            print(f"  Insufficient data for {period_name}")
            continue

        # Blanchett-style
        blanchett_res, _ = blanchett_style_estimation(period_data, period_name)
        results.append(blanchett_res)

        # Pooled OLS
        pooled_res = panel_ols_pooled(period_data, period_name)
        results.append(pooled_res)

        # Fixed Effects WITHOUT time effects (for Hausman comparison with RE)
        fe_res_no_time = panel_fixed_effects(period_data, period_name, time_effects=False)
        results.append(fe_res_no_time)

        # Random Effects (no time effects by default in linearmodels)
        re_res = panel_random_effects(period_data, period_name)
        results.append(re_res)

        # Hausman test: Compare FE (no time) vs RE for valid comparison
        print("\n--- Hausman Test (FE no time vs RE - valid comparison) ---")
        hausman = hausman_test(fe_res_no_time, re_res, period_name=period_name)

        # Fixed Effects WITH time effects
        # NOTE: With FE + time effects, age is not identified separately from period
        # (APC problem). We include this for ln(Spending) sensitivity only.
        fe_res_with_time = panel_fixed_effects(period_data, period_name, time_effects=True)
        results.append(fe_res_with_time)

        print("\nNote: FE with time effects cannot identify age separately from period")
        print("      (APC problem). Use FE without time effects for age-profile inference.")

        # Correlated Random Effects (Mundlak) - only for full period
        if period_name == '2001-2021':
            cre_res = correlated_random_effects(period_data, period_name)
            results.append(cre_res)

    return results


def create_comparison_figure(results_df, age_means_dict):
    """Create figure comparing methods and periods."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel A: Age coefficient comparison
    ax = axes[0, 0]
    methods = results_df['method'].unique()
    x = np.arange(len(methods))
    width = 0.25

    plot_periods = ['2001-2009', '2005-2021', '2001-2021']
    width = 0.25
    for i, period in enumerate(plot_periods):
        period_data = results_df[results_df['period'] == period]
        if len(period_data) > 0:
            values = [period_data[period_data['method'] == m]['age_sq'].values[0]
                     if len(period_data[period_data['method'] == m]) > 0 else 0
                     for m in methods]
            ax.bar(x + i*width, values, width, label=period)

    ax.axhline(y=BLANCHETT['age_sq'], color='red', linestyle='--', label='Blanchett (2014)')
    ax.set_ylabel('Age² Coefficient')
    ax.set_title('A. Age² Coefficient by Method and Period')
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()

    # Panel B: ln(Spending) coefficient comparison
    ax = axes[0, 1]
    for i, period in enumerate(plot_periods):
        period_data = results_df[results_df['period'] == period]
        if len(period_data) > 0:
            values = [period_data[period_data['method'] == m]['ln_exp'].values[0]
                     if len(period_data[period_data['method'] == m]) > 0 else 0
                     for m in methods]
            ax.bar(x + i*width, values, width, label=period)

    ax.axhline(y=BLANCHETT['ln_exp'], color='red', linestyle='--', label='Blanchett (2014)')
    ax.set_ylabel('ln(Spending) Coefficient')
    ax.set_title('B. ln(Spending) Coefficient by Method and Period')
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()

    # Panel C: Spending curves by period (Blanchett-style)
    ax = axes[1, 0]
    ages = np.arange(60, 91)
    avg_spend = 50000  # Reference spending level (same as REFERENCE_SPENDING in 08)

    # Blanchett's curve
    blanchett_curve = [BLANCHETT['age_sq']*a**2 + BLANCHETT['age']*a +
                       BLANCHETT['ln_exp']*np.log(avg_spend) + BLANCHETT['constant']
                       for a in ages]
    ax.plot(ages, np.array(blanchett_curve)*100, 'r--', linewidth=2, label='Blanchett (2014)')

    # Our curves by period
    colors = {'2001-2009': 'blue', '2005-2021': 'orange', '2011-2021': 'green', '2001-2021': 'purple'}
    for period in ['2001-2009', '2005-2021', '2001-2021']:
        period_data = results_df[(results_df['period'] == period) &
                                 (results_df['method'] == 'Blanchett-style')]
        if len(period_data) > 0:
            row = period_data.iloc[0]
            curve = [row['age_sq']*a**2 + row['age']*a +
                    row['ln_exp']*np.log(avg_spend) + row['constant']
                    for a in ages]
            ax.plot(ages, np.array(curve)*100, color=colors[period], linewidth=2, label=period)

    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Spending Change (%)')
    ax.set_title('C. Spending "Smile" by Period (Blanchett Method)')
    ax.legend()
    ax.set_xlim(60, 90)

    # Panel D: Age polynomial comparison (excluding ln(spending) to avoid scale issues)
    # NOTE: Per reviewer feedback, we plot only the age component to enable fair comparison
    # across methods. FE absorbs household-level variation, making full prediction misleading.
    ax = axes[1, 1]

    # Compare ONLY the age polynomial component across methods for full period
    full_blanchett = results_df[(results_df['period'] == '2001-2021') &
                                (results_df['method'] == 'Blanchett-style')]
    full_pooled = results_df[(results_df['period'] == '2001-2021') &
                             (results_df['method'] == 'Pooled OLS')]
    # Handle various FE method names
    fe_methods = ['Fixed Effects (time)', 'Fixed Effects', 'Fixed Effects (approx)']
    full_fe = results_df[(results_df['period'] == '2001-2021') &
                         (results_df['method'].isin(fe_methods))]
    full_re = results_df[(results_df['period'] == '2001-2021') &
                         (results_df['method'].str.contains('Random'))]

    # Plot AGE POLYNOMIAL ONLY (no ln(spending) term) - normalized to mean of reference
    ref_age = 72  # Reference age for normalization

    if len(full_blanchett) > 0:
        row = full_blanchett.iloc[0]
        # Age polynomial only, normalized to zero at reference age
        curve_agg = [(row['age_sq']*a**2 + row['age']*a) -
                     (row['age_sq']*ref_age**2 + row['age']*ref_age)
                     for a in ages]
        ax.plot(ages, np.array(curve_agg)*100, 'b-', linewidth=2, label='Blanchett-style')

    if len(full_pooled) > 0:
        row = full_pooled.iloc[0]
        curve_pooled = [(row['age_sq']*a**2 + row['age']*a) -
                        (row['age_sq']*ref_age**2 + row['age']*ref_age)
                        for a in ages]
        ax.plot(ages, np.array(curve_pooled)*100, 'c--', linewidth=2, label='Pooled OLS')

    if len(full_fe) > 0:
        row = full_fe.iloc[0]
        curve_fe = [(row['age_sq']*a**2 + row['age']*a) -
                    (row['age_sq']*ref_age**2 + row['age']*ref_age)
                    for a in ages]
        ax.plot(ages, np.array(curve_fe)*100, 'g-', linewidth=2, label='Fixed Effects')

    if len(full_re) > 0:
        row = full_re.iloc[0]
        curve_re = [(row['age_sq']*a**2 + row['age']*a) -
                    (row['age_sq']*ref_age**2 + row['age']*ref_age)
                    for a in ages]
        ax.plot(ages, np.array(curve_re)*100, 'm:', linewidth=2, label='Random Effects')

    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Age')
    ax.set_ylabel('Age Effect on Spending Change (%)\n(relative to age 72)')
    ax.set_title('D. Age Polynomial Comparison (2001-2021)')
    ax.legend(loc='upper left')
    ax.set_xlim(60, 90)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURES, 'rigorous_extension_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to: {OUTPUT_FIGURES}/rigorous_extension_comparison.png")


def main():
    """Run rigorous extension analysis."""
    print("="*70)
    print("RIGOROUS EXTENSION ANALYSIS")
    print("="*70)

    # Clear Hausman results file at start (ensures idempotent outputs)
    hausman_file = os.path.join(OUTPUT_TABLES, 'hausman_test_results.csv')
    if os.path.exists(hausman_file):
        os.remove(hausman_file)

    # Load panel - exit gracefully if data not available
    try:
        panel = load_panel()
    except FileNotFoundError as e:
        print(f"\nWARNING: {e}")
        print("\nExtension analysis requires the 2001-2021 panel dataset.")
        print("This is optional - the core replication runs without it.")
        print("\nTo run extension analysis:")
        print("  1. Ensure you have RAND HRS and CAMS data files")
        print("  2. Run: python 01_build_panel.py")
        print("\nSkipping extension analysis.")
        return 0  # Exit successfully - extension is optional

    # Run analysis by period
    results = period_analysis(panel)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    output_path = os.path.join(OUTPUT_TABLES, 'extension_panel_models.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Create comparison figure
    # Get age means for plotting (from full period)
    _, age_means = blanchett_style_estimation(panel, "Full Period (for plotting)")
    create_comparison_figure(results_df, {'2001-2021': age_means})

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    # Compare Age² coefficient across methods
    print("\nAge² Coefficient Comparison (2001-2021):")
    full_period = results_df[results_df['period'] == '2001-2021']
    for _, row in full_period.iterrows():
        print(f"  {row['method']}: {row['age_sq']:.6f}")
    print(f"  Blanchett (2014): {BLANCHETT['age_sq']:.6f}")

    # Key insight: Does aggregation vs micro-level make a difference?
    agg_age_sq = full_period[full_period['method'] == 'Blanchett-style']['age_sq'].values[0]
    # Use FE with time effects as the preferred specification
    fe_methods = ['Fixed Effects (time)', 'Fixed Effects']
    fe_age_sq = None
    for fe_method in fe_methods:
        if fe_method in full_period['method'].values:
            fe_age_sq = full_period[full_period['method'] == fe_method]['age_sq'].values[0]
            break

    print("\n" + "="*70)
    print("METHODOLOGICAL INSIGHT")
    print("="*70)
    if fe_age_sq is not None:
        if np.sign(agg_age_sq) != np.sign(fe_age_sq):
            print("Note: The 'smile' curvature (Age²) has DIFFERENT SIGNS")
            print("         between aggregation and micro-level panel models!")
            print("         This suggests the 'smile' may be an aggregation artifact.")
        elif abs(agg_age_sq - fe_age_sq) / abs(agg_age_sq) > 0.5:
            print("IMPORTANT: The 'smile' curvature (Age²) differs substantially")
            print("          between aggregation and micro-level panel models.")
            print(f"          Aggregation: {agg_age_sq:.6f}")
            print(f"          Fixed Effects: {fe_age_sq:.6f}")
        else:
            print("The 'smile' curvature is similar between methods.")

    return results_df


if __name__ == "__main__":
    import sys
    results = main()
    # main() returns 0 (int) if optional data missing, or DataFrame if successful
    if isinstance(results, int):
        sys.exit(results)
    sys.exit(0)

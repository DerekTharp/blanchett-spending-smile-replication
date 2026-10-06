#!/usr/bin/env python3
"""
08_generate_figures.py
======================
Generate manuscript figures.

Reads this study's estimated coefficients from pre-computed CSV tables and
combines them with Blanchett's published coefficients for comparison figures:
- Figure 1: Replication of Blanchett's spending smile
- Figure 2: ln(Spending) definition sensitivity
- Figure 3: Panel model comparison
- Figure 4: Bootstrap confidence intervals

All estimated coefficients from this study are read from tables/*.csv files.
Blanchett's published coefficients and the $50,000 scenario baseline are
fixed comparison inputs defined in this script.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
# Suppress dependency warnings (matplotlib, pandas) that clutter pipeline output.
# Analysis-level warnings are not suppressed.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # peer_review/
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # Blanchett Smile Replication/
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
TABLES_DIR = os.path.join(BASE_DIR, "tables")
OUTPUT_FIGURES = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_FIGURES, exist_ok=True)

# Blanchett's published coefficients
BLANCHETT = {
    'age_sq': 0.00008,
    'age': -0.0125,
    'ln_exp': -0.0066,
    'constant': 0.546
}

# Reference spending level for projections and curve comparisons ($50,000 at age 60/65)
REFERENCE_SPENDING = 50000

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'blanchett': '#E74C3C',  # Red
    'replication': '#3498DB',  # Blue
    'extension': '#27AE60',  # Green
    'fe': '#9B59B6',  # Purple
    'lagged': '#27AE60',  # Green
    'current': '#E74C3C',  # Red
    'baseline': '#3498DB',  # Blue
    'mean': '#F39C12',  # Orange
}


def load_replication_results():
    """Load primary replication results from CSV."""
    path = os.path.join(TABLES_DIR, "replication_results.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Replication results not found: {path}")
    return pd.read_csv(path)


def load_sign_frequency():
    """Load sign frequency analysis from CSV."""
    path = os.path.join(TABLES_DIR, "sign_frequency_analysis.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sign frequency results not found: {path}")
    return pd.read_csv(path)


def load_extension_results():
    """Load extension panel model results from CSV."""
    path = os.path.join(TABLES_DIR, "extension_panel_models.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Extension results not found: {path}")
    return pd.read_csv(path)


def load_bootstrap_cis():
    """Load bootstrap confidence intervals from CSV."""
    path = os.path.join(TABLES_DIR, "bootstrap_coefficient_cis.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bootstrap CIs not found: {path}")
    return pd.read_csv(path)


def load_age_means_data():
    """Load pre-computed age-level means from the tables directory."""
    # Load pre-computed age means from 02_replication.py
    data_path = os.path.join(TABLES_DIR, "age_means_2001_2009.csv")

    if not os.path.exists(data_path):
        print(f"  Warning: Age means table not found at {data_path}")
        print("           Run 02_replication.py first to generate this table.")
        return None

    age_means = pd.read_csv(data_path)

    # Filter to ages with sufficient observations (consistent with replication)
    age_means = age_means[age_means['n'] >= 20].copy()

    # Rename for compatibility with figure code
    age_means = age_means.rename(columns={'mean_change': 'annual_change'})

    return age_means


def generate_figure1(output_path):
    """
    Figure 1: Replication of Blanchett's Spending Smile

    Panel A: Age-level means with fitted curves
    Panel B: Coefficient comparison (reading from replication_results.csv)
    """
    # Load replication results
    replication = load_replication_results()

    # Extract our coefficients from the CSV
    our_age_sq = float(replication[replication['Coefficient'] == 'age_sq']['This_Study'].values[0])
    our_age = float(replication[replication['Coefficient'] == 'age']['This_Study'].values[0])
    our_ln_exp = float(replication[replication['Coefficient'] == 'ln_exp']['This_Study'].values[0])
    our_constant = float(replication[replication['Coefficient'] == 'constant']['This_Study'].values[0])

    print(f"  Coefficients from CSV: Age²={our_age_sq:.6f}, Age={our_age:.4f}, ln(Spending)={our_ln_exp:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Curves comparison
    ax1 = axes[0]

    age_smooth = np.linspace(60, 85, 100)
    ln_spending = np.log(REFERENCE_SPENDING)

    # Blanchett's curve
    y_blanchett = (BLANCHETT['age_sq'] * age_smooth**2 + BLANCHETT['age'] * age_smooth +
                   BLANCHETT['ln_exp'] * ln_spending + BLANCHETT['constant']) * 100

    # Our curve (using coefficients from CSV)
    y_ours = (our_age_sq * age_smooth**2 + our_age * age_smooth +
              our_ln_exp * ln_spending + our_constant) * 100

    ax1.plot(age_smooth, y_blanchett, '--', color=COLORS['blanchett'],
             linewidth=2.5, label='Blanchett (2014)')
    ax1.plot(age_smooth, y_ours, '-', color=COLORS['replication'],
             linewidth=2.5, label='This Study')

    # Try to add age-level means as scatter (from pre-computed table)
    age_means = load_age_means_data()
    if age_means is not None:
        sizes = np.sqrt(age_means['n']) * 5
        ax1.scatter(age_means['age'], age_means['annual_change'] * 100,
                    s=sizes, alpha=0.4, c=COLORS['replication'],
                    label='Age means (size = sqrt(n))')

    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Age', fontsize=11)
    ax1.set_ylabel('Annual Spending Change (%)', fontsize=11)
    ax1.set_title('A. Spending Change Curves', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(59, 86)
    y_plotted = np.concatenate([y_blanchett, y_ours])
    if age_means is not None:
        y_plotted = np.concatenate([y_plotted, age_means['annual_change'].values * 100])
    pad = 0.05 * (y_plotted.max() - y_plotted.min())
    ax1.set_ylim(y_plotted.min() - pad, y_plotted.max() + pad)

    # Panel B: Coefficient comparison bar chart
    ax2 = axes[1]

    width = 0.35

    # Age and ln(Spending) share the primary (left) axis; Age² is ~1000x smaller,
    # so it gets a secondary (right) axis to make its magnitude visible.
    main_b = [BLANCHETT['age'], BLANCHETT['ln_exp']]
    main_o = [our_age, our_ln_exp]
    xm = np.array([0.0, 1.0])
    bars1 = ax2.bar(xm - width/2, main_b, width, label='Blanchett (2014)',
                    color=COLORS['blanchett'], alpha=0.8)
    bars2 = ax2.bar(xm + width/2, main_o, width, label='This Study',
                    color=COLORS['replication'], alpha=0.8)
    ax2.set_ylabel('Coefficient value: Age, ln(Spending)', fontsize=10)
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    ax2b = ax2.twinx()
    xs = np.array([2.0])
    bars3 = ax2b.bar(xs - width/2, [BLANCHETT['age_sq']], width,
                     color=COLORS['blanchett'], alpha=0.8)
    bars4 = ax2b.bar(xs + width/2, [our_age_sq], width,
                     color=COLORS['replication'], alpha=0.8)
    agesq_max = max(BLANCHETT['age_sq'], our_age_sq)
    ax2b.set_ylim(-0.4 * agesq_max, 1.5 * agesq_max)
    ax2b.set_ylabel('Age² coefficient (right axis)', fontsize=10, color='#555555')
    ax2b.tick_params(axis='y', labelcolor='#555555')
    ax2b.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    ax2.set_title('B. Coefficient Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['Age', 'ln(Spending)', 'Age²\n(right axis)'], fontsize=9)
    ax2.legend(loc='lower left', fontsize=9)

    def _label(ax, bars, sci=False):
        for bar in bars:
            h = bar.get_height()
            txt = f'{h:.1e}' if sci else f'{h:.4f}'
            ax.annotate(txt, xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3 if h >= 0 else -12), textcoords="offset points",
                        ha='center', va='bottom' if h >= 0 else 'top', fontsize=8)

    _label(ax2, bars1)
    _label(ax2, bars2)
    _label(ax2b, bars3, sci=True)
    _label(ax2b, bars4, sci=True)
    ax2.set_ylim(min(main_b + main_o) * 1.2, 0.0005)  # headroom for below-bar labels

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_figure2(output_path):
    """
    Figure 2: Sensitivity to ln(Spending) Definition

    Shows how coefficient sign depends on spending measure definition.
    Reads from sign_frequency_analysis.csv.
    """
    # Load sign frequency data
    sign_freq = load_sign_frequency()

    fig, ax1 = plt.subplots(1, 1, figsize=(5, 4))

    # Bar chart of coefficients by definition

    # Get mean coefficients for each definition from sign_frequency
    definitions = ['lagged', 'current', 'baseline', 'mean']
    labels = ['Lagged', 'Current', 'Baseline', 'Mean']

    values = []
    for defn in definitions:
        row = sign_freq[sign_freq['ln_spending_def'] == defn]
        if len(row) > 0:
            values.append(row['ln_exp_mean'].values[0])
        else:
            values.append(0)

    colors = [COLORS['lagged'] if v < 0 else COLORS['current'] for v in values]

    bars = ax1.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    # Add Blanchett reference line
    ax1.axhline(y=BLANCHETT['ln_exp'], color=COLORS['blanchett'], linestyle='--',
                linewidth=2, label=f"Blanchett (2014): {BLANCHETT['ln_exp']:.4f}")
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.7)

    ax1.set_ylabel('ln(Spending) Coefficient', fontsize=10)
    ax1.set_title('ln(Spending) Coefficient by Definition', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=9)
    ax1.legend(loc='lower left', fontsize=8, framealpha=0.9)

    # Add value labels - inside tall bars, below near-zero bars (a white in-bar
    # label on a near-zero bar would be invisible against the background)
    label_cutoff = 0.15 * max(abs(v) for v in values)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        if height >= 0:
            # Positive bars: label above
            ax1.annotate((f'{val:+.4f}' if abs(val) >= 0.0005 else f'{val:+.1e}'),
                         xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=9, fontweight='bold')
        elif abs(height) >= label_cutoff:
            # Negative bars: label inside the bar
            ax1.annotate((f'{val:+.4f}' if abs(val) >= 0.0005 else f'{val:+.1e}'),
                         xy=(bar.get_x() + bar.get_width()/2, height/2),
                         ha='center', va='center',
                         fontsize=8, fontweight='bold', color='white')
        else:
            # Near-zero bars: label below the bar in default text color
            ax1.annotate((f'{val:+.4f}' if abs(val) >= 0.0005 else f'{val:+.1e}'),
                         xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, -12),
                         textcoords="offset points",
                         ha='center', va='top',
                         fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_figure3(output_path):
    """
    Figure 3: Panel Model Comparison

    Panel A: Shows how smile curvature (Age² coefficient) changes with different
    econometric methods. Reads from extension_panel_models.csv.

    Panel B: Projects spending trajectories for a hypothetical retiree starting
    with $50,000 at age 60. Shows Blanchett-style models alongside constant
    decline scenarios (0%, 1%, 2%) to illustrate the range of assumptions
    practitioners might use based on the FE findings.

    Note: Colorblind-accessible design using distinct line styles, markers, and
    hatching patterns in addition to colors.
    """
    # Load extension results
    extension = load_extension_results()

    # Panel A uses 2001-2021 to show smile attenuation in FE (key finding)
    # Panel B uses 2001-2009 to match primary replication period
    period_panel_a = extension[extension['period'] == '2001-2021'].copy()
    period_panel_b = extension[extension['period'] == '2001-2009'].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Age² coefficients by method (2001-2021)
    ax1 = axes[0]

    # Four-method comparison; FE+Time curvature is reported in Table 8.
    methods_order = ['Blanchett-style', 'Pooled OLS', 'Fixed Effects', 'Random Effects']
    method_labels = ['Blanchett-\nstyle', 'Pooled\nOLS', 'Fixed\nEffects', 'Random\nEffects']

    age_sq_vals = []
    for method in methods_order:
        row = period_panel_a[period_panel_a['method'] == method]
        if len(row) > 0:
            val = row['age_sq'].values[0]
            # Handle NaN or missing values
            if pd.isna(val):
                age_sq_vals.append(0)
            else:
                age_sq_vals.append(val)
        else:
            age_sq_vals.append(0)

    # Colorblind-accessible: use hatching patterns in addition to colors
    # Blanchett-style = red with diagonal hatching
    # Pooled OLS = blue with horizontal hatching
    # Fixed Effects = green with cross hatching
    # Random Effects = purple with dots
    hatches = ['///', '---', 'xxx', '...']
    colors_panel_a = [COLORS['blanchett'], COLORS['replication'], COLORS['extension'], '#9B59B6']

    bars1 = ax1.bar(method_labels, age_sq_vals, color=colors_panel_a, alpha=0.7,
                    edgecolor='black', linewidth=1.5)

    # Add hatching patterns for colorblind accessibility
    for bar, hatch in zip(bars1, hatches):
        bar.set_hatch(hatch)

    ax1.axhline(y=BLANCHETT['age_sq'], color=COLORS['blanchett'], linestyle='--',
                linewidth=2, label=f"Blanchett: {BLANCHETT['age_sq']:.6f}")
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.7)

    # Add value labels above each bar for clarity
    for bar, val in zip(bars1, age_sq_vals):
        height = bar.get_height()
        ax1.annotate(f'{val:.1e}',
                     xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom',
                     fontsize=8, fontweight='bold')

    ax1.set_ylabel('Age² Coefficient', fontsize=11)
    ax1.set_title('A. Smile Curvature (Age²) by Method', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.tick_params(axis='x', labelsize=9)

    # Panel B: Projected Spending Trajectories with Constant Decline Scenarios
    # Shows Blanchett-style models alongside 0%, 1%, 2% constant decline scenarios
    # to illustrate the range of assumptions practitioners might use
    ax2 = axes[1]

    # Starting conditions
    start_age = 60
    end_age = 90
    start_spending = REFERENCE_SPENDING

    ages = np.arange(start_age, end_age + 1)

    def project_spending(age_sq_coef, age_coef, ln_spending_coef, constant, start_spending, ages):
        """Project spending trajectory with dynamic ln(Spending) updates."""
        spending = np.zeros(len(ages))
        spending[0] = start_spending

        for i in range(1, len(ages)):
            age = ages[i]
            ln_spend = np.log(spending[i-1])

            # Annual change rate from model
            annual_change = (age_sq_coef * age**2 + age_coef * age +
                           ln_spending_coef * ln_spend + constant)

            # Apply change to get new spending
            spending[i] = spending[i-1] * (1 + annual_change)

        return spending

    # Blanchett (2014) projection - dashed line with triangle markers
    spending_blanchett = project_spending(
        BLANCHETT['age_sq'], BLANCHETT['age'], BLANCHETT['ln_exp'], BLANCHETT['constant'],
        start_spending, ages
    )
    ax2.plot(ages, spending_blanchett / 1000, '--', color=COLORS['blanchett'], linewidth=2.5,
             marker='^', markevery=5, markersize=6, label='Blanchett (2014)')

    # Our replication projection (Table 5 coefficients) - solid line with circle markers
    replication = load_replication_results()
    our_age_sq = float(replication[replication['Coefficient'] == 'age_sq']['This_Study'].values[0])
    our_age = float(replication[replication['Coefficient'] == 'age']['This_Study'].values[0])
    our_ln_exp = float(replication[replication['Coefficient'] == 'ln_exp']['This_Study'].values[0])
    our_constant = float(replication[replication['Coefficient'] == 'constant']['This_Study'].values[0])
    spending_our = project_spending(
        our_age_sq, our_age, our_ln_exp, our_constant,
        start_spending, ages
    )
    ax2.plot(ages, spending_our / 1000, '-', color=COLORS['replication'], linewidth=2.5,
             marker='o', markevery=5, markersize=6, label='This study (Blanchett-style)')

    # Constant decline scenarios (0%, 1%, 2%) - dotted lines with distinct patterns
    spending_0pct = start_spending * (1.00 ** np.arange(len(ages)))
    spending_1pct = start_spending * (0.99 ** np.arange(len(ages)))
    spending_2pct = start_spending * (0.98 ** np.arange(len(ages)))

    # Use different dash patterns for colorblind accessibility
    ax2.plot(ages, spending_0pct / 1000, ':', color='#555555', linewidth=1.5, alpha=0.9,
             marker='s', markevery=10, markersize=4, label='0% constant decline')
    ax2.plot(ages, spending_1pct / 1000, '-.', color='#333333', linewidth=2,
             marker='d', markevery=10, markersize=4, label='1% constant decline')
    ax2.plot(ages, spending_2pct / 1000, '--', color='#555555', linewidth=1.5, alpha=0.9,
             marker='v', markevery=10, markersize=4, label='2% constant decline')

    # Add shaded region for 0-2% range
    ax2.fill_between(ages, spending_0pct / 1000, spending_2pct / 1000, color='gray', alpha=0.15,
                     label='_nolegend_')  # Hide from legend

    ax2.set_xlabel('Age', fontsize=11)
    ax2.set_ylabel('Annual Spending ($000s)', fontsize=11)
    ax2.set_title('B. Projected Spending Trajectories', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlim(59, 91)
    ax2.set_ylim(25, 55)

    # Add note explaining setup
    ax2.text(0.02, 0.02, 'Starting: $50,000 at age 60\nShaded: 0-2% constant decline range',
             transform=ax2.transAxes, fontsize=8, va='bottom', ha='left',
             style='italic', alpha=0.7,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_figure4(output_path):
    """
    Figure 4: Bootstrap Confidence Intervals

    Shows uncertainty in key estimates.
    Reads from bootstrap_coefficient_cis.csv.
    """
    # Load bootstrap results
    boot_df = load_bootstrap_cis()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    # Panel A: Age² CI on its own scale (the central curvature parameter)
    ax0 = axes[0]
    agesq_data = boot_df[boot_df['coefficient'] == 'Age²']
    if len(agesq_data) == 0:
        agesq_data = boot_df[boot_df['coefficient'].str.contains('Age', na=False) &
                             boot_df['coefficient'].str.contains('2|²', na=False)]
    if len(agesq_data) > 0:
        row = agesq_data.iloc[0]
        est, ci_lo, ci_hi = row['estimate'], row['ci_lower'], row['ci_upper']
        color = COLORS['replication']  # all points are This Study's estimates
        ax0.errorbar(est, 0, xerr=[[est - ci_lo], [ci_hi - est]],
                     fmt='o', color=color, capsize=5, capthick=2, markersize=8)
        ax0.axvline(x=0, color='gray', linestyle=':', alpha=0.7)
        ax0.set_yticks([0])
        ax0.set_yticklabels(['Age²'])
        ax0.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax0.set_xlabel('Coefficient Value', fontsize=11)
        ax0.set_title('A. Age-squared 95% Bootstrap CI', fontsize=12, fontweight='bold')
        ax0.text(0.02, 0.98, 'CI straddles zero:\ncurvature not\nstatistically detectable',
                 transform=ax0.transAxes, fontsize=8, va='top', style='italic')
    else:
        ax0.text(0.5, 0.5, 'Age-squared CI not available', ha='center', va='center',
                 transform=ax0.transAxes, fontsize=12)

    # Panel B: Age & ln(Spending) CIs (shared scale)
    ax1 = axes[1]

    coef_names = ['Age', 'ln(Spending)']
    coef_data = boot_df[boot_df['coefficient'].isin(coef_names)]

    if len(coef_data) == 0:
        print("  Warning: No coefficient data in bootstrap results")
        ax1.text(0.5, 0.5, 'Bootstrap data not available', ha='center', va='center',
                 transform=ax1.transAxes, fontsize=12)
    else:
        y_pos = np.arange(len(coef_data))

        for i, (_, row) in enumerate(coef_data.iterrows()):
            est = row['estimate']
            ci_lo = row['ci_lower']
            ci_hi = row['ci_upper']

            color = COLORS['replication']  # all points are This Study's estimates

            ax1.errorbar(est, i, xerr=[[est - ci_lo], [ci_hi - est]],
                         fmt='o', color=color, capsize=5, capthick=2, markersize=8)

        ax1.axvline(x=0, color='gray', linestyle=':', alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(coef_data['coefficient'].values)
        ax1.set_xlabel('Coefficient Value', fontsize=11)
        ax1.set_title('B. Age & ln(Spending) 95% Bootstrap CIs', fontsize=12, fontweight='bold')

    # Panel C: Turning point distribution
    ax2 = axes[2]

    tp_row = boot_df[boot_df['coefficient'] == 'Smile Minimum Age']
    if len(tp_row) > 0:
        tp_est = tp_row['estimate'].values[0]
        tp_lo = tp_row['ci_lower'].values[0]
        tp_hi = tp_row['ci_upper'].values[0]

        # Create a visual representation
        ax2.axvline(x=tp_est, color=COLORS['replication'], linewidth=3,
                    label=f'Point estimate: {tp_est:.1f}')
        ax2.axvspan(tp_lo, tp_hi, alpha=0.3, color=COLORS['replication'],
                    label=f'Conditional central 95% range: [{tp_lo:.1f}, {tp_hi:.1f}]')
        blanchett_tp = -BLANCHETT['age'] / (2 * BLANCHETT['age_sq'])  # = 78.125
        ax2.axvline(x=blanchett_tp, color=COLORS['blanchett'], linestyle='--', linewidth=2,
                    label=f'Blanchett (2014): {blanchett_tp:.1f}')

        ax2.set_xlim(70, 120)
        ax2.set_xlabel('Age', fontsize=11)
        ax2.set_title('C. Turning Point (Minimum of Fitted Change Rate)', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.set_yticks([])

        # Add annotation about the wide conditional range
        ax2.text(95, 0.5, 'Wide conditional range\nindicates high uncertainty about\nwhen the decline rate eases',
                 fontsize=10, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'Turning point data not available', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


def generate_projection_table(output_path):
    """
    Save the Section 6.3 spending projection milestones to CSV.

    Uses Table 5 coefficients (from replication_results.csv)
    and Blanchett's published coefficients to project spending from age 65
    to 95, recording milestone ages (65, 75, 85, 95).
    """
    replication = load_replication_results()

    our_age_sq = float(replication[replication['Coefficient'] == 'age_sq']['This_Study'].values[0])
    our_age = float(replication[replication['Coefficient'] == 'age']['This_Study'].values[0])
    our_ln_exp = float(replication[replication['Coefficient'] == 'ln_exp']['This_Study'].values[0])
    our_constant = float(replication[replication['Coefficient'] == 'constant']['This_Study'].values[0])

    def project_spending(age_sq_coef, age_coef, ln_spending_coef, constant, start_spending, ages):
        """Project spending trajectory with dynamic ln(Spending) updates."""
        spending = np.zeros(len(ages))
        spending[0] = start_spending
        for i in range(1, len(ages)):
            age = ages[i]
            ln_spend = np.log(spending[i-1])
            annual_change = (age_sq_coef * age**2 + age_coef * age +
                           ln_spending_coef * ln_spend + constant)
            spending[i] = spending[i-1] * (1 + annual_change)
        return spending

    start_spending = REFERENCE_SPENDING
    ages = np.arange(65, 96)
    milestones = [65, 75, 85, 95]

    spending_blanchett = project_spending(
        BLANCHETT['age_sq'], BLANCHETT['age'], BLANCHETT['ln_exp'], BLANCHETT['constant'],
        start_spending, ages
    )
    spending_ours = project_spending(
        our_age_sq, our_age, our_ln_exp, our_constant,
        start_spending, ages
    )

    rows = []
    for age in milestones:
        idx = age - 65
        b_val = round(spending_blanchett[idx] / 100) * 100  # Round to nearest $100
        o_val = round(spending_ours[idx] / 100) * 100
        rows.append({
            'age': age,
            'blanchett_spending': b_val,
            'this_study_spending': o_val,
            'difference': b_val - o_val
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved projection table: {output_path}")


def main():
    """Generate all figures from CSV tables."""
    print("=" * 70)
    print("GENERATING MANUSCRIPT FIGURES (from CSV tables)")
    print("=" * 70)

    # Verify table files exist
    print("\nVerifying table files...")
    required_files = [
        "replication_results.csv",
        "sign_frequency_analysis.csv",
        "extension_panel_models.csv",
        "bootstrap_coefficient_cis.csv"
    ]

    errors = []
    for f in required_files:
        path = os.path.join(TABLES_DIR, f)
        if os.path.exists(path):
            print(f"  Found: {f}")
        else:
            print(f"  MISSING: {f}")
            errors.append(f"Missing required file: {f}")

    # Generate figures
    print("\n" + "-" * 70)
    print("Generating Figure 1: Replication comparison...")
    try:
        generate_figure1(os.path.join(OUTPUT_FIGURES, 'figure1_replication.png'))
    except Exception as e:
        print(f"  ERROR: {e}")
        errors.append(f"Figure 1 failed: {e}")

    print("\n" + "-" * 70)
    print("Generating Figure 2: ln(Spending) sensitivity...")
    try:
        generate_figure2(os.path.join(OUTPUT_FIGURES, 'figure2_ln_spending.png'))
    except Exception as e:
        print(f"  ERROR: {e}")
        errors.append(f"Figure 2 failed: {e}")

    print("\n" + "-" * 70)
    print("Generating Figure 3: Panel model comparison...")
    try:
        generate_figure3(os.path.join(OUTPUT_FIGURES, 'figure3_panel_models.png'))
    except Exception as e:
        print(f"  ERROR: {e}")
        errors.append(f"Figure 3 failed: {e}")

    print("\n" + "-" * 70)
    print("Generating Figure 4: Bootstrap confidence intervals...")
    try:
        generate_figure4(os.path.join(OUTPUT_FIGURES, 'figure4_bootstrap_ci.png'))
    except Exception as e:
        print(f"  ERROR: {e}")
        errors.append(f"Figure 4 failed: {e}")

    print("\n" + "-" * 70)
    print("Generating projection summary table (Section 6.3)...")
    try:
        generate_projection_table(os.path.join(TABLES_DIR, 'projection_summary.csv'))
    except Exception as e:
        print(f"  ERROR: {e}")
        errors.append(f"Projection table failed: {e}")

    print("\n" + "=" * 70)
    if errors:
        print("FIGURE GENERATION COMPLETED WITH ERRORS")
        for err in errors:
            print(f"  - {err}")
    else:
        print("FIGURE GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nFigures saved to: {OUTPUT_FIGURES}")

    return len(errors) == 0  # Return True if no errors


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)

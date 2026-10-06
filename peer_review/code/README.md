# Blanchett (2014) Spending Smile Replication Code

This directory contains all analysis code for replicating and extending Blanchett's (2014) retirement spending smile analysis.

## Data Requirements

### Required Data Files

The analysis requires two RAND datasets, which must be obtained from the Health and Retirement Study (HRS):

1. **RAND HRS Longitudinal File**
   - File: `randhrs1992_2022v1.dta`
   - Location: `data/raw/randhrs1992_2022v1_STATA/`
   - Source: https://hrsdata.isr.umich.edu/

2. **RAND CAMS (Consumption and Activities Mail Survey)**
   - File: `randcams_2001_2021v1.dta`
   - Location: `data/raw/randcams_2001_2021v1/`
   - Source: https://hrsdata.isr.umich.edu/

### Data Access Instructions

1. Register for an account at https://hrsdata.isr.umich.edu/
2. Download the RAND HRS Longitudinal File (STATA format)
3. Download the RAND CAMS data file (STATA format)
4. Extract files to the locations specified above

**Note:** HRS data are free but require registration and agreement to terms of use.

## Data Layout

These scripts expect raw RAND files to live in a `data/` directory *next to* the `peer_review/` folder:

```
<project_root>/
  peer_review/
    code/
    tables/
    figures/
  data/
    raw/
      randhrs1992_2022v1_STATA/
        randhrs1992_2022v1.dta
      randcams_2001_2021v1/
        randcams_2001_2021v1.dta
    processed/
      panel_analysis_2001_2021.csv  (built by 01_build_panel.py)
```

## Directory Structure

```
peer_review/
├── code/
│   ├── README.md                         # This file
│   ├── requirements.txt                  # Python dependencies
│   ├── run_all.py                        # Master pipeline script
│   ├── 01_build_panel.py                 # Build panel from raw RAND data
│   ├── 02_replication.py                 # Primary replication
│   ├── 03_robustness_grid.py             # Robustness analysis
│   ├── 04_panel_extension.py             # Panel model extension
│   ├── 05_robustness_additions.py        # Additional robustness (DV form, panel support)
│   ├── 06_weighted_sensitivity.py        # Weighted sensitivity analysis
│   ├── 07_twoway_clustering.py           # Two-way clustering sensitivity
│   ├── 08_generate_figures.py            # Manuscript figures
│   ├── 09_additional_analyses.py         # Descriptive statistics, controls, stratification, survivorship, spline
│   └── validate_outputs.py               # Output-verification harness
├── tables/                               # Output CSV tables
└── figures/                              # Output PNG figures
```

## Installation

### Prerequisites

- Python 3.12 or higher (tested on Python 3.13)
- pip package manager

### Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (broad compatibility)
pip install -r requirements.txt

# Or install exact versions from a known-good run
pip install -r requirements-lock.txt
```

## Running the Analysis

### Full Pipeline

To reproduce all results:

```bash
cd peer_review/code
python run_all.py
```

Optional: set a per-stage timeout (seconds) via `FPR_REPLICATION_TIMEOUT`, e.g.:

```bash
FPR_REPLICATION_TIMEOUT=1800 python run_all.py
```

This runs all scripts in order and generates:

**Tables:**
- `tables/attrition_table.csv` - Sample construction
- `tables/replication_results.csv` - Main coefficient comparison
- `tables/bootstrap_coefficient_cis.csv` - 95% CIs from cluster bootstrap
- `tables/bootstrap_draw_composition.csv` - Turning-point status counts across bootstrap draws
- `tables/sign_frequency_analysis.csv` - ln(Spending) sign by definition
- `tables/full_robustness_variants.csv` - All 48 specification variants
- `tables/robustness_grid.csv` - Summary robustness grid
- `tables/extension_panel_models.csv` - Panel FE/RE/CRE model results
- `tables/hausman_test_results.csv` - FE vs RE specification diagnostic
- `tables/dv_robustness_comparison.csv` - Table S7: DV specification comparison
- `tables/panel_support_diagnostic.csv` - Table S8: Panel support statistics
- `tables/weighted_sensitivity.csv` - Table S6: Weighted vs unweighted results
- `tables/descriptive_statistics.csv` - Table S1: Descriptive statistics
- `tables/retirement_filter_sensitivity.csv` - Retired-only extension sensitivity (Section 3.4)
- `tables/specification_decomposition.csv` - Table 3: Specification decomposition (Section 5.7)
- `tables/fe_time_varying_controls.csv` - Table S2: Time-varying FE controls
- `tables/stratified_analysis.csv` - Table S3: Wealth and health stratification
- `tables/survivorship_robustness.csv` - Table S4: Survivorship restrictions
- `tables/spline_age_profile.csv` - Table S5: Spline vs quadratic profiles
- `tables/table4_symmetric_check.csv` - Section 5.2: symmetric lagged-vs-current timing check
- `tables/singleton_household_diagnostic.csv` - Table S8 prose: FE with/without single-interval households

**Figures:**
- `figures/figure1_replication.png` - Spending smile comparison with Blanchett
- `figures/figure2_ln_spending.png` - ln(Spending) definition sensitivity
- `figures/figure3_panel_models.png` - Panel model comparison
- `figures/figure4_bootstrap_ci.png` - Bootstrap confidence intervals
- `figures/figure5_spline_comparison.png` - Spline vs quadratic age profiles
- `figures/replication_strict_minobs20.png` - Replication detail
- `figures/extension_comparison.png` - Extension comparison

### Individual Scripts

Scripts can also be run individually:

```bash
# Build extension panel from raw RAND data (prerequisite for extension)
python 01_build_panel.py

# Primary replication (requires raw data)
python 02_replication.py

# Robustness grid
python 03_robustness_grid.py

# Panel model extension (requires processed panel data)
python 04_panel_extension.py

# Additional robustness checks
python 05_robustness_additions.py
python 06_weighted_sensitivity.py
python 07_twoway_clustering.py

# Generate manuscript figures (requires tables from above scripts)
python 08_generate_figures.py

# Additional analyses (Tables S1-S5, Table 1, Figure 4)
python 09_additional_analyses.py

# Verify the shipped output tables against the reported values
python validate_outputs.py
```

## Output Files

### Tables

| File | Manuscript | Description |
|------|------------|-------------|
| `attrition_table.csv` | Table 4 | Sample construction showing filter effects |
| `replication_results.csv` | Table 9 | Main coefficient comparison with Blanchett |
| `full_robustness_variants.csv` | Table 8 | All 48 specification variants |
| `extension_panel_models.csv` | Table 5 | Panel FE/RE/CRE model results |
| `dv_robustness_comparison.csv` | Table S7 | DV specification comparison |
| `panel_support_diagnostic.csv` | Table S8 | Panel support statistics |
| `weighted_sensitivity.csv` | Table S6 | Weighted vs unweighted sensitivity |
| `bootstrap_coefficient_cis.csv` | - | 95% CIs from cluster bootstrap |
| `sign_frequency_analysis.csv` | Table 6 | ln(Spending) sign by definition |
| `hausman_test_results.csv` | - | FE vs RE specification diagnostic |
| `robustness_grid.csv` | - | Summary robustness grid (see note below) |
| `outlier_filter_sensitivity.csv` | - | Outlier filter sensitivity (with/without) |
| `twoway_clustering_comparison.csv` | - | Two-way clustering SE comparison |
| `age_means_2001_2009.csv` | - | Age-level means for Figure 1 visualization |
| `change_filter_comparison.csv` | - | Annualized vs between-survey filter sample sizes |
| `projection_summary.csv` | - | Section 6.3 spending projection milestones |
| `descriptive_statistics.csv` | Table S1 | Descriptive statistics for the extension sample |
| `fe_time_varying_controls.csv` | Table S2 | FE models with time-varying controls |
| `stratified_analysis.csv` | Table S3 | Wealth and health subgroup FE estimates |
| `survivorship_robustness.csv` | Table S4 | Survivorship restrictions by interval count |
| `spline_age_profile.csv` | Table S5 | Spline vs quadratic age profiles |
| `table4_symmetric_check.csv` | - | Section 5.2 symmetric lagged-vs-current timing check |
| `singleton_household_diagnostic.csv` | - | Table S8 prose, FE estimates with/without single-interval households |
| `specification_decomposition.csv` | Table 3 | Section 5.7 pooled/FE x with/without ln(Spending) decomposition |
| `retirement_filter_sensitivity.csv` | - | Section 3.4 retirement-filter sensitivity |

**Note on `robustness_grid.csv` vs. `full_robustness_variants.csv`:** These two tables use different change-filter implementations. `full_robustness_variants.csv` (from `02_replication.py`) applies a *geometric annualized* change filter (|annualized change| < 50%), yielding 679 HH for the primary specification. `robustness_grid.csv` (from `03_robustness_grid.py`) applies a *raw two-year between-survey* change filter (|raw change| < 50%), which is a stricter adjacent-survey interpretation of Blanchett's wording, yielding ~325 HH for the strict retirement specification. This difference is intentional: the robustness grid explores sensitivity to this ambiguity in the original methodology.

### Figures

| File | Manuscript | Description |
|------|------------|-------------|
| `figure1_replication.png` | Figure 1 | Spending smile comparison with Blanchett |
| `figure2_ln_spending.png` | Figure 2 | ln(Spending) definition sensitivity |
| `figure3_panel_models.png` | Figure 3 | Panel model comparison |
| `figure4_bootstrap_ci.png` | Figure S1 | Bootstrap confidence intervals |
| `figure5_spline_comparison.png` | Figure 4 | Spline vs quadratic age profiles |

## Key Implementation Details

### Inflation Adjustment and Wealth Timing

All real dollar amounts use a 2009 base and [BLS CPI-U annual averages](https://www.bls.gov/regions/northeast/data/consumerpriceindex_us_table.htm). Spending (`CSTOT`) and consumption (`CCTOT`) use the odd-year CAMS CPI. Total household wealth (`ATOTB`) comes from the preceding HRS core wave and uses its nominal even-year CPI: H5 wealth uses 2000, H6 uses 2002, and so forth through H15 in 2020. This convention uses the nominal core-wave year rather than the individual's actual interview year.

`01_build_panel.py` stores `wealth_year` and `wealth_cpi` separately from the CAMS `year` and `cpi`, and calculates `real_wealth = total_wealth * CPI_U[2009] / wealth_cpi`. Wealth subgroups use each household's first nonmissing real wealth value in the processed panel; tercile cutoffs are computed over one baseline value per household.

### ln(Spending) Definition

The sign of the ln(Spending) coefficient depends critically on the timing definition:
- **Lagged** (`S_{t-1}`): Produces negative coefficient (predetermined, implementable input; couples with the DV through the denominator)
- **Current** (`S_t`): Produces positive coefficient (couples with the DV through the numerator; the mechanical derivatives are roughly equal-and-opposite to lagged)
- **Baseline** (first wave): Produces negative coefficient
- **Mean** (across waves): Produces near-zero coefficient

We use **lagged spending** as the primary specification.

### Panel Models

The extension analysis (2001-2021) includes:
- **Blanchett-style aggregation**: Replicates original cross-sectional methodology
- **Pooled OLS**: Ignores panel structure
- **Fixed Effects (FE)**: Controls for household heterogeneity
- **Random Effects (RE)**: Allows between-household variation
- **Correlated RE (Mundlak)**: Tests RE orthogonality assumption under clustering
- **2005+ Robustness**: Drops early CAMS waves (2001, 2003) that had fewer spending categories

### Estimator Choices

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Polynomial weighting | Weighted by √n | Standard for heteroscedastic errors |
| ln estimation | Conditional on age | Matches Blanchett's multivariate form |
| Min observations per age | 20 | Balance precision and coverage |
| Change filter | Annualized < 50% | Closer to Blanchett's N=591 |

## Troubleshooting

### "Data not found" errors

Ensure RAND data files are in the correct locations under `data/raw/`.

### "linearmodels not installed"

The extension analysis requires the `linearmodels` package:
```bash
pip install linearmodels
```

### Memory issues

The bootstrap procedure (1,000 replications) requires ~4GB RAM. Reduce `n_bootstrap` in the script if needed.

### Timeout issues

Set a longer per-stage timeout:
```bash
FPR_REPLICATION_TIMEOUT=3600 python run_all.py
```

## Citation

If you use this code, please cite:

Tharp, D. (2026). The Retirement Spending Smile Revisited: Cross-Sectional Patterns versus Within-Household Dynamics. Working paper.

## License

This code is provided for research and educational purposes. The underlying HRS/CAMS data are subject to their own terms of use.

## Contact

Derek Tharp, Ph.D., CFP®

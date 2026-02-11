# Replication and Extension of Blanchett's (2014) Retirement Spending Smile

Replication code and materials for:

> Tharp, D. (2026). The Retirement Spending Smile Revisited: Cross-Sectional Patterns versus Within-Household Dynamics. Working paper.

This study replicates and extends the analysis in:

> Blanchett, D. (2014). Exploring the Retirement Consumption Puzzle. *Journal of Financial Planning*, 27(5), 34-42.

---

## Key Findings

### 1. Sign Replication (2001-2009)

Using RAND HRS/CAMS data with a strict retirement filter and lagged ln(Spending), all coefficient signs match Blanchett's published equation:

| Coefficient | Blanchett (2014) | This Study | Sign Match |
|-------------|------------------|------------|------------|
| Age² | +0.000080 | +0.000046 | YES |
| Age | -0.0125 | -0.0082 | YES |
| ln(Spending) | -0.0066 | -0.0261 | YES |
| Mean change | -0.96% | -0.92% | YES |
| N households | 591 | 679 | -- |

### 2. The ln(Spending) Definition Matters

The ln(Spending) coefficient sign depends on the spending measure used, an implementation choice not specified in the original paper:

- **Lagged (start-of-interval):** Negative (matches Blanchett) -- 100% of 12 specifications
- **Baseline:** Negative -- 100% of 12 specifications
- **Mean:** Mixed -- 50% negative
- **Current (end-of-interval):** Positive -- mechanical correlation with the dependent variable

Using lagged spending is essential for interpretable results.

### 3. Smile Attenuates in Fixed Effects

The spending smile curvature (positive Age²) is present in cross-sectional aggregated data but largely disappears in household fixed-effects models:

| Method | Age² Coefficient | Interpretation |
|--------|-----------------|----------------|
| Blanchett-style | +0.000151 | Clear positive curvature ("smile") |
| Pooled OLS | +0.000059 | Mild curvature |
| Fixed Effects | +0.000011 | Near zero |
| FE + Time | [not identified] | APC collinearity |

For individual client projections, a constant annual decline (~1%, with sensitivity around 0-2%) is more defensible than assuming a U-shaped smile.

### 4. Turning Point Uncertainty

The smile minimum age has a wide conditional 95% bootstrap confidence interval:
- Point estimate: 88.7 years
- Conditional 95% CI: [75.3, 109.9]
- Computed over the 75.3% of bootstrap draws where Age² > 0; in the remaining 24.7% the turning point is undefined or outside [50, 120]

---

## Repository Contents

### Code

| Script | Description |
|--------|-------------|
| `code/run_all.py` | Master pipeline (runs all stages in order) |
| `code/01_build_panel.py` | Build panel dataset from raw RAND data |
| `code/02_replication.py` | Primary replication (2001-2009) |
| `code/03_robustness_grid.py` | Robustness analysis (48 specifications) |
| `code/04_panel_extension.py` | Panel models (2001-2021), including CRE/Mundlak |
| `code/05_robustness_additions.py` | DV robustness and panel support diagnostics |
| `code/06_weighted_sensitivity.py` | Weighted sensitivity analysis |
| `code/07_twoway_clustering.py` | Two-way clustering sensitivity |
| `code/08_generate_figures.py` | Manuscript figures |

### Output Tables

| File | Description |
|------|-------------|
| `tables/attrition_table.csv` | Sample construction |
| `tables/replication_results.csv` | Primary replication results |
| `tables/bootstrap_coefficient_cis.csv` | Bootstrap 95% CIs |
| `tables/sign_frequency_analysis.csv` | ln(Spending) sign by definition |
| `tables/full_robustness_variants.csv` | All 48 specification variants |
| `tables/robustness_grid.csv` | Robustness grid summary |
| `tables/extension_panel_models.csv` | Panel FE/RE/CRE results (2001-2021) |
| `tables/hausman_test_results.csv` | FE vs RE diagnostic |
| `tables/dv_robustness_comparison.csv` | DV specification comparison |
| `tables/panel_support_diagnostic.csv` | Panel support statistics |
| `tables/weighted_sensitivity.csv` | Weighted vs unweighted sensitivity |
| `tables/outlier_filter_sensitivity.csv` | Outlier filter sensitivity |
| `tables/twoway_clustering_comparison.csv` | Two-way clustering SE comparison |

### Output Figures

| File | Description |
|------|-------------|
| `figures/figure1_replication.png` | Spending smile comparison with Blanchett |
| `figures/figure2_ln_spending.png` | ln(Spending) definition sensitivity |
| `figures/figure3_panel_models.png` | Panel model comparison |
| `figures/figure4_bootstrap_ci.png` | Bootstrap confidence intervals |

---

## Replication Instructions

### Data Requirements

The analysis uses two RAND datasets, available (free, registration required) from https://hrsdata.isr.umich.edu/:

1. **RAND HRS Longitudinal File** -- `randhrs1992_2022v1.dta`
2. **RAND CAMS** -- `randcams_2001_2021v1.dta`

### Setup

```
# Directory structure
<project_root>/
  peer_review/
    code/
    tables/
    figures/
    manuscript_FPR.md
  data/
    raw/
      randhrs1992_2022v1_STATA/
        randhrs1992_2022v1.dta
      randcams_2001_2021v1/
        randcams_2001_2021v1.dta
    processed/            # Created by scripts
```

### Install and Run

```bash
# Install dependencies
pip install -r code/requirements.txt

# Or use exact versions from a known-good run (Python 3.13.2)
pip install -r code/requirements-lock.txt

# Run the full pipeline
cd peer_review/code
python run_all.py
```

See `code/README.md` for individual script usage and troubleshooting.

---

## Data Availability

Data come from the Health and Retirement Study (HRS), sponsored by the National Institute on Aging (grant NIA U01AG009740) and conducted by the University of Michigan. Data are available to registered users at https://hrsdata.isr.umich.edu/.

---

## Contact

Derek Tharp, Ph.D., CFP® -- derek.tharp@maine.edu

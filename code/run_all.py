#!/usr/bin/env python3
"""
run_all.py - Master Pipeline for Blanchett (2014) Replication

This script reproduces all outputs in the correct order:
1. Build extension panel dataset (from raw RAND data)
2. Primary replication (main specification)
3. Robustness grid (sensitivity analysis)
4. Extension analysis (panel models for 2001-2021)
5. Additional robustness (DV form, panel support diagnostics)
6. Weighted sensitivity analysis
7. Two-way clustering sensitivity
8. Manuscript figure generation

Prerequisites:
- Raw RAND HRS and CAMS data in data/raw/
- Python packages from requirements.txt

Environment Variables:
- FPR_REPLICATION_TIMEOUT: Per-stage timeout in seconds (default: 1800)

Author: Derek Tharp
Date: 2026
"""

import os
import sys
import subprocess
from datetime import datetime

# Paths - peer_review/ is the base for the replication package
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # peer_review/
CODE_DIR = os.path.join(BASE_DIR, 'code')
# Output goes to peer_review/tables and peer_review/figures (matches package structure)
OUTPUT_TABLES = os.path.join(BASE_DIR, 'tables')
OUTPUT_FIGURES = os.path.join(BASE_DIR, 'figures')
# Ensure output directories exist
os.makedirs(OUTPUT_TABLES, exist_ok=True)
os.makedirs(OUTPUT_FIGURES, exist_ok=True)

# Default per-stage timeout (seconds). Override with environment variable FPR_REPLICATION_TIMEOUT.
TIMEOUT_SECONDS = int(os.environ.get('FPR_REPLICATION_TIMEOUT', '1800'))


def run_script(script_path, description):
    """Run a Python script and report status."""
    print(f"\n{'='*70}")
    print(f"RUNNING: {description}")
    print(f"Script: {script_path}")
    print(f"{'='*70}")

    if not os.path.exists(script_path):
        print(f"ERROR: Script not found: {script_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS  # per-stage timeout
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode != 0:
            print(f"ERROR: Script exited with code {result.returncode}")
            return False

        print(f"SUCCESS: {description}")
        return True

    except subprocess.TimeoutExpired:
        print(f"ERROR: Script timed out after {TIMEOUT_SECONDS} seconds")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    """Run the full replication pipeline."""
    print("="*70)
    print("BLANCHETT (2014) REPLICATION PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Per-stage timeout: {TIMEOUT_SECONDS} seconds")
    print("="*70)

    # Track results
    results = []

    # Pipeline stages
    stages = [
        # Stage 1: Build extension panel from raw RAND data
        {
            'script': os.path.join(CODE_DIR, '01_build_panel.py'),
            'description': 'Build Extension Panel (from raw RAND data)',
            'outputs': [
                '../data/processed/panel_analysis_2001_2021.csv',
                '../data/processed/full_panel_unfiltered.csv'
            ],
            'optional': True  # Skip if raw data not available
        },

        # Stage 2: Primary replication (uses raw data, proper attrition)
        {
            'script': os.path.join(CODE_DIR, '02_replication.py'),
            'description': 'Primary Replication (from raw RAND data)',
            'outputs': [
                'tables/replication_results.csv',
                'tables/attrition_table.csv',
                'tables/sign_frequency_analysis.csv',
                'tables/full_robustness_variants.csv',
                'tables/bootstrap_coefficient_cis.csv',
                'tables/age_means_2001_2009.csv',
                'tables/change_filter_comparison.csv',
                'figures/replication_strict_minobs20.png'
            ]
        },

        # Stage 3: Robustness grid
        {
            'script': os.path.join(CODE_DIR, '03_robustness_grid.py'),
            'description': 'Robustness Grid Analysis',
            'outputs': [
                'tables/robustness_grid.csv'
            ]
        },

        # Stage 4: Extension analysis (includes 2005+ robustness, Mundlak CRE)
        {
            'script': os.path.join(CODE_DIR, '04_panel_extension.py'),
            'description': 'Extension: Panel Models (2001-2021)',
            'outputs': [
                'tables/extension_panel_models.csv',
                'tables/hausman_test_results.csv',
                'figures/rigorous_extension_comparison.png'
            ],
            'optional': True  # Skip if panel data not available
        },

        # Stage 5: Additional robustness (DV form, panel support diagnostics)
        {
            'script': os.path.join(CODE_DIR, '05_robustness_additions.py'),
            'description': 'Additional Robustness (DV form, panel support)',
            'outputs': [
                'tables/dv_robustness_comparison.csv',
                'tables/panel_support_diagnostic.csv'
            ],
            'optional': True
        },

        # Stage 6: Weighted sensitivity analysis
        {
            'script': os.path.join(CODE_DIR, '06_weighted_sensitivity.py'),
            'description': 'Weighted Sensitivity Analysis',
            'outputs': [
                'tables/weighted_sensitivity.csv'
            ],
            'optional': True
        },

        # Stage 7: Two-way clustering sensitivity
        {
            'script': os.path.join(CODE_DIR, '07_twoway_clustering.py'),
            'description': 'Two-Way Clustering Sensitivity',
            'outputs': [
                'tables/twoway_clustering_comparison.csv'
            ],
            'optional': True
        },

        # Stage 8: Manuscript figures (reads from tables/*.csv and writes figures/*.png)
        {
            'script': os.path.join(CODE_DIR, '08_generate_figures.py'),
            'description': 'Manuscript Figures (from CSV tables)',
            'outputs': [
                'figures/figure1_replication.png',
                'figures/figure2_ln_spending.png',
                'figures/figure3_panel_models.png',
                'figures/figure4_bootstrap_ci.png',
                'tables/projection_summary.csv'
            ]
        },
    ]

    # Run each stage
    for i, stage in enumerate(stages, 1):
        print(f"\n\n{'#'*70}")
        print(f"STAGE {i}/{len(stages)}: {stage['description']}")
        print(f"{'#'*70}")

        success = run_script(stage['script'], stage['description'])

        # Handle optional stages
        if not success and stage.get('optional', False):
            print(f"  (Optional stage - continuing pipeline)")
            success = True  # Don't fail pipeline for optional stages

        results.append({
            'stage': i,
            'description': stage['description'],
            'success': success,
            'optional': stage.get('optional', False)
        })

        # Check outputs exist
        if success:
            for output in stage['outputs']:
                output_path = os.path.normpath(os.path.join(BASE_DIR, output))
                if os.path.exists(output_path):
                    print(f"  Output: {output}")
                else:
                    print(f"  WARNING: Expected output not found: {output}")

    # Summary
    print("\n\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)

    all_success = True
    for r in results:
        status = "SUCCESS" if r['success'] else "FAILED"
        optional_note = " (optional)" if r.get('optional', False) else ""
        print(f"  Stage {r['stage']}: {r['description']} - {status}{optional_note}")
        if not r['success'] and not r.get('optional', False):
            all_success = False

    print("\n" + "-"*70)
    if all_success:
        print("ALL REQUIRED STAGES COMPLETED SUCCESSFULLY")
    else:
        print("SOME REQUIRED STAGES FAILED - Review output above")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())

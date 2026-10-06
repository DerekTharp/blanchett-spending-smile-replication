from __future__ import annotations

import unittest
from pathlib import Path

from harness_utils import ROOT, csv_columns, find_row, parse_number, read_csv_rows


class ArtifactOutputTests(unittest.TestCase):
    def test_required_output_files_exist(self) -> None:
        expected_files = [
            "peer_review/tables/attrition_table.csv",
            "peer_review/tables/replication_results.csv",
            "peer_review/tables/bootstrap_coefficient_cis.csv",
            "peer_review/tables/sign_frequency_analysis.csv",
            "peer_review/tables/table4_symmetric_check.csv",
            "peer_review/tables/full_robustness_variants.csv",
            "peer_review/tables/robustness_grid.csv",
            "peer_review/tables/extension_panel_models.csv",
            "peer_review/tables/hausman_test_results.csv",
            "peer_review/tables/dv_robustness_comparison.csv",
            "peer_review/tables/panel_support_diagnostic.csv",
            "peer_review/tables/weighted_sensitivity.csv",
            "peer_review/tables/twoway_clustering_comparison.csv",
            "peer_review/tables/projection_summary.csv",
            "peer_review/tables/descriptive_statistics.csv",
            "peer_review/tables/fe_time_varying_controls.csv",
            "peer_review/tables/stratified_analysis.csv",
            "peer_review/tables/survivorship_robustness.csv",
            "peer_review/tables/spline_age_profile.csv",
            "peer_review/figures/figure1_replication.png",
            "peer_review/figures/figure2_ln_spending.png",
            "peer_review/figures/figure3_panel_models.png",
            "peer_review/figures/figure4_bootstrap_ci.png",
            "peer_review/figures/figure5_spline_comparison.png",
        ]
        missing = [path for path in expected_files if not (ROOT / path).exists()]
        self.assertEqual([], missing, f"Missing expected outputs: {missing}")

    def test_key_csv_schemas(self) -> None:
        expected_columns = {
            "peer_review/tables/replication_results.csv": [
                "Coefficient",
                "Blanchett",
                "This_Study",
                "Ratio",
                "Sign_Match",
            ],
            "peer_review/tables/bootstrap_coefficient_cis.csv": [
                "coefficient",
                "estimate",
                "bootstrap_se",
                "ci_lower",
                "ci_upper",
                "n_valid",
            ],
            "peer_review/tables/extension_panel_models.csv": [
                "method",
                "period",
                "age_sq",
                "age",
                "ln_exp",
                "n_obs",
                "n_households",
            ],
            "peer_review/tables/fe_time_varying_controls.csv": [
                "Model",
                "Age_sq",
                "Age_sq_SE",
                "Age",
                "Age_SE",
                "ln_Spending",
                "ln_Spending_SE",
                "N_obs",
                "N_hh",
            ],
            "peer_review/tables/survivorship_robustness.csv": [
                "Sample",
                "Age_sq",
                "Age_sq_SE",
                "Age",
                "Age_SE",
                "ln_Spending",
                "ln_Spending_SE",
                "N_obs",
                "N_hh",
            ],
            "peer_review/tables/spline_age_profile.csv": [
                "age",
                "blanchett_quadratic",
                "blanchett_spline",
                "fe_quadratic",
                "fe_spline",
            ],
        }

        for relative_path, columns in expected_columns.items():
            with self.subTest(path=relative_path):
                observed = csv_columns(relative_path)
                for column in columns:
                    self.assertIn(column, observed)

    def test_replication_coefficients_match_expected_values(self) -> None:
        rows = read_csv_rows("peer_review/tables/replication_results.csv")
        age_sq = find_row(rows, "Coefficient", "age_sq")
        age = find_row(rows, "Coefficient", "age")
        ln_exp = find_row(rows, "Coefficient", "ln_exp")

        self.assertAlmostEqual(parse_number(age_sq["This_Study"]), 0.000046, places=6)
        self.assertAlmostEqual(parse_number(age["This_Study"]), -0.008187, places=6)
        self.assertAlmostEqual(parse_number(ln_exp["This_Study"]), -0.026075, places=6)

    def test_panel_support_summary_matches_current_outputs(self) -> None:
        rows = read_csv_rows("peer_review/tables/panel_support_diagnostic.csv")
        total_households = find_row(rows, "Statistic", "Total households")
        total_observations = find_row(rows, "Statistic", "Total observations")
        mean_intervals = find_row(rows, "Statistic", "Mean intervals per HH")

        self.assertEqual(parse_number(total_households["Value"]), 4900)
        self.assertEqual(parse_number(total_observations["Value"]), 20189)
        self.assertAlmostEqual(parse_number(mean_intervals["Value"]), 4.12, places=2)

    def test_fe_controls_table_matches_expected_values(self) -> None:
        rows = read_csv_rows("peer_review/tables/fe_time_varying_controls.csv")
        baseline = find_row(rows, "Model", "Baseline FE")
        all_controls = find_row(rows, "Model", "FE + All Controls")

        self.assertAlmostEqual(parse_number(baseline["Age_sq"]), 0.000011, places=6)
        self.assertAlmostEqual(parse_number(all_controls["Age_sq"]), 0.000035, places=6)
        self.assertAlmostEqual(parse_number(all_controls["partnered"]), 0.0742, places=4)

    def test_survivorship_table_uses_interval_labels(self) -> None:
        rows = read_csv_rows("peer_review/tables/survivorship_robustness.csv")
        labels = [row["Sample"] for row in rows]
        self.assertEqual(
            labels,
            [
                "Full sample",
                "HH with 3+ intervals",
                "HH with 4+ intervals",
                "HH with 5+ intervals",
            ],
        )

    def test_projection_summary_matches_current_table(self) -> None:
        rows = read_csv_rows("peer_review/tables/projection_summary.csv")
        age_95 = find_row(rows, "age", "95")

        self.assertEqual(parse_number(age_95["blanchett_spending"]), 41200)
        self.assertEqual(parse_number(age_95["this_study_spending"]), 29200)
        self.assertEqual(parse_number(age_95["difference"]), 12000)

    def test_output_tables_do_not_contain_serialized_model_objects(self) -> None:
        columns = csv_columns("peer_review/tables/extension_panel_models.csv")
        self.assertNotIn(
            "model_obj",
            columns,
            "Output tables should store scalar results, not serialized model summaries.",
        )

    def test_time_effects_curvature_is_reported(self) -> None:
        extension = read_csv_rows("peer_review/tables/extension_panel_models.csv")
        time_rows = [row for row in extension if row["method"] == "Fixed Effects (time)"]
        self.assertEqual(len(time_rows), 4)
        for row in time_rows:
            with self.subTest(period=row["period"]):
                self.assertGreater(parse_number(row["age_sq_se"]), 0)
                self.assertLess(abs(parse_number(row["age_sq"])), 0.01)
                self.assertIn(row["age"], ["", "nan"])
                self.assertIn(row["age_se"], ["", "nan"])

        full = find_row(time_rows, "period", "2001-2021")
        dv_rows = read_csv_rows("peer_review/tables/dv_robustness_comparison.csv")
        dv_time = [row for row in dv_rows if row["Method"] == "FE + Time"]
        self.assertEqual(len(dv_time), 2)
        for row in dv_time:
            self.assertGreater(parse_number(row["Age² SE"]), 0)
            self.assertIn(row["Age"], ["", "nan"])
        ratio = find_row(dv_time, "DV", "Ratio")
        self.assertAlmostEqual(parse_number(ratio["Age²"]), parse_number(full["age_sq"]), places=12)
        self.assertAlmostEqual(parse_number(ratio["Age² SE"]), parse_number(full["age_sq_se"]), places=12)


if __name__ == "__main__":
    unittest.main()

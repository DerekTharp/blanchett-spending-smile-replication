"""Numerical-invariance guard for the panel estimator.

Centering age is a pure reparameterization: it cannot change the Age-squared
coefficient or any fitted value. A violation means the least-squares solve is
numerically unstable. The test
estimates the most ill-conditioned model in the paper (the 2001-2009
conditional FE) with uncentered and centered age and asserts the Age-squared
estimates agree with each other and with the shipped CSV.

Skipped when the restricted-access panel data are not present.
"""

from __future__ import annotations

import os
import unittest

from harness_utils import ROOT, find_row, parse_number, read_csv_rows

PANEL_PATH = ROOT / "data" / "processed" / "panel_analysis_2001_2021.csv"


@unittest.skipUnless(PANEL_PATH.exists(), "restricted panel data not available")
class EstimatorInvarianceTests(unittest.TestCase):
    def test_age_centering_invariance_2001_2009_fe(self) -> None:
        import numpy as np
        import pandas as pd
        from linearmodels.panel import PanelOLS

        panel = pd.read_csv(PANEL_PATH)
        df = panel.dropna(
            subset=["pct_change_total", "lag_real_total", "age_int", "wave"]
        ).copy()
        df = df[(df["age_int"] >= 60) & (df["age_int"] <= 90)]
        df = df[df["year"] <= 2009]
        df["ln_spending"] = np.log(df["lag_real_total"])

        def age_sq_estimate(center: float) -> float:
            d = df.copy()
            a = d["age_int"].astype(float) - center
            d["A2"], d["A1"] = a**2, a
            d = d.set_index(["hhidpn", "wave"])
            fit = PanelOLS(
                d["pct_change_total"], d[["A2", "A1", "ln_spending"]],
                entity_effects=True,
            ).fit(cov_type="clustered", cluster_entity=True)
            return float(fit.params["A2"])

        uncentered = age_sq_estimate(0.0)
        centered = age_sq_estimate(75.0)
        self.assertAlmostEqual(
            uncentered, centered, places=10,
            msg="Age-centering changed the Age-squared estimate: the "
                "least-squares solve is numerically unstable.",
        )

        rows = read_csv_rows("peer_review/tables/extension_panel_models.csv")
        stored = find_row(
            [r for r in rows if r["period"] == "2001-2009"], "method", "Fixed Effects"
        )
        self.assertAlmostEqual(uncentered, parse_number(stored["age_sq"]), places=8)


class PriceYearTests(unittest.TestCase):
    def test_wealth_and_spending_use_their_own_measurement_year(self) -> None:
        import importlib.util
        import pandas as pd

        path = ROOT / "peer_review" / "code" / "01_build_panel.py"
        spec = importlib.util.spec_from_file_location("build_panel_price_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        # Equal purchasing power in 2000/2020 wealth and 2001/2021 spending.
        raw = pd.DataFrame({
            "hhidpn": [1], "r5agey_e": [60], "r15agey_e": [80],
            "h5atotb": [172200.0], "h15atotb": [258811.0],
            "h5cstot": [177100.0], "h15cstot": [270970.0],
        })
        panel = module.reshape_to_panel(raw).set_index("wave")
        for wave in [5, 15]:
            with self.subTest(wave=wave):
                self.assertAlmostEqual(panel.loc[wave, "real_wealth"], 214537.0, places=6)
                self.assertAlmostEqual(panel.loc[wave, "real_total"], 214537.0, places=6)


if __name__ == "__main__":
    unittest.main()

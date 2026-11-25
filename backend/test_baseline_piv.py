"""
Acceptance tests for baseline_piv.py.  Run: python -m unittest -v test_baseline_piv

Sources of the two published-style examples: MEREC paper (Keshavarz-Ghorabaee et al. 2021) and PIV
paper (Mufazzal & Muzakkir 2018). The table / equation numbers they come from are UNVERIFIED: no PDF
was attached. What IS verified here is that the spec formulas reproduce the supplied numbers.
"""
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_piv as bp
import schema as sch

HERE = Path(__file__).resolve().parent
CFG = sch.load_config(HERE / "ppiv_config.json")
READY = sch.resolve_path(CFG, "ready_csv")


class TestMerec(unittest.TestCase):
    X = np.array([[450, 8000, 54, 145], [10, 9100, 2, 160], [100, 8200, 31, 153],
                  [220, 9300, 1, 162], [5, 8400, 23, 158]], float)
    DIRS = [1, 1, -1, -1]
    EXPECTED = [0.5752, 0.0141, 0.4016, 0.0091]

    def test_acceptance_example(self):
        w = bp.merec_weights(pd.DataFrame(self.X, columns=list("ABCD")), self.DIRS)
        np.testing.assert_allclose(w.to_numpy(), self.EXPECTED, atol=1e-3)
        self.assertAlmostEqual(w.sum(), 1.0, places=12)
        self.assertEqual(w.attrs["constant_columns"], [])

    def test_accepts_ndarray_and_dict_directions(self):
        w1 = bp.merec_weights(self.X, self.DIRS)
        w2 = bp.merec_weights(pd.DataFrame(self.X, columns=list("ABCD")), dict(zip("ABCD", self.DIRS)))
        np.testing.assert_allclose(w1.to_numpy(), w2.to_numpy())

    def test_row_order_invariance(self):
        w1 = bp.merec_weights(self.X, self.DIRS)
        w2 = bp.merec_weights(self.X[::-1], self.DIRS)
        np.testing.assert_allclose(w1.to_numpy(), w2.to_numpy(), atol=1e-12)

    def test_m_is_number_of_criteria_not_alternatives(self):
        # hand check on a 2x2 case, benefit both: X=[[1,2],[2,1]]
        # n = [[1, .5],[.5, 1]] ; |ln n| = [[0,a],[a,0]] with a=ln2 ; m=2
        # S_i = ln(1+a/2) for both rows ; S'_i,1 (drop crit 1) = ln(1+L_i2/2), S'_i,2 = ln(1+L_i1/2)
        a = np.log(2)
        S = np.log(1 + a / 2)
        E1 = abs(np.log(1 + a / 2) - S) + abs(np.log(1 + 0) - S)       # row1: L_12=a, row2: L_22=0
        E2 = abs(np.log(1 + 0) - S) + abs(np.log(1 + a / 2) - S)
        w = bp.merec_weights(np.array([[1, 2], [2, 1]], float), [1, 1])
        np.testing.assert_allclose(w.to_numpy(), [E1 / (E1 + E2), E2 / (E1 + E2)], atol=1e-12)

    def test_rejects_nonpositive_nan_inf(self):
        for bad in (0.0, -1.0, np.nan, np.inf):
            X = self.X.copy(); X[2, 1] = bad
            with self.assertRaises(ValueError, msg=str(bad)):
                bp.merec_weights(X, self.DIRS)

    def test_rejects_bad_directions(self):
        for bad in ([1, 1, -1], [1, 1, -1, 0], [1, 1, -1, True], [1, 1, -1, "x"], [1, 1, -1, 1.0]):
            with self.assertRaises(ValueError, msg=str(bad)):
                bp.merec_weights(self.X, bad)

    def test_constant_column_explicit(self):
        X = self.X.copy(); X[:, 1] = 7.0
        with self.assertWarns(UserWarning):
            w = bp.merec_weights(pd.DataFrame(X, columns=list("ABCD")), self.DIRS)
        self.assertEqual(w.attrs["constant_columns"], ["B"])
        self.assertEqual(w["B"], 0.0)
        self.assertAlmostEqual(w.sum(), 1.0, places=12)

    def test_all_constant_raises_zero_total_E(self):
        with self.assertRaises(ValueError) as cm:
            bp.merec_weights(np.full((4, 3), 5.0), [1, 1, -1])
        self.assertIn("zero", str(cm.exception))


class TestPiv(unittest.TestCase):
    def test_acceptance_example(self):
        X = [[1, 9, 1, 3], [9, 1, 9, 1], [8, 1, 4, 5]]
        out = bp.piv_rank(X, [0.25] * 4, [1, 1, 1, 1], "vector")
        np.testing.assert_allclose(out["scores"].to_numpy(), [0.452, 0.389, 0.366], atol=5e-4)
        self.assertEqual(out["ranks"].tolist(), [3, 2, 1])
        # normalised matrix hand check for column 1: 1/sqrt(1+81+64)
        self.assertAlmostEqual(out["normalized_matrix"].iloc[0, 0], 1 / np.sqrt(146), places=12)
        self.assertAlmostEqual(out["weighted_matrix"].iloc[0, 0], 0.25 / np.sqrt(146), places=12)

    def test_cost_criterion_hand_computed(self):
        # X = [[2,10],[4,5],[3,8]]; C1 benefit w=.6; C2 cost w=.4; vector normalisation.
        # norms: sqrt(29)=5.385165, sqrt(189)=13.747727
        # r1 = [.371391,.742781,.557086]  r2 = [.727393,.363696,.581914]
        # v1 = .6*r1 = [.222834,.445669,.334252]   v2 = .4*r2 = [.290957,.145479,.232766]
        # u1 = max(v1)-v1 = [.222834,0,.111417]    u2 = v2-min(v2) = [.145479,0,.087287]
        # d = [.368313, 0, .198704]  -> ranks [3,1,2]
        out = bp.piv_rank([[2, 10], [4, 5], [3, 8]], [0.6, 0.4], [1, -1], "vector")
        np.testing.assert_allclose(out["scores"].to_numpy(), [0.368313, 0.0, 0.198704], atol=1e-6)
        self.assertEqual(out["ranks"].tolist(), [3, 1, 2])

    def test_normalization_none_hand_computed(self):
        # same X, no normalisation: v1=[1.2,2.4,1.8], v2=[4,2,3.2]
        # u1=[1.2,0,.6], u2=[2,0,1.2] -> d=[3.2,0,1.8]
        out = bp.piv_rank([[2, 10], [4, 5], [3, 8]], [0.6, 0.4], [1, -1], "none")
        np.testing.assert_allclose(out["scores"].to_numpy(), [3.2, 0.0, 1.8], atol=1e-12)
        self.assertEqual(out["ranks"].tolist(), [3, 1, 2])

    def test_ties_use_min_rank(self):
        out = bp.piv_rank([[5, 5], [5, 5], [1, 1]], [0.5, 0.5], [1, 1], "none")
        self.assertEqual(out["ranks"].tolist(), [1, 1, 3])

    def test_float_noise_does_not_split_ties(self):
        out = bp.piv_rank([[0.1 + 0.2, 1.0], [0.3, 1.0], [0.0, 0.0]], [0.5, 0.5], [1, 1], "none")
        self.assertEqual(out["ranks"].tolist()[:2], [1, 1])

    def test_zero_norm_guard(self):
        out = bp.piv_rank([[0, 1], [0, 2]], [0.5, 0.5], [1, 1], "vector")
        self.assertEqual(out["meta"]["zero_norm_columns"], [0])
        self.assertTrue(np.isfinite(out["scores"]).all())

    def test_weights_validated_and_renormalised(self):
        X = [[1, 2], [2, 1], [3, 3]]
        out = bp.piv_rank(X, [2.0, 2.0], [1, 1])
        self.assertTrue(out["meta"]["weights_renormalised"])
        np.testing.assert_allclose(out["weights"].to_numpy(), [0.5, 0.5])
        for bad in ([-1, 2], [0, 0], [np.nan, 1], [1, 2, 3]):
            with self.assertRaises(ValueError, msg=str(bad)):
                bp.piv_rank(X, bad, [1, 1])

    def test_rejects_nan_inf_bad_normalization(self):
        with self.assertRaises(ValueError):
            bp.piv_rank([[1, np.nan], [2, 3]], [.5, .5], [1, 1])
        with self.assertRaises(ValueError):
            bp.piv_rank([[1, np.inf], [2, 3]], [.5, .5], [1, 1])
        with self.assertRaises(ValueError):
            bp.piv_rank([[1, 2], [2, 3]], [.5, .5], [1, 1], normalization="zscore")

    def test_all_cost_flips_ranking_of_all_benefit(self):
        X = [[1, 9], [9, 1], [5, 5.5]]
        a = bp.piv_rank(X, [.5, .5], [1, 1], "none")["ranks"].tolist()
        b = bp.piv_rank(X, [.5, .5], [-1, -1], "none")["ranks"].tolist()
        self.assertEqual(a, [2, 2, 1])
        self.assertEqual(b, [1, 1, 3])


@unittest.skipUnless(READY.is_file(), f"{READY} not found; run data_prep.py first")
class TestClassicalFromConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(READY)

    def test_columns_directions_raw_and_weights(self):
        out = bp.classical_piv_from_config(self.df, CFG)
        srcs = sch.all_sources(CFG)
        self.assertEqual(out["columns"], [s["column"] for s in srcs])      # raw, no __adj
        self.assertEqual(out["directions"].tolist(), [s["direction"] for s in srcs])
        self.assertAlmostEqual(out["weights"].sum(), 1.0, places=12)
        self.assertEqual(out["weights_source"], "merec")
        self.assertEqual(len(out["ranks"]), len(self.df))
        self.assertEqual(out["ranks"].min(), 1)

    def test_matches_direct_composition(self):
        out = bp.classical_piv_from_config(self.df, CFG)
        cols = out["columns"]
        w = bp.merec_weights(self.df[cols], out["directions"])
        ref = bp.piv_rank(self.df[cols], w, out["directions"], CFG["piv"]["normalization"])
        pd.testing.assert_series_equal(out["scores"], ref["scores"])

    def test_custom_weights_and_normalization_from_config(self):
        cols = bp.classical_columns(CFG)[0]
        w = {c: 1.0 for c in cols}
        out = bp.classical_piv_from_config(self.df, CFG, weights=w)
        self.assertEqual(out["weights_source"], "custom")
        np.testing.assert_allclose(out["weights"].to_numpy(), 1 / len(cols))
        cfg2 = {**CFG, "piv": {**CFG["piv"], "normalization": "none"}}
        self.assertEqual(bp.classical_piv_from_config(self.df, cfg2)["meta"]["normalization"], "none")

    def test_subset_recomputes_weights_and_missing_column_raises(self):
        sub = self.df.iloc[:200]
        out = bp.classical_piv_from_config(sub, CFG)
        full = bp.classical_piv_from_config(self.df, CFG)
        self.assertFalse(np.allclose(out["weights"].to_numpy(), full["weights"].to_numpy()))
        with self.assertRaises(ValueError):
            bp.classical_piv_from_config(self.df.drop(columns=[out["columns"][0]]), CFG)

    def test_nonpositive_raw_value_raises(self):
        bad = self.df.copy()
        bad.loc[bad.index[0], bp.classical_columns(CFG)[0][0]] = 0
        with self.assertRaises(ValueError):
            bp.classical_piv_from_config(bad, CFG)


if __name__ == "__main__":
    unittest.main(verbosity=2)

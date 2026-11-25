"""
Tests for validation.py. The rank functions here are tiny stand-ins or the classical baseline used
only as a convenient concrete method; validation.py itself imports no method.
Run: python -m unittest -v test_validation
"""
import copy
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_piv as bp
import schema as sch
import validation as val

HERE = Path(__file__).resolve().parent
CFG = sch.load_config(HERE / "ppiv_config.json")
READY = sch.resolve_path(CFG, "ready_csv")


def classical_fn(cfg, normalization=None):
    c = cfg if normalization is None else {**cfg, "piv": {**cfg["piv"], "normalization": normalization}}
    return lambda df, weights=None: bp.classical_piv_from_config(df, c, weights)


def classical_factory(cfg_variant, df_full):
    return classical_fn(cfg_variant)


class TestImports(unittest.TestCase):
    def test_validation_imports_no_method(self):
        src = (HERE / "validation.py").read_text()
        for mod in ("baseline_piv", "ppiv", "merec", "plithogenic"):
            self.assertNotIn(f"import {mod}", src)
            self.assertNotIn(f"from {mod}", src)


class TestMetrics(unittest.TestCase):
    def test_identical_and_reversed(self):
        r = pd.Series(np.arange(1, 21), index=list("abcdefghijklmnopqrst"))
        m = val.compare_rankings(r, r, [5, 10])
        self.assertEqual((m["spearman"], m["kendall"]), (1.0, 1.0))
        self.assertEqual(m["top_5_overlap_fraction"], 1.0)
        m2 = val.compare_rankings(r, 21 - r, [5, 10])
        self.assertAlmostEqual(m2["spearman"], -1.0)
        self.assertEqual(m2["top_5_overlap_count"], 0)

    def test_partial_overlap_hand_check(self):
        r1 = pd.Series([1, 2, 3, 4, 5, 6], index=list("abcdef"))
        r2 = pd.Series([2, 1, 6, 5, 4, 3], index=list("abcdef"))
        m = val.compare_rankings(r1, r2, [3])
        self.assertEqual(m["top_3_overlap_count"], 2)             # {a,b,c} vs {a,b,f}
        self.assertAlmostEqual(m["top_3_overlap_fraction"], 2 / 3)
        self.assertFalse(m["top_3_tie_at_cutoff"])

    def test_constant_ranks_give_none_not_nan(self):
        r = pd.Series([1, 1, 1], index=list("abc"))
        m = val.compare_rankings(r, pd.Series([1, 2, 3], index=list("abc")), [2])
        self.assertIsNone(m["spearman"])
        json.dumps(m, allow_nan=False)

    def test_ties_at_cutoff_flagged(self):
        r = pd.Series([1, 2, 2, 4], index=list("abcd"))
        self.assertTrue(val.compare_rankings(r, r, [2])["top_2_tie_at_cutoff"])


class TestReversalSynthetic(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.df = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("xyz"), index=[f"r{i}" for i in range(120)])
        self.cfg = {"validation": {"reversal": {"n_iterations": 6, "drop_fraction": 0.1, "max_pairs": 500},
                                   "top_k": [5, 10]}}

    def additive(self, df, weights=None):
        w = pd.Series([0.5, 0.3, 0.2], index=list("xyz")) if weights is None else weights
        score = (df[list("xyz")] * w).sum(axis=1)
        return {"ranks": score.rank(ascending=False, method="min"), "weights": w}

    def data_dependent(self, df, weights=None):
        # weights recomputed from the subset's variances, and each column scaled by the subset's std
        # (like a column norm) -> the order can move when rows are dropped, in either mode
        cols = list(df[list("xyz")].columns)
        sd = df[cols].std()
        w = sd ** 2 / (sd ** 2).sum() if weights is None else weights
        score = ((df[cols] / sd) * w).sum(axis=1)
        return {"ranks": score.rank(ascending=False, method="min"), "weights": w}

    def test_frozen_additive_is_zero_by_construction(self):
        res = val.evaluate(self.df, self.cfg, {"add": self.additive}, seed=3)
        fz = res["rank_reversal"]["add"][val.MODE_FROZEN]
        self.assertEqual(fz["reversals"], 0)
        self.assertGreater(fz["valid_pairs"], 0)
        self.assertIn("by construction", fz["note"])

    def test_recomputed_mode_runs_separately_and_can_flip(self):
        res = val.evaluate(self.df, self.cfg, {"dd": self.data_dependent}, seed=3)
        rc = res["rank_reversal"]["dd"][val.MODE_RECOMPUTED]
        fz = res["rank_reversal"]["dd"][val.MODE_FROZEN]
        self.assertEqual(rc["mode"], val.MODE_RECOMPUTED)
        self.assertEqual(fz["mode"], val.MODE_FROZEN)
        self.assertGreater(rc["reversals"], 0)                    # measured on this synthetic set
        self.assertLessEqual(rc["valid_pairs"], 6 * 500)
        for it in rc["iterations"]:
            self.assertEqual(it["rows_dropped"], 12)
            self.assertEqual(it["valid_pairs"] + it["tied_pairs_excluded"], it["pairs_sampled"])
            self.assertLessEqual(it["pairs_sampled"], 500)

    def test_seeded_reproducible_and_seed_matters(self):
        a = val.evaluate(self.df, self.cfg, {"dd": self.data_dependent}, seed=7)
        b = val.evaluate(self.df, self.cfg, {"dd": self.data_dependent}, seed=7)
        c = val.evaluate(self.df, self.cfg, {"dd": self.data_dependent}, seed=8)
        self.assertEqual(a, b)
        self.assertNotEqual(a["rank_reversal"], c["rank_reversal"])

    def test_same_subsets_for_all_methods_and_modes(self):
        res = val.evaluate(self.df, self.cfg, {"a": self.additive, "d": self.data_dependent}, seed=2)
        ref = [(i["pairs_sampled"], i["rows_kept"]) for i in res["rank_reversal"]["a"][val.MODE_RECOMPUTED]["iterations"]]
        for m in ("a", "d"):
            for mode in (val.MODE_RECOMPUTED, val.MODE_FROZEN):
                got = [(i["pairs_sampled"], i["rows_kept"]) for i in res["rank_reversal"][m][mode]["iterations"]]
                self.assertEqual(got, ref)

    def test_pair_sampler_caps_and_distinct(self):
        i, j = val._sample_pairs(400, 5000, np.random.default_rng(0))
        self.assertEqual(len(i), 5000)
        self.assertTrue((i < j).all())
        self.assertEqual(len(set(zip(i.tolist(), j.tolist()))), 5000)
        i, j = val._sample_pairs(10, 5000, np.random.default_rng(0))
        self.assertEqual(len(i), 45)

    def test_pair_flip_counting_hand_check(self):
        # one iteration forced by hand: full ranks a<b<c<d ; subset ranks reverse b,c
        full = pd.Series([1, 2, 3, 4], index=list("abcd"), dtype=float)
        sub = pd.Series([1, 3, 2, 4], index=list("abcd"), dtype=float)
        sf = np.sign(full.to_numpy()[:, None] - full.to_numpy()[None, :])
        ss = np.sign(sub.to_numpy()[:, None] - sub.to_numpy()[None, :])
        iu = np.triu_indices(4, 1)
        self.assertEqual(int((sf[iu] != ss[iu]).sum()), 1)         # only the (b,c) pair flips

    def test_ranks_only_method_cannot_run_frozen_mode(self):
        bare = lambda df, weights=None: self.additive(df)["ranks"]
        with self.assertRaises(ValueError):
            val.evaluate(self.df, self.cfg, {"bare": bare})

    def test_bad_inputs(self):
        dup = self.df.copy(); dup.index = ["r0"] * len(dup)
        with self.assertRaises(ValueError):
            val.evaluate(dup, self.cfg, {"a": self.additive})
        with self.assertRaises(ValueError):
            val.evaluate(self.df, self.cfg, {})
        bad = lambda df, weights=None: pd.Series(1.0, index=df.index[:-1])
        with self.assertRaises(ValueError):
            val.evaluate(self.df, self.cfg, {"b": bad})


@unittest.skipUnless(READY.is_file(), "run data_prep.py first")
class TestOnRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(READY)

    def test_normalization_none_frozen_is_zero_vector_is_measured(self):
        cfg_none = {**CFG, "piv": {**CFG["piv"], "normalization": "none"}}
        cfg_vec = {**CFG, "piv": {**CFG["piv"], "normalization": "vector"}}
        res = val.evaluate(self.df, CFG, {"classical_none": classical_fn(cfg_none),
                                          "classical_vector": classical_fn(cfg_vec)}, seed=0)
        self.assertEqual(res["rank_reversal"]["classical_none"][val.MODE_FROZEN]["reversals"], 0)
        rec = res["rank_reversal"]["classical_none"][val.MODE_RECOMPUTED]
        self.assertGreater(rec["valid_pairs"], 0)
        self.assertLessEqual(rec["valid_pairs"], 20 * 5000)
        self.assertEqual(rec["n_iterations"], 20)
        self.assertIn("classical_none__vs__classical_vector", res["agreement"])
        json.dumps(res, allow_nan=False)

    def test_sensitivity_with_classical_factory(self):
        res = val.sensitivity(classical_factory, None, self.df, CFG)
        m = len(sch.all_sources(CFG))
        self.assertEqual(len(res["weight_perturbation"]), 2 * m)
        self.assertTrue(all(r["status"] == "ok" for r in res["weight_perturbation"]))
        skipped = [r for r in res["leave_one_value_out"] if r["status"] == "skipped"]
        self.assertEqual([r["removed"] for r in skipped], [CFG["plithogenic"]["dominant_value"]])
        self.assertEqual(set(res["param_sweeps"]), set(CFG["validation"]["sensitivity"]["param_grid"]))
        # classical baseline ignores percentiles/epsilon/operator/contradiction -> identical ranking
        for name in ("percentiles", "epsilon", "operator", "contradiction"):
            for r in res["param_sweeps"][name]:
                self.assertEqual(r["status"], "ok", r)
                self.assertAlmostEqual(r["spearman"], 1.0)
        # normalization 'none' is the one that can change the baseline; the config's own setting is identical
        norm = {r["overrides"]["piv.normalization"]: r for r in res["param_sweeps"]["normalization"]}
        self.assertTrue(norm[CFG["piv"]["normalization"]]["identical_to_base"])
        json.dumps(res, allow_nan=False)

    def test_sensitivity_records_errors_instead_of_hiding(self):
        grid = {"typo": [{"floor.epsilonn": 0.1}], "badvalue": [{"appurtenance.lower_percentile": 99}]}
        res = val.sensitivity(classical_factory, grid, self.df, CFG)
        self.assertEqual(res["param_sweeps"]["typo"][0]["status"], "error")
        self.assertEqual(res["param_sweeps"]["badvalue"][0]["status"], "error")      # fails schema (lower>=upper)

    def test_contradiction_uniform_expansion_satisfies_axioms(self):
        v = val.apply_overrides(CFG, {"contradiction.uniform": 0.3})
        self.assertEqual(v["contradiction"]["method"], "expert")
        sch.validate_config(v)                                      # axioms checked by schema
        names = sch.value_names(CFG)
        mat = v["contradiction"]["expert_matrix"]
        self.assertTrue(all(mat[a][a] == 0 and mat[a][b] == 0.3 for a in names for b in names if a != b))
        self.assertEqual(CFG["contradiction"]["method"], "spearman_on_mean_degrees")   # original untouched

    def test_overrides_apply_together(self):
        v = val.apply_overrides(CFG, {"appurtenance.lower_percentile": 1, "appurtenance.upper_percentile": 99})
        self.assertEqual((v["appurtenance"]["lower_percentile"], v["appurtenance"]["upper_percentile"]), (1, 99))

    def test_leave_one_out_uses_variant_config(self):
        seen = []
        def factory(cfg_variant, df):
            seen.append(len(cfg_variant["plithogenic"]["values"]))
            return classical_fn(cfg_variant)
        val.sensitivity(factory, {}, self.df, CFG)
        n = len(CFG["plithogenic"]["values"])
        self.assertEqual(seen[0], n)                                # base
        self.assertEqual(sorted(set(seen[1:])), [n - 1])           # each leave-one-out has one value fewer


if __name__ == "__main__":
    unittest.main(verbosity=2)

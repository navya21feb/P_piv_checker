"""
Acceptance tests for plithogenic_piv.py.   Run: python -m unittest -v test_plithogenic_piv

Table 5 numbers: Smarandache (2018), p. 165 (data Tables 3-4, pp. 163-164) -- confirmed in the PDF.
Everything else is a property test or a team-made check; none of it is a published result.
"""
import copy
import itertools
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import baseline_piv as bp
import plithogenic_piv as pp
import schema as sch

HERE = Path(__file__).resolve().parent
CFG = sch.load_config(HERE / "ppiv_config.json")
DF = pd.read_csv(sch.resolve_path(CFG, "ready_csv"))
PARAMS = pp.fit_frozen_params(DF, CFG)


# ---------------------------------------------------------------- synthetic config helper
def synth(n=60, seed=0, dominant="V1", sources=None):
    """Small synthetic problem with 3 values; returns (df, cfg). Column names are arbitrary."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"College_ID": [f"C{i}" for i in range(n)]})
    base = rng.normal(size=n)
    for c in ("a1", "a2", "b1", "b2", "c1"):
        df[c] = np.exp(0.5 * base + rng.normal(size=n)) + 1.0
    cfg = copy.deepcopy(CFG)
    cfg["shrinkage"]["columns"] = {}
    cfg["derived"] = []
    cfg["quality_flags"]["rules"] = []
    cfg["missing_values"]["strategies"] = {}
    cfg["scale_inspection"]["columns"] = []
    srcs = sources or {"V1": [("a1", 1, "identity"), ("a2", -1, "log10")],
                       "V2": [("b1", 1, "log10"), ("b2", 1, "identity")],
                       "V3": [("c1", -1, "identity")]}
    cfg["plithogenic"]["values"] = [
        {"name": v, "sources": [{"column": c, "direction": d, "transform": t, "use_adjusted": False}
                                for c, d, t in lst]} for v, lst in srcs.items()]
    cfg["plithogenic"]["dominant_value"] = dominant
    return df, cfg


class TestTable5(unittest.TestCase):
    A = np.array([.6, .2, .7, .8, .5]); B = np.array([.7, .4, .6, .6, .4])
    C = np.array([0, 1 / 3, 2 / 3, 0, 1 / 2])

    def test_intersection(self):
        r = pp.combine_two(self.A, self.B, self.C, "intersection")
        np.testing.assert_array_equal(np.round(r, 2), [.42, .23, .73, .48, .45])

    def test_union(self):
        r = pp.combine_two(self.A, self.B, self.C, "union")
        np.testing.assert_array_equal(np.round(r, 2), [.88, .37, .57, .92, .45])

    def test_text_value_0_2266(self):
        # p. 166: 0.23 ~ 0.2266... = (2/3)*0.08 + (1/3)*0.52
        self.assertAlmostEqual(float(pp.combine_two(.2, .4, 1 / 3, "intersection")), 2 / 3 * .08 + 1 / 3 * .52, 12)


class TestOperatorProperties(unittest.TestCase):
    PAIRS = [("product", "probabilistic_sum"), ("minimum", "maximum"), ("lukasiewicz", "bounded_sum")]
    rng = np.random.default_rng(1)
    A, B = rng.random(200), rng.random(200)

    def test_c0_and_c1(self):
        for tn, tc in self.PAIRS:
            T, S = pp.T_NORMS[tn](self.A, self.B), pp.T_CONORMS[tc](self.A, self.B)
            np.testing.assert_allclose(pp.combine_two(self.A, self.B, 0.0, "intersection", tn, tc), T)
            np.testing.assert_allclose(pp.combine_two(self.A, self.B, 0.0, "union", tn, tc), S)
            np.testing.assert_allclose(pp.combine_two(self.A, self.B, 1.0, "intersection", tn, tc), S)
            np.testing.assert_allclose(pp.combine_two(self.A, self.B, 1.0, "union", tn, tc), T)

    def test_c_half_is_mean_for_product_probsum(self):
        for op in ("intersection", "union"):
            np.testing.assert_allclose(pp.combine_two(self.A, self.B, .5, op), (self.A + self.B) / 2)

    def test_range_and_monotone(self):
        grid = np.linspace(0, 1, 11)
        for tn, tc in self.PAIRS:
            for op in ("intersection", "union"):
                for c in (0, .25, .5, 1):
                    Z = pp.combine_two(grid[:, None], grid[None, :], c, op, tn, tc)
                    self.assertTrue((Z >= 0).all() and (Z <= 1).all())
                    self.assertTrue((np.diff(Z, axis=0) >= -1e-12).all())    # non-decreasing in A
                    self.assertTrue((np.diff(Z, axis=1) >= -1e-12).all())    # non-decreasing in B

    def test_registry_matches_schema_vocabulary(self):
        self.assertEqual(set(pp.T_NORMS), set(sch.VALID_T_NORMS))
        self.assertEqual(set(pp.T_CONORMS), set(sch.VALID_T_CONORMS))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError): pp.combine_two(.2, .4, 1.5, "union")
        with self.assertRaises(ValueError): pp.combine_two(.2, .4, .5, "xor")


class TestAggregateValues(unittest.TestCase):
    def test_one_source_unchanged(self):
        df, cfg = synth()
        P = pp.fit_frozen_params(df, cfg)
        deg = pp.appurtenance_degrees(df, P)
        vd = pp.aggregate_values(deg, pd.Series({"V1": 0, "V2": .3, "V3": .9}), cfg)
        np.testing.assert_array_equal(vd["V3"].to_numpy(), deg["V3"]["c1"].to_numpy())

    def test_two_sources_match_formula(self):
        df, cfg = synth()
        P = pp.fit_frozen_params(df, cfg)
        deg = pp.appurtenance_degrees(df, P)
        c = pd.Series({"V1": 0, "V2": 1 / 3, "V3": 0})
        for op in ("intersection", "union"):
            vd = pp.aggregate_values(deg, c, cfg, operator=op)
            a, b = deg["V2"].iloc[:, 0].to_numpy(), deg["V2"].iloc[:, 1].to_numpy()
            T, S = a * b, a + b - a * b
            exp = (2 / 3) * T + (1 / 3) * S if op == "intersection" else (2 / 3) * S + (1 / 3) * T
            np.testing.assert_allclose(vd["V2"].to_numpy(), exp)

    def test_more_than_two_sources_raises(self):
        df, cfg = synth()
        df["a3"] = df["a1"] * 1.3
        deg = {"V1": pd.DataFrame(np.full((3, 3), .5), columns=["x", "y", "z"])}
        with self.assertRaisesRegex(ValueError, "two-assessment"):
            pp.aggregate_values(deg, pd.Series({"V1": .2}), cfg)

    def test_nary_needs_label_then_works(self):
        cfg = copy.deepcopy(CFG)
        deg = {"V": pd.DataFrame(np.full((2, 3), .5), columns=["x", "y", "z"])}
        cfg["aggregation"].update(allow_nary_extension=True, nary_label=None)
        with self.assertRaises(ValueError):
            pp.aggregate_values(deg, pd.Series({"V": .2}), cfg)
        cfg["aggregation"].update(nary_label="[D]", max_sources_per_value=3)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = pp.aggregate_values(deg, pd.Series({"V": 0.0}), cfg)
        self.assertTrue(any("[D]" in str(x.message) for x in w))
        np.testing.assert_allclose(r["V"], .125)          # c=0 -> product of three .5

    def test_config_rejects_unapproved_nary(self):
        df, cfg = synth(sources={"V1": [("a1", 1, "identity"), ("a2", 1, "identity"), ("b1", 1, "identity")]})
        with self.assertRaises(ValueError):
            sch.validate_config(cfg)

    def test_degrees_out_of_range_rejected(self):
        deg = {"V": pd.DataFrame({"x": [1.2], "y": [.5]})}
        with self.assertRaises(ValueError):
            pp.aggregate_values(deg, pd.Series({"V": .1}), CFG)

    def test_floor(self):
        d = pd.DataFrame({"a": [0.0, .004, .62]})
        np.testing.assert_allclose(pp.apply_floor(d, .01)["a"], [.01, .01, .62])
        self.assertEqual(d["a"].iloc[0], 0.0)             # input untouched (unfloored kept)


class TestContradiction(unittest.TestCase):
    def test_real_matrix_axioms_and_dominant(self):
        C = pd.DataFrame(PARAMS["contradiction"]["matrix"])
        a = C.to_numpy()
        self.assertTrue((np.diag(a) == 0).all())
        np.testing.assert_array_equal(a, a.T)
        self.assertTrue(((a >= 0) & (a <= 1)).all())
        self.assertEqual(C.loc[CFG["plithogenic"]["dominant_value"], CFG["plithogenic"]["dominant_value"]], 0.0)

    def test_matches_manual_spearman(self):
        from scipy import stats
        df, cfg = synth()
        P = pp.fit_frozen_params(df, cfg)
        deg = pp.appurtenance_degrees(df, P)
        m = {v: d.mean(axis=1) for v, d in deg.items()}
        rho = stats.spearmanr(m["V1"], m["V2"]).statistic
        self.assertAlmostEqual(P["contradiction"]["matrix"]["V1"]["V2"], (1 - rho) / 2, 12)

    def test_change_dominant_no_code_edit(self):
        df, cfg1 = synth(dominant="V1")
        _, cfg2 = synth(dominant="V2")
        P1, P2 = pp.fit_frozen_params(df, cfg1), pp.fit_frozen_params(df, cfg2)
        r1 = pp.run_plithogenic_piv(df, P1, cfg1)
        r2 = pp.run_plithogenic_piv(df, P2, cfg2)
        self.assertEqual(r1["dominant_value"], "V1"); self.assertEqual(r2["dominant_value"], "V2")
        self.assertEqual(r1["contradiction_used"]["V1"], 0.0); self.assertEqual(r2["contradiction_used"]["V2"], 0.0)
        self.assertFalse(np.allclose(r1["contradiction_used"].to_numpy(), r2["contradiction_used"].to_numpy()))

    def test_stale_params_after_dominant_change_raise(self):
        df, cfg1 = synth(dominant="V1")
        _, cfg2 = synth(dominant="V2")
        P1 = pp.fit_frozen_params(df, cfg1)
        with self.assertRaisesRegex(ValueError, "stale"):
            pp.run_plithogenic_piv(df, P1, cfg2)

    def test_nan_rho_raises(self):
        # dominant value has sources x (+) and x (-): mean degree is exactly 0.5 for everyone -> constant
        rng = np.random.default_rng(3)
        df = pd.DataFrame({"College_ID": [f"C{i}" for i in range(30)], "p": rng.random(30) + 1, "q": rng.random(30) + 1})
        df["p2"] = df["p"]
        _, cfg = synth()
        cfg["plithogenic"]["values"] = [
            {"name": "D", "sources": [{"column": "p", "direction": 1, "transform": "identity", "use_adjusted": False},
                                      {"column": "p2", "direction": -1, "transform": "identity", "use_adjusted": False}]},
            {"name": "E", "sources": [{"column": "q", "direction": 1, "transform": "identity", "use_adjusted": False}]}]
        cfg["plithogenic"]["dominant_value"] = "D"
        with self.assertRaisesRegex(ValueError, "NaN"):
            pp.fit_frozen_params(df, cfg)

    def test_expert_method(self):
        df, cfg = synth()
        names = ["V1", "V2", "V3"]
        M = {a: {b: (0 if a == b else .4) for b in names} for a in names}
        cfg["contradiction"].update(method="expert", expert_matrix=M)
        P = pp.fit_frozen_params(df, cfg)
        self.assertEqual(P["contradiction"]["matrix"]["V1"]["V3"], .4)

    def test_override_validation(self):
        names = list(PARAMS["values"])
        good = pd.DataFrame(PARAMS["contradiction"]["matrix"]).loc[names, names]
        bad_diag = good.copy(); bad_diag.iloc[0, 0] = .1
        bad_sym = good.copy(); bad_sym.iloc[0, 1] = min(1.0, bad_sym.iloc[0, 1] + .2)
        bad_rng = good.copy(); bad_rng.iloc[0, 1] = bad_rng.iloc[1, 0] = 1.5
        bad_nan = good.copy(); bad_nan.iloc[0, 1] = bad_nan.iloc[1, 0] = np.nan
        for bad in (bad_diag, bad_sym, bad_rng, bad_nan, good.iloc[:-1]):
            with self.assertRaises(ValueError):
                pp.run_plithogenic_piv(DF, PARAMS, CFG, contradiction_override=bad)
        ok = pp.run_plithogenic_piv(DF, PARAMS, CFG, contradiction_override=good * 0)
        self.assertEqual(ok["meta"]["contradiction_source"], "override")
        self.assertTrue((ok["contradiction_used"] == 0).all())

    def test_override_can_be_disabled(self):
        cfg = copy.deepcopy(CFG); cfg["contradiction"]["allow_runtime_override"] = False
        with self.assertRaises(ValueError):
            pp.run_plithogenic_piv(DF, PARAMS, cfg, contradiction_override=pd.DataFrame(PARAMS["contradiction"]["matrix"]))


class TestFrozenAnchors(unittest.TestCase):
    def test_same_college_same_degrees_full_vs_filtered(self):
        mask = DF["Institution_Type"] == DF["Institution_Type"].iloc[0]
        sub = DF[mask]
        self.assertLess(len(sub), len(DF))
        full = pp.appurtenance_degrees(DF, PARAMS, CFG)
        part = pp.appurtenance_degrees(sub, PARAMS, CFG)
        for v in full:
            pd.testing.assert_frame_equal(full[v].loc[part[v].index], part[v])

    def test_anchors_do_not_change_with_input(self):
        before = json.dumps(PARAMS, sort_keys=True)
        pp.run_plithogenic_piv(DF.sample(100, random_state=1), PARAMS, CFG)
        self.assertEqual(before, json.dumps(PARAMS, sort_keys=True))

    def test_lo_eq_hi_raises(self):
        df, cfg = synth()
        df["a1"] = 5.0
        with self.assertRaisesRegex(ValueError, "degenerate"):
            pp.fit_frozen_params(df, cfg)
        P = copy.deepcopy(PARAMS)
        k = next(iter(P["sources"])); P["sources"][k]["hi"] = P["sources"][k]["lo"]
        with self.assertRaises(ValueError):
            pp.appurtenance_degrees(DF, P)

    def test_log10_nonpositive_and_nan_raise(self):
        df, cfg = synth(); P = pp.fit_frozen_params(df, cfg)
        bad = df.copy(); bad.loc[0, "a2"] = 0
        with self.assertRaisesRegex(ValueError, "log10"):
            pp.appurtenance_degrees(bad, P)
        bad = df.copy(); bad.loc[0, "a1"] = np.nan
        with self.assertRaises(ValueError):
            pp.appurtenance_degrees(bad, P)

    def test_degree_direction_and_clip(self):
        df, cfg = synth(); P = pp.fit_frozen_params(df, cfg)
        d = pp.appurtenance_degrees(df, P)["V3"]["c1"]            # direction -1
        self.assertTrue(((d >= 0) & (d <= 1)).all())
        self.assertEqual(d[df["c1"].idxmax()], 0.0)
        self.assertEqual(d[df["c1"].idxmin()], 1.0)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            p = pp.save_frozen_params(PARAMS, Path(t) / "frozen_params.json")
            self.assertEqual(pp.load_frozen_params(p), PARAMS)

    def test_stale_percentiles_raise(self):
        cfg = copy.deepcopy(CFG); cfg["appurtenance"]["lower_percentile"] = 10
        with self.assertRaisesRegex(ValueError, "stale"):
            pp.run_plithogenic_piv(DF, PARAMS, cfg)


class TestRun(unittest.TestCase):
    R = pp.run_plithogenic_piv(DF, PARAMS, CFG)

    def test_keys_and_index(self):
        for k in ("scores", "ranks", "weights", "source_degrees", "value_degrees", "matrix_M",
                  "contradiction_matrix", "contradiction_used", "normalized_matrix", "weighted_matrix",
                  "dominant_value", "operator"):
            self.assertIn(k, self.R)
        for t in ("scores", "ranks", "value_degrees", "matrix_M", "normalized_matrix", "weighted_matrix"):
            self.assertEqual(self.R[t].index.name, CFG["id_column"])
            self.assertEqual(len(self.R[t]), len(DF))

    def test_ranking_is_computed_from_matrix_M(self):
        M = self.R["matrix_M"]
        piv = bp.piv_rank(M, self.R["weights"], pd.Series(1, index=M.columns),
                          CFG["piv"]["normalization"], CFG["piv"]["tie_decimals"])
        pd.testing.assert_series_equal(piv["scores"], self.R["scores"])
        pd.testing.assert_series_equal(piv["ranks"], self.R["ranks"])

    def test_M_is_floored_value_degrees_and_unfloored_kept(self):
        eps = PARAMS["epsilon"]
        np.testing.assert_allclose(self.R["matrix_M"].to_numpy(), np.maximum(self.R["value_degrees"].to_numpy(), eps))
        self.assertGreater(self.R["matrix_M"].to_numpy().min(), 0)
        self.assertGreater(self.R["meta"]["n_floored_cells"], 0)
        self.assertLess(self.R["value_degrees"].to_numpy().min(), eps)     # unfloored really kept

    def test_default_weights_are_merec_on_M(self):
        M = self.R["matrix_M"]
        w = bp.merec_weights(M, pd.Series(1, index=M.columns))
        np.testing.assert_allclose(self.R["weights"].to_numpy(), w.to_numpy())
        self.assertEqual(self.R["weights_source"], "merec")
        self.assertAlmostEqual(self.R["weights"].sum(), 1.0, 12)

    def test_weights_recomputed_on_filtered_subset(self):
        sub = DF[DF["Institution_Type"] == DF["Institution_Type"].iloc[0]]
        r = pp.run_plithogenic_piv(sub, PARAMS, CFG)
        self.assertFalse(np.allclose(r["weights"].to_numpy(), self.R["weights"].to_numpy()))
        self.assertTrue(r["meta"]["weights_recomputed_on_current_P"])

    def test_frozen_weights_equal_full_data_weights_and_dont_change(self):
        fw = pp.fit_frozen_weights(DF, PARAMS, CFG)
        sub = DF.sample(200, random_state=0)
        r = pp.run_plithogenic_piv(sub, PARAMS, CFG, frozen_weights=fw)
        np.testing.assert_allclose(r["weights"].to_numpy(), fw.to_numpy())
        self.assertEqual(r["weights_source"], "frozen")

    def test_recompute_flag_false_needs_frozen(self):
        cfg = copy.deepcopy(CFG); cfg["weights"]["recompute_on_filter"] = False
        with self.assertRaisesRegex(ValueError, "frozen_weights"):
            pp.run_plithogenic_piv(DF, PARAMS, cfg)

    def test_custom_weights(self):
        v = list(PARAMS["values"])
        r = pp.run_plithogenic_piv(DF, PARAMS, CFG, weight_mode="custom", custom_weights=[2, 1, 1, 0, 0])
        np.testing.assert_allclose(r["weights"].to_numpy(), [.5, .25, .25, 0, 0])
        r2 = pp.run_plithogenic_piv(DF, PARAMS, CFG, weight_mode="custom", custom_weights=dict(zip(v, [.2] * 5)))
        np.testing.assert_allclose(r2["weights"].to_numpy(), [.2] * 5)
        for bad in ([1, 2], [1, 1, 1, 1, -1], [1, 1, 1, 1, np.nan], [0] * 5, [np.inf, 1, 1, 1, 1], dict(zip(v[:4], [1] * 4))):
            with self.assertRaises(ValueError):
                pp.run_plithogenic_piv(DF, PARAMS, CFG, weight_mode="custom", custom_weights=bad)
        with self.assertRaises(ValueError):
            pp.run_plithogenic_piv(DF, PARAMS, CFG, weight_mode="custom")
        with self.assertRaises(ValueError):
            pp.run_plithogenic_piv(DF, PARAMS, CFG, weight_mode="custom", custom_weights=[1] * 5,
                                   frozen_weights=[1] * 5)

    def test_exclude_values(self):
        ex = list(PARAMS["values"])[-1]
        r = pp.run_plithogenic_piv(DF, PARAMS, CFG, exclude_values=[ex])
        self.assertNotIn(ex, r["matrix_M"].columns)
        self.assertNotIn(ex, r["weights"].index)
        self.assertEqual(r["excluded_values"], [ex])
        with self.assertRaises(ValueError):
            pp.run_plithogenic_piv(DF, PARAMS, CFG, exclude_values=["nope"])
        with self.assertRaises(ValueError):
            pp.run_plithogenic_piv(DF, PARAMS, CFG, exclude_values=list(PARAMS["values"]))

    def test_operator_and_normalization_overrides(self):
        a = pp.run_plithogenic_piv(DF, PARAMS, CFG, operator="intersection")
        b = pp.run_plithogenic_piv(DF, PARAMS, CFG, operator="union")
        self.assertFalse(np.allclose(a["matrix_M"].to_numpy(), b["matrix_M"].to_numpy()))
        self.assertEqual(b["operator"], "union")
        n = pp.run_plithogenic_piv(DF, PARAMS, CFG, normalization="none")
        self.assertEqual(n["meta"]["normalization"], "none")
        with self.assertRaises(ValueError): pp.run_plithogenic_piv(DF, PARAMS, CFG, operator="xor")
        with self.assertRaises(ValueError): pp.run_plithogenic_piv(DF, PARAMS, CFG, normalization="zscore")

    def test_ranks_valid(self):
        r = self.R["ranks"]
        self.assertEqual(r.min(), 1)
        self.assertTrue((r[self.R["scores"].round(CFG["piv"]["tie_decimals"]) == self.R["scores"].round(CFG["piv"]["tie_decimals"]).min()] == 1).all())

    def test_no_neutrosophic_columns_used(self):
        d2 = DF.copy()
        for c in CFG["columns"]["display_only"]:
            d2[c] = np.random.default_rng(0).random(len(d2))
        r2 = pp.run_plithogenic_piv(d2, PARAMS, CFG)
        pd.testing.assert_series_equal(r2["ranks"], self.R["ranks"])

    def test_missing_or_duplicate_id(self):
        with self.assertRaises(ValueError):
            pp.run_plithogenic_piv(DF.drop(columns=[CFG["id_column"]]).reset_index(drop=True), PARAMS, CFG)
        dup = pd.concat([DF.head(3), DF.head(3)])
        with self.assertRaises(ValueError):
            pp.run_plithogenic_piv(dup, PARAMS, CFG)

    def test_plithogenic_set_view(self):
        S = pp.as_plithogenic_set(self.R, PARAMS)
        self.assertEqual(S.a, CFG["plithogenic"]["attribute"])
        self.assertEqual(S.v_D, CFG["plithogenic"]["dominant_value"])
        self.assertEqual(list(S.V), sch.value_names(CFG))
        self.assertEqual(len(S.P), len(DF))


if __name__ == "__main__":
    unittest.main()

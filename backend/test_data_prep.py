"""
Tests for schema.py and data_prep.py. Run:  python -m unittest -v test_data_prep   (pytest also works)

Column names are read from ppiv_config.json wherever possible; the raw CSV is the one named in
paths.input_csv (tests are skipped if it is not next to the config).
"""
import copy
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import data_prep as dp
import schema as sch

HERE = Path(__file__).resolve().parent
CFG0 = sch.load_config(HERE / "ppiv_config.json")
RAW_PATH = sch.resolve_path(CFG0, "input_csv")


@unittest.skipUnless(RAW_PATH.is_file(), f"raw CSV {RAW_PATH} not found")
class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw0 = pd.read_csv(RAW_PATH)

    def setUp(self):
        self.cfg = copy.deepcopy(CFG0)
        self.raw = self.raw0.copy()
        # convenient config-derived names
        self.srcs = sch.all_sources(self.cfg)
        self.adj_src = next(s for s in self.srcs if s["use_adjusted"])
        self.log_src = next(s for s in self.srcs if s["transform"] == "log10")
        self.conf_col = self.cfg["shrinkage"]["columns"][self.adj_src["column"]]


class TestConfigValidation(Base):
    def test_valid_config_passes_both_stages(self):
        sch.validate_config(self.cfg, self.raw, stage="raw")
        ready = dp.prepare(self.raw, self.cfg, verbose=False)
        sch.validate_config(self.cfg, ready, stage="prepared")

    def test_raw_stage_does_not_require_generated_columns(self):
        gen = set(sch.generated_columns(self.cfg))
        self.assertTrue(gen)                              # there are generated columns...
        self.assertFalse(gen & set(self.raw.columns))     # ...and none are in the raw CSV
        sch.validate_config(self.cfg, self.raw, stage="raw")   # must not raise

    def test_prepared_stage_requires_generated_columns(self):
        with self.assertRaises(ValueError) as cm:
            sch.validate_config(self.cfg, self.raw, stage="prepared")
        self.assertIn(self.cfg["shrinkage"]["suffix"], str(cm.exception))

    def test_missing_column(self):
        raw = self.raw.drop(columns=[self.log_src["column"]])
        with self.assertRaises(ValueError) as cm:
            sch.validate_config(self.cfg, raw, stage="raw")
        self.assertIn(self.log_src["column"], str(cm.exception))

    def test_bad_direction(self):
        for bad in (0, 2, "1", True, None, 1.0):
            cfg = copy.deepcopy(self.cfg)
            cfg["plithogenic"]["values"][0]["sources"][0]["direction"] = bad
            with self.assertRaises(ValueError, msg=f"direction={bad!r}") as cm:
                sch.validate_config(cfg)
            self.assertIn("direction", str(cm.exception))

    def test_bad_dominant_value(self):
        self.cfg["plithogenic"]["dominant_value"] = "NoSuchValue"
        with self.assertRaises(ValueError) as cm:
            sch.validate_config(self.cfg)
        self.assertIn("dominant_value", str(cm.exception))

    def test_duplicate_value_names(self):
        vals = self.cfg["plithogenic"]["values"]
        vals[1]["name"] = vals[0]["name"]
        with self.assertRaises(ValueError) as cm:
            sch.validate_config(self.cfg)
        self.assertIn("Duplicate value names", str(cm.exception))

    def test_bad_percentiles(self):
        for lo, hi in ((0, 95), (5, 100), (95, 5), (5, 5), (-1, 50), (5, "x")):
            cfg = copy.deepcopy(self.cfg)
            cfg["appurtenance"]["lower_percentile"], cfg["appurtenance"]["upper_percentile"] = lo, hi
            with self.assertRaises(ValueError, msg=f"{lo},{hi}"):
                sch.validate_config(cfg)

    def test_bad_epsilon_and_names(self):
        for path, key, bad in (("floor", "epsilon", 0), ("floor", "epsilon", 1), ("aggregation", "operator", "xor"),
                               ("aggregation", "t_norm", "nope"), ("aggregation", "t_conorm", "nope"),
                               ("piv", "normalization", "zscore")):
            cfg = copy.deepcopy(self.cfg)
            cfg[path][key] = bad
            with self.assertRaises(ValueError, msg=f"{path}.{key}={bad}"):
                sch.validate_config(cfg)

    def test_more_than_two_sources_rejected_unless_approved(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["plithogenic"]["values"][0]["sources"].append(
            {"column": "X", "direction": 1, "transform": "identity", "use_adjusted": False})
        with self.assertRaises(ValueError) as cm:
            sch.validate_config(cfg)
        self.assertIn("at most 2", str(cm.exception))

    def test_expert_matrix_axioms(self):
        names = sch.value_names(self.cfg)
        good = {a: {b: (0.0 if a == b else 0.5) for b in names} for a in names}
        cfg = copy.deepcopy(self.cfg)
        cfg["contradiction"].update(method="expert", expert_matrix=good)
        sch.validate_config(cfg)                                   # ok
        bad = copy.deepcopy(good); bad[names[0]][names[1]] = 0.9   # asymmetric
        cfg["contradiction"]["expert_matrix"] = bad
        with self.assertRaises(ValueError):
            sch.validate_config(cfg)
        bad = copy.deepcopy(good); bad[names[0]][names[0]] = 0.1   # c(v,v) != 0
        cfg["contradiction"]["expert_matrix"] = bad
        with self.assertRaises(ValueError):
            sch.validate_config(cfg)


class TestPrepare(Base):
    def test_nan_in_source_raises_without_strategy(self):
        self.raw.loc[3, self.log_src["column"]] = np.nan
        with self.assertRaises(ValueError) as cm:
            dp.prepare(self.raw, self.cfg, verbose=False)
        self.assertIn("NaN", str(cm.exception))

    def test_nan_imputed_only_when_configured_and_flagged(self):
        col = self.log_src["column"]
        self.raw.loc[3, col] = np.nan
        median = float(self.raw[col].median())          # median of the remaining rows
        self.cfg["missing_values"]["strategies"] = {col: {"method": "median"}}
        ready = dp.prepare(self.raw, self.cfg, verbose=False)
        self.assertEqual(len(ready), len(self.raw))                         # row kept
        self.assertAlmostEqual(float(ready.loc[3, col]), median)
        self.assertIn(f"imputed:{col}", ready.loc[3, self.cfg["quality_flags"]["column"]])
        self.assertTrue(np.isnan(self.raw.loc[3, col]))                     # input untouched

    def test_inf_raises(self):
        self.raw.loc[0, self.log_src["column"]] = np.inf
        with self.assertRaises(ValueError) as cm:
            dp.prepare(self.raw, self.cfg, verbose=False)
        self.assertIn("inf", str(cm.exception))

    def test_non_numeric_source_raises(self):
        self.raw[self.log_src["column"]] = self.raw[self.log_src["column"]].astype(str)
        with self.assertRaises(ValueError) as cm:
            dp.prepare(self.raw, self.cfg, verbose=False)
        self.assertIn("numeric", str(cm.exception))

    def test_log10_of_zero_raises(self):
        self.raw.loc[0, self.log_src["column"]] = 0
        with self.assertRaises(ValueError) as cm:
            dp.prepare(self.raw, self.cfg, verbose=False)
        self.assertIn("log10", str(cm.exception))
        self.assertIn(self.log_src["column"], str(cm.exception))

    def test_log10_of_negative_raises(self):
        self.raw.loc[0, self.log_src["column"]] = -5
        with self.assertRaises(ValueError):
            dp.prepare(self.raw, self.cfg, verbose=False)

    def test_shrinkage_hand_check(self):
        col, conf = self.adj_src["column"], self.conf_col
        small = self.raw.head(4).copy()
        small[col] = [0.2, 0.4, 0.6, 0.8]            # mean over these 4 rows = 0.5
        small[conf] = [1.0, 0.0, 0.5, 0.25]
        # the other shrinkage columns must still be in the scale range; they are (real data)
        ready = dp.prepare(small, self.cfg, verbose=False)
        # expected: conf*x + (1-conf)*0.5
        expected = [1.0 * 0.2 + 0.0 * 0.5,      # 0.2
                    0.0 * 0.4 + 1.0 * 0.5,      # 0.5
                    0.5 * 0.6 + 0.5 * 0.5,      # 0.55
                    0.25 * 0.8 + 0.75 * 0.5]    # 0.575
        np.testing.assert_allclose(ready[sch.adjusted_name(self.cfg, col)].to_numpy(), expected, atol=1e-12)
        np.testing.assert_allclose(ready[col].to_numpy(), small[col].to_numpy())    # original kept
        self.assertAlmostEqual(ready.attrs["prep_stats"]["shrinkage_means"][col], 0.5)

    def test_shrinkage_uses_all_rows_mean(self):
        col = self.adj_src["column"]
        ready = dp.prepare(self.raw, self.cfg, verbose=False)
        self.assertAlmostEqual(ready.attrs["prep_stats"]["shrinkage_means"][col], float(self.raw[col].mean()))

    def test_scale_check_raises_outside_expected_range(self):
        self.raw.loc[0, self.adj_src["column"]] = 7.0
        with self.assertRaises(ValueError) as cm:
            dp.prepare(self.raw, self.cfg, verbose=False)
        self.assertIn("Scale check", str(cm.exception))

    def test_confidence_outside_unit_interval_raises(self):
        self.raw.loc[0, self.conf_col] = 1.5
        with self.assertRaises(ValueError):
            dp.prepare(self.raw, self.cfg, verbose=False)

    def test_facility_count(self):
        d = self.cfg["derived"][0]
        small = self.raw.head(6).copy()
        small[d["from"]] = ["a, b, c", "a", "", " , ,a,, b ,", "x,y,z,w,v", "a,b"]
        ready = dp.prepare(small, self.cfg, verbose=False)
        self.assertEqual(ready[d["name"]].tolist(), [3, 1, 0, 2, 5, 2])
        self.assertEqual(str(ready[d["name"]].dtype), "int64")
        # and on the full data it matches an independent count
        full = dp.prepare(self.raw, self.cfg, verbose=False)
        indep = self.raw[d["from"]].map(lambda s: len([t for t in s.split(d["delimiter"]) if t.strip()]))
        self.assertTrue((full[d["name"]].to_numpy() == indep.to_numpy()).all())

    def test_never_drops_rows_or_edits_originals(self):
        ready = dp.prepare(self.raw, self.cfg, verbose=False)
        self.assertEqual(len(ready), len(self.raw))
        for c in self.raw.columns:
            pd.testing.assert_series_equal(ready[c], self.raw[c], check_names=True)

    def test_flags_semicolon_separated_and_ops(self):
        fcol = self.cfg["quality_flags"]["column"]
        ready = dp.prepare(self.raw, self.cfg, verbose=False)
        for rule in self.cfg["quality_flags"]["rules"]:
            s = ready[rule["column"]]
            if rule["op"] == "lt":
                exp = s < rule["value"]
            elif rule["op"] == "gt":
                exp = s > rule["value"]
            elif rule["op"] == "startswith":
                exp = s.str.startswith(rule["value"])
            else:
                exp = s.str.contains(rule["value"], regex=False)
            got = ready[fcol].map(lambda c: rule["name"] in c.split(";"))
            self.assertTrue((got.to_numpy() == exp.to_numpy()).all(), rule["name"])
        multi = ready[fcol][ready[fcol].str.contains(";")]
        self.assertTrue(len(multi) > 0 and not multi.str.contains("; ").any())

    def test_unknown_flag_op_rejected(self):
        self.cfg["quality_flags"]["rules"][0]["op"] = "regex"
        with self.assertRaises(ValueError):
            sch.validate_config(self.cfg)


class TestAuditAndDummy(Base):
    def test_full_audit_raw_and_ready(self):
        t = sch.audit_columns(self.raw, self.cfg, quiet=True)
        self.assertEqual(len(t), self.raw.shape[1])
        self.assertTrue((t["Role"] != "").all())
        ready = dp.prepare(self.raw, self.cfg, verbose=False)
        t2 = sch.audit_columns(ready, self.cfg, quiet=True)
        self.assertEqual(len(t2), ready.shape[1])

    def test_audit_lists_all_unclassified(self):
        self.raw["Surprise_A"] = 1
        self.raw["Surprise_B"] = 2
        with self.assertRaises(ValueError) as cm:
            sch.audit_columns(self.raw, self.cfg, quiet=True)
        self.assertIn("Surprise_A", str(cm.exception))
        self.assertIn("Surprise_B", str(cm.exception))

    def test_excluded_needs_reason(self):
        k = next(iter(self.cfg["columns"]["excluded"]))
        self.cfg["columns"]["excluded"][k] = ""
        with self.assertRaises(ValueError):
            sch.validate_config(self.cfg)

    def test_dummy_schema_dtypes_and_both_types(self):
        ready = dp.prepare(self.raw, self.cfg, verbose=False)
        dummy = dp.make_dummy(ready, self.cfg)
        self.assertEqual(len(dummy), self.cfg["dummy"]["n_rows"])
        self.assertEqual(list(dummy.columns), list(ready.columns))
        self.assertTrue((dummy.dtypes == ready.dtypes).all())
        strat = self.cfg["dummy"]["stratify_column"]
        self.assertEqual(set(dummy[strat]), set(ready[strat]))        # Government and Private both present
        pd.testing.assert_frame_equal(dummy, dp.make_dummy(ready, self.cfg))   # deterministic

    def test_csv_roundtrip_dtypes_match(self):
        import tempfile
        ready = dp.prepare(self.raw, self.cfg, verbose=False)
        dummy = dp.make_dummy(ready, self.cfg)
        with tempfile.TemporaryDirectory() as td:
            ready.to_csv(Path(td) / "r.csv", index=False)
            dummy.to_csv(Path(td) / "d.csv", index=False)
            r, d = dp.load_ready(Path(td) / "r.csv", self.cfg), dp.load_ready(Path(td) / "d.csv", self.cfg)
        self.assertTrue((r.dtypes == d.dtypes).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)

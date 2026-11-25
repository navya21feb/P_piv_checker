"""
schema.py -- configuration loading, validation and column audit for P-PIV.

Scope: single-valued fuzzy plithogenic PIV. NOT neutrosophic.
Nothing about the data (column names, directions, thresholds) is hard-coded here;
all of it comes from ppiv_config.json or the DataFrame. The only constants below are
the *vocabularies* of operator / transform names that the later modules implement.

Tags: [S] = from an original paper, [D] = team design choice.
Equation / table numbers are UNVERIFIED (no PDF was available when this was written).

Public API
----------
load_config(path)                      -> dict
validate_config(cfg, df=None, stage)   -> None (raises ValueError listing ALL problems)
audit_columns(df, cfg, quiet=False)    -> DataFrame (raises ValueError on unclassified columns)
check_source_values(df, cfg, stage)    -> list[str] of problems (used by data_prep)
all_sources(cfg), value_names(cfg), get_value(cfg, name), effective_column(cfg, src),
adjusted_name(cfg, col), generated_columns(cfg), resolve_path(cfg, key)

Validation order expected by the pipeline:
  1. validate_config(cfg, df_raw,  stage="raw")       raw columns only, generated ones skipped
  2. prepare(df_raw, cfg)                              (data_prep.py)
  3. validate_config(cfg, df_ready, stage="prepared")  full config, generated columns required
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- vocabularies
# Names that downstream modules must implement. Changing one here is an interface change.
VALID_DIRECTIONS = (1, -1)
VALID_TRANSFORMS = ("identity", "log10")
VALID_OPERATORS = ("intersection", "union")
VALID_T_NORMS = ("product", "minimum", "lukasiewicz")
VALID_T_CONORMS = ("probabilistic_sum", "maximum", "bounded_sum")
VALID_CONTRADICTION_METHODS = ("spearman_on_mean_degrees", "expert")
VALID_NORMALIZATIONS = ("vector", "none")
VALID_WEIGHT_METHODS = ("merec",)
VALID_FLAG_OPS = ("contains", "startswith", "lt", "gt")
VALID_IMPUTE_METHODS = ("median", "mean", "constant")
VALID_DERIVED_TYPES = ("count_delimited",)

ROLE_KEYS = ("labels", "filters", "display_only", "reliability_metadata", "excluded")
_TOL = 1e-12


# ---------------------------------------------------------------- small helpers
def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_nonempty_str(x) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


# ---------------------------------------------------------------- loading + accessors
def load_config(path) -> dict:
    """Read ppiv_config.json. Remembers its folder so relative paths can be resolved."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"{p}: top level of the config must be a JSON object.")
    cfg["_config_dir"] = str(p.parent.resolve())
    return cfg


def resolve_path(cfg: dict, key: str, base_dir=None) -> Path:
    """cfg['paths'][key] resolved against base_dir (default: the config's own folder)."""
    rel = Path(cfg["paths"][key])
    if rel.is_absolute():
        return rel
    return Path(base_dir or cfg.get("_config_dir", ".")) / rel


def value_names(cfg: dict) -> list:
    return [v["name"] for v in cfg["plithogenic"]["values"]]


def get_value(cfg: dict, name: str) -> dict:
    for v in cfg["plithogenic"]["values"]:
        if v["name"] == name:
            return v
    raise KeyError(f"Value '{name}' not in plithogenic.values")


def adjusted_name(cfg: dict, col: str) -> str:
    return col + cfg["shrinkage"]["suffix"]


def effective_column(cfg: dict, src: dict) -> str:
    """Column P-PIV reads for this source: the __adj column if use_adjusted, else the raw one."""
    return adjusted_name(cfg, src["column"]) if src.get("use_adjusted", False) else src["column"]


def all_sources(cfg: dict) -> list:
    """Flat list of source dicts (copies) in config order, each tagged with its value name and
    'effective_column'. The classical baseline uses src['column'] (raw); P-PIV uses
    src['effective_column']."""
    out = []
    for v in cfg["plithogenic"]["values"]:
        for s in v["sources"]:
            d = dict(s)
            d["value"] = v["name"]
            d["effective_column"] = effective_column(cfg, s)
            out.append(d)
    return out


def generated_columns(cfg: dict) -> list:
    """Columns that prepare() creates (derived, __adj, flags). They may legitimately be absent
    from the raw CSV."""
    gen = [d["name"] for d in cfg.get("derived", [])]
    gen += [adjusted_name(cfg, c) for c in cfg.get("shrinkage", {}).get("columns", {})]
    flag_col = cfg.get("quality_flags", {}).get("column")
    if flag_col:
        gen.append(flag_col)
    return gen


def _referenced_columns(cfg: dict) -> set:
    """Every column name the config mentions, generated ones included."""
    cols = set()
    if cfg.get("id_column"):
        cols.add(cfg["id_column"])
    c = cfg.get("columns", {})
    for k in ("labels", "filters", "display_only", "reliability_metadata"):
        cols.update(c.get(k, []))
    cols.update(c.get("excluded", {}).keys())
    for v in cfg.get("plithogenic", {}).get("values", []):
        for s in v.get("sources", []):
            cols.add(s.get("column"))
    for d in cfg.get("derived", []):
        cols.add(d.get("from"))
        cols.add(d.get("name"))
    sh = cfg.get("shrinkage", {}).get("columns", {})
    cols.update(sh.keys())
    cols.update(sh.values())
    for r in cfg.get("quality_flags", {}).get("rules", []):
        cols.add(r.get("column"))
    cols.update(cfg.get("missing_values", {}).get("strategies", {}).keys())
    cols.update(cfg.get("scale_inspection", {}).get("columns", []))
    if cfg.get("dummy", {}).get("stratify_column"):
        cols.add(cfg["dummy"]["stratify_column"])
    cols.discard(None)
    return cols


# ---------------------------------------------------------------- role map (audit + conflicts)
def _role_map(cfg: dict) -> dict:
    roles: dict = {}

    def add(col, role, detail=""):
        roles.setdefault(col, []).append((role, detail))

    if cfg.get("id_column"):
        add(cfg["id_column"], "label", "id")
    c = cfg.get("columns", {})
    for k, role in (("labels", "label"), ("filters", "filter"),
                    ("display_only", "display_only"), ("reliability_metadata", "reliability_metadata")):
        for col in c.get(k, []):
            add(col, role)
    for col, reason in c.get("excluded", {}).items():
        add(col, "excluded", reason)
    for v in cfg.get("plithogenic", {}).get("values", []):
        for s in v.get("sources", []):
            adj = " (uses __adj)" if s.get("use_adjusted") else ""
            add(s.get("column"), "source",
                f"value={v.get('name')} dir={s.get('direction')} transform={s.get('transform')}{adj}")
    for d in cfg.get("derived", []):
        add(d.get("name"), "derived", f"{d.get('type')} of {d.get('from')}")
        add(d.get("from"), "derived_input", f"-> {d.get('name')}")
    sh = cfg.get("shrinkage", {})
    for col, conf in sh.get("columns", {}).items():
        add(col + sh.get("suffix", "__adj"), "derived", f"shrinkage of {col} by {conf}")
    qf = cfg.get("quality_flags", {})
    if qf.get("column"):
        add(qf["column"], "derived", "data-quality flags")
    return roles


_EXCLUSIVE = [("excluded", None),            # excluded may not combine with anything
              ("source", "display_only"), ("source", "reliability_metadata"),
              ("display_only", "reliability_metadata")]


# ---------------------------------------------------------------- structural validation
def _structure_errors(cfg: dict) -> list:
    errs: list = []

    def section(name):
        s = cfg.get(name)
        if not isinstance(s, dict):
            errs.append(f"Config section '{name}' is missing or not an object.")
            return {}
        return s

    if not _is_nonempty_str(cfg.get("id_column")):
        errs.append("id_column must be a non-empty string.")
    section("paths")
    cols = section("columns")
    for k in ROLE_KEYS:
        if k not in cols:
            errs.append(f"columns.{k} is missing.")
    for k in ROLE_KEYS[:-1]:
        if k in cols and not (isinstance(cols[k], list) and all(_is_nonempty_str(x) for x in cols[k])):
            errs.append(f"columns.{k} must be a list of column names.")
    ex = cols.get("excluded", {})
    if not isinstance(ex, dict):
        errs.append("columns.excluded must be an object {column: reason}.")
    else:
        for col, reason in ex.items():
            if not _is_nonempty_str(reason):
                errs.append(f"columns.excluded['{col}'] needs a non-empty reason.")

    # ---- derived
    derived = cfg.get("derived", [])
    if not isinstance(derived, list):
        errs.append("derived must be a list.")
        derived = []
    seen = set()
    for d in derived:
        if not isinstance(d, dict) or not _is_nonempty_str(d.get("name")) or not _is_nonempty_str(d.get("from")):
            errs.append(f"derived entry {d!r} needs 'name' and 'from'.")
            continue
        if d["name"] in seen:
            errs.append(f"Duplicate derived name '{d['name']}'.")
        seen.add(d["name"])
        if d.get("type") not in VALID_DERIVED_TYPES:
            errs.append(f"derived '{d['name']}': type {d.get('type')!r} not in {VALID_DERIVED_TYPES}.")
        if d.get("type") == "count_delimited" and not _is_nonempty_str(d.get("delimiter")):
            errs.append(f"derived '{d['name']}': count_delimited needs a non-empty 'delimiter'.")

    # ---- shrinkage
    sh = section("shrinkage")
    if not _is_nonempty_str(sh.get("suffix")):
        errs.append("shrinkage.suffix must be a non-empty string.")
    shcols = sh.get("columns", {})
    if not isinstance(shcols, dict):
        errs.append("shrinkage.columns must be an object {column: confidence_column}.")
        shcols = {}

    # ---- plithogenic set
    pl = section("plithogenic")
    if not _is_nonempty_str(pl.get("attribute")):
        errs.append("plithogenic.attribute must be a non-empty string.")
    values = pl.get("values")
    if not isinstance(values, list) or not values:
        errs.append("plithogenic.values must be a non-empty list.")
        values = []
    names = [v.get("name") if isinstance(v, dict) else None for v in values]
    if any(not _is_nonempty_str(n) for n in names):
        errs.append("every plithogenic value needs a non-empty 'name'.")
    dup = sorted({n for n in names if n is not None and names.count(n) > 1})
    if dup:
        errs.append(f"Duplicate value names: {dup}")
    if pl.get("dominant_value") not in names:
        errs.append(f"plithogenic.dominant_value {pl.get('dominant_value')!r} is not one of the values {names}.")

    agg = section("aggregation")
    max_src = agg.get("max_sources_per_value")
    if not (isinstance(max_src, int) and not isinstance(max_src, bool) and max_src >= 1):
        errs.append("aggregation.max_sources_per_value must be an integer >= 1.")
        max_src = 2
    nary_ok = agg.get("allow_nary_extension") is True
    if nary_ok and agg.get("nary_label") != "[D]":
        errs.append("aggregation.allow_nary_extension=true requires nary_label '[D]' (an n-ary extension "
                    "is a team design choice and must be labelled as such).")
    if not nary_ok and max_src > 2:
        errs.append("aggregation.max_sources_per_value > 2 needs allow_nary_extension=true with nary_label '[D]'.")

    seen_cols = {}
    for v in values:
        if not isinstance(v, dict):
            continue
        vn = v.get("name")
        srcs = v.get("sources")
        if not isinstance(srcs, list) or not srcs:
            errs.append(f"value '{vn}' needs a non-empty 'sources' list.")
            continue
        limit = max_src if nary_ok else min(max_src, 2)
        if len(srcs) > limit:
            errs.append(f"value '{vn}' has {len(srcs)} sources; at most {limit} are allowed "
                        "(the [S] formulas are two-assessment; an n-ary extension must be approved and labelled [D]).")
        for s in srcs:
            col = s.get("column") if isinstance(s, dict) else None
            if not _is_nonempty_str(col):
                errs.append(f"value '{vn}': a source has no 'column'.")
                continue
            d = s.get("direction")
            if not (type(d) is int and d in VALID_DIRECTIONS):
                errs.append(f"source '{col}' (value '{vn}'): direction {d!r} must be 1 or -1.")
            if s.get("transform") not in VALID_TRANSFORMS:
                errs.append(f"source '{col}' (value '{vn}'): transform {s.get('transform')!r} not in {VALID_TRANSFORMS}.")
            if not isinstance(s.get("use_adjusted"), bool):
                errs.append(f"source '{col}' (value '{vn}'): use_adjusted must be true or false.")
            elif s["use_adjusted"] and col not in shcols:
                errs.append(f"source '{col}' (value '{vn}'): use_adjusted=true but '{col}' is not in shrinkage.columns.")
            if col in seen_cols:
                errs.append(f"source column '{col}' is used in both '{seen_cols[col]}' and '{vn}'; "
                            "a column may feed only one value (the baseline would double-count it).")
            seen_cols[col] = vn

    # ---- appurtenance percentiles
    ap = section("appurtenance")
    lo, hi = ap.get("lower_percentile"), ap.get("upper_percentile")
    if not (_is_num(lo) and _is_num(hi) and 0 < lo < hi < 100):
        errs.append(f"appurtenance percentiles must satisfy 0 < lower < upper < 100 (got {lo!r}, {hi!r}).")

    # ---- contradiction
    ct = section("contradiction")
    if ct.get("method") not in VALID_CONTRADICTION_METHODS:
        errs.append(f"contradiction.method {ct.get('method')!r} not in {VALID_CONTRADICTION_METHODS}.")
    if ct.get("method") == "expert":
        errs += _expert_matrix_errors(ct.get("expert_matrix"), [n for n in names if n])

    # ---- aggregation names
    if agg.get("operator") not in VALID_OPERATORS:
        errs.append(f"aggregation.operator {agg.get('operator')!r} not in {VALID_OPERATORS}.")
    if agg.get("t_norm") not in VALID_T_NORMS:
        errs.append(f"aggregation.t_norm {agg.get('t_norm')!r} not in {VALID_T_NORMS}.")
    if agg.get("t_conorm") not in VALID_T_CONORMS:
        errs.append(f"aggregation.t_conorm {agg.get('t_conorm')!r} not in {VALID_T_CONORMS}.")

    # ---- floor, weights, piv
    eps = section("floor").get("epsilon")
    if not (_is_num(eps) and 0 < eps < 1):
        errs.append(f"floor.epsilon must be in (0,1) (got {eps!r}).")
    w = section("weights")
    if w.get("method") not in VALID_WEIGHT_METHODS:
        errs.append(f"weights.method {w.get('method')!r} not in {VALID_WEIGHT_METHODS}.")
    if not isinstance(w.get("recompute_on_filter"), bool):
        errs.append("weights.recompute_on_filter must be true or false.")
    pv = section("piv")
    if pv.get("normalization") not in VALID_NORMALIZATIONS:
        errs.append(f"piv.normalization must be one of {VALID_NORMALIZATIONS}.")
    td = pv.get("tie_decimals")
    if not (isinstance(td, int) and not isinstance(td, bool) and 0 <= td <= 15):
        errs.append(f"piv.tie_decimals must be an integer in [0,15] (got {td!r}).")

    # ---- validation settings (used by validation.py)
    va = section("validation")
    rv = va.get("reversal", {})
    if not isinstance(rv, dict):
        errs.append("validation.reversal must be an object.")
        rv = {}
    for k in ("n_iterations", "max_pairs"):
        v = rv.get(k)
        if not (isinstance(v, int) and not isinstance(v, bool) and v >= 1):
            errs.append(f"validation.reversal.{k} must be an integer >= 1 (got {v!r}).")
    df_ = rv.get("drop_fraction")
    if not (_is_num(df_) and 0 < df_ < 1):
        errs.append(f"validation.reversal.drop_fraction must be in (0,1) (got {df_!r}).")
    tk = va.get("top_k")
    if not (isinstance(tk, list) and tk and all(isinstance(k, int) and not isinstance(k, bool) and k >= 1 for k in tk)):
        errs.append("validation.top_k must be a non-empty list of integers >= 1.")
    sv = va.get("sensitivity", {})
    if not isinstance(sv, dict):
        errs.append("validation.sensitivity must be an object.")
        sv = {}
    wp = sv.get("weight_perturbation")
    if not (_is_num(wp) and 0 < wp < 1):
        errs.append(f"validation.sensitivity.weight_perturbation must be in (0,1) (got {wp!r}).")
    pg = sv.get("param_grid", {})
    if not isinstance(pg, dict):
        errs.append("validation.sensitivity.param_grid must be an object {name: [override, ...]}.")
        pg = {}
    for name, entries in pg.items():
        if not (isinstance(entries, list) and entries and all(
                isinstance(e, dict) and e and all(_is_nonempty_str(p) for p in e) for e in entries)):
            errs.append(f"param_grid['{name}'] must be a non-empty list of non-empty {{dotted.path: value}} objects.")
    if not isinstance(section("baseline").get("use_adjusted_columns"), bool):
        errs.append("baseline.use_adjusted_columns must be true or false.")

    # ---- scale inspection
    si = section("scale_inspection")
    if not (isinstance(si.get("columns"), list) and _is_num(si.get("expected_min")) and _is_num(si.get("expected_max"))
            and si["expected_min"] < si["expected_max"]):
        errs.append("scale_inspection needs columns (list) and expected_min < expected_max.")

    # ---- missing values
    mv = section("missing_values").get("strategies", {})
    if not isinstance(mv, dict):
        errs.append("missing_values.strategies must be an object.")
        mv = {}
    for col, st in mv.items():
        if not isinstance(st, dict) or st.get("method") not in VALID_IMPUTE_METHODS:
            errs.append(f"missing_values.strategies['{col}']: method must be one of {VALID_IMPUTE_METHODS}.")
        elif st["method"] == "constant" and "value" not in st:
            errs.append(f"missing_values.strategies['{col}']: constant needs a 'value'.")

    # ---- quality flags
    qf = section("quality_flags")
    if not _is_nonempty_str(qf.get("column")):
        errs.append("quality_flags.column must be a non-empty string.")
    if not isinstance(qf.get("flag_imputed"), bool):
        errs.append("quality_flags.flag_imputed must be true or false.")
    rules = qf.get("rules", [])
    if not isinstance(rules, list):
        errs.append("quality_flags.rules must be a list.")
        rules = []
    rnames = [r.get("name") for r in rules if isinstance(r, dict)]
    for n in sorted({n for n in rnames if rnames.count(n) > 1}):
        errs.append(f"Duplicate quality flag name '{n}'.")
    for r in rules:
        if not isinstance(r, dict) or not _is_nonempty_str(r.get("name")) or not _is_nonempty_str(r.get("column")):
            errs.append(f"quality flag {r!r} needs 'name' and 'column'.")
            continue
        if r.get("op") not in VALID_FLAG_OPS:
            errs.append(f"quality flag '{r['name']}': op {r.get('op')!r} not in {VALID_FLAG_OPS}.")
        elif r["op"] in ("lt", "gt") and not _is_num(r.get("value")):
            errs.append(f"quality flag '{r['name']}': op {r['op']} needs a numeric value.")
        elif r["op"] in ("contains", "startswith") and not _is_nonempty_str(r.get("value")):
            errs.append(f"quality flag '{r['name']}': op {r['op']} needs a non-empty text value.")

    # ---- dummy
    dm = section("dummy")
    if not (isinstance(dm.get("n_rows"), int) and not isinstance(dm.get("n_rows"), bool) and dm["n_rows"] >= 1):
        errs.append("dummy.n_rows must be an integer >= 1.")
    if not _is_nonempty_str(dm.get("stratify_column")):
        errs.append("dummy.stratify_column must be a non-empty string.")

    # ---- role conflicts
    if not errs:
        for col, rl in _role_map(cfg).items():
            kinds = {r for r, _ in rl}
            if "excluded" in kinds and len(kinds) > 1:
                errs.append(f"column '{col}' is 'excluded' but also has role(s) {sorted(kinds - {'excluded'})}.")
            for a, b in _EXCLUSIVE[1:]:
                if a in kinds and b in kinds:
                    errs.append(f"column '{col}' cannot be both '{a}' and '{b}'.")
    return errs


def _expert_matrix_errors(mat, names) -> list:
    """[S] axioms for a contradiction function: c(v,v)=0, symmetric, in [0,1]."""
    if not isinstance(mat, dict):
        return ["contradiction.expert_matrix must be an object {value: {value: c}} when method='expert'."]
    errs = []
    for a in names:
        row = mat.get(a)
        if not isinstance(row, dict):
            errs.append(f"expert_matrix has no row for '{a}'.")
            continue
        for b in names:
            x = row.get(b)
            if not _is_num(x) or not (0.0 <= x <= 1.0):
                errs.append(f"expert_matrix[{a}][{b}]={x!r} must be a number in [0,1].")
    if not errs:
        for a in names:
            if abs(mat[a][a]) > _TOL:
                errs.append(f"expert_matrix[{a}][{a}] must be 0 (axiom c(v,v)=0).")
            for b in names:
                if abs(mat[a][b] - mat[b][a]) > _TOL:
                    errs.append(f"expert_matrix is not symmetric at ({a},{b}).")
    return errs


# ---------------------------------------------------------------- source value checks
def check_source_values(df: pd.DataFrame, cfg: dict, stage: str) -> list:
    """
    stage="raw_numeric": raw source / shrinkage columns present in df must be numeric dtype
                         (NaN allowed, imputation may follow).
    stage="final":       effective AND raw source columns must be numeric, finite, no NaN;
                         log10 sources strictly > 0; confidence columns in [0,1].
    Returns a list of problem strings (empty = fine).
    """
    errs = []
    sh = cfg["shrinkage"]["columns"]
    if stage == "raw_numeric":
        cols = {s["column"] for s in all_sources(cfg)} | set(sh) | set(sh.values())
        for c in sorted(cols):
            if c in df.columns and not _is_numeric_series(df[c]):
                errs.append(f"column '{c}' must be numeric but has dtype {df[c].dtype}.")
        return errs
    if stage != "final":
        raise ValueError(f"unknown stage {stage!r}")

    for s in all_sources(cfg):
        for c in dict.fromkeys([s["effective_column"], s["column"]]):   # effective first, raw second
            if c not in df.columns:
                errs.append(f"source column '{c}' is missing.")
                continue
            x = df[c]
            if not _is_numeric_series(x):
                errs.append(f"source column '{c}' must be numeric but has dtype {x.dtype}.")
                continue
            xf = x.astype(float)
            n_nan, n_inf = int(xf.isna().sum()), int(np.isinf(xf).sum())
            if n_nan:
                errs.append(f"source column '{c}' has {n_nan} NaN (no imputation strategy in config).")
            if n_inf:
                errs.append(f"source column '{c}' has {n_inf} inf values.")
            if s["transform"] == "log10":
                n_bad = int(((xf <= 0) & xf.notna()).sum())
                if n_bad:
                    errs.append(f"source column '{c}' uses log10 but has {n_bad} value(s) <= 0 "
                                f"(min={xf.min()}); log10 needs strictly positive data.")
    for conf in sorted(set(sh.values())):
        if conf not in df.columns:
            errs.append(f"confidence column '{conf}' is missing.")
            continue
        if not _is_numeric_series(df[conf]):
            errs.append(f"confidence column '{conf}' must be numeric.")
            continue
        cf = df[conf].astype(float)
        if cf.isna().any() or ((cf < 0) | (cf > 1)).any():
            errs.append(f"confidence column '{conf}' must be non-NaN and within [0,1] "
                        f"(min={cf.min()}, max={cf.max()}).")
    return errs


# ---------------------------------------------------------------- public validation
def validate_config(cfg: dict, df: pd.DataFrame | None = None, stage: str = "raw") -> None:
    """
    Structural checks always run. With a DataFrame:
      stage="raw":      every RAW column the config references must exist; generated columns
                        (derived, __adj, flags) are NOT required and their absence never fails.
      stage="prepared": every referenced column, generated ones included, must exist and the
                        source values must be numeric / finite / log10-safe.
    Raises one ValueError listing every problem found.
    """
    if stage not in ("raw", "prepared"):
        raise ValueError("stage must be 'raw' or 'prepared'")
    errs = _structure_errors(cfg)
    if not errs and df is not None:
        generated = set(generated_columns(cfg))
        referenced = _referenced_columns(cfg)
        required = referenced if stage == "prepared" else referenced - generated
        if stage == "prepared":
            required |= {s["effective_column"] for s in all_sources(cfg)}
        missing = sorted(c for c in required if c not in df.columns)
        if missing:
            errs.append(f"Columns referenced by the config but missing from the {stage} data: {missing}")
        if stage == "prepared" and not missing:
            errs += check_source_values(df, cfg, "final")
    if errs:
        raise ValueError("Config validation failed:\n  - " + "\n  - ".join(errs))


# ---------------------------------------------------------------- audit
def audit_columns(df: pd.DataFrame, cfg: dict, quiet: bool = False) -> pd.DataFrame:
    """
    Every column of df must be classified as source, derived (incl. derived_input and
    __adj / flags), display_only, reliability_metadata, excluded-with-reason, or label/filter.
    Unclassified columns raise ValueError (all of them listed). Prints the audit table
    unless quiet=True and returns it.
    """
    errs = _structure_errors(cfg)
    if errs:
        raise ValueError("Cannot audit: config is invalid:\n  - " + "\n  - ".join(errs))
    roles = _role_map(cfg)
    unclassified = [c for c in df.columns if c not in roles]
    if unclassified:
        raise ValueError(f"{len(unclassified)} column(s) have no role in the config "
                         f"(add each as source, derived, display_only, reliability_metadata, "
                         f"excluded-with-reason, or label/filter): {unclassified}")
    rows = []
    for c in df.columns:
        rl = roles[c]
        rows.append({
            "Column": c,
            "Role": "+".join(dict.fromkeys(r for r, _ in rl)),
            "Detail": " | ".join(d for _, d in rl if d),
            "Dtype": str(df[c].dtype),
        })
    table = pd.DataFrame(rows)
    if not quiet:
        with pd.option_context("display.max_colwidth", 90, "display.width", 250):
            print(f"\nCOLUMN AUDIT ({len(table)} columns)")
            print(table.to_string(index=False))
    return table

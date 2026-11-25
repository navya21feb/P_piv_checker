"""
data_prep.py -- data preparation for P-PIV (single-valued fuzzy plithogenic PIV, NOT neutrosophic).

Contains NO MEREC, PIV, appurtenance or plithogenic code. It only turns the raw CSV into the
"ready" frame the later modules consume. Every column name, threshold and strategy comes from
ppiv_config.json; rows are never dropped.

Order inside prepare():
  report NaN -> derived features -> imputation (only if configured) -> scale inspection
  -> shrinkage (x__adj) -> quality flags -> final source checks.

Tags: [D] = team design choice (all of this module's transformations are [D]).

CLI:  python data_prep.py [--config ppiv_config.json] [--input raw.csv] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import schema as sch


# ---------------------------------------------------------------- derived features
def count_delimited(series: pd.Series, delimiter: str) -> pd.Series:
    """[D] Number of non-empty (after strip) items in a delimited text cell.
    NaN stays NaN (unknown), '' gives 0. Returns int64 when no NaN, else float64."""
    def f(v):
        if pd.isna(v):
            return np.nan
        return float(sum(1 for t in str(v).split(delimiter) if t.strip()))
    out = series.map(f).astype("float64")
    return out.astype("int64") if not out.isna().any() else out


# ---------------------------------------------------------------- imputation
def _impute(df: pd.DataFrame, cfg: dict, stats: dict) -> dict:
    """Impute ONLY columns listed in missing_values.strategies. Returns {col: bool mask of imputed rows}."""
    masks = {}
    for col, st in cfg["missing_values"]["strategies"].items():
        if col not in df.columns:
            raise ValueError(f"missing_values strategy for '{col}' but the column does not exist.")
        mask = df[col].isna()
        if not mask.any():
            continue
        m = st["method"]
        if m == "constant":
            fill = st["value"]
        else:
            if not sch._is_numeric_series(df[col]):
                raise ValueError(f"imputation method '{m}' needs a numeric column; '{col}' is {df[col].dtype}.")
            if df[col].notna().sum() == 0:
                raise ValueError(f"cannot impute '{col}' with {m}: every value is NaN.")
            fill = float(df[col].median() if m == "median" else df[col].mean())
        df[col] = df[col].fillna(fill)
        masks[col] = mask
        stats["imputation"][col] = {"method": m, "fill_value": fill, "n_rows": int(mask.sum())}
    return masks


# ---------------------------------------------------------------- scale inspection
def inspect_scales(df: pd.DataFrame, cfg: dict, verbose: bool = True) -> pd.DataFrame:
    si = cfg["scale_inspection"]
    lo, hi = si["expected_min"], si["expected_max"]
    rows, bad = [], []
    for c in si["columns"]:
        if c not in df.columns:
            raise ValueError(f"scale_inspection column '{c}' not found.")
        if not sch._is_numeric_series(df[c]):
            raise ValueError(f"scale_inspection column '{c}' is not numeric.")
        mn, mx = float(df[c].min()), float(df[c].max())
        ok = mn >= lo and mx <= hi
        rows.append({"Column": c, "min": mn, "max": mx, "mean": float(df[c].mean()),
                     f"within [{lo},{hi}]": ok})
        if not ok:
            bad.append(f"{c} (min={mn}, max={mx})")
    t = pd.DataFrame(rows)
    if verbose:
        print("\nSENTIMENT / SCALE INSPECTION")
        print(t.to_string(index=False))
    if bad:
        raise ValueError(f"Scale check failed: outside [{lo},{hi}]: {bad}. "
                         "Shrinkage mixes these with a mean, so the scale must be known.")
    return t


# ---------------------------------------------------------------- quality flags
def _flag_mask(series: pd.Series, op: str, value) -> pd.Series:
    if op in ("contains", "startswith"):
        s = series.astype("string")
        m = s.str.contains(value, regex=False) if op == "contains" else s.str.startswith(value)
        return m.fillna(False).astype(bool)
    x = pd.to_numeric(series, errors="coerce")
    return ((x < value) if op == "lt" else (x > value)).fillna(False).astype(bool)


def build_flags(df: pd.DataFrame, cfg: dict, imputed: dict) -> list:
    """[D] Semicolon-separated flag string per row ('' when none)."""
    qf = cfg["quality_flags"]
    names, masks = [], []
    for r in qf["rules"]:
        if r["column"] not in df.columns:
            raise ValueError(f"quality flag '{r['name']}' references missing column '{r['column']}'.")
        names.append(r["name"])
        masks.append(_flag_mask(df[r["column"]], r["op"], r["value"]).to_numpy())
    if qf["flag_imputed"]:
        for col, m in imputed.items():
            names.append(f"imputed:{col}")
            masks.append(m.to_numpy())
    if not names:
        return [""] * len(df)
    arr = np.column_stack(masks)
    return [";".join(n for n, hit in zip(names, row) if hit) for row in arr]


# ---------------------------------------------------------------- prepare
def prepare(df_raw: pd.DataFrame, cfg: dict, verbose: bool = True) -> pd.DataFrame:
    """
    Build the ready frame from the raw frame. Never drops rows, never edits original columns.
    Adds: derived features, x__adj for each shrinkage column, and the quality-flag column.
    Side info (shrinkage means, imputation values, flag counts) is in result.attrs['prep_stats'].
    """
    df = df_raw.copy()
    n_rows = len(df)
    stats = {"n_rows": n_rows, "missing_before": {}, "imputation": {}, "shrinkage_means": {}, "flag_counts": {}}

    # raw columns must at least have numeric dtype where numbers are expected
    errs = sch.check_source_values(df, cfg, "raw_numeric")
    if errs:
        raise ValueError("Raw data problems:\n  - " + "\n  - ".join(errs))

    # 1. missing values (report only)
    miss = df.isna().sum()
    stats["missing_before"] = {c: int(n) for c, n in miss.items() if n}

    # 2. derived features
    overwritten = [c for c in sch.generated_columns(cfg) if c in df.columns]
    if overwritten and verbose:
        print(f"NOTE: raw data already contains generated column(s) {overwritten}; they are recomputed.")
    for d in cfg["derived"]:
        if d["from"] not in df.columns:
            raise ValueError(f"derived '{d['name']}': source column '{d['from']}' not found.")
        df[d["name"]] = count_delimited(df[d["from"]], d["delimiter"])

    # 3. imputation, ONLY where the config gives a strategy
    imputed = _impute(df, cfg, stats)

    # 4. scale inspection (sentiment columns) -- before shrinkage
    inspect_scales(df, cfg, verbose=verbose)

    # 5. shrinkage: x__adj = conf*x + (1-conf)*mean(x over ALL rows)
    for col, conf in cfg["shrinkage"]["columns"].items():
        for c in (col, conf):
            if c not in df.columns:
                raise ValueError(f"shrinkage needs column '{c}' which is missing.")
        cf = df[conf].astype(float)
        if cf.isna().any() or ((cf < 0) | (cf > 1)).any():
            raise ValueError(f"confidence column '{conf}' must be non-NaN and within [0,1].")
        x = df[col].astype(float)
        if x.isna().any():
            raise ValueError(f"shrinkage column '{col}' has NaN; configure an imputation strategy or fix the data.")
        mean_all = float(x.mean())
        df[sch.adjusted_name(cfg, col)] = cf * x + (1.0 - cf) * mean_all
        stats["shrinkage_means"][col] = mean_all

    # 6. quality flags
    flag_col = cfg["quality_flags"]["column"]
    df[flag_col] = build_flags(df, cfg, imputed)
    counts = {}
    for cell in df[flag_col]:
        for f in filter(None, cell.split(";")):
            counts[f] = counts.get(f, 0) + 1
    stats["flag_counts"] = counts

    # 7. final source checks: numeric, no NaN, no inf, log10 strictly > 0
    errs = sch.check_source_values(df, cfg, "final")
    if errs:
        raise ValueError("Prepared data problems:\n  - " + "\n  - ".join(errs))

    if len(df) != n_rows:  # defensive: prepare must never drop rows
        raise AssertionError("prepare() changed the number of rows")
    df.attrs["prep_stats"] = stats
    return df


# ---------------------------------------------------------------- IO helpers
def load_ready(path, cfg: dict) -> pd.DataFrame:
    """Read a ready/dummy CSV. Empty flag cells come back as '' (not NaN) so dtype is stable."""
    df = pd.read_csv(path)
    fc = cfg["quality_flags"]["column"]
    if fc in df.columns:
        df[fc] = df[fc].fillna("").astype(str)
    return df


def make_dummy(df_ready: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """[D] Small stratified sample of the READY frame (identical schema and dtypes).
    One random row from every category of dummy.stratify_column first (sorted, so Government and
    Private are both present when the data has them), then random fill up to n_rows. Seeded."""
    dm = cfg["dummy"]
    n, strat, seed = dm["n_rows"], dm["stratify_column"], dm["seed"]
    if n > len(df_ready):
        raise ValueError(f"dummy.n_rows={n} exceeds the number of rows ({len(df_ready)}).")
    picked = []
    for g in sorted(df_ready[strat].dropna().unique().tolist())[:n]:
        picked.append(df_ready[df_ready[strat] == g].sample(1, random_state=seed).index[0])
    if len(picked) < n:
        rest = df_ready.drop(index=picked).sample(n - len(picked), random_state=seed).index.tolist()
        picked += rest
    return df_ready.loc[picked].reset_index(drop=True)


# ---------------------------------------------------------------- report
def data_quality_report(df_raw: pd.DataFrame, df_ready: pd.DataFrame, cfg: dict) -> None:
    idc = cfg["id_column"]
    st = df_ready.attrs.get("prep_stats", {})
    print("\nDATA-QUALITY REPORT")
    print(f"  rows                : {len(df_ready)}")
    print(f"  columns             : raw={df_raw.shape[1]}  ready={df_ready.shape[1]}")
    print(f"  null cells (raw)    : {int(df_raw.isna().sum().sum())}  "
          f"{st.get('missing_before') or '(none)'}")
    print(f"  null cells (ready)  : {int(df_ready.isna().sum().sum())}")
    print(f"  imputation          : {st.get('imputation') or 'none configured / none needed'}")
    dup = int(df_ready[idc].duplicated().sum())
    print(f"  duplicate {idc:<10}: {dup}" + ("   <-- WARNING: rows kept, but IDs are not unique" if dup else ""))
    print("  flag counts         :")
    if st.get("flag_counts"):
        for k, v in st["flag_counts"].items():
            print(f"      {k:<32}{v:>5}  ({100*v/len(df_ready):.1f}%)")
    else:
        print("      (no flags fired)")
    print(f"      rows with >=1 flag{'':<14}{int((df_ready[cfg['quality_flags']['column']] != '').sum()):>5}")
    rows = []
    for s in sch.all_sources(cfg):
        r, e = df_ready[s["column"]], df_ready[s["effective_column"]]
        rows.append({"Value": s["value"], "Source": s["column"], "dir": s["direction"], "transform": s["transform"],
                     "raw_min": r.min(), "raw_max": r.max(), "used_column": s["effective_column"],
                     "used_min": e.min(), "used_max": e.max()})
    print("  source ranges:")
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print(pd.DataFrame(rows).to_string(index=False))


# ---------------------------------------------------------------- pipeline
def run(config_path="ppiv_config.json", input_csv=None, out_dir=None, verbose=True) -> pd.DataFrame:
    cfg = sch.load_config(config_path)
    out_dir = Path(out_dir) if out_dir else Path(cfg["_config_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(input_csv) if input_csv else sch.resolve_path(cfg, "input_csv")
    df_raw = pd.read_csv(src)
    print(f"Loaded {src} : {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")

    sch.validate_config(cfg, df_raw, stage="raw")                 # (1) raw columns only
    sch.audit_columns(df_raw, cfg)                                # audit table (raw)
    ready = prepare(df_raw, cfg, verbose=verbose)                 # (2)
    sch.validate_config(cfg, ready, stage="prepared")             # (3) full config
    sch.audit_columns(ready, cfg, quiet=True)

    data_quality_report(df_raw, ready, cfg)

    ready_path = sch.resolve_path(cfg, "ready_csv", out_dir)
    ready.to_csv(ready_path, index=False)
    dummy = make_dummy(ready, cfg)
    dummy_path = sch.resolve_path(cfg, "dummy_csv", out_dir)
    dummy.to_csv(dummy_path, index=False)
    stats_path = sch.resolve_path(cfg, "prep_stats_json", out_dir)
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(ready.attrs["prep_stats"], fh, indent=2)

    # same schema and dtypes after a CSV round trip
    r, d = load_ready(ready_path, cfg), load_ready(dummy_path, cfg)
    if list(r.columns) != list(d.columns) or not (r.dtypes == d.dtypes).all():
        raise AssertionError("dummy and ready frames differ in schema or dtypes after reload")
    strat = cfg["dummy"]["stratify_column"]
    print(f"\nSaved {ready_path.name} ({len(ready)} rows), {dummy_path.name} ({len(dummy)} rows; "
          f"{strat} = {dummy[strat].tolist()}), {stats_path.name}")
    return ready


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="P-PIV data preparation")
    ap.add_argument("--config", default="ppiv_config.json")
    ap.add_argument("--input", default=None, help="raw CSV (default: paths.input_csv in the config)")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    run(a.config, a.input, a.out_dir)

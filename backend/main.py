"""
main.py -- FastAPI service for the Plithogenic-PIV (P-PIV) Decision Engine.

Nothing here knows a column name, value name, direction, threshold or label: every one comes from
ppiv_config.json (path in env PPIV_CONFIG, default: next to this file) or from the data.
The ranking itself is done only by plithogenic_piv.run_plithogenic_piv (P-PIV) and
baseline_piv.classical_piv_from_config (classical PIV); validation by validation.evaluate / sensitivity.

Frozen-parameter rule (spec item 2/3): appurtenance anchors and the contradiction matrix are fitted
ONCE on the FULL prepared data (frozen_params.json) and never refit on filtered rows. The only
refits happen (a) at startup when the file is missing, (b) on an uploaded dataset, which is its own
full dataset, and (c) inside the sensitivity sweep, where a config variant changes the anchors by
design -- those refits also use the FULL data, never the filtered subset.

Run:  uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import copy
import io
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import schema as sch
import data_prep as dp
import plithogenic_piv as pp
import baseline_piv as bp
import validation as val

HERE = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("PPIV_CONFIG", HERE / "ppiv_config.json"))

PPIV_NAME = "P-PIV"
CLASSICAL_NAME = "Classical PIV"


# =============================================================================== helpers
def jsonable(o: Any) -> Any:
    """Recursively turn numpy / pandas objects into plain JSON (NaN/inf -> None)."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, pd.Series):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, pd.DataFrame):
        return [jsonable(r) for r in o.to_dict(orient="records")]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if o is pd.NA or o is pd.NaT:
        return None
    return o


def err(msg: str, status: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": msg})


def api_cfg(cfg: dict) -> dict:
    a = cfg.get("api")
    return a if isinstance(a, dict) else {}


def label_column(cfg: dict) -> str:
    a = api_cfg(cfg)
    return a.get("label_column") or (cfg["columns"]["labels"] or [cfg["id_column"]])[0]


def review_column(cfg: dict) -> str | None:
    return api_cfg(cfg).get("review_count_column")


def flag_column(cfg: dict) -> str:
    return cfg["quality_flags"]["column"]


def collect_caveats(node: Any, path: str = "") -> list:
    """Every '_caveat' entry anywhere in the config, with where it was found."""
    out = []
    if isinstance(node, dict):
        where = node.get("name") or path
        if isinstance(node.get("_caveat"), str):
            out.append({"source": where, "text": node["_caveat"]})
        for k, v in node.items():
            if k != "_caveat":
                out += collect_caveats(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += collect_caveats(v, f"{path}[{i}]")
    return out


def flag_counts(df: pd.DataFrame, cfg: dict) -> dict:
    fc = flag_column(cfg)
    counts: dict = {}
    if fc not in df.columns:
        return counts
    for cell in df[fc].fillna("").astype(str):
        for f in filter(None, cell.split(";")):
            counts[f] = counts.get(f, 0) + 1
    return counts


def split_flags(cell) -> list:
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return []
    return [f for f in str(cell).split(";") if f]


# =============================================================================== dataset state
class Dataset:
    """A prepared dataset + config + frozen params. One for the default data; one per upload."""

    def __init__(self, cfg: dict, df: pd.DataFrame, params: dict, origin: str):
        sch.validate_config(cfg, df, stage="prepared")
        sch.audit_columns(df, cfg, quiet=True)
        pp._check_params_match_cfg(params, cfg)
        idc = cfg["id_column"]
        if df[idc].duplicated().any():
            dups = df.loc[df[idc].duplicated(), idc].astype(str).tolist()[:5]
            raise ValueError(f"'{idc}' is not unique (e.g. {dups}).")
        self.cfg, self.df, self.params, self.origin = cfg, df.reset_index(drop=True), params, origin
        self.idc = idc
        self._fit_cache: dict = {}

    # ---- filtering
    def filter_specs(self) -> list:
        out = []
        for col in self.cfg["columns"]["filters"]:
            s = self.df[col]
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                out.append({"column": col, "type": "numeric",
                            "min": jsonable(s.min()), "max": jsonable(s.max())})
            else:
                opts = sorted(s.dropna().astype(str).unique().tolist())
                out.append({"column": col, "type": "categorical", "options": opts})
        return out

    def apply_filters(self, filters: dict | None, min_reviews: float | None) -> pd.DataFrame:
        d = self.df
        allowed = set(self.cfg["columns"]["filters"])
        for col, v in (filters or {}).items():
            if col not in allowed:
                raise ValueError(f"'{col}' is not a filter column in the config (allowed: {sorted(allowed)}).")
            if v is None or v == "" or v == []:
                continue
            s = d[col]
            numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
            if numeric:
                if not (isinstance(v, (list, tuple)) and len(v) == 2):
                    raise ValueError(f"filter '{col}' is numeric: send [min, max].")
                lo, hi = v
                if lo is not None:
                    d = d[s >= float(lo)]
                    s = d[col]
                if hi is not None:
                    d = d[s <= float(hi)]
            else:
                wanted = [str(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
                d = d[s.astype(str).isin(wanted)]
        if min_reviews is not None and float(min_reviews) > 0:
            rc = review_column(self.cfg)
            if not rc or rc not in d.columns:
                raise ValueError("min_reviews needs api.review_count_column in the config.")
            d = d[d[rc].astype(float) >= float(min_reviews)]
        return d

    # ---- frozen-param refits for sensitivity variants (FULL data only, cached)
    def params_for(self, cfg_variant: dict) -> dict:
        key = json.dumps({k: v for k, v in cfg_variant.items() if not k.startswith("_")},
                         sort_keys=True, default=str)
        if key not in self._fit_cache:
            try:
                pp._check_params_match_cfg(self.params, cfg_variant)
                self._fit_cache[key] = self.params
            except ValueError:
                self._fit_cache[key] = pp.fit_frozen_params(self.df, cfg_variant)
        return self._fit_cache[key]

    # ---- meta
    def meta(self) -> dict:
        cfg, params = self.cfg, self.params
        pl = cfg["plithogenic"]
        dom = pl["dominant_value"]
        C = params["contradiction"]["matrix"]
        a = api_cfg(cfg)
        labels = a.get("display_only_labels") or {}
        values = []
        for v in pl["values"]:
            srcs = []
            for s in v["sources"]:
                eff = sch.effective_column(cfg, s)
                fp = params["sources"].get(eff, {})
                srcs.append({"column": s["column"], "effective_column": eff,
                             "direction": s["direction"], "transform": s.get("transform", "identity"),
                             "use_adjusted": bool(s.get("use_adjusted", False)),
                             "lo": fp.get("lo"), "hi": fp.get("hi")})
            values.append({"name": v["name"], "dominant": v["name"] == dom, "sources": srcs,
                           "n_sources": len(srcs), "frozen_c": C[dom][v["name"]],
                           "caveat": v.get("_caveat")})
        agg = cfg["aggregation"]
        qf = cfg["quality_flags"]
        flag_names = [r["name"] for r in qf["rules"]]
        counts = flag_counts(self.df, cfg)
        if qf.get("flag_imputed"):
            flag_names += [k for k in counts if k.startswith("imputed:")]
        classical_cols, classical_dirs = bp.classical_columns(cfg)
        return jsonable({
            "title": a.get("title"),
            "origin": self.origin,
            "n_rows": len(self.df),
            "id_column": self.idc,
            "label_column": label_column(cfg),
            "label_columns": cfg["columns"]["labels"],
            "filters": self.filter_specs(),
            "attribute": pl["attribute"],
            "dominant_value": dom,
            "values": values,
            "contradiction": {"method": params["contradiction"]["method"], "matrix": C,
                              "allow_runtime_override": bool(cfg["contradiction"].get("allow_runtime_override"))},
            "appurtenance": {"lower_percentile": params["percentiles"]["lower"],
                             "upper_percentile": params["percentiles"]["upper"],
                             "fitted_on_rows": params.get("fit_info", {}).get("n_rows")},
            "epsilon": params["epsilon"],
            "defaults": {"operator": agg["operator"], "normalization": cfg["piv"]["normalization"],
                         "t_norm": agg["t_norm"], "t_conorm": agg["t_conorm"],
                         "weight_method": cfg["weights"]["method"],
                         "recompute_on_filter": bool(cfg["weights"].get("recompute_on_filter", True))},
            "options": {"operators": list(sch.VALID_OPERATORS),
                        "normalizations": list(sch.VALID_NORMALIZATIONS),
                        "weight_modes": ["merec", "custom"]},
            "display_only": [{"column": c, "label": labels.get(c, c)} for c in cfg["columns"]["display_only"]],
            "display_only_group": a.get("display_only_group"),
            "reliability_columns": cfg["columns"]["reliability_metadata"],
            "review_count_column": review_column(cfg),
            "low_review_threshold": a.get("low_review_threshold"),
            "quality_flags": [{"name": r["name"], "column": r["column"], "op": r["op"], "value": r["value"]}
                              for r in qf["rules"]],
            "quality_flag_names": flag_names,
            "flag_counts": counts,
            "excluded_columns": cfg["columns"]["excluded"],
            "classical": {"columns": classical_cols, "directions": classical_dirs,
                          "normalization": cfg["piv"]["normalization"]},
            "caveats": collect_caveats({k: v for k, v in cfg.items() if k != "api"}) +
                       [{"source": "Note", "text": t} for t in a.get("caveats", [])],
        })


# =============================================================================== request model
class RankRequest(BaseModel):
    filters: dict[str, Any] = Field(default_factory=dict)
    weight_mode: str = "merec"
    custom_weights: dict[str, float] | None = None
    contradiction_override: dict[str, float] | None = None
    exclude_values: list[str] = Field(default_factory=list)
    operator: str | None = None
    normalization: str | None = None
    frozen_weights: bool = False
    min_reviews: float | None = None


def override_matrix(ds: Dataset, override: dict | None) -> dict | None:
    """{value: c} -> full V x V matrix: the frozen matrix with c(v_D, v) = c(v, v_D) replaced."""
    if not override:
        return None
    if not ds.cfg["contradiction"].get("allow_runtime_override", False):
        raise ValueError("contradiction.allow_runtime_override is false in the config.")
    dom = ds.params["dominant_value"]
    M = copy.deepcopy(ds.params["contradiction"]["matrix"])
    for v, c in override.items():
        if v not in M:
            raise ValueError(f"contradiction_override: unknown value '{v}'.")
        c = float(c)
        if v == dom:
            if c != 0.0:
                raise ValueError("c(v_D, v_D) must be 0 (axiom).")
            continue
        if not (math.isfinite(c) and 0.0 <= c <= 1.0):
            raise ValueError(f"contradiction_override['{v}'] must be in [0,1], got {c}.")
        M[dom][v] = M[v][dom] = c
    return M


def check_request(ds: Dataset, req: RankRequest) -> None:
    if req.weight_mode not in ("merec", "custom"):
        raise ValueError("weight_mode must be 'merec' or 'custom'.")
    if req.operator is not None and req.operator not in sch.VALID_OPERATORS:
        raise ValueError(f"operator must be one of {sch.VALID_OPERATORS}.")
    if req.normalization is not None and req.normalization not in sch.VALID_NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {sch.VALID_NORMALIZATIONS}.")
    names = sch.value_names(ds.cfg)
    bad = [v for v in req.exclude_values if v not in names]
    if bad:
        raise ValueError(f"exclude_values {bad} are not values in the config.")
    if req.weight_mode == "custom":
        if not req.custom_weights:
            raise ValueError("weight_mode 'custom' needs custom_weights {value: w}.")
        bad = [k for k in req.custom_weights if k not in names]
        if bad:
            raise ValueError(f"custom_weights: unknown value(s) {bad}.")


def custom_for(req: RankRequest, values: list) -> dict | None:
    if req.weight_mode != "custom":
        return None
    return {v: float(req.custom_weights.get(v, 0.0)) for v in values if v not in req.exclude_values}


def ppiv_kwargs(ds: Dataset, req: RankRequest) -> dict:
    return dict(contradiction_override=override_matrix(ds, req.contradiction_override),
                exclude_values=req.exclude_values or None,
                operator=req.operator, normalization=req.normalization)


# =============================================================================== ranking
def rank(ds: Dataset, req: RankRequest) -> dict:
    check_request(ds, req)
    sub = ds.apply_filters(req.filters, req.min_reviews)
    if len(sub) < 2:
        return {"error": f"only {len(sub)} alternative(s) match the filters; at least 2 are needed to rank.",
                "n_matched": int(len(sub))}
    cfg, params, idc = ds.cfg, ds.params, ds.idc
    kw = ppiv_kwargs(ds, req)
    values_all = sch.value_names(cfg)
    notes = []

    # ---- P-PIV weights: custom | MEREC on the filtered P | MEREC frozen from the FULL data
    frozen_w = None
    if req.weight_mode == "custom":
        if req.frozen_weights:
            notes.append("freeze-weights ignored: custom weights are already fixed.")
        res = pp.run_plithogenic_piv(sub, params, cfg, weight_mode="custom",
                                     custom_weights=custom_for(req, values_all), **kw)
    else:
        if req.frozen_weights:
            full = pp.run_plithogenic_piv(ds.df, params, cfg, **kw)
            frozen_w = full["weights"]
        res = pp.run_plithogenic_piv(sub, params, cfg, frozen_weights=frozen_w, **kw)

    # ---- classical PIV (spec item 8): raw columns, MEREC on the same rows (or frozen from full data)
    cl_w = None
    if req.frozen_weights and req.weight_mode != "custom":
        cl_w = bp.classical_piv_from_config(ds.df.set_index(idc), cfg)["weights"]
    cl = bp.classical_piv_from_config(sub.set_index(idc), cfg, weights=cl_w)

    # ---- items
    s = sub.set_index(idc)
    lab = label_column(cfg)
    filt_cols = cfg["columns"]["filters"]
    disp = cfg["columns"]["display_only"]
    rc = review_column(cfg)
    fc = flag_column(cfg)
    vd = res["value_degrees"]
    items = []
    for i in res["ranks"].sort_values(kind="stable").index:
        row = s.loc[i]
        pr, cr = int(res["ranks"][i]), int(cl["ranks"][i])
        items.append({
            "id": str(i),
            "label": row[lab] if lab in s.columns else str(i),
            "labels": {c: row[c] for c in cfg["columns"]["labels"] if c in s.columns},
            "filters": {c: row[c] for c in filt_cols},
            "ppiv_rank": pr, "classical_rank": cr, "rank_diff": cr - pr,
            "ppiv_score": float(res["scores"][i]), "classical_score": float(cl["scores"][i]),
            "value_degrees": {v: float(vd.loc[i, v]) for v in vd.columns},
            "source_degrees": {v: {c: float(dfv.loc[i, c]) for c in dfv.columns}
                               for v, dfv in res["source_degrees"].items()},
            "display_only": {c: row[c] for c in disp},
            "data_quality_flags": split_flags(row[fc]) if fc in s.columns else [],
            "review_count": row[rc] if rc and rc in s.columns else None,
        })
    m = res["meta"]
    return jsonable({
        "n_matched": int(len(sub)),
        "items": items,
        "weights_used": res["weights"],
        "weights_source": "frozen_from_full_data" if frozen_w is not None else res["weights_source"],
        "contradiction_used": res["contradiction_used"],
        "contradiction_source": m["contradiction_source"],
        "dominant_value": res["dominant_value"],
        "operator": res["operator"], "t_norm": res["t_norm"], "t_conorm": res["t_conorm"],
        "normalization": m["normalization"],
        "excluded_values": res["excluded_values"],
        "n_floored_cells": m["n_floored_cells"], "epsilon": m["epsilon"],
        "constant_columns": m["constant_columns"],
        "classical": {"weights_used": cl["weights"], "weights_source":
                      "frozen_from_full_data" if cl_w is not None else cl["weights_source"],
                      "columns": cl["columns"], "constant_columns": cl["constant_columns"],
                      "normalization": cl["meta"]["normalization"]},
        "flag_counts": flag_counts(sub, cfg),
        "notes": notes,
    })


# =============================================================================== stress test
def stress(ds: Dataset, req: RankRequest) -> dict:
    """validation.evaluate (agreement + rank reversal in BOTH weight modes, never mixed) and
    validation.sensitivity (weight perturbation, leave-one-value-out, parameter sweeps) on the
    filtered rows. Anchors stay frozen from the full data. The freeze-weights toggle is not used
    here: evaluate() itself runs the recomputed and frozen modes side by side."""
    check_request(ds, req)
    sub = ds.apply_filters(req.filters, req.min_reviews)
    rv = ds.cfg["validation"]["reversal"]
    n_drop = max(1, int(round(rv["drop_fraction"] * len(sub))))
    if len(sub) - n_drop < 2:
        return {"error": f"only {len(sub)} alternative(s) match the filters; too few for the stress test "
                         f"(it drops {n_drop} row(s) per iteration and needs at least 2 left).",
                "n_matched": int(len(sub))}
    cfg, idc = ds.cfg, ds.idc
    seed = int(api_cfg(cfg).get("validation_seed", 0))
    sub_i = sub.set_index(idc)
    kw = ppiv_kwargs(ds, req)
    values_all = sch.value_names(cfg)

    def ppiv_fn(df, weights=None):
        if weights is not None:
            return pp.run_plithogenic_piv(df, ds.params, cfg, frozen_weights=weights, **kw)
        if req.weight_mode == "custom":
            return pp.run_plithogenic_piv(df, ds.params, cfg, weight_mode="custom",
                                          custom_weights=custom_for(req, values_all), **kw)
        return pp.run_plithogenic_piv(df, ds.params, cfg, **kw)

    def classical_fn(df, weights=None):
        return bp.classical_piv_from_config(df, cfg, weights=weights)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ev = val.evaluate(sub_i, cfg, {PPIV_NAME: ppiv_fn, CLASSICAL_NAME: classical_fn}, seed=seed)

    # ---- sensitivity: the request's operator / normalisation / contradiction are baked into the
    # base config so that sweeps can vary them; anchors for any variant are refit on the FULL data.
    base = copy.deepcopy(cfg)
    if req.operator:
        base["aggregation"]["operator"] = req.operator
    if req.normalization:
        base["piv"]["normalization"] = req.normalization
    cm = override_matrix(ds, req.contradiction_override)
    if cm is not None:
        base["contradiction"]["method"] = "expert"
        base["contradiction"]["expert_matrix"] = cm

    def factory(cfg_v, _df_rows):
        params_v = ds.params_for(cfg_v)                    # FULL-data fit, never the subset
        names_v = sch.value_names(cfg_v)
        excl = [v for v in req.exclude_values if v in names_v] or None
        cw = custom_for(req, names_v)

        def fn(df, weights=None):
            if weights is not None:
                return pp.run_plithogenic_piv(df, params_v, cfg_v, frozen_weights=weights, exclude_values=excl)
            if cw is not None:
                return pp.run_plithogenic_piv(df, params_v, cfg_v, weight_mode="custom",
                                              custom_weights=cw, exclude_values=excl)
            return pp.run_plithogenic_piv(df, params_v, cfg_v, exclude_values=excl)
        return fn

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sens = val.sensitivity(factory, None, sub_i, base)

    notes = []
    if req.weight_mode == "custom":
        notes.append("P-PIV uses fixed custom weights, so its 'recomputed' and 'frozen' reversal modes "
                     "use the same weights.")
    if (req.normalization or cfg["piv"]["normalization"]) == "none":
        notes.append(val.NOTE_FROZEN)
    return jsonable({"n_matched": int(len(sub)), "methods": [PPIV_NAME, CLASSICAL_NAME],
                     "mode_keys": {"recomputed": val.MODE_RECOMPUTED, "frozen": val.MODE_FROZEN},
                     "evaluate": ev, "sensitivity": sens, "notes": notes})


# =============================================================================== startup
def load_default_dataset() -> Dataset:
    cfg = sch.load_config(CONFIG_PATH)
    sch.validate_config(cfg)
    ready_path = sch.resolve_path(cfg, "ready_csv")
    if ready_path.is_file():
        df = dp.load_ready(ready_path, cfg)
    else:
        raw = sch.resolve_path(cfg, "input_csv")
        if not raw.is_file():
            raise FileNotFoundError(f"neither {ready_path} nor the raw input {raw} exists.")
        df = dp.prepare(pd.read_csv(raw), cfg, verbose=False)
        df.to_csv(ready_path, index=False)
    fp_path = sch.resolve_path(cfg, "frozen_params_json")
    if fp_path.is_file():
        params = pp.load_frozen_params(fp_path)
    else:
        params = pp.fit_frozen_params(df, cfg)          # FULL data, once
        pp.save_frozen_params(params, fp_path)
    return Dataset(cfg, df, params, origin=str(ready_path.name))


app = FastAPI(title="Plithogenic-PIV Decision Engine API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])
STATE: dict = {}


@app.on_event("startup")
def _startup():
    STATE["ds"] = load_default_dataset()


def _ds() -> Dataset:
    if "ds" not in STATE:
        STATE["ds"] = load_default_dataset()
    return STATE["ds"]


# =============================================================================== routes
@app.get("/api/meta")
def get_meta():
    return _ds().meta()


@app.post("/api/recommend")
def post_recommend(req: RankRequest):
    try:
        return rank(_ds(), req)
    except ValueError as e:
        return err(str(e))


@app.post("/api/stress-test")
def post_stress(req: RankRequest):
    try:
        return stress(_ds(), req)
    except ValueError as e:
        return err(str(e))


@app.post("/api/upload-csv")
async def post_upload(file: UploadFile = File(...), config: UploadFile | None = File(None),
                      config_json: str | None = Form(None), request: str | None = Form(None)):
    """CSV + a config JSON in the SAME schema as ppiv_config.json (as a file part 'config' or a text
    part 'config_json'). Runs schema validation, prepare(), fit_frozen_params() on the uploaded
    data (its own full dataset) and ranks. No column auto-detection: every column must be declared."""
    limit = int(api_cfg(_ds().cfg).get("max_upload_mb", 25)) * 1024 * 1024
    raw = await file.read()
    if len(raw) > limit:
        return err(f"CSV larger than {limit // (1024 * 1024)} MB.")
    try:
        if config is not None:
            cfg_text = (await config.read()).decode("utf-8")
        elif config_json:
            cfg_text = config_json
        else:
            return err("send a config JSON (file part 'config' or text part 'config_json').")
        cfg = json.loads(cfg_text)
        if not isinstance(cfg, dict):
            return err("config must be a JSON object.")
        req = RankRequest(**(json.loads(request) if request else {}))
        df_raw = pd.read_csv(io.BytesIO(raw))
    except (UnicodeDecodeError, json.JSONDecodeError, pd.errors.ParserError) as e:
        return err(f"could not parse input: {e}")
    except Exception as e:                                         # pydantic / pandas edge cases
        return err(f"bad input: {e}")
    try:
        sch.validate_config(cfg, df_raw, stage="raw")
        ready = dp.prepare(df_raw, cfg, verbose=False)
        audit = sch.audit_columns(ready, cfg, quiet=True)
        params = pp.fit_frozen_params(ready, cfg)
        ds = Dataset(cfg, ready, params, origin=f"upload:{file.filename}")
        result = rank(ds, req)
    except ValueError as e:
        return err(str(e))
    return jsonable({"meta": ds.meta(), "result": result,
                     "prep_stats": ready.attrs.get("prep_stats", {}),
                     "audit": audit, "frozen_params": params})

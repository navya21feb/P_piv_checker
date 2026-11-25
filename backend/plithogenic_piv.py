"""
plithogenic_piv.py -- single-valued fuzzy Plithogenic PIV (P-PIV).  NOT neutrosophic.

Truth_T / Indeterminacy_I / Falsity_F are display-only and never read here.
Nothing in this file knows a column name, direction, percentile, epsilon, operator or threshold:
all of it comes from cfg['plithogenic'] / cfg['appurtenance'] / cfg['contradiction'] /
cfg['aggregation'] / cfg['floor'] / cfg['piv'] (ppiv_config.json) or from the data.

TAGS
----
[S] taken from an original paper      [D] team design choice (must be justified in the report)
CONFIRMED SOURCE REFERENCES (checked against the three attached PDFs; page = printed page)
------------------------------------------------------------------------------------------
Smarandache (2018), Neutrosophic Sets and Systems 21, pp. 153-166:
  * (P, a, V, d, c) is defined in Sec. 2.5, p. 154; V = Sec. 2.2, v_D = Sec. 2.3 (p. 154).
  * d: appurtenance degree function, Eq. (1) p. 154; single-valued fuzzy case d: P x V -> [0,1],
    Eq. (36), Sec. 4.2.1, p. 159.
  * c: V x V -> [0,1] with axioms c(v,v)=0 and c(v1,v2)=c(v2,v1), Sec. 2.5, p. 154.
  * General linear-combination operators with the dominant value: Eqs. (2)-(5), Sec. 2.6, p. 155.
  * Two-expert single-valued intersection / union: Eq. (44) / Eq. (45), Secs. 5.6.1-5.6.2, p. 163.
  * Numerical example used as acceptance test: Table 5, p. 165 (union worked in Sec. 5.7.2, p. 165;
    0.23 = (2/3)*0.08 + (1/3)*0.52 stated on p. 166). Data: Tables 3-4, pp. 163-164.
  * The paper itself calls the two degrees "Expert A" and "Expert B" (p. 163). Using two DATA
    SOURCES in that role is a team choice [D], not something the paper does.
  * The paper says contradiction is also defined w.r.t. other values, not only v_D (Sec. 2.6,
    p. 155), which supports storing the full V x V matrix. It gives NO estimator for c from data
    (c is "designed in each field", p. 154): the Spearman estimator is [D].
Keshavarz-Ghorabaee et al. (2021), Symmetry 13, 525 (page "x of 20"):
  * x_ij > 0 required: Step 1, p. 7.   Normalisation n_ij: Eq. (2), p. 8 (benefit min_k x_kj/x_ij,
    non-beneficial x_ij/max_k x_kj).   S_i: Eq. (3), p. 8.   S'_ij: Eq. (4), p. 9.
    E_j: Eq. (5), p. 9.   w_j: Eq. (6), p. 9.
  * Illustrative example (Tables 2-4, pp. 9-10; weights 0.5752, 0.0141, 0.4016, 0.0091, pp. 10-11)
    is reproduced by baseline_piv.merec_weights (test_baseline_piv.py).
Mufazzal & Muzakkir (2018), Computers & Industrial Engineering 119, 427-438:
  * Vector normalisation Eq. (1), weighted values Eq. (2), proximity index u_i Eq. (3),
    d_i Eq. (4), ranking by smallest d_i (Step 7): all p. 431.
  * Example: Table 11, Case 1 (d = 0.45, 0.39, 0.37; ranks 3, 2, 1), p. 433, reproduced by
    baseline_piv.piv_rank. Tied d values get tied ranks there (Case 2: ranks 4, 3, 1, 1), which
    is what baseline_piv's rank(method='min') does.
  * Rank reversal: "less prone ... not always" (p. 433) and "does not conclusively eliminate"
    (Conclusion, p. 437). This module makes no stronger claim.

THE OBJECT (P, a, V, d, c) AND v_D  -- kept separate on purpose
---------------------------------------------------------------
  P    alternatives (colleges) that pass the filters; the row index of the data you pass in.
  a    the ATTRIBUTE: one name, cfg['plithogenic']['attribute'] ("College_Quality").   [S structure]
  V    the attribute VALUES: cfg['plithogenic']['values'][*]['name'] (Reputation, ...). [S structure;
       the choice of values and which columns feed them is D]
  raw  RAW COLUMNS ("sources"): the data columns feeding a value. They are NOT values.
  d    appurtenance DEGREE of an alternative to a value, in [0,1]. Two stages:
         source degree  : one per (alternative, source column)         appurtenance_degrees()
         value  degree  : one per (alternative, value), after the contradiction-weighted
                          aggregation of its source degrees              aggregate_values()
       Unfloored value degrees are kept for display; matrix M = max(degree, epsilon) is ranked.
  c    contradiction function V x V -> [0,1]: c(v,v)=0, symmetric.            [S axioms]
       Estimation (Spearman on mean degrees, or expert matrix) is [D].
  v_D  dominant value, cfg['plithogenic']['dominant_value']. Only the row c(v_D, .) enters the
       aggregation (spec item 4). The full V x V matrix is stored so that the dominant value can
       be changed by editing the config and refitting, with no code edit.

PIPELINE AND TAGS
-----------------
 1. fit_frozen_params         full data, once. lo/hi percentiles per source [D], contradiction
                              matrix [S axioms / D estimation], epsilon [D], dominant value. Frozen.
 2. appurtenance_degrees      [D] x' = transform(x); r = clip((x'-lo)/(hi-lo),0,1);
                              degree = r (dir +1) or 1-r (dir -1). Anchors are NEVER recomputed.
 3. aggregate_values          [S] two-assessment formulas, Smarandache (2018) Eqs. (44)-(45) p. 163, with
                              c = c(v_D, v), A and B = the two source degrees:
                                intersection = (1-c)*(A AND_F B) + c*(A OR_F B)
                                union        = (1-c)*(A OR_F B)  + c*(A AND_F B)
                              [D] treating the two DATA SOURCES of a value as the paper's "experts
                              A and B"; [D] choice of t-norm/t-conorm pair and of operator;
                              one source -> unchanged [D]; >2 sources -> error unless
                              aggregation.allow_nary_extension is true with nary_label "[D]".
 4. apply_floor               [D] M_ij = max(a_ij, epsilon): MEREC needs x > 0.
 5. MEREC weights             [S] baseline_piv.merec_weights on M (MEREC Eqs. (2)-(6), pp. 8-9), all benefit.
 6. PIV ranking               [S] baseline_piv.piv_rank on M (PIV Eqs. (1)-(4), p. 431).
 7. Weights are recomputed on the filtered set by default (MEREC on the current P);
    appurtenance anchors and contradiction stay frozen. (cfg weights.recompute_on_filter)

[D] extensions that are NOT in the spec's [S] text and are labelled as such here:
  * contradiction between two NON-dominant values uses the same Spearman formula (spec item 3
    defines only c(v_D, v)); these entries are stored for completeness/display and for a later
    change of dominant value, and do not affect the ranking while v_D is unchanged.
  * n-ary aggregation (only if explicitly enabled): (1-c)*T_fold + c*S_fold over all k source
    degrees. Reduces to the [S] formula at k = 2 and is order-independent; it is NOT published.
  * extra t-norm/t-conorm pairs (lukasiewicz / bounded_sum) exist because schema.py lists them.

LIMITATIONS (to be stated in the report)
----------------------------------------
  * Spearman-estimated contradictions are data-driven: a high c only means the two values rank
    colleges differently on this dataset; it is not a semantic judgement.
  * Percentile anchors (5/95 by default) are [D]; degrees saturate at 0/1 outside them and
    different anchors change the numbers (see the sensitivity grid in validation.py).
  * The floor epsilon changes M; MEREC uses |ln(n_ij)|, which is sensitive to small values.
  * "Source = expert" is a modelling analogy [D]; the sources here are data columns, not
    independent assessors.
  * Frozen-weight rank reversal can be 0 by construction for an additive score; MEREC-recomputed
    weights can reverse ranks. The PIV paper claims rank reversal is minimised, not eliminated.
  * No claim is made here that P-PIV is better or worse than the classical baseline.

EXAMPLE CALCULATIONS (reproduced by the tests)
----------------------------------------------
  Spearman -> c:   rho = 0.2  ->  c = (1-0.2)/2 = 0.4;   rho = -1 -> c = 1;   rho = 1 -> c = 0.
  Intersection, product / prob-sum, A=.2, B=.4, c=1/3:
       A AND B = .08 ; A OR B = .2+.4-.08 = .52
       (1-1/3)*.08 + (1/3)*.52 = .05333 + .17333 = .22667  -> 0.23
  Union, same inputs:  (2/3)*.52 + (1/3)*.08 = .34667 + .02667 = .37333 -> 0.37
  Floor (epsilon .01): degree 0.004 -> M = 0.01; degree 0.62 -> M = 0.62 (display keeps 0.004).
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import schema as sch
from baseline_piv import merec_weights, piv_rank, _weights as _validate_weights

PARAMS_VERSION = 1
_TOL = 1e-12


# =============================================================================== registries [D]
def _t_product(a, b): return a * b
def _t_min(a, b): return np.minimum(a, b)
def _t_luk(a, b): return np.maximum(0.0, a + b - 1.0)
def _s_prob_sum(a, b): return np.clip(a + b - a * b, 0.0, 1.0)
def _s_max(a, b): return np.maximum(a, b)
def _s_bounded(a, b): return np.minimum(1.0, a + b)


#: t-norm registry (AND_F).  Names must equal schema.VALID_T_NORMS.
T_NORMS = {"product": _t_product, "minimum": _t_min, "lukasiewicz": _t_luk}
#: t-conorm registry (OR_F). Names must equal schema.VALID_T_CONORMS.
T_CONORMS = {"probabilistic_sum": _s_prob_sum, "maximum": _s_max, "bounded_sum": _s_bounded}


def _resolve_ops(cfg: dict, operator=None) -> tuple:
    agg = cfg["aggregation"]
    op = agg["operator"] if operator is None else operator
    if op not in sch.VALID_OPERATORS:
        raise ValueError(f"operator must be one of {sch.VALID_OPERATORS}, got {op!r}.")
    if agg["t_norm"] not in T_NORMS:
        raise ValueError(f"unknown t_norm {agg['t_norm']!r}; registry has {sorted(T_NORMS)}.")
    if agg["t_conorm"] not in T_CONORMS:
        raise ValueError(f"unknown t_conorm {agg['t_conorm']!r}; registry has {sorted(T_CONORMS)}.")
    return op, agg["t_norm"], agg["t_conorm"]


def combine_two(A, B, c, operator: str, t_norm: str = "product", t_conorm: str = "probabilistic_sum"):
    """[S] Smarandache (2018) Eq. (44) intersection / Eq. (45) union, p. 163. Works on arrays/scalars.
         intersection = (1-c)*T(A,B) + c*S(A,B)         union = (1-c)*S(A,B) + c*T(A,B)"""
    if operator not in sch.VALID_OPERATORS:
        raise ValueError(f"operator must be one of {sch.VALID_OPERATORS}, got {operator!r}.")
    A, B, c = np.asarray(A, float), np.asarray(B, float), np.asarray(c, float)
    if np.any((c < 0) | (c > 1)) or not np.isfinite(c).all():
        raise ValueError("contradiction c must be finite and in [0,1].")
    T, S = T_NORMS[t_norm](A, B), T_CONORMS[t_conorm](A, B)
    out = (1 - c) * T + c * S if operator == "intersection" else (1 - c) * S + c * T
    return np.clip(out, 0.0, 1.0)


def _combine_nary(cols, c, operator, t_norm, t_conorm):
    """[D] n-ary extension (only when approved in the config): (1-c)*T_fold + c*S_fold."""
    T = cols[0].copy(); S = cols[0].copy()
    for x in cols[1:]:
        T = T_NORMS[t_norm](T, x); S = T_CONORMS[t_conorm](S, x)
    out = (1 - c) * T + c * S if operator == "intersection" else (1 - c) * S + c * T
    return np.clip(out, 0.0, 1.0)


# =============================================================================== helpers
def _ensure_id_index(df: pd.DataFrame, cfg: dict, strict: bool) -> pd.DataFrame:
    idc = cfg["id_column"]
    if idc in df.columns:
        out = df.set_index(idc, drop=True)
    elif df.index.name == idc:
        out = df
    elif strict:
        raise ValueError(f"data has no '{idc}' column and its index is not named '{idc}'.")
    else:
        out = df
    if out.index.has_duplicates:
        raise ValueError(f"alternative identifiers in '{idc}' are not unique.")
    return out


def _transform(x: pd.Series, name: str, col: str) -> np.ndarray:
    v = x.to_numpy(dtype=float)
    if np.isnan(v).any() or np.isinf(v).any():
        raise ValueError(f"source column '{col}' has NaN/inf; impute or fix it in data_prep.")
    if name == "identity":
        return v
    if name == "log10":
        if (v <= 0).any():
            raise ValueError(f"source column '{col}' uses log10 but has value(s) <= 0 (min={v.min()}).")
        return np.log10(v)
    raise ValueError(f"unknown transform {name!r}; valid: {sch.VALID_TRANSFORMS}.")


def _source_degree(x: pd.Series, spec: dict) -> pd.Series:
    """[D] r = clip((x'-lo)/(hi-lo),0,1); degree = r (dir +1) or 1-r (dir -1). lo==hi raises."""
    col, lo, hi = spec["effective_column"], spec["lo"], spec["hi"]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"frozen anchors for '{col}' are not finite (lo={lo}, hi={hi}).")
    if hi == lo:
        raise ValueError(f"degenerate anchors for '{col}': lo == hi == {lo}. A degree cannot be "
                         "computed without dividing by zero; widen the percentiles or drop the source.")
    r = np.clip((_transform(x, spec["transform"], col) - lo) / (hi - lo), 0.0, 1.0)
    d = r if spec["direction"] == 1 else 1.0 - r
    return pd.Series(d, index=x.index, name=col)


def _fit_anchor_table(df: pd.DataFrame, cfg: dict) -> dict:
    ap = cfg["appurtenance"]
    lo_p, hi_p = float(ap["lower_percentile"]), float(ap["upper_percentile"])
    table = {}
    for s in sch.all_sources(cfg):
        col = s["effective_column"]
        if col not in df.columns:
            raise ValueError(f"source column '{col}' (value '{s['value']}') is missing from the data.")
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            raise ValueError(f"source column '{col}' must be numeric (dtype {df[col].dtype}).")
        xt = _transform(df[col], s["transform"], col)
        lo, hi = float(np.percentile(xt, lo_p)), float(np.percentile(xt, hi_p))
        if hi == lo:
            raise ValueError(f"degenerate anchors for '{col}': the {lo_p}th and {hi_p}th percentiles of "
                             f"its transformed values are both {lo}. Cannot fit a degree scale.")
        table[col] = {"value": s["value"], "column": s["column"], "direction": int(s["direction"]),
                      "transform": s["transform"], "use_adjusted": bool(s["use_adjusted"]),
                      "lo": lo, "hi": hi}
    return table


def _sources_by_value(params: dict) -> dict:
    out = {v: [] for v in params["values"]}
    for col, s in params["sources"].items():
        out[s["value"]].append(dict(s, effective_column=col))
    return out


# =============================================================================== contradiction
def validate_contradiction_matrix(mat, values) -> pd.DataFrame:
    """[S axioms] Return a float DataFrame (values x values). Raises on: wrong/missing keys,
    non-finite, outside [0,1], non-zero diagonal, not symmetric."""
    values = list(values)
    if isinstance(mat, pd.DataFrame):
        df = mat.copy()
    elif isinstance(mat, dict):
        if not all(isinstance(v, dict) for v in mat.values()):
            raise ValueError("contradiction matrix must be a dict of dicts {value: {value: c}}.")
        df = pd.DataFrame({k: pd.Series(v) for k, v in mat.items()}).T      # rows = outer key
    else:
        arr = np.asarray(mat, float)
        if arr.shape != (len(values), len(values)):
            raise ValueError(f"contradiction matrix must be {len(values)}x{len(values)}, got {arr.shape}.")
        df = pd.DataFrame(arr, index=values, columns=values)
    miss_r = [v for v in values if v not in df.index]
    miss_c = [v for v in values if v not in df.columns]
    extra = [v for v in set(df.index) | set(df.columns) if v not in values]
    if miss_r or miss_c or extra:
        raise ValueError(f"contradiction matrix keys must equal the values {values} "
                         f"(missing rows {miss_r}, missing cols {miss_c}, unexpected {sorted(map(str, extra))}).")
    df = df.loc[values, values]
    try:
        a = df.to_numpy(dtype=float)
    except (TypeError, ValueError):
        raise ValueError("contradiction matrix must be numeric.")
    if not np.isfinite(a).all():
        raise ValueError("contradiction matrix contains NaN or inf.")
    if ((a < 0) | (a > 1)).any():
        raise ValueError("contradiction matrix entries must lie in [0,1].")
    if np.abs(np.diag(a)).max() > _TOL:
        raise ValueError("contradiction matrix diagonal must be 0 (axiom c(v,v)=0).")
    if np.abs(a - a.T).max() > _TOL:
        i, j = np.unravel_index(np.argmax(np.abs(a - a.T)), a.shape)
        raise ValueError(f"contradiction matrix is not symmetric at ({values[i]}, {values[j]}).")
    return pd.DataFrame(a, index=values, columns=values)


def _estimate_contradiction(df: pd.DataFrame, table: dict, cfg: dict) -> pd.DataFrame:
    """[D] method; [S] axioms. 'spearman_on_mean_degrees' or 'expert' (config matrix)."""
    values = sch.value_names(cfg)
    ct = cfg["contradiction"]
    method = ct["method"]
    if method == "expert":
        return validate_contradiction_matrix(ct.get("expert_matrix"), values)
    if method != "spearman_on_mean_degrees":
        raise ValueError(f"unknown contradiction.method {method!r}; valid: {sch.VALID_CONTRADICTION_METHODS}.")
    if len(df) < 3:
        raise ValueError("Spearman contradiction needs at least 3 alternatives.")
    mean_deg = {}
    for v in values:
        cols = [c for c, s in table.items() if s["value"] == v]
        mean_deg[v] = pd.concat(
            [_source_degree(df[c], dict(table[c], effective_column=c)) for c in cols], axis=1).mean(axis=1)
    C = np.zeros((len(values), len(values)))
    for i, a in enumerate(values):
        for j in range(i + 1, len(values)):
            b = values[j]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", stats.ConstantInputWarning)   # NaN is raised just below
                rho = stats.spearmanr(mean_deg[a].to_numpy(), mean_deg[b].to_numpy()).statistic
            if not np.isfinite(rho):
                raise ValueError(f"Spearman rho between '{a}' and '{b}' is NaN (a constant mean-degree "
                                 "column on the full data). Contradiction cannot be estimated; "
                                 "no number is assigned. Fix the data/config or use method 'expert'.")
            C[i, j] = C[j, i] = float(np.clip((1.0 - rho) / 2.0, 0.0, 1.0))
    return pd.DataFrame(C, index=values, columns=values)


# =============================================================================== fit / save / load
def fit_frozen_params(df_full: pd.DataFrame, cfg: dict) -> dict:
    """[D anchors/epsilon/estimation; S axioms] Fit ONCE on the FULL prepared dataset. Returns a
    JSON-serialisable dict: per-source lo/hi (keyed by the effective column), contradiction matrix
    (full V x V, symmetric, zero diagonal), epsilon, dominant value. Never call it on a filtered or
    resampled frame unless you deliberately want new anchors."""
    sch.validate_config(cfg)
    pl = cfg["plithogenic"]
    table = _fit_anchor_table(df_full, cfg)
    C = _estimate_contradiction(df_full, table, cfg)
    return {
        "schema_version": PARAMS_VERSION,
        "scope": "P-PIV single-valued fuzzy (not neutrosophic)",
        "attribute": pl["attribute"],
        "dominant_value": pl["dominant_value"],
        "values": sch.value_names(cfg),
        "epsilon": float(cfg["floor"]["epsilon"]),
        "percentiles": {"lower": float(cfg["appurtenance"]["lower_percentile"]),
                        "upper": float(cfg["appurtenance"]["upper_percentile"])},
        "sources": table,
        "contradiction": {"method": cfg["contradiction"]["method"],
                          "matrix": {a: {b: float(C.loc[a, b]) for b in C.columns} for a in C.index}},
        "fit_info": {"n_rows": int(len(df_full)), "id_column": cfg["id_column"]},
    }


def save_frozen_params(params: dict, path) -> Path:
    p = Path(path)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2, allow_nan=False)
    return p


def load_frozen_params(path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"frozen params not found: {p} (run fit_frozen_params and save it).")
    with open(p, "r", encoding="utf-8") as fh:
        params = json.load(fh)
    _validate_params(params)
    return params


def _validate_params(params: dict) -> None:
    need = ("schema_version", "attribute", "dominant_value", "values", "epsilon", "percentiles",
            "sources", "contradiction")
    miss = [k for k in need if k not in params]
    if miss:
        raise ValueError(f"frozen params missing keys {miss}.")
    if params["schema_version"] != PARAMS_VERSION:
        raise ValueError(f"frozen params schema_version {params['schema_version']} != {PARAMS_VERSION}.")
    if params["dominant_value"] not in params["values"]:
        raise ValueError("frozen params: dominant_value is not one of the values.")
    if not (0 < params["epsilon"] < 1):
        raise ValueError("frozen params: epsilon must be in (0,1).")
    for col, s in params["sources"].items():
        for k in ("value", "column", "direction", "transform", "lo", "hi"):
            if k not in s:
                raise ValueError(f"frozen params: source '{col}' lacks '{k}'.")
        if s["value"] not in params["values"]:
            raise ValueError(f"frozen params: source '{col}' belongs to unknown value '{s['value']}'.")
        if not (np.isfinite(s["lo"]) and np.isfinite(s["hi"]) and s["hi"] > s["lo"]):
            raise ValueError(f"frozen params: source '{col}' needs finite lo < hi (got {s['lo']}, {s['hi']}).")
    validate_contradiction_matrix(params["contradiction"]["matrix"], params["values"])


def _check_params_match_cfg(params: dict, cfg: dict) -> None:
    """The frozen snapshot must describe the same model as the config; otherwise it is stale."""
    _validate_params(params)
    pl = cfg["plithogenic"]
    problems = []
    if params["attribute"] != pl["attribute"]:
        problems.append("attribute differs")
    if params["dominant_value"] != pl["dominant_value"]:
        problems.append(f"dominant_value: frozen {params['dominant_value']!r} vs config {pl['dominant_value']!r}")
    if params["values"] != sch.value_names(cfg):
        problems.append("value names/order differ")
    cfg_src = {s["effective_column"]: (s["value"], s["column"], int(s["direction"]), s["transform"])
               for s in sch.all_sources(cfg)}
    fro_src = {c: (s["value"], s["column"], int(s["direction"]), s["transform"]) for c, s in params["sources"].items()}
    if cfg_src != fro_src:
        problems.append("sources (column/direction/transform/use_adjusted) differ")
    ap = cfg["appurtenance"]
    if (params["percentiles"]["lower"], params["percentiles"]["upper"]) != (float(ap["lower_percentile"]), float(ap["upper_percentile"])):
        problems.append("percentiles differ")
    if params["epsilon"] != float(cfg["floor"]["epsilon"]):
        problems.append("epsilon differs")
    if params["contradiction"]["method"] != cfg["contradiction"]["method"]:
        problems.append("contradiction method differs")
    elif cfg["contradiction"]["method"] == "expert":
        exp = validate_contradiction_matrix(cfg["contradiction"].get("expert_matrix"), params["values"])
        fro = validate_contradiction_matrix(params["contradiction"]["matrix"], params["values"])
        if np.abs(exp.to_numpy() - fro.to_numpy()).max() > _TOL:
            problems.append("expert matrix differs")
    if problems:
        raise ValueError("frozen_params.json is stale relative to ppiv_config.json (" + "; ".join(problems) +
                         "). Refit on the full data with fit_frozen_params and save again.")


# =============================================================================== degrees
def appurtenance_degrees(df: pd.DataFrame, params: dict, cfg: dict | None = None) -> dict:
    """[D] value name -> DataFrame of SOURCE degrees (columns = effective source columns, index =
    alternatives). Anchors come ONLY from `params`; df never influences lo/hi, so the same
    college has identical degrees on the full data and on any filtered subset. Raises on lo==hi,
    missing columns, NaN/inf, log10 of x<=0. cfg is optional (used only for the id index)."""
    _validate_params(params)
    d = _ensure_id_index(df, cfg, strict=False) if cfg is not None else df
    out = {}
    for v, srcs in _sources_by_value(params).items():
        cols = {}
        for s in srcs:
            col = s["effective_column"]
            if col not in d.columns:
                raise ValueError(f"source column '{col}' (value '{v}') is missing from the data.")
            cols[col] = _source_degree(d[col], s)
        out[v] = pd.DataFrame(cols, index=d.index)
    return out


# =============================================================================== aggregation
def aggregate_values(degrees: dict, contradiction, cfg: dict, operator: str | None = None) -> pd.DataFrame:
    """[S Eqs. (44)-(45) p. 163; D mapping sources->assessments] value name -> UNFLOORED value
    degree DataFrame (rows = alternatives, columns = values, in `degrees` order).

    contradiction: Series {value: c(v_D, value)}  (what run_plithogenic_piv passes), or the full
    V x V matrix (DataFrame/dict), in which case the row of cfg's dominant value is used.
    k = 1 source: degree unchanged.  k = 2: formulas of item 4.  k > 2: ValueError unless
    cfg aggregation.allow_nary_extension is true and nary_label == '[D]' (then the labelled [D]
    n-ary extension is used and a UserWarning is issued)."""
    op, tn, tc = _resolve_ops(cfg, operator)
    agg = cfg["aggregation"]
    nary_ok = agg.get("allow_nary_extension") is True and agg.get("nary_label") == "[D]"
    if isinstance(contradiction, (pd.DataFrame, dict)):
        full = validate_contradiction_matrix(contradiction, sch.value_names(cfg))
        contradiction = full.loc[cfg["plithogenic"]["dominant_value"], :]
    c_row = pd.Series(contradiction, dtype=float)
    result = {}
    for v, dfv in degrees.items():
        arr = dfv.to_numpy(dtype=float)
        if not np.isfinite(arr).all() or (arr < 0).any() or (arr > 1).any():
            raise ValueError(f"source degrees of '{v}' must be finite and in [0,1].")
        k = dfv.shape[1]
        if k == 0:
            raise ValueError(f"value '{v}' has no source degrees.")
        if k == 1:
            result[v] = dfv.iloc[:, 0].astype(float).to_numpy()
            continue
        if v not in c_row.index:
            raise ValueError(f"no contradiction supplied for value '{v}'.")
        c = float(c_row[v])
        if not (np.isfinite(c) and 0.0 <= c <= 1.0):
            raise ValueError(f"contradiction for '{v}' must be in [0,1], got {c}.")
        if k == 2:
            result[v] = combine_two(arr[:, 0], arr[:, 1], c, op, tn, tc)
        elif nary_ok:
            warnings.warn(f"[D] n-ary extension used for value '{v}' ({k} sources); not a published formula.",
                          UserWarning)
            result[v] = _combine_nary([arr[:, i] for i in range(k)], c, op, tn, tc)
        else:
            raise ValueError(f"value '{v}' has {k} sources; the [S] formulas are two-assessment. Reduce to "
                             "<= 2 sources, or approve an n-ary extension (aggregation.allow_nary_extension "
                             "= true with nary_label '[D]').")
    idx = next(iter(degrees.values())).index
    return pd.DataFrame(result, index=idx)


def apply_floor(value_degrees: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    """[D] M_ij = max(a_ij, epsilon). Needed because MEREC requires x > 0."""
    if not (np.isfinite(epsilon) and 0 < epsilon < 1):
        raise ValueError(f"epsilon must be in (0,1), got {epsilon}.")
    return value_degrees.clip(lower=epsilon)


# =============================================================================== frozen weights
def fit_frozen_weights(df_full: pd.DataFrame, params: dict, cfg: dict, operator=None) -> pd.Series:
    """MEREC weights of matrix M on the FULL data, to be saved and re-used as `frozen_weights`."""
    res = run_plithogenic_piv(df_full, params, cfg, operator=operator)
    w = res["weights"].copy()
    w.name = "frozen_weight"
    return w


# =============================================================================== main entry
def run_plithogenic_piv(df, params, cfg, weight_mode="merec", custom_weights=None,
                        contradiction_override=None, exclude_values=None, frozen_weights=None,
                        operator=None, normalization=None) -> dict:
    """Run P-PIV on `df` (the alternatives P that passed the filters) with FROZEN anchors/contradiction.

    weight_mode 'merec'  : MEREC on matrix M of the current P [S] (default, recompute_on_filter);
                           or, if frozen_weights is given, those weights (validated, renormalised).
                           If cfg weights.recompute_on_filter is false, frozen_weights are required.
    weight_mode 'custom' : custom_weights (validated: non-negative, finite, right keys/length,
                           renormalised to sum 1 [D]); frozen_weights must not be given.
    custom/frozen weights may be a dict/Series keyed by value name or a list in the order of the
    included values; keys of excluded values are dropped, then the rest is renormalised.
    contradiction_override: full V x V matrix (validated: range, symmetry, zero diagonal) replacing
                            the frozen one for this call only; allowed if contradiction.allow_runtime_override.
    exclude_values: value names removed from the criteria (and from all returned tables).
    Returns: scores, ranks, weights, source_degrees, value_degrees (unfloored), matrix_M,
    contradiction_matrix, contradiction_matrix_used, contradiction_used, normalized_matrix, weighted_matrix, dominant_value,
    operator, plus t_norm, t_conorm, weights_source, excluded_values, meta.
    Index of every table = college ID (cfg['id_column'])."""
    if weight_mode not in ("merec", "custom"):
        raise ValueError(f"weight_mode must be 'merec' or 'custom', got {weight_mode!r}.")
    sch.validate_config(cfg)
    _check_params_match_cfg(params, cfg)
    op, tn, tc = _resolve_ops(cfg, operator)
    norm = cfg["piv"]["normalization"] if normalization is None else normalization
    if norm not in sch.VALID_NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {sch.VALID_NORMALIZATIONS}, got {norm!r}.")

    d = _ensure_id_index(df, cfg, strict=True)
    if len(d) < 1:
        raise ValueError("no alternatives to rank (empty P).")

    values_all = list(params["values"])
    excl = [] if exclude_values is None else list(dict.fromkeys(exclude_values))
    bad = [v for v in excl if v not in values_all]
    if bad:
        raise ValueError(f"exclude_values {bad} are not among the values {values_all}.")
    values = [v for v in values_all if v not in excl]
    if not values:
        raise ValueError("exclude_values removes every value; nothing left to rank.")

    # --- contradiction (frozen, or validated override)
    dom = params["dominant_value"]
    frozen_C = validate_contradiction_matrix(params["contradiction"]["matrix"], values_all)
    if contradiction_override is not None:
        if not cfg["contradiction"].get("allow_runtime_override", False):
            raise ValueError("contradiction.allow_runtime_override is false in the config.")
        C = validate_contradiction_matrix(contradiction_override, values_all)
        c_source = "override"
    else:
        C, c_source = frozen_C, "frozen"
    c_used = C.loc[dom, values].copy()
    c_used.name = f"c({dom}, .)"

    # --- degrees, aggregation, floor
    all_src = appurtenance_degrees(d, params)
    source_degrees = {v: all_src[v] for v in values}
    value_degrees = aggregate_values(source_degrees, c_used, cfg, operator=op)       # unfloored
    M = apply_floor(value_degrees, params["epsilon"])
    dirs = pd.Series(1, index=M.columns, name="direction")                            # all benefit

    # --- weights
    if weight_mode == "custom":
        if custom_weights is None:
            raise ValueError("weight_mode='custom' needs custom_weights.")
        if frozen_weights is not None:
            raise ValueError("pass either custom_weights or frozen_weights, not both.")
        w_in, w_source = _prepare_weights(custom_weights, values, values_all, "custom_weights"), "custom"
    elif frozen_weights is not None:
        w_in, w_source = _prepare_weights(frozen_weights, values, values_all, "frozen_weights"), "frozen"
    else:
        if custom_weights is not None:
            raise ValueError("custom_weights given but weight_mode is 'merec'; use weight_mode='custom'.")
        if not cfg["weights"]["recompute_on_filter"]:
            raise ValueError("weights.recompute_on_filter is false: pass frozen_weights "
                             "(see fit_frozen_weights).")
        w_in, w_source = None, "merec"
    constant_cols = []
    if w_in is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            w_raw = merec_weights(M, dirs)
        constant_cols = w_raw.attrs["constant_columns"]
        w_in = w_raw
    renorm_note = None
    w, renorm = _validate_weights(M.columns, w_in)
    if renorm:
        renorm_note = "weights renormalised to sum 1"

    # --- PIV on M
    piv = piv_rank(M, w, dirs, normalization=norm, tie_decimals=cfg["piv"]["tie_decimals"])
    return {
        "scores": piv["scores"], "ranks": piv["ranks"], "weights": piv["weights"],
        "source_degrees": source_degrees, "value_degrees": value_degrees, "matrix_M": M,
        "contradiction_matrix": frozen_C,            # the frozen V x V matrix, always
        "contradiction_matrix_used": C,              # == frozen unless an override was passed
        "contradiction_used": c_used,                # row c(v_D, .) over the included values
        "normalized_matrix": piv["normalized_matrix"], "weighted_matrix": piv["weighted_matrix"],
        "dominant_value": dom, "operator": op, "t_norm": tn, "t_conorm": tc,
        "weights_source": w_source, "excluded_values": excl,
        "meta": {"normalization": norm, "contradiction_source": c_source, "epsilon": params["epsilon"],
                 "n_floored_cells": int((value_degrees.to_numpy() < params["epsilon"]).sum()),
                 "constant_columns": constant_cols, "weights_note": renorm_note,
                 "weights_recomputed_on_current_P": w_source == "merec",
                 "piv_meta": piv["meta"], "n_alternatives": int(len(M)),
                 "tags": "anchors/epsilon/aggregation mapping [D]; MEREC, PIV, two-assessment formulas [S]"},
    }


def _prepare_weights(w, included, all_values, what) -> pd.Series:
    """dict/Series keyed by value, or list in the order of `included`. Keys of excluded values are
    dropped. Validation of sign/NaN/sum is done by baseline_piv._weights afterwards."""
    if isinstance(w, (dict, pd.Series)):
        s = pd.Series(w)
        unknown = [k for k in s.index if k not in all_values]
        if unknown:
            raise ValueError(f"{what}: unknown value name(s) {unknown}.")
        missing = [v for v in included if v not in s.index]
        if missing:
            raise ValueError(f"{what}: missing weight(s) for {missing}.")
        return s.loc[included]
    arr = list(w)
    if len(arr) != len(included):
        raise ValueError(f"{what}: expected {len(included)} weights (one per included value), got {len(arr)}.")
    return pd.Series(arr, index=included)


# =============================================================================== (P,a,V,d,c) view
@dataclass(frozen=True)
class PlithogenicSet:
    """Read-only view of the plithogenic set behind one run. Fields are deliberately separate."""
    P: pd.Index                    # alternatives passing the filters
    a: str                         # attribute
    V: tuple                       # attribute values (included)
    d_sources: dict                # value -> source-degree DataFrame (raw columns -> degrees)
    d_values: pd.DataFrame         # unfloored value degrees d(p, v)
    c: pd.DataFrame                # contradiction matrix V x V (full, as frozen/overridden)
    v_D: str                       # dominant value


def as_plithogenic_set(result: dict, params: dict) -> PlithogenicSet:
    return PlithogenicSet(P=result["matrix_M"].index, a=params["attribute"],
                          V=tuple(result["matrix_M"].columns), d_sources=result["source_degrees"],
                          d_values=result["value_degrees"], c=result["contradiction_matrix_used"],
                          v_D=result["dominant_value"])

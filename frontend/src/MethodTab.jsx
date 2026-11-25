import React from "react";
import { BookOpen, Crown, Sigma } from "lucide-react";
import { Card, Badge, human, fmt } from "./ui.jsx";

const TNORM = { product: "∏ d_s", minimum: "min_s d_s", lukasiewicz: "max(0, Σ d_s − (k−1))" };
const TCONORM = { probabilistic_sum: "1 − ∏ (1 − d_s)", maximum: "max_s d_s", bounded_sum: "min(1, Σ d_s)" };

export default function MethodTab({ meta }) {
  const names = meta.values.map((v) => v.name);
  const { t_norm, t_conorm, operator } = meta.defaults;
  return (
    <div className="space-y-4">
      <Card title="Plithogenic set (P, a, V, d, c)" icon={BookOpen}>
        <dl className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-2 text-sm">
          <div><dt className="text-slate-500 text-xs">P (alternatives)</dt><dd>rows passing the filters (of {meta.n_rows})</dd></div>
          <div><dt className="text-slate-500 text-xs">a (attribute)</dt><dd className="font-medium">{human(meta.attribute)}</dd></div>
          <div><dt className="text-slate-500 text-xs">V (values = criteria)</dt><dd>{names.map(human).join(", ")}</dd></div>
          <div><dt className="text-slate-500 text-xs">v_D (dominant value)</dt><dd className="font-medium flex items-center gap-1"><Crown className="w-4 h-4 text-sky-700" />{human(meta.dominant_value)}</dd></div>
          <div><dt className="text-slate-500 text-xs">d (appurtenance)</dt>
            <dd>single-valued fuzzy, percentile ramp P{meta.appurtenance.lower_percentile}–P{meta.appurtenance.upper_percentile}, fitted once on {meta.appurtenance.fitted_on_rows} rows</dd></div>
          <div><dt className="text-slate-500 text-xs">c (contradiction)</dt><dd>method: {human(meta.contradiction.method)}; frozen, overridable at run time</dd></div>
        </dl>
      </Card>

      <Card title="Values and their sources" icon={Sigma}>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
          {meta.values.map((v) => (
            <div key={v.name} className="border border-slate-200 rounded-lg p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-slate-800">{human(v.name)}</span>
                <span className="flex gap-1">
                  {v.dominant ? <Badge tone="blue">dominant</Badge> : <Badge>c = {fmt(v.frozen_c, 3)}</Badge>}
                  <Badge>{v.n_sources} source{v.n_sources > 1 ? "s" : ""}</Badge>
                </span>
              </div>
              <table className="w-full text-xs">
                <thead><tr className="text-slate-500 text-left"><th className="py-1">Column read</th><th>dir</th><th>transform</th><th>lo</th><th>hi</th></tr></thead>
                <tbody>
                  {v.sources.map((s) => (
                    <tr key={s.effective_column} className="border-t border-slate-100">
                      <td className="py-1 font-mono">{s.effective_column}</td>
                      <td>{s.direction > 0 ? "+1 (higher = higher degree)" : "−1 (lower = higher degree)"}</td>
                      <td>{s.transform}</td>
                      <td className="tabular-nums">{fmt(s.lo, 4)}</td>
                      <td className="tabular-nums">{fmt(s.hi, 4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {v.n_sources === 1 && <p className="text-[11px] text-slate-500 mt-1">One source: the degree passes through unchanged; c is not used.</p>}
            </div>
          ))}
        </div>
      </Card>

      <Card title="Aggregation, weights and ranking" icon={Sigma}>
        <div className="space-y-3 text-sm text-slate-700">
          <div className="font-mono text-xs bg-slate-50 rounded p-3 space-y-1">
            <div>T = {TNORM[t_norm] || t_norm}   <span className="text-slate-400">(t-norm: {t_norm})</span></div>
            <div>S = {TCONORM[t_conorm] || t_conorm}   <span className="text-slate-400">(t-conorm: {t_conorm})</span></div>
            <div className={operator === "intersection" ? "font-semibold" : ""}>intersection: a = (1 − c)·T + c·S</div>
            <div className={operator === "union" ? "font-semibold" : ""}>union:        a = (1 − c)·S + c·T</div>
            <div>c = c(v_D, v); k = 1 source → a = d; k = 2 → paper formula</div>
            <div>M_ij = max(a_ij, ε), ε = {meta.epsilon}</div>
          </div>
          <p>Config default operator: <b>{operator}</b>. Weights: <b>{meta.defaults.weight_method.toUpperCase()}</b> on M (all values benefit-type), or custom weights renormalised to 1.</p>
          <p>PIV: normalisation <b>{meta.defaults.normalization}</b>; v_ij = w_j·r_ij; u_ij = max_i v_ij − v_ij; d_i = Σ_j u_ij; rank ascending (smallest d_i = rank 1).</p>
          <div>
            <p>Classical baseline: these source columns as separate criteria on raw values, MEREC on raw values, PIV with {meta.classical.normalization} normalisation:</p>
            <div className="flex flex-wrap gap-1 mt-1">
              {meta.classical.columns.map((c) => <code key={c} className="text-xs bg-slate-100 rounded px-1">{c}</code>)}
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}

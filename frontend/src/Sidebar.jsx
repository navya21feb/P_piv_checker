import React from "react";
import { Play, RotateCcw, Lock, Unlock, SlidersHorizontal, Filter, GitMerge, Loader2 } from "lucide-react";
import { human, fmt } from "./ui.jsx";

export function initialSettings(meta) {
  const values = meta.values.map((v) => v.name);
  const eq = 1 / values.length;
  return {
    filters: Object.fromEntries(meta.filters.map((f) => [f.column, f.type === "numeric" ? [f.min, f.max] : ""])),
    minReviews: "",
    weightMode: "merec",
    customWeights: Object.fromEntries(values.map((v) => [v, eq])),
    included: Object.fromEntries(values.map((v) => [v, true])),
    cOverride: Object.fromEntries(meta.values.map((v) => [v.name, v.frozen_c])),
    operator: meta.defaults.operator,
    normalization: meta.defaults.normalization,
    frozenWeights: false,
  };
}

export function buildRequest(meta, s) {
  const filters = {};
  for (const f of meta.filters) {
    const v = s.filters[f.column];
    if (f.type === "numeric") {
      const [lo, hi] = v || [];
      if ((lo !== "" && lo !== null && Number(lo) > f.min) || (hi !== "" && hi !== null && Number(hi) < f.max))
        filters[f.column] = [lo === "" ? null : Number(lo), hi === "" ? null : Number(hi)];
    } else if (v) filters[f.column] = v;
  }
  const exclude_values = meta.values.filter((v) => !s.included[v.name]).map((v) => v.name);
  const contradiction_override = {};
  for (const v of meta.values) {
    if (v.dominant || v.n_sources < 2) continue;
    if (Math.abs(Number(s.cOverride[v.name]) - v.frozen_c) > 1e-9) contradiction_override[v.name] = Number(s.cOverride[v.name]);
  }
  const req = {
    filters,
    weight_mode: s.weightMode,
    exclude_values,
    operator: s.operator,
    normalization: s.normalization,
    frozen_weights: s.weightMode === "merec" && s.frozenWeights,
    min_reviews: s.minReviews === "" ? null : Number(s.minReviews),
  };
  if (Object.keys(contradiction_override).length) req.contradiction_override = contradiction_override;
  if (s.weightMode === "custom") req.custom_weights = s.customWeights;
  return req;
}

const Label = ({ children }) => <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500 mb-1.5">{children}</div>;
const Section = ({ icon: Icon, title, children }) => (
  <div className="py-4 border-b border-slate-200 last:border-0">
    <div className="flex items-center gap-2 text-sm font-semibold text-slate-800 mb-3">
      <Icon className="w-4 h-4 text-slate-500" /> {title}
    </div>
    <div className="space-y-3">{children}</div>
  </div>
);

export default function Sidebar({ meta, s, setS, onRun, running, lastResult }) {
  const set = (patch) => setS((p) => ({ ...p, ...patch }));
  const setIn = (key, k, v) => setS((p) => ({ ...p, [key]: { ...p[key], [k]: v } }));
  const usedW = lastResult?.weights_used || {};
  const customSum = meta.values.filter((v) => s.included[v.name]).reduce((a, v) => a + Number(s.customWeights[v.name] || 0), 0);
  const cValues = meta.values.filter((v) => !v.dominant && v.n_sources >= 2);

  return (
    <aside className="w-80 shrink-0 bg-white border-r border-slate-200 h-screen sticky top-0 overflow-y-auto">
      <div className="px-4 pt-4">
        <button onClick={onRun} disabled={running}
          className="w-full inline-flex items-center justify-center gap-2 rounded-lg bg-slate-900 text-white text-sm font-semibold py-2.5 hover:bg-slate-800 disabled:opacity-60">
          {running ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />} Run P-PIV
        </button>
      </div>
      <div className="px-4">
        <Section icon={Filter} title="Filters">
          {meta.filters.map((f) => (
            <div key={f.column}>
              <Label>{human(f.column)}</Label>
              {f.type === "categorical" ? (
                <select value={s.filters[f.column]} onChange={(e) => setIn("filters", f.column, e.target.value)}
                  className="w-full rounded-md border border-slate-300 text-sm px-2 py-1.5 bg-white">
                  <option value="">Any</option>
                  {f.options.map((o) => <option key={o} value={o}>{o}</option>)}
                </select>
              ) : (
                <div className="flex gap-2">
                  {[0, 1].map((i) => (
                    <input key={i} type="number" value={s.filters[f.column]?.[i] ?? ""} placeholder={i ? f.max : f.min}
                      onChange={(e) => { const r = [...(s.filters[f.column] || [f.min, f.max])]; r[i] = e.target.value; setIn("filters", f.column, r); }}
                      className="w-1/2 rounded-md border border-slate-300 text-sm px-2 py-1.5" />
                  ))}
                </div>
              )}
            </div>
          ))}
          {meta.review_count_column && (
            <div>
              <Label>Minimum {human(meta.review_count_column)}</Label>
              <input type="number" min="0" value={s.minReviews} placeholder="none"
                onChange={(e) => set({ minReviews: e.target.value })}
                className="w-full rounded-md border border-slate-300 text-sm px-2 py-1.5" />
            </div>
          )}
        </Section>

        <Section icon={SlidersHorizontal} title={`Weights — ${human(meta.attribute)} values`}>
          <div className="flex rounded-md border border-slate-300 overflow-hidden text-xs">
            {meta.options.weight_modes.map((m) => (
              <button key={m} onClick={() => set({ weightMode: m })}
                className={`flex-1 py-1.5 font-medium ${s.weightMode === m ? "bg-slate-900 text-white" : "bg-white text-slate-600"}`}>
                {m === "merec" ? "MEREC" : "Custom"}
              </button>
            ))}
          </div>
          {meta.values.map((v) => {
            const inc = s.included[v.name];
            const custom = s.weightMode === "custom";
            const shown = custom ? Number(s.customWeights[v.name] || 0) : usedW[v.name] ?? 0;
            return (
              <div key={v.name} className={inc ? "" : "opacity-50"}>
                <div className="flex items-center justify-between text-xs mb-1">
                  <label className="flex items-center gap-1.5 font-medium text-slate-700">
                    <input type="checkbox" checked={inc} onChange={(e) => setIn("included", v.name, e.target.checked)} />
                    {human(v.name)} {v.dominant && <span className="text-[10px] text-sky-700">(dominant)</span>}
                  </label>
                  <span className="tabular-nums text-slate-500">
                    {custom ? (customSum > 0 && inc ? fmt(shown / customSum, 3) : "–") : (usedW[v.name] !== undefined ? fmt(shown, 3) : "–")}
                  </span>
                </div>
                <input type="range" min="0" max="1" step="0.01" value={shown} disabled={!custom || !inc}
                  onChange={(e) => setIn("customWeights", v.name, Number(e.target.value))}
                  className="w-full accent-slate-900 disabled:opacity-60" />
              </div>
            );
          })}
          <p className="text-[11px] text-slate-500">
            {s.weightMode === "merec"
              ? "MEREC weights are computed by the engine; sliders show the last run's weights."
              : "Custom weights are renormalised to sum to 1 over the included values."}
          </p>
          <label className={`flex items-center justify-between text-sm ${s.weightMode === "custom" ? "opacity-50" : ""}`}>
            <span className="flex items-center gap-1.5 text-slate-700">
              {s.frozenWeights ? <Lock className="w-4 h-4" /> : <Unlock className="w-4 h-4" />} Freeze weights
            </span>
            <input type="checkbox" disabled={s.weightMode === "custom"} checked={s.frozenWeights}
              onChange={(e) => set({ frozenWeights: e.target.checked })} />
          </label>
          <p className="text-[11px] text-slate-500">Frozen: MEREC weights fitted on all {meta.n_rows} rows, reused for the filtered subset.</p>
        </Section>

        <Section icon={GitMerge} title={`Contradiction c(${human(meta.dominant_value)}, ·)`}>
          {cValues.length === 0 && <p className="text-xs text-slate-500">No non-dominant value has two or more sources.</p>}
          {!meta.contradiction.allow_runtime_override && <p className="text-xs text-amber-700">Runtime override is disabled in the config.</p>}
          {cValues.map((v) => {
            const cur = Number(s.cOverride[v.name]);
            const changed = Math.abs(cur - v.frozen_c) > 1e-9;
            return (
              <div key={v.name}>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="font-medium text-slate-700">{human(v.name)}</span>
                  <span className="flex items-center gap-2">
                    <span className={`tabular-nums ${changed ? "text-amber-700 font-semibold" : "text-slate-500"}`}>{fmt(cur, 3)}</span>
                    <button title={`Reset to frozen c = ${fmt(v.frozen_c, 3)}`} disabled={!changed}
                      onClick={() => setIn("cOverride", v.name, v.frozen_c)} className="text-slate-400 hover:text-slate-700 disabled:opacity-30">
                      <RotateCcw className="w-3.5 h-3.5" />
                    </button>
                  </span>
                </div>
                <input type="range" min="0" max="1" step="0.01" value={cur} disabled={!meta.contradiction.allow_runtime_override}
                  onChange={(e) => setIn("cOverride", v.name, Number(e.target.value))} className="w-full accent-slate-900" />
              </div>
            );
          })}
        </Section>

        <Section icon={SlidersHorizontal} title="Operators">
          <div>
            <Label>Aggregation operator</Label>
            <select value={s.operator} onChange={(e) => set({ operator: e.target.value })}
              className="w-full rounded-md border border-slate-300 text-sm px-2 py-1.5 bg-white">
              {meta.options.operators.map((o) => <option key={o} value={o}>{o}{o === meta.defaults.operator ? " (config default)" : ""}</option>)}
            </select>
          </div>
          <div>
            <Label>PIV normalisation</Label>
            <select value={s.normalization} onChange={(e) => set({ normalization: e.target.value })}
              className="w-full rounded-md border border-slate-300 text-sm px-2 py-1.5 bg-white">
              {meta.options.normalizations.map((o) => <option key={o} value={o}>{o}{o === meta.defaults.normalization ? " (config default)" : ""}</option>)}
            </select>
          </div>
        </Section>
      </div>
    </aside>
  );
}

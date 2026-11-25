import React from "react";
import { FlaskConical, Loader2, Shuffle, SlidersHorizontal } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LabelList } from "recharts";
import { Card, KPI, ErrorBox, SERIES, fmt, pct, human } from "./ui.jsx";

function MetricTable({ rows, topK, extra = [] }) {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-xs">
        <thead>
          <tr className="text-left text-slate-500 border-b border-slate-200">
            <th className="py-1.5 pr-3">Variant</th>
            {extra.map((e) => <th key={e.key} className="pr-3">{e.label}</th>)}
            <th className="pr-3">Spearman</th><th className="pr-3">Kendall</th>
            {topK.map((k) => <th key={k} className="pr-3">Top-{k} overlap</th>)}
            <th className="pr-3">Max |rank shift|</th><th>Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-slate-100">
              <td className="py-1.5 pr-3 font-mono">{r.label ?? "–"}</td>
              {extra.map((e) => <td key={e.key} className="pr-3">{r[e.key] ?? "–"}</td>)}
              <td className="pr-3 tabular-nums">{fmt(r.spearman)}</td>
              <td className="pr-3 tabular-nums">{fmt(r.kendall)}</td>
              {topK.map((k) => <td key={k} className="pr-3 tabular-nums">{r[`top_${k}_overlap_count`] ?? "–"}{r[`top_${k}_tie_at_cutoff`] ? " (tie)" : ""}</td>)}
              <td className="pr-3 tabular-nums">{r.max_abs_rank_shift ?? "–"}</td>
              <td className={r.status === "ok" ? "text-slate-500" : "text-amber-700"} title={r.error || r.reason || ""}>{r.status}{r.reason ? `: ${r.reason}` : ""}{r.error ? `: ${r.error}` : ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ValidationTab({ data, running, onRun, error }) {
  const ev = data?.evaluate;
  const sens = data?.sensitivity;
  const topK = ev?.settings.top_k || [];
  const agree = ev ? Object.entries(ev.agreement)[0] : null;
  const modes = data?.mode_keys;
  const chart = ev ? data.methods.map((m) => ({
    method: m,
    recomputed: ev.rank_reversal[m][modes.recomputed].reversal_pct,
    frozen: ev.rank_reversal[m][modes.frozen].reversal_pct,
  })) : [];

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <button onClick={onRun} disabled={running}
          className="inline-flex items-center gap-2 rounded-lg bg-slate-900 text-white text-sm font-semibold px-4 py-2 hover:bg-slate-800 disabled:opacity-60">
          {running ? <Loader2 className="w-4 h-4 animate-spin" /> : <FlaskConical className="w-4 h-4" />} Run stress test
        </button>
        <span className="text-xs text-slate-500">Uses the current sidebar filters and settings. Anchors and contradiction stay frozen from the full data.</span>
      </div>
      <ErrorBox msg={error} />
      {!ev && !running && !error && <p className="text-sm text-slate-500">No stress test run yet.</p>}

      {ev && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <KPI label="Spearman ρ (P-PIV vs classical)" value={fmt(agree[1].spearman)} sub={`n = ${agree[1].n}`} />
            <KPI label="Kendall τ-b" value={fmt(agree[1].kendall)} sub="rank agreement, full filtered set" />
            {topK.map((k) => (
              <KPI key={k} label={`Top-${k} overlap`} value={`${agree[1][`top_${k}_overlap_count`]} / ${Math.min(k, agree[1].n)}`}
                sub={`${pct(agree[1][`top_${k}_overlap_fraction`], 0)}${agree[1][`top_${k}_tie_at_cutoff`] ? " · tie at cutoff" : ""}`} />
            ))}
          </div>

          <Card title="Rank reversal after dropping rows" icon={Shuffle}>
            <p className="text-xs text-slate-500 mb-3">
              {ev.settings.validation.n_iterations} iterations, each dropping {pct(ev.settings.validation.drop_fraction, 0)} of the rows (seed {ev.settings.seed});
              share of sampled pairs (up to {ev.settings.validation.max_pairs} per iteration) whose order flips against the full ranking.
              Recomputed and frozen weights are separate runs and are never combined.
            </p>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chart} margin={{ top: 20, right: 16, left: 0, bottom: 0 }} barGap={2}>
                  <CartesianGrid vertical={false} stroke="#eeeeee" />
                  <XAxis dataKey="method" tick={{ fontSize: 12, fill: "#666666" }} axisLine={{ stroke: "#d4d4d4" }} tickLine={false} />
                  <YAxis tick={{ fontSize: 11, fill: "#666666" }} axisLine={false} tickLine={false} unit="%" />
                  <Tooltip formatter={(v) => `${fmt(v, 3)}%`} cursor={{ fill: "#f5f5f5" }} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="recomputed" name="Weights recomputed on subset" fill={SERIES[0]} radius={[4, 4, 0, 0]} maxBarSize={48}>
                    <LabelList dataKey="recomputed" position="top" formatter={(v) => `${fmt(v, 2)}%`} style={{ fontSize: 11, fill: "#191919" }} />
                  </Bar>
                  <Bar dataKey="frozen" name="Weights frozen from full data" fill={SERIES[1]} radius={[4, 4, 0, 0]} maxBarSize={48}>
                    <LabelList dataKey="frozen" position="top" formatter={(v) => `${fmt(v, 2)}%`} style={{ fontSize: 11, fill: "#191919" }} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <table className="min-w-full text-xs mt-3">
              <thead><tr className="text-left text-slate-500 border-b border-slate-200">
                <th className="py-1.5 pr-3">Method</th><th className="pr-3">Weights</th><th className="pr-3">Valid pairs</th>
                <th className="pr-3">Reversals</th><th className="pr-3">Reversal %</th><th className="pr-3">Mean / max per iteration</th></tr></thead>
              <tbody>
                {data.methods.flatMap((m) => [modes.recomputed, modes.frozen].map((mk) => {
                  const r = ev.rank_reversal[m][mk];
                  return (
                    <tr key={m + mk} className="border-b border-slate-100">
                      <td className="py-1.5 pr-3">{m}</td><td className="pr-3">{human(mk)}</td>
                      <td className="pr-3 tabular-nums">{r.valid_pairs}</td><td className="pr-3 tabular-nums">{r.reversals}</td>
                      <td className="pr-3 tabular-nums">{fmt(r.reversal_pct, 3)}</td>
                      <td className="pr-3 tabular-nums">{fmt(r.per_iteration_pct_mean, 3)} / {fmt(r.per_iteration_pct_max, 3)}</td>
                    </tr>
                  );
                }))}
              </tbody>
            </table>
            {[...new Set(data.methods.map((m) => ev.rank_reversal[m][modes.frozen].note).filter(Boolean)).values(), ...(data.notes || [])]
              .map((n, i) => <p key={i} className="text-xs text-slate-500 mt-2">{n}</p>)}
          </Card>

          <Card title="Parameter sweeps (P-PIV vs its own base ranking)" icon={SlidersHorizontal}>
            <div className="space-y-4">
              {Object.entries(sens.param_sweeps).map(([name, rows]) => (
                <div key={name}>
                  <h4 className="text-xs font-semibold text-slate-600 mb-1">{human(name)}</h4>
                  <MetricTable rows={rows} topK={topK} />
                </div>
              ))}
            </div>
            <p className="text-xs text-slate-500 mt-3">Sweep ranges come from the config and are team choices. Variants that change the anchors are refitted on the full data, never the filtered subset.</p>
          </Card>
        </>
      )}
    </div>
  );
}

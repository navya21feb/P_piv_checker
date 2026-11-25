import React from "react";

// Validated categorical order (dataviz reference palette, light mode). Assigned by position, never cycled.
export const SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"];
export const SEQ = "#257de0"; // single hue for magnitudes (degrees in [0,1])

export const API_BASE = (import.meta.env.VITE_API_BASE || "http://localhost:8000").replace(/\/$/, "");

export async function api(path, { method = "GET", body, form } = {}) {
  const opts = { method };
  if (form) opts.body = form;
  else if (body !== undefined) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, opts);
  } catch (e) {
    return { error: `Cannot reach the API at ${API_BASE} (${e.message}).` };
  }
  let data;
  try {
    data = await res.json();
  } catch {
    return { error: `API returned ${res.status} with a non-JSON body.` };
  }
  if (!res.ok && !data.error) data = { error: data.detail ? JSON.stringify(data.detail) : `HTTP ${res.status}` };
  return data;
}

// Display transform only: underscores -> spaces. Names themselves come from the API.
export const human = (s) => String(s ?? "").replace(/_/g, " ");
export const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));
export const pct = (x, d = 1) => (x === null || x === undefined ? "–" : `${(100 * x).toFixed(d)}%`);

export function Card({ title, icon: Icon, right, children, className = "" }) {
  return (
    <section className={`bg-white border border-slate-200 rounded-xl shadow-sm ${className}`}>
      {(title || right) && (
        <header className="flex items-center justify-between px-4 py-3 border-b border-slate-100">
          <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-800">
            {Icon && <Icon className="w-4 h-4 text-slate-500" />}
            {title}
          </h3>
          {right}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

export function Badge({ tone = "slate", children, title }) {
  const tones = {
    slate: "bg-slate-100 text-slate-700 border-slate-200",
    green: "bg-emerald-50 text-emerald-800 border-emerald-200",
    red: "bg-rose-50 text-rose-800 border-rose-200",
    amber: "bg-amber-50 text-amber-900 border-amber-200",
    blue: "bg-sky-50 text-sky-800 border-sky-200",
  };
  return (
    <span title={title} className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded border text-xs font-medium whitespace-nowrap ${tones[tone]}`}>
      {children}
    </span>
  );
}

// rank_diff = classical_rank - ppiv_rank. Positive: P-PIV places it higher up the list.
export function RankDiff({ d }) {
  if (d === 0) return <Badge title="Same position in both rankings">0</Badge>;
  const up = d > 0;
  return (
    <Badge tone={up ? "green" : "red"}
      title={`classical rank − P-PIV rank = ${d}. ${up ? "Higher" : "Lower"} in the P-PIV list than in the classical list.`}>
      {up ? "+" : "−"}{Math.abs(d)}
    </Badge>
  );
}

export function MiniBar({ value, title }) {
  const v = Math.max(0, Math.min(1, Number(value) || 0));
  return (
    <div className="flex items-center gap-1.5" title={title ?? fmt(value)}>
      <div className="w-14 h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${v * 100}%`, background: SEQ }} />
      </div>
      <span className="text-[11px] tabular-nums text-slate-500 w-7">{fmt(value, 2)}</span>
    </div>
  );
}

// Stacked share bar for display-only columns whose values are shares. Legend labels come from meta.
export function ShareBar({ parts }) {
  const total = parts.reduce((a, p) => a + (Number(p.value) || 0), 0) || 1;
  const tip = parts.map((p) => `${p.label}: ${pct((Number(p.value) || 0))}`).join("\n");
  return (
    <div className="flex w-28 h-2 rounded overflow-hidden gap-[2px] bg-white" title={tip}>
      {parts.map((p, i) => (
        <div key={p.column} style={{ width: `${(100 * (Number(p.value) || 0)) / total}%`, background: SERIES[i] }} />
      ))}
    </div>
  );
}

export function ShareLegend({ items }) {
  return (
    <div className="flex flex-wrap gap-3 text-xs text-slate-600">
      {items.map((d, i) => (
        <span key={d.column} className="inline-flex items-center gap-1">
          <span className="w-2.5 h-2.5 rounded-sm" style={{ background: SERIES[i] }} />
          {d.label}
        </span>
      ))}
    </div>
  );
}

export function KPI({ label, value, sub }) {
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
      <div className="text-xs font-medium text-slate-500">{label}</div>
      <div className="text-2xl font-semibold text-slate-900 tabular-nums mt-1">{value}</div>
      {sub && <div className="text-xs text-slate-500 mt-1">{sub}</div>}
    </div>
  );
}

export function ErrorBox({ msg }) {
  if (!msg) return null;
  return <div className="rounded-lg border border-rose-200 bg-rose-50 text-rose-800 text-sm px-3 py-2 whitespace-pre-wrap">{msg}</div>;
}

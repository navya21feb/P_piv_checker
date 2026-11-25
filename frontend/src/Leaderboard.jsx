import React, { useMemo, useState } from "react";
import { Award, MessageSquareWarning, Search, Table2 } from "lucide-react";
import { Card, Badge, RankDiff, MiniBar, ShareBar, ShareLegend, human, fmt } from "./ui.jsx";

function DisplayOnly({ meta, item }) {
  const cols = meta.display_only;
  if (!cols.length) return null;
  if (meta.display_only_group?.type === "share_bar")
    return <ShareBar parts={cols.map((d) => ({ column: d.column, label: d.label, value: item.display_only[d.column] }))} />;
  return (
    <div className="text-[11px] text-slate-600 space-y-0.5">
      {cols.map((d) => <div key={d.column}>{d.label}: {fmt(item.display_only[d.column], 2)}</div>)}
    </div>
  );
}

function LowReview({ meta, item }) {
  const t = meta.low_review_threshold;
  if (t === null || t === undefined || item.review_count === null || item.review_count >= t) return null;
  return (
    <Badge tone="amber" title={`${human(meta.review_count_column)} = ${item.review_count} (< ${t}); sentiment-based degrees rest on few reviews.`}>
      <MessageSquareWarning className="w-3 h-3" /> {item.review_count}
    </Badge>
  );
}

function Winner({ meta, result }) {
  const w = result.items[0];
  if (!w) return null;
  return (
    <Card>
      <div className="flex flex-col md:flex-row md:items-start gap-6">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 text-xs font-semibold text-sky-700 uppercase tracking-wide">
            <Award className="w-4 h-4" /> P-PIV rank 1 of {result.n_matched} under the current settings
          </div>
          <h2 className="text-xl font-semibold text-slate-900 mt-1 truncate">{w.label}</h2>
          <div className="text-sm text-slate-500 mt-0.5">
            {Object.entries(w.labels).filter(([c]) => c !== meta.label_column).map(([, v]) => v).join(" · ")}
          </div>
          <div className="flex flex-wrap items-center gap-2 mt-3 text-sm">
            <Badge tone="blue">P-PIV d = {fmt(w.ppiv_score, 4)}</Badge>
            <Badge>Classical rank {w.classical_rank}</Badge>
            <RankDiff d={w.rank_diff} />
            <LowReview meta={meta} item={w} />
          </div>
        </div>
        <div className="md:w-72 space-y-1.5">
          {Object.entries(w.value_degrees).map(([v, d]) => (
            <div key={v} className="flex items-center justify-between gap-3 text-xs">
              <span className="text-slate-600 truncate">{human(v)}</span>
              <MiniBar value={d} title={`degree d(${w.label}, ${v}) = ${fmt(d, 4)}`} />
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}

export function RankTable({ meta, result }) {
  const [q, setQ] = useState("");
  const [limit, setLimit] = useState(25);
  const values = Object.keys(result.items[0]?.value_degrees || {});
  const rows = useMemo(() => {
    const t = q.trim().toLowerCase();
    return t ? result.items.filter((i) => [i.label, ...Object.values(i.labels), ...Object.values(i.filters)].join(" ").toLowerCase().includes(t)) : result.items;
  }, [q, result]);
  const filterCols = meta.filters.map((f) => f.column);

  return (
    <Card title={`Ranking (${result.n_matched} alternatives)`} icon={Table2}
      right={
        <div className="relative">
          <Search className="w-4 h-4 absolute left-2 top-1.5 text-slate-400" />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search"
            className="pl-7 pr-2 py-1 text-sm rounded-md border border-slate-300 w-48" />
        </div>
      }>
      {meta.display_only_group?.type === "share_bar" && (
        <div className="flex items-center gap-3 mb-3 text-xs text-slate-500">
          <span className="font-medium">{meta.display_only_group.label}:</span>
          <ShareLegend items={meta.display_only} />
        </div>
      )}
      <div className="overflow-x-auto -mx-4">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="text-left text-[11px] uppercase tracking-wide text-slate-500 border-b border-slate-200">
              <th className="px-3 py-2">P-PIV</th>
              <th className="px-3 py-2">Classical</th>
              <th className="px-3 py-2" title="classical rank − P-PIV rank">Δ rank</th>
              <th className="px-3 py-2">{human(meta.label_column)}</th>
              {filterCols.map((c) => <th key={c} className="px-3 py-2">{human(c)}</th>)}
              {values.map((v) => <th key={v} className="px-3 py-2" title={`appurtenance degree to ${v} (unfloored)`}>{human(v)}</th>)}
              {meta.display_only.length > 0 && <th className="px-3 py-2">{meta.display_only_group?.type === "share_bar" ? "Sentiment mix" : "Display only"}</th>}
              <th className="px-3 py-2">Reviews</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, limit).map((i) => (
              <tr key={i.id} className="border-b border-slate-100 hover:bg-slate-50">
                <td className="px-3 py-2 font-semibold tabular-nums">{i.ppiv_rank}</td>
                <td className="px-3 py-2 tabular-nums text-slate-600">{i.classical_rank}</td>
                <td className="px-3 py-2"><RankDiff d={i.rank_diff} /></td>
                <td className="px-3 py-2 max-w-[18rem]">
                  <div className="font-medium text-slate-800 truncate" title={i.label}>{i.label}</div>
                  <div className="text-[11px] text-slate-400 font-mono">{i.id}</div>
                </td>
                {filterCols.map((c) => <td key={c} className="px-3 py-2 text-slate-600 whitespace-nowrap">{String(i.filters[c] ?? "")}</td>)}
                {values.map((v) => (
                  <td key={v} className="px-3 py-2">
                    <MiniBar value={i.value_degrees[v]}
                      title={`${v}: ${fmt(i.value_degrees[v], 4)}\n` + Object.entries(i.source_degrees[v] || {}).map(([c, d]) => `  ${c}: ${fmt(d, 4)}`).join("\n")} />
                  </td>
                ))}
                {meta.display_only.length > 0 && <td className="px-3 py-2"><DisplayOnly meta={meta} item={i} /></td>}
                <td className="px-3 py-2 whitespace-nowrap">
                  <span className="tabular-nums text-slate-600 mr-1">{i.review_count ?? "–"}</span>
                  <LowReview meta={meta} item={i} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > limit && (
        <button onClick={() => setLimit((l) => l + 50)} className="mt-3 text-sm text-sky-700 hover:underline">
          Show more ({rows.length - limit} remaining)
        </button>
      )}
    </Card>
  );
}

export default function Leaderboard({ meta, result }) {
  if (!result) return <div className="text-sm text-slate-500">Set the controls on the left and press <b>Run P-PIV</b>.</div>;
  return (
    <div className="space-y-4">
      <Winner meta={meta} result={result} />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card title="Weights used (P-PIV)">
          <div className="text-xs text-slate-500 mb-2">source: {human(result.weights_source)} · operator: {result.operator} · normalisation: {result.normalization}</div>
          {Object.entries(result.weights_used).map(([v, w]) => (
            <div key={v} className="flex items-center justify-between text-sm py-0.5">
              <span className="text-slate-700">{human(v)}</span><span className="tabular-nums">{fmt(w, 4)}</span>
            </div>
          ))}
        </Card>
        <Card title="Contradiction used c(v_D, v)">
          <div className="text-xs text-slate-500 mb-2">source: {result.contradiction_source} · dominant: {human(result.dominant_value)}</div>
          {Object.entries(result.contradiction_used).map(([v, c]) => (
            <div key={v} className="flex items-center justify-between text-sm py-0.5">
              <span className="text-slate-700">{human(v)}</span><span className="tabular-nums">{fmt(c, 4)}</span>
            </div>
          ))}
          {result.n_floored_cells > 0 && <p className="text-xs text-slate-500 mt-2">{result.n_floored_cells} degree cell(s) floored to ε = {result.epsilon} before MEREC/PIV.</p>}
        </Card>
      </div>
      {result.notes?.length > 0 && <div className="text-xs text-slate-500">{result.notes.join(" ")}</div>}
      <RankTable meta={meta} result={result} />
    </div>
  );
}

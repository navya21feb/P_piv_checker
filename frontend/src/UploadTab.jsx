import React, { useState } from "react";
import { Upload, Loader2, FileSpreadsheet, FileJson } from "lucide-react";
import { Card, ErrorBox, api } from "./ui.jsx";
import { RankTable } from "./Leaderboard.jsx";

export default function UploadTab() {
  const [csv, setCsv] = useState(null);
  const [cfg, setCfg] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [out, setOut] = useState(null);

  async function submit() {
    setError(""); setOut(null);
    if (!csv || !cfg) { setError("Choose both a CSV file and a config JSON file."); return; }
    const form = new FormData();
    form.append("file", csv);
    form.append("config", cfg);
    setBusy(true);
    const r = await api("/api/upload-csv", { method: "POST", form });
    setBusy(false);
    if (r.error) setError(r.error);
    else if (r.result?.error) { setError(r.result.error); setOut(r); }
    else setOut(r);
  }

  const FilePick = ({ icon: Icon, label, accept, file, onPick }) => (
    <label className="flex-1 flex items-center gap-3 border border-dashed border-slate-300 rounded-lg p-4 cursor-pointer hover:bg-slate-50">
      <Icon className="w-6 h-6 text-slate-500" />
      <div className="min-w-0">
        <div className="text-sm font-medium text-slate-800">{label}</div>
        <div className="text-xs text-slate-500 truncate">{file ? file.name : "No file chosen"}</div>
      </div>
      <input type="file" accept={accept} className="hidden" onChange={(e) => onPick(e.target.files[0] || null)} />
    </label>
  );

  return (
    <div className="space-y-4">
      <Card title="Rank your own dataset" icon={Upload}>
        <p className="text-sm text-slate-600 mb-3">
          Upload a raw CSV and a config JSON in the same schema as <code className="text-xs bg-slate-100 px-1 rounded">ppiv_config.json</code>.
          Every column must be declared in the config (no auto-detection). The server validates the schema, prepares the data,
          fits the frozen anchors and contradiction on this upload, then ranks it with the config defaults.
        </p>
        <div className="flex flex-col md:flex-row gap-3">
          <FilePick icon={FileSpreadsheet} label="Data CSV" accept=".csv,text/csv" file={csv} onPick={setCsv} />
          <FilePick icon={FileJson} label="Config JSON" accept=".json,application/json" file={cfg} onPick={setCfg} />
        </div>
        <button onClick={submit} disabled={busy}
          className="mt-3 inline-flex items-center gap-2 rounded-lg bg-slate-900 text-white text-sm font-semibold px-4 py-2 disabled:opacity-60">
          {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />} Validate and rank
        </button>
      </Card>
      <ErrorBox msg={error} />
      {out?.meta && (
        <Card title={`Prepared ${out.meta.n_rows} rows from ${out.meta.origin}`}>
          <details className="mt-3 text-xs">
            <summary className="cursor-pointer text-slate-600">Column audit ({out.audit.length} columns)</summary>
            <table className="mt-2 text-xs">
              <tbody>{out.audit.map((a) => (
                <tr key={a.Column} className="border-b border-slate-100"><td className="pr-3 py-1 font-mono">{a.Column}</td><td className="pr-3">{a.Role}</td><td className="text-slate-500">{a.Detail}</td></tr>
              ))}</tbody>
            </table>
          </details>
        </Card>
      )}
      {out?.result?.items && <RankTable meta={out.meta} result={out.result} />}
    </div>
  );
}

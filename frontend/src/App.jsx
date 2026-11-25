import React, { useEffect, useState } from "react";
import { BookOpen, FlaskConical, Layers, Loader2, Trophy, Upload } from "lucide-react";
import Sidebar, { initialSettings, buildRequest } from "./Sidebar.jsx";
import Leaderboard from "./Leaderboard.jsx";
import MethodTab from "./MethodTab.jsx";
import ValidationTab from "./ValidationTab.jsx";
import UploadTab from "./UploadTab.jsx";
import { api, human, ErrorBox, API_BASE } from "./ui.jsx";

const TABS = [
  { id: "leaderboard", label: "Leaderboard", icon: Trophy },
  { id: "method", label: "Method", icon: BookOpen },
  { id: "validation", label: "Validation", icon: FlaskConical },
  { id: "upload", label: "Upload", icon: Upload },
];

export default function App() {
  const [meta, setMeta] = useState(null);
  const [metaErr, setMetaErr] = useState("");
  const [s, setS] = useState(null);
  const [tab, setTab] = useState("leaderboard");
  const [result, setResult] = useState(null);
  const [running, setRunning] = useState(false);
  const [runErr, setRunErr] = useState("");
  const [stress, setStress] = useState(null);
  const [stressRunning, setStressRunning] = useState(false);
  const [stressErr, setStressErr] = useState("");

  async function run(m = meta, settings = s) {
    setRunning(true); setRunErr("");
    const r = await api("/api/recommend", { method: "POST", body: buildRequest(m, settings) });
    setRunning(false);
    if (r.error) { setRunErr(r.error); setResult(null); } else setResult(r);
  }

  async function runStress() {
    setStressRunning(true); setStressErr("");
    const r = await api("/api/stress-test", { method: "POST", body: buildRequest(meta, s) });
    setStressRunning(false);
    if (r.error) { setStressErr(r.error); setStress(null); } else setStress(r);
  }

  useEffect(() => {
    (async () => {
      const m = await api("/api/meta");
      if (m.error) { setMetaErr(m.error); return; }
      const init = initialSettings(m);
      setMeta(m); setS(init);
      document.title = m.title || document.title;
      run(m, init);
    })();
  }, []);

  if (metaErr) return <div className="p-8 max-w-xl"><ErrorBox msg={`${metaErr}\nStart the backend (uvicorn main:app --port 8000) or set VITE_API_BASE.`} /></div>;
  if (!meta || !s) return <div className="p-8 flex items-center gap-2 text-slate-500"><Loader2 className="w-4 h-4 animate-spin" /> Loading configuration from {API_BASE}…</div>;

  return (
    <div className="flex min-h-screen text-slate-900">
      <Sidebar meta={meta} s={s} setS={setS} onRun={() => run()} running={running} lastResult={result} />
      <main className="flex-1 min-w-0">
        <header className="bg-white border-b border-slate-200 px-6 py-4">
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-sky-700" />
            <h1 className="text-lg font-semibold">{meta.title || "Plithogenic-PIV Decision Engine"}</h1>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            {meta.n_rows} alternatives from {meta.origin} · attribute <b>{human(meta.attribute)}</b> · dominant value <b>{human(meta.dominant_value)}</b>
            {result && <> · showing {result.n_matched} after filters</>}
          </p>
          <nav className="flex gap-1 mt-3 -mb-4">
            {TABS.map((t) => (
              <button key={t.id} onClick={() => setTab(t.id)}
                className={`inline-flex items-center gap-1.5 px-3 py-2 text-sm border-b-2 ${tab === t.id ? "border-slate-900 text-slate-900 font-medium" : "border-transparent text-slate-500 hover:text-slate-700"}`}>
                <t.icon className="w-4 h-4" /> {t.label}
              </button>
            ))}
          </nav>
        </header>
        <div className="p-6 space-y-4 max-w-[1600px]">
          {tab === "leaderboard" && (<><ErrorBox msg={runErr} /><Leaderboard meta={meta} result={result} /></>)}
          {tab === "method" && <MethodTab meta={meta} />}
          {tab === "validation" && <ValidationTab data={stress} running={stressRunning} onRun={runStress} error={stressErr} />}
          {tab === "upload" && <UploadTab />}
        </div>
      </main>
    </div>
  );
}

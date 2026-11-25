# Plithogenic-PIV Decision Engine

## Backend (FastAPI)
    cd backend
    pip install fastapi uvicorn python-multipart pandas numpy scipy
    uvicorn main:app --reload --port 8000

- Config: `ppiv_config.json` next to `main.py` (override with env `PPIV_CONFIG`).
- At startup it loads `colleges_ready.csv` (or prepares it from `paths.input_csv`) and
  `frozen_params.json` (fitted ONCE on the full data and saved if missing; a stale file is an error, never silently refit).
- New optional config block `api`: presentation-only settings (label column, review-count column,
  low-review threshold, display-only labels, extra caveats, upload limit, validation seed). Nothing in it enters a ranking.

Endpoints: `GET /api/meta`, `POST /api/recommend`, `POST /api/stress-test`, `POST /api/upload-csv`
(multipart: `file` = CSV, `config` = JSON file or `config_json` = JSON text, optional `request` = JSON of the recommend body).

Recommend body:
    {"filters": {"State": "Kerala", "Institution_Type": ["Private"]},   // numeric filter: [min, max]
     "weight_mode": "merec" | "custom", "custom_weights": {"Reputation": 0.3, ...},
     "contradiction_override": {"Faculty_Quality": 0.2}, "exclude_values": ["Academic_Breadth"],
     "operator": "intersection" | "union", "normalization": "vector" | "none",
     "frozen_weights": false, "min_reviews": 5}

- `frozen_weights: true` = MEREC weights computed on ALL rows under the same settings, reused on the filtered subset
  (the classical baseline gets its own full-data MEREC weights).
- `contradiction_override {value: c}` replaces c(v_D, v) = c(v, v_D) in the frozen matrix for that call only.
- Stress test = `validation.evaluate` (agreement + rank reversal, recomputed and frozen modes kept separate)
  + `validation.sensitivity` (leave-one-value-out, weight perturbation, parameter sweeps). Sweep variants that change
  anchors are refit on the FULL data, never on the filtered rows.

Tests: `python -m pytest -q` (existing suite; 30 tests skip without the raw master CSV).

## Frontend (React + Tailwind + lucide-react + recharts)
    cd frontend
    npm install
    cp .env.example .env      # set VITE_API_BASE if the API is not on localhost:8000
    npm run dev               # http://localhost:5173

All criteria, filters, labels, sliders and caveats come from `GET /api/meta`.

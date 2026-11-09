# LogiTrack

LogiTrack is a React and TypeScript logistics analytics dashboard with a local FastAPI service for real delivery-duration estimates.

## Quick start on Windows

Install Node.js and Python, then run these commands from the project folder in PowerShell:

```powershell
npm install
npm run setup:api
npm run dev:all
```

`npm run dev:all` starts both services together:

- Dashboard: `http://127.0.0.1:5173`
- Prediction API health check: `http://127.0.0.1:8000/health`

The web server waits for FastAPI to report a ready model before it starts Vite, so the first Predictive Analytics request does not race model loading.

The API launcher checks for `machine-learning/models/delivery_time_pipeline.joblib`. The file is intentionally not committed because it is large; if it is missing after a fresh clone, the launcher trains the reproducible Random Forest pipeline from `amazon_delivery.csv` before starting FastAPI. Backend startup and training errors remain visible in the same terminal.

To stop both services, press `Ctrl+C` once in that terminal.

## Prediction API configuration

Local development uses Vite's `/api` proxy by default, so the React client does not hard-code a port or trigger a browser CORS request. Optional settings are documented in [.env.example](.env.example):

```powershell
Copy-Item .env.example .env.local
```

- `LOGITRACK_API_PROXY_TARGET` changes Vite's local proxy target (default: `http://127.0.0.1:8000`).
- `LOGITRACK_API_HEALTH_URL` changes the readiness check used by `npm run dev:all` when the API is hosted at a different local address.
- `VITE_PREDICTION_API_URL` sets a direct API URL at frontend build time for a deployed dashboard.
- `LOGITRACK_CORS_ORIGINS` configures FastAPI origins when a direct browser-to-API URL is used.

The Predictive Analytics page first checks `/health`, then reads `/metadata`. When the API is offline, verified static model metadata and historical held-out evaluation charts remain visible; only live estimates and scenario comparisons are disabled. Use **Retry connection** after starting or restoring the API.

## Model workflow

Regenerate the trained local artifact and metadata when the dataset or pipeline changes:

```powershell
python machine-learning/train_model.py
```

The dashboard's static evaluation view bundles the training-generated `machine-learning/models/model_metadata.json`, so it remains traceable to the same reproducible model run even while the API is offline.

The Predictive Analytics page supports one delivery estimate and a two-scenario comparison. Both call the same inference API; differences are model-based associations, not causal claims. The API validates categorical fields against the training schema and numeric fields against training bounds.

## Verification

```powershell
python -m pytest api/test_api.py machine-learning/test_consistency.py -q
npm run build
```

## Project folders

- `src/` dashboard source, including the prediction API client and offline predictive view
- `machine-learning/` reproducible training script, notebook, model metadata, and evaluation predictions
- `api/` FastAPI inference service
- `scripts/` local development launchers
- `dataset/` data dictionary; the original CSV remains in the project root
- `documentation/` project plan and methodology
- `presentation/` presentation script and recording checklist

## Model scope

The current model predicts `Delivery_Time` in minutes using historical features available at order time. It uses a chronological holdout split and training-only preprocessing. It is an estimate, not a guarantee or causal model. Demand forecasting is not implemented because the source covers only 44 dates.

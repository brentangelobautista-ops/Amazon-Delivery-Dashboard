# LogiTrack

LogiTrack is a React and TypeScript logistics analytics dashboard with a local FastAPI prediction service for delivery-duration estimates.

## Run the dashboard

```powershell
npm install
npm run dev
```

## Train and serve the model

```powershell
python -m pip install -r machine-learning/requirements.txt
python machine-learning/train_model.py
python -m uvicorn api.main:app --reload
```

The dashboard fetches model metadata and predictions from `http://127.0.0.1:8000`. If the service is unavailable, the Predictive Analytics page clearly reports that state and does not fabricate inference results.

## Project folders

- `src/` dashboard source
- `machine-learning/` reproducible training script, notebook, model artifacts, and evaluation predictions
- `api/` FastAPI inference service
- `dataset/` data dictionary; the original CSV remains in the project root
- `documentation/` project plan and methodology
- `presentation/` presentation script and recording checklist

## Model scope

The current model predicts `Delivery_Time` in minutes using historical features available at order time. It uses a chronological holdout split and training-only preprocessing. It is an estimate, not a guarantee or causal model. Demand forecasting is not implemented because the source covers only 44 dates.

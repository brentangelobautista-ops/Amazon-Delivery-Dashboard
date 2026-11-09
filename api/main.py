"""Local inference service for the exported LogiTrack regression pipeline."""
from __future__ import annotations
import json, os
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT=Path(__file__).resolve().parents[1]; MODELS=ROOT/'machine-learning'/'models'
MODEL_PATH=Path(os.getenv('LOGITRACK_MODEL_PATH',MODELS/'delivery_time_pipeline.joblib'))
META_PATH=Path(os.getenv('LOGITRACK_METADATA_PATH',MODELS/'model_metadata.json'))
model=joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
metadata=json.loads(META_PATH.read_text(encoding='utf-8')) if META_PATH.exists() else None
origins=[item.strip() for item in os.getenv('LOGITRACK_CORS_ORIGINS','http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174').split(',') if item.strip()]
app=FastAPI(title='LogiTrack Prediction API',version='1.1.0')
app.add_middleware(CORSMiddleware,allow_origins=origins,allow_methods=['GET','POST'],allow_headers=['Content-Type'])

class DeliveryInput(BaseModel):
    Agent_Age: float|None=None
    Agent_Rating: float|None=None
    Distance_Model_km: float=Field(ge=0,le=50000)
    Order_Hour:int=Field(ge=-1,le=23)
    Day_of_Week:int=Field(ge=0,le=6)
    Weather:str; Traffic:str; Vehicle:str; Area:str; Category:str

def ready():
    if model is None or metadata is None: raise HTTPException(503,'Prediction model is unavailable. Run machine-learning/train_model.py.')

def validate_payload(row:dict):
    for feature,choices in metadata['allowed_categories'].items():
        if row[feature] not in choices: raise HTTPException(422,f'Unsupported {feature}: {row[feature]}.')
    for feature,bounds in metadata['numeric_bounds'].items():
        if feature=='Distance_Model_km': continue
        value=row.get(feature)
        if value is not None and not bounds['min']<=value<=bounds['max']:
            raise HTTPException(422,f'{feature} must be between {bounds["min"]} and {bounds["max"]} based on training data.')

@app.get('/health')
def health(): return {'ready':model is not None and metadata is not None,'model':metadata.get('model') if metadata else None}

@app.get('/metadata')
def get_metadata(): ready(); return metadata

@app.get('/evaluation')
def evaluation(): ready(); return {'held_out_metrics':metadata['held_out_metrics'],'candidate_models':metadata['candidate_models'],'diagnostics':metadata['evaluation_diagnostics'],'importance_method':metadata['importance_method']}

@app.post('/predict')
def predict(payload:DeliveryInput):
    ready(); row=payload.model_dump(); validate_payload(row)
    original=row['Distance_Model_km']; row['Distance_Model_km']=min(original,metadata['distance_model_cap_p99_km'])
    try: value=float(model.predict(pd.DataFrame([row]))[0])
    except Exception: raise HTTPException(500,'Prediction could not be generated.')
    return {'predicted_delivery_minutes':round(max(0,value),1),'model':metadata['model'],'distance_input_capped':original!=row['Distance_Model_km'],'disclaimer':'This is a model-based estimate from historical data, not an actual outcome or causal claim.'}

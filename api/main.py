from pathlib import Path
import json, joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT=Path(__file__).resolve().parents[1]; MODELS=ROOT/'machine-learning'/'models'
MODEL_PATH=MODELS/'delivery_time_pipeline.joblib'; META_PATH=MODELS/'model_metadata.json'
model=joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
metadata=json.loads(META_PATH.read_text()) if META_PATH.exists() else None
app=FastAPI(title='LogiTrack Prediction API',version='1.0.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173','http://localhost:5174','http://127.0.0.1:5174'],allow_methods=['*'],allow_headers=['*'])
class DeliveryInput(BaseModel):
    Agent_Age: float|None=None; Agent_Rating: float|None=None; Distance_Model_km: float=Field(ge=0,le=50000); Order_Hour:int=Field(ge=-1,le=23); Day_of_Week:int=Field(ge=0,le=6)
    Weather:str; Traffic:str; Vehicle:str; Area:str; Category:str
@app.get('/health')
def health(): return {'ready':model is not None,'model':metadata.get('model') if metadata else None}
@app.get('/metadata')
def get_metadata():
    if not metadata: raise HTTPException(503,'Model metadata is unavailable. Run machine-learning/train_model.py first.')
    return metadata
@app.post('/predict')
def predict(payload:DeliveryInput):
    if model is None or metadata is None: raise HTTPException(503,'Model unavailable. Train the pipeline first.')
    row=payload.model_dump(); row['Distance_Model_km']=min(row['Distance_Model_km'],metadata['distance_model_cap_p99_km'])
    import pandas as pd
    value=float(model.predict(pd.DataFrame([row]))[0])
    return {'predicted_delivery_minutes':round(max(0,value),1),'model':metadata['model'],'distance_input_capped':payload.Distance_Model_km!=row['Distance_Model_km'],'disclaimer':'This is a model-based estimate from historical data, not an actual outcome or causal claim.'}

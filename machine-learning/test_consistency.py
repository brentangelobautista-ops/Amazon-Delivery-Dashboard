import json
from pathlib import Path
import joblib
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
meta=json.loads((ROOT/'machine-learning/models/model_metadata.json').read_text())
model=joblib.load(ROOT/'machine-learning/models/delivery_time_pipeline.joblib')

def test_artifact_and_metadata_feature_schema_match():
    assert list(model.feature_names_in_)==meta['features']

def test_pipeline_prediction_is_finite():
    row={'Agent_Age':30,'Agent_Rating':4.5,'Distance_Model_km':8,'Order_Hour':12,'Day_of_Week':2,'Weather':'Sunny','Traffic':'Medium','Vehicle':'motorcycle','Area':'Urban','Category':'Electronics'}
    value=float(model.predict(pd.DataFrame([row]))[0])
    assert value>=0 and value<1000

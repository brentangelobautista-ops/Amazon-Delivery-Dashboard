from fastapi.testclient import TestClient
from api.main import app, metadata

client=TestClient(app)
payload={'Agent_Age':30,'Agent_Rating':4.5,'Distance_Model_km':8,'Order_Hour':12,'Day_of_Week':2,'Weather':'Sunny','Traffic':'Medium','Vehicle':'motorcycle','Area':'Urban','Category':'Electronics'}

def test_health_and_metadata():
    assert client.get('/health').json()['ready'] is True
    response=client.get('/metadata'); assert response.status_code==200
    assert response.json()['model']==metadata['model']

def test_valid_prediction_has_stable_contract():
    response=client.post('/predict',json=payload); assert response.status_code==200
    body=response.json(); assert body['model']==metadata['model']; assert body['predicted_delivery_minutes']>=0

def test_invalid_category_is_rejected():
    response=client.post('/predict',json={**payload,'Traffic':'Impossible traffic'})
    assert response.status_code==422

def test_invalid_numeric_edge_is_rejected():
    response=client.post('/predict',json={**payload,'Order_Hour':24})
    assert response.status_code==422

def test_evaluation_has_held_out_diagnostics():
    response=client.get('/evaluation'); assert response.status_code==200
    assert response.json()['held_out_metrics']['mae']==17.595

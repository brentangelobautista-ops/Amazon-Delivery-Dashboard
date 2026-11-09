"""Reproducible delivery-time regression training for LogiTrack."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(__file__).resolve().parents[1]
CSV=ROOT/'amazon_delivery.csv'; MODELS=Path(__file__).parent/'models'; EVAL=Path(__file__).parent/'evaluation'
MODELS.mkdir(exist_ok=True); EVAL.mkdir(exist_ok=True)
SEED=42

def haversine(lat1, lon1, lat2, lon2):
    r=6371.0
    p1,p2=np.radians(lat1),np.radians(lat2)
    dp=np.radians(lat2-lat1); dl=np.radians(lon2-lon1)
    a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*r*np.arcsin(np.sqrt(np.clip(a,0,1)))

def prepare(frame:pd.DataFrame)->pd.DataFrame:
    df=frame.copy()
    df['Order_Date']=pd.to_datetime(df['Order_Date'], errors='coerce')
    for c in ['Agent_Age','Agent_Rating','Store_Latitude','Store_Longitude','Drop_Latitude','Drop_Longitude','Delivery_Time']:
        df[c]=pd.to_numeric(df[c],errors='coerce')
    valid_coords=df[['Store_Latitude','Store_Longitude','Drop_Latitude','Drop_Longitude']].notna().all(axis=1)
    valid_coords &= df.Store_Latitude.between(-90,90)&df.Drop_Latitude.between(-90,90)&df.Store_Longitude.between(-180,180)&df.Drop_Longitude.between(-180,180)
    df=df.loc[valid_coords & df.Order_Date.notna() & df.Delivery_Time.notna()].copy()
    df['Distance_km']=haversine(df.Store_Latitude,df.Store_Longitude,df.Drop_Latitude,df.Drop_Longitude)
    time=pd.to_datetime(df['Order_Time'],format='%H:%M:%S',errors='coerce')
    df['Order_Hour']=time.dt.hour.fillna(-1)
    df['Day_of_Week']=df.Order_Date.dt.day_of_week
    # Geographic extremes remain valid source records; cap only the model input at training-set P99 to reduce leverage.
    for c in ['Weather','Traffic','Vehicle','Area','Category']:
        df[c]=df[c].fillna('Unknown').astype(str).str.strip().replace({'NaN':'Unknown','nan':'Unknown','':'Unknown'})
    return df

def metrics(y, pred):
    return {'mae':round(float(mean_absolute_error(y,pred)),3),'rmse':round(float(mean_squared_error(y,pred)**.5),3),'r2':round(float(r2_score(y,pred)),3)}

raw=pd.read_csv(CSV)
df=prepare(raw).sort_values('Order_Date').reset_index(drop=True)
split_date=df.Order_Date.sort_values().iloc[int(len(df)*.8)]
train=df[df.Order_Date<split_date].copy(); test=df[df.Order_Date>=split_date].copy()
cap=float(train.Distance_km.quantile(.99))
train['Distance_Model_km']=train.Distance_km.clip(upper=cap); test['Distance_Model_km']=test.Distance_km.clip(upper=cap)
num=['Agent_Age','Agent_Rating','Distance_Model_km','Order_Hour','Day_of_Week']
cat=['Weather','Traffic','Vehicle','Area','Category']
features=num+cat; target='Delivery_Time'
preprocess=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
models={'Dummy mean':DummyRegressor(strategy='mean'),'Ridge':Pipeline([('prep',preprocess),('model',Ridge(alpha=3.0))]),'Random Forest':Pipeline([('prep',preprocess),('model',RandomForestRegressor(n_estimators=160,min_samples_leaf=3,max_features=.8,n_jobs=-1,random_state=SEED))]),'Gradient Boosting':Pipeline([('prep',preprocess),('model',GradientBoostingRegressor(n_estimators=220,learning_rate=.045,max_depth=3,min_samples_leaf=8,random_state=SEED))])}
results=[]; fitted={}
for name,model in models.items():
    model.fit(train[features],train[target]); pred=model.predict(test[features]); results.append({'model':name,**metrics(test[target],pred)}); fitted[name]=model
best_name=min(results,key=lambda r:r['mae'])['model']; best=fitted[best_name]; predictions=best.predict(test[features])
joblib.dump(best,MODELS/'delivery_time_pipeline.joblib')

# Aggregate tree-based transformed feature importances to source variables.
importance=[]
if best_name in {'Random Forest','Gradient Boosting'}:
    names=best.named_steps['prep'].get_feature_names_out(); values=best.named_steps['model'].feature_importances_; grouped={}
    for name,value in zip(names,values):
        source=name.split('__',1)[-1].split('_',1)[0] if name.startswith('cat__') else name.split('__',1)[-1]
        grouped[source]=grouped.get(source,0)+float(value)
    importance=[{'feature':k,'importance':round(v,4)} for k,v in sorted(grouped.items(),key=lambda kv:kv[1],reverse=True)]
residuals=test[target].to_numpy()-predictions
hist_counts,hist_edges=np.histogram(residuals,bins=24)
sample_indices=np.linspace(0,len(test)-1,min(1200,len(test)),dtype=int)
metadata={'project':'LogiTrack','task':'Delivery duration prediction','target':'Delivery_Time (minutes)','model':best_name,'features':features,'categorical_features':cat,'numeric_features':num,'train_rows':int(len(train)),'test_rows':int(len(test)),'chronological_split_date':split_date.strftime('%Y-%m-%d'),'distance_model_cap_p99_km':round(cap,3),'candidate_models':results,'held_out_metrics':metrics(test[target],predictions),'feature_importance':importance,'importance_method':'Native Random Forest impurity importance aggregated from one-hot encoded features to source variables.','allowed_categories':{c:sorted(train[c].unique().tolist()) for c in cat},'numeric_bounds':{c:{'min':round(float(train[c].min()),3),'max':round(float(train[c].max()),3)} for c in num},'evaluation_diagnostics':{'actual_vs_predicted_sample':[{'actual':round(float(test[target].iloc[i]),2),'predicted':round(float(predictions[i]),2)} for i in sample_indices],'residual_histogram':[{'range':f'{hist_edges[i]:.1f} to {hist_edges[i+1]:.1f}','count':int(hist_counts[i])} for i in range(len(hist_counts))]},'notes':['The pipeline fits imputers, scaling, and one-hot encoding on training data only.','Distance is capped at the training P99 only for model input robustness; the original source data remains unchanged.','Predictions estimate historical patterns and do not establish causality.','Prediction intervals are not displayed because no independent calibration experiment has validated interval coverage.']}
(MODELS/'model_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
pd.DataFrame({'actual':test[target].round(3),'predicted':predictions.round(3),'residual':(test[target]-predictions).round(3)}).to_csv(EVAL/'held_out_predictions.csv',index=False)
print(json.dumps(metadata,indent=2))

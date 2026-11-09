# LogiTrack Presentation Script

## 0:00–1:00 Project introduction

Introduce LogiTrack as a logistics analytics dashboard that combines historical monitoring with delivery-time prediction. State that the prediction estimates delivery duration from historical operational characteristics.

## 1:00–2:00 Dataset and exploratory findings

Show the Overview page. Explain that the project uses 43,739 delivery records and includes traffic, weather, vehicle, area, category, timing, ratings, and coordinates. Mention the geographic anomaly handling: charts use the 99th percentile for readability while overall metrics retain all valid routes.

## 2:00–3:00 Dashboard walkthrough

Demonstrate filters, Overview KPIs, Performance comparisons, Operations distributions, Trends, Geographic Analysis, and the paginated Data Explorer.

## 3:00–5:30 Machine-learning methodology

Explain the prediction target, Haversine distance, order-hour and weekday features, and leakage prevention through excluding pickup time. Explain the chronological train/test split. Show the candidate comparison: Random Forest has held-out MAE 17.595 minutes, RMSE 22.647 minutes, and R² 0.814.

## 5:30–7:30 Predictive Analytics demo

Open Predictive Analytics. Choose a scenario, generate an estimate, then show model comparison and feature-importance panels. Explain that the output is a model estimate, not a guaranteed result or causal finding.

## 7:30–9:00 Conclusion

Summarize the integration of descriptive, diagnostic, and predictive analytics. Note constraints: a static historical dataset, geographic anomalies, unobserved factors, and too few dates for reliable demand forecasting.

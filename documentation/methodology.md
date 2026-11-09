# LogiTrack Predictive Analytics Methodology

## Objective

LogiTrack predicts delivery duration in minutes using historical operational and logistics characteristics. The prediction target is `Delivery_Time`.

## Data preparation

The source dataset contains 43,739 delivery orders. Training keeps rows with valid coordinates, a parseable order date, and an available delivery-time target. Haversine distance uses store latitude/longitude and delivery latitude/longitude. Coordinate values must lie within latitude [-90, 90] and longitude [-180, 180].

The dataset contains extreme but range-valid route distances. The training feature caps distance at the training-set 99th percentile (20.853 km) to reduce leverage from geographic anomalies. This cap applies only to model input. It does not edit or delete original historical records.

The model uses agent age, agent rating, capped route distance, order hour, day of week, weather, traffic, vehicle, area, and category. It excludes `Pickup_Time` because it can reflect a later operational stage and could introduce leakage for an order-time prediction task.

## Evaluation design

Records are sorted by order date. The model trains on 34,682 earlier records and evaluates on 9,057 later records from 2022-03-29 onward. This chronological holdout better matches prospective delivery prediction than a random split. Preprocessing fits only on training data through a scikit-learn pipeline.

## Results

| Model | MAE (min) | RMSE (min) | R² |
|---|---:|---:|---:|
| Dummy mean | 41.785 | 52.493 | -0.001 |
| Ridge | 25.394 | 32.232 | 0.623 |
| Random Forest | 17.595 | 22.647 | 0.814 |
| Gradient Boosting | 19.364 | 24.854 | 0.776 |

Random Forest was selected using held-out MAE. Feature importance is model-based and does not demonstrate causality.

## Limitations

This source covers 44 order dates, which is not enough for a reliable demand-forecasting feature. Predictions estimate patterns in historical data and cannot guarantee actual outcomes. Geographic anomalies and unmeasured factors may affect delivery duration.

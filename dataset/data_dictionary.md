# LogiTrack Data Dictionary

Source: `amazon_delivery.csv`, preserved without replacement or synthetic rows. It contains 43,739 records and 16 fields.

| Field | Type | Use |
|---|---|---|
| Order_ID | string | Unique delivery order identifier |
| Agent_Age | numeric | Candidate prediction input |
| Agent_Rating | numeric | Candidate prediction input; missing values are imputed inside the training pipeline |
| Store_Latitude, Store_Longitude | numeric | Store coordinates used for Haversine distance |
| Drop_Latitude, Drop_Longitude | numeric | Delivery coordinates used for Haversine distance |
| Order_Date | date | Trend analysis and chronological split |
| Order_Time | time | Engineered order-hour feature |
| Pickup_Time | time | Excluded from prediction because it may reflect an operational stage after order placement |
| Weather | category | Candidate prediction input |
| Traffic | category | Candidate prediction input |
| Vehicle | category | Candidate prediction input |
| Area | category | Candidate prediction input |
| Delivery_Time | numeric | Prediction target in minutes |
| Category | category | Candidate prediction input |

Derived fields include `Distance_km` from a Haversine calculation, `Order_Hour`, and `Day_of_Week`. Invalid coordinate ranges are excluded from distance/model computation. Source categorical `NaN` strings are normalized to `Unknown`.

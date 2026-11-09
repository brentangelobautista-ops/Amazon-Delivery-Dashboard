# LogiTrack Project Plan

## Findings Summary

LogiTrack analyzes 43,739 historical delivery orders with 16 source columns. The dataset combines order timing, agent characteristics, operational conditions, vehicle and area information, product category, delivery duration, and origin/destination coordinates. The dashboard calculates delivery distance using the Haversine formula.

The observed date range is 2022-02-11 through 2022-04-06. Agent rating has 54 missing values. The model pipeline imputes this feature using training data only. Geographic analysis found a mean distance of 38.56 km and a median of 9.22 km. A small set of range-valid but geographically anomalous coordinates creates distances up to 19,692.67 km; the dashboard retains them in overall statistics and limits visual axes to the 99th percentile. Traffic and weather categories have different average delivery-time patterns, which motivates operational analysis without claiming causation.

## Problem Statement

Delivery managers must estimate delivery duration while traffic, weather, vehicle choice, route distance, and order characteristics vary. Historical monitoring explains what has happened, but it does not provide an estimate for a new scenario. This project evaluates supervised regression models that estimate delivery duration from characteristics available at order time. These estimates can support planning and scenario discussion, subject to the limits of observational data.

## Intended Audience

Logistics managers can compare historical operating conditions with prediction estimates. Delivery supervisors can explore scenario inputs before assigning or monitoring work. Fleet coordinators can inspect vehicle and area patterns. Business and data analysts can review held-out model performance, feature importance, and data-quality limitations.

## Chosen Advanced Analytics Type

The primary approach is predictive analytics through supervised machine-learning regression. Regression fits this problem because delivery duration is continuous in minutes. Classification would require arbitrary duration bands and would discard information. Demand forecasting was assessed but not implemented because the dataset spans only 44 dates.

## Project Plan

1. Audit and profile source data.
2. Prepare valid modeling records and engineered features.
3. Train baseline, linear, bagged-tree, and boosted-tree regressors.
4. Select a model on a chronological held-out evaluation set.
5. Export the pipeline and metadata.
6. Serve asynchronous predictions through FastAPI.
7. Integrate prediction, scenario, evaluation, and limitations panels into LogiTrack.

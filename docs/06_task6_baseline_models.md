# Task 6 – Dummy, Linear Regression and Random Forest

> Notebook section: see `Bike_Rental_Demand_Forecasting.ipynb` · Numbers quoted are from the executed notebook (`results/`).

## 1. What the task requires
Train DummyRegressor, Linear Regression and Random Forest; record configurations and comparison metrics; explain differences.

## 2. Why this approach
The dummy defines the floor, Linear Regression a simple interpretable benchmark and Random Forest a flexible non-linear model. MAE (robust, interpretable in rentals/day), RMSE (penalises big misses) and R² are the standard regression metrics.

## 3. How it was implemented
Fit each model on the training matrix; compute train and validation metrics; table, bar charts and printed parameters (`DummyRegressor(strategy='mean')`, `LinearRegression()` defaults, `RandomForestRegressor(n_estimators=300, random_state=42)`).

## 4. Logic behind the implementation
Comparing train and validation error per model reveals over-fitting (large gap) and under-fitting; the test block is not yet used.

## 5. Important decisions, assumptions and considerations
* 300 trees – enough for stable averaging at low cost.
* R² on a narrow validation block can be strongly negative even for decent absolute errors, so MAE/RMSE are primary.

## 6. Expected output and how to read it
Validation RMSE: Dummy 3,076, Linear 959, Forest 847. Forest train RMSE 228 ⇒ strong over-fitting.

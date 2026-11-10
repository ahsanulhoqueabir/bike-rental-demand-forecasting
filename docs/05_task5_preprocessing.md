# Task 5 – Train-only encoding and numerical preprocessing

> Notebook section: see `Bike_Rental_Demand_Forecasting.ipynb` · Numbers quoted are from the executed notebook (`results/`).

## 1. What the task requires
Build train-only encoding and numerical preprocessing; record method, settings and libraries; demonstrate that no held-out information is used.

## 2. Why this approach
Fitting scalers/encoders on all data lets validation/test statistics influence training (a subtle leak). Nominal codes (month, weekday, season, weather) need one-hot encoding because their numeric order is arbitrary for linear models.

## 3. How it was implemented
`ColumnTransformer`: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` for `season, mnth, weekday, weathersit`; passthrough for `yr, holiday, workingday`; `StandardScaler` for `temp, hum, windspeed`. `fit` on `X_train` only; `transform` on all three blocks → 32 columns.

## 4. Logic behind the implementation
`fit` learns means/stds/categories from training only; `transform` applies them unchanged to later blocks. Evidence: (1) fitted `scaler.mean_` equals train means, not full-data means; (2) train columns standardised to mean 0/std 1 while val/test are not; (3) editing val/test values cannot change a preprocessor fitted on train; (4) category levels in val/test are all present in train.

## 5. Important decisions, assumptions and considerations
* Random Forest does not need scaling, but one common matrix keeps the model comparison fair.
* `yr` is treated as a numeric 0/1 trend indicator.
* Validation is much hotter than training (+1.35 σ) – left as is; correcting it would require using validation statistics.

## 6. Expected output and how to read it
Shapes (511,32)/(109,32)/(110,32), no NaNs, the evidence tables and assertions all pass.

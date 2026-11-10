# Task 3 – Identify and remove leakage

> Notebook section: see `Bike_Rental_Demand_Forecasting.ipynb` · Numbers quoted are from the executed notebook (`results/`).

## 1. What the task requires
Remove `casual` and `registered` when predicting `cnt`; document the steps and show the significance.

## 2. Why this approach
Target leakage inflates validation scores and yields a model that cannot work in production, because the leaked inputs do not exist at forecast time.

## 3. How it was implemented
Prove `cnt == casual + registered` for all rows; run a throw-away experiment (Linear Regression, chronological 70/30) with and without the two columns; drop them plus `instant` and `atemp`; assert that none of them remain among the features.

## 4. Logic behind the implementation
A model given both components can reproduce the target by addition (MAE = 0, R² = 1.000), which demonstrates the leakage. `instant` is a row counter, i.e. a proxy of time that would let tree models memorise positions; `atemp` (r = 0.992 with `temp`) is redundant and destabilises linear coefficients.

## 5. Important decisions, assumptions and considerations
* `dteday` is kept for sorting/plotting but is **not** a predictor.
* The assignment's list of candidate inputs is followed exactly (10 predictors).
* The leakage experiment is not part of the modelling pipeline; it only illustrates the effect.

## 6. Expected output and how to read it
Leaky model MAE 0.00 / R² 1.000 vs clean model MAE 834 / RMSE 1,128 / R² 0.53. The final feature list is printed.

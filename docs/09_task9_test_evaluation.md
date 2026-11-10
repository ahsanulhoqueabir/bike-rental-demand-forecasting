# Task 9 – Final evaluation on the held-out test block

> Notebook section: see `Bike_Rental_Demand_Forecasting.ipynb` · Numbers quoted are from the executed notebook (`results/`).

## 1. What the task requires
Evaluate MAE and RMSE on the most recent test block with a results table and labelled visualisation; interpret and state limitations.

## 2. Why this approach
The test block gives the least biased estimate of future performance, because no choice was made with it.

## 3. How it was implemented
Predict the test block with all models trained on train only; table MAE/RMSE/R², bias, median absolute % error; extras clearly marked: seasonal-naive lag-364 and the tuned forest refit on train+validation (preprocessor refit on train+val only). Figures: actual vs predicted lines + bar chart (Fig. 10), predicted-vs-actual scatter and residual histogram (Fig. 11).

## 4. Logic behind the implementation
Extra baselines test whether the learned models add value (seasonal naive) and whether more recent data helps (refit). MAPE is replaced by the median APE because one test day has cnt = 22 (its percentage error is in the thousands of per cent).

## 5. Important decisions, assumptions and considerations
* Primary result = tuned forest as chosen on validation (MAE 1,229, RMSE 1,402).
* Linear Regression is better (MAE 863, RMSE 1,158); results are reported honestly, not re-tuned on test.
* Limitation: single 110-day block.

## 6. Expected output and how to read it
Results table (also `results/task9_test_metrics.csv/json`) and Figs 10–11. Bias column explains the ranking: trees −726, linear +10.

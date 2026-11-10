# Task 7 – Tuning depth and minimum samples per leaf

> Notebook section: see `Bike_Rental_Demand_Forecasting.ipynb` · Numbers quoted are from the executed notebook (`results/`).

## 1. What the task requires
Tune `max_depth` and `min_samples_leaf` on the chronological validation block; show a comparison table/plot; explain the choice.

## 2. Why this approach
These two parameters control tree complexity (the main over-fitting lever of a forest). Selecting them on the later validation block respects time order; shuffled K-fold CV would leak the future.

## 3. How it was implemented
Grid `max_depth ∈ {2,3,4,5,6,8,10,None}` × `min_samples_leaf ∈ {1,2,5,10,20,30}` (48 forests, 300 trees, seed 42), each trained on train only and scored on validation; ranking by validation RMSE with ties broken to the simpler model; heat-maps of validation and training RMSE and a depth line plot; full grid saved as CSV.

## 4. Logic behind the implementation
Exhaustive search is cheap (511 rows) and transparent; reporting train RMSE next to validation RMSE shows the complexity/over-fit trade-off.

## 5. Important decisions, assumptions and considerations
* Selection uses validation only; test is untouched.
* The winner (depth 10, leaf 5, val RMSE 825.8) sits on a flat plateau – differences of ~5 RMSE points are within noise.
* Validation is one summer block, so the choice can be sensitive to it.

## 6. Expected output and how to read it
Top-10 table, heat-maps (Fig. 7). Tuned vs default: val RMSE 825.8 vs 847.1.

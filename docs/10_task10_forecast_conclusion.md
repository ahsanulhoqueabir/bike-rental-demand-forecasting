# Task 10 – Forecast five test days, seasonal change and conclusion

> Notebook section: see `Bike_Rental_Demand_Forecasting.ipynb` · Numbers quoted are from the executed notebook (`results/`).

## 1. What the task requires
Forecast five test-period days, plot actual vs predicted, discuss seasonal change, summarise findings, unexpected outcomes and two real-world limitations.

## 2. Why this approach
A concrete day-level view and a monthly breakdown make the errors tangible and show how accuracy varies with season.

## 3. How it was implemented
Five days chosen deterministically by `np.linspace(0, 109, 5)` (13 Sep, 10 Oct, 6 Nov, 4 Dec, 31 Dec); table with weather, actual, predicted, error; Fig. 12; monthly table/Fig. 13 for Sep–Dec; impurity and permutation importance (Fig. 14).

## 4. Logic behind the implementation
Deterministic spacing avoids cherry-picking; monthly aggregation shows seasonal behaviour; importances show what the model uses (with the caveat that permutation importance on a shifted validation block disagrees with impurity importance).

## 5. Important decisions, assumptions and considerations
* Four of five days under-predicted; the best is 31 Dec (+83).
* Demand falls 47 % from Sep to Dec and the model follows the shape; relative error rises from 15.5 % to 26.5 %.
* Two limitations: non-stationary growth with few seasonal cycles; reliance on actual (not forecast) weather and invisible events.

## 6. Expected output and how to read it
Five-day table, Figs 12–14, concluding summary in the notebook and `results/RESULTS_ANALYSIS.md`.

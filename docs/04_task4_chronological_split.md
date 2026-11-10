# Task 4 – Chronological 70/15/15 split

> Notebook section: see `Bike_Rental_Demand_Forecasting.ipynb` · Numbers quoted are from the executed notebook (`results/`).

## 1. What the task requires
Sort by date and create contiguous 70/15/15 train/validation/test blocks without shuffling.

## 2. Why this approach
Forecasting must use the past to predict the future; random splits leak neighbouring days and overestimate accuracy.

## 3. How it was implemented
Sort by `dteday`; `n_tr = round(0.70·730) = 511`, `n_va = int(0.15·730) = 109`, test = remainder = 110; slice with `iloc`; assert strict date ordering; table + coloured timeline figure.

## 4. Logic behind the implementation
Three disjoint, ordered blocks give three roles: *train* (fit), *validation* (choose hyper-parameters – later period than train), *test* (touched once at the end – the most recent period).

## 5. Important decisions, assumptions and considerations
* 730 × 0.15 = 109.5 cannot be split evenly: validation 109, test 110 days (70.0/14.9/15.1 %).
* The blocks differ in season and level (validation = May–Sep 2019 mean 6,757; test = Sep–Dec 2019 mean 5,552; train contains 2019 only for Jan–May). This is realistic but important for interpretation.

## 6. Expected output and how to read it
Table with block dates/sizes/means and Fig. 5. Date ranges: train 2018-01-01…2019-05-26, validation 2019-05-27…2019-09-12, test 2019-09-13…2019-12-31.

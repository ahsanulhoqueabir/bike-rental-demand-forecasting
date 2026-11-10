# Task 2 – Exploratory data analysis

> Notebook section: see `Bike_Rental_Demand_Forecasting.ipynb` · Numbers quoted are from the executed notebook (`results/`).

## 1. What the task requires
Plot demand over time and by season and weather; report descriptive statistics, missing values and the target distribution; at least three informative plots and three observations.

## 2. Why this approach
EDA shows what structure a model must capture (trend, seasonality, weather effects) and what pitfalls to expect (shift, outliers) *before* choosing models.

## 3. How it was implemented
Label maps for season/weather (after verifying month ↔ season code in a cross-tab: code 1 = Dec 21–Mar 20 = Winter, unlike the UCI codebook) → `describe()`, missing counts, skew/kurtosis → Fig. 1 time series with 7-day rolling mean, Fig. 2 boxplots by season and weather with group tables, Fig. 3 histogram/temperature scatter/monthly means per year, Fig. 4 correlation matrix.

## 4. Logic behind the implementation
Boxplots show distribution *within* groups (not just means); the rolling mean removes the day-to-day noise to reveal the trend; month-by-year curves expose the year-over-year growth; the correlation heat-map shows which numeric inputs relate to the target and which are redundant (`temp` vs `atemp`).

## 5. Important decisions, assumptions and considerations
* Using verified season labels rather than assuming the UCI codes.
* The rain/snow category has only 21 days – its statistics are less reliable.
* The EDA used the whole data set for description only; no modelling decision uses statistics of the test block.

## 6. Expected output and how to read it
Zero missing values; skew −0.05; season means 2,608/4,992/5,644/4,728; weather means 4,877/4,045/1,803; 2018 mean 3,406 vs 2019 5,610. Three written observations: strong seasonality & weather effect, strong year-over-year growth, large within-group spread.

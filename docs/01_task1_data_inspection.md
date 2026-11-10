# Task 1 – Find and inspect the daily observations file

> Notebook section: see `Bike_Rental_Demand_Forecasting.ipynb` · Numbers quoted are from the executed notebook (`results/`).

## 1. What the task requires
Find `day.csv`, inspect dates and count ranges, include descriptive tables and state at least one important observation.

## 2. Why this approach
Before modelling, the structure, date coding, time ordering and field names must be verified – the assignment explicitly says to check actual file/field names and document differences.

## 3. How it was implemented
`pd.read_csv` → `info()`/`head()` → parse `dteday` with `format='%d-%m-%Y'` → checks for sorting, duplicates, day gaps, per-year counts → min/max of `cnt`, `casual`, `registered` → a table mapping assignment field names to actual columns.

## 4. Logic behind the implementation
Dates are text in day-first format; letting pandas guess could swap day and month (e.g. 02-01-2018). An explicit format removes the ambiguity. A time-series needs a complete, regular daily grid, so gaps (none: 729 one-day steps) and duplicates (none) are verified.

## 5. Important decisions, assumptions and considerations
Findings recorded in the notebook:
* The file covers **2018-01-01 → 2019-12-31**, not 2011–2012 as in the original UCI data.
* Columns are `yr` (0/1), `mnth`, `weathersit`, `temp`, `hum`, plus extra `instant`, `atemp`.
* `temp/hum/windspeed` are already in physical units (no de-normalisation).
* `weathersit` has only levels 1–3.
* `cnt` ranges 22 → 8,714 (mean 4,508).

## 6. Expected output and how to read it
Shape (730, 16), dtypes, first five rows, date range and count ranges, the schema-difference table. Observation: strong range of demand ⇒ judge by absolute errors.

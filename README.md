# Bike Rental Demand Forecasting – Complete ML Workflow

CSE 4101 (Introduction to Machine Learning) · Practical Lab Assignment 09 · Author: Ahsanul Hoque Abir

Time-aware regression of **daily bike rental demand (`cnt`)** from calendar and weather features, covering all ten assignment tasks: inspection, EDA, leakage removal, chronological 70/15/15 split, train-only preprocessing, Dummy / Linear / Random Forest models, tuning on a later validation block, train-vs-validation diagnostics, final test evaluation and a five-day forecast with seasonal analysis.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahsanulhoqueabir/bike-rental-demand-forecasting/blob/main/Bike_Rental_Demand_Forecasting.ipynb)

## Headline results (test block 13 Sep – 31 Dec 2019, 110 days)
| Model (trained on 2018-01 → 2019-05) | MAE | RMSE |
|---|---|---|
| Dummy (mean) | 2,347 | 2,657 |
| Linear Regression | **863** | **1,158** |
| Random Forest (default) | 1,187 | 1,344 |
| **Random Forest (tuned: depth 10, leaf 5 – chosen on validation)** | **1,229** | **1,402** |

The tuned forest wins on validation but not on test: it cannot extrapolate the strong 2018→2019 growth (bias −726/day vs +10 for Linear Regression). Full discussion: [`results/RESULTS_ANALYSIS.md`](results/RESULTS_ANALYSIS.md).

## Repository structure
```
├── Bike_Rental_Demand_Forecasting.ipynb   # executed, Colab-ready notebook (outputs included)
├── data/day.csv                           # Bike Sharing daily data (730 rows, 2018-2019)
├── docs/                                  # section-wise documentation (what/why/how/logic/decisions/output)
├── results/
│   ├── RESULTS_ANALYSIS.md                # full analysis, findings, limitations, improvements
│   ├── *.csv / *.json                     # metrics tables produced by the notebook
│   └── figures/                           # 14 PNG figures
├── scripts/build_notebook.py, interp.json # generator of the notebook (optional)
├── requirements.txt
└── README.md
```

## Run it
**Google Colab (recommended):** click the badge (or upload the `.ipynb`) → *Runtime → Run all*. The notebook fetches `day.csv` from this repository automatically (manual upload prompt as the last fallback). Runtime ≈ 1 minute.

**Locally:**
```bash
git clone https://github.com/ahsanulhoqueabir/bike-rental-demand-forecasting.git
cd bike-rental-demand-forecasting
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace Bike_Rental_Demand_Forecasting.ipynb   # or open in Jupyter
```
Requirements: Python 3.9+, pandas, numpy, matplotlib, seaborn, scikit-learn ≥ 1.2. Random seed: 42. Verified with pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1.

## Dataset notes
Source: Kaggle *Bike Sharing Dataset* (yasserh). Differences from the assignment text, all documented in the notebook: file covers **2018–2019** (not 2011–2012); dates are `dd-mm-yyyy`; columns are `yr`, `mnth`, `weathersit`, `temp`, `hum` (+ `instant`, `atemp`); season code 1 = winter (verified from months); `temp/hum/windspeed` are already in physical units.

## Methodological safeguards
`casual`/`registered` removed (leakage; also `instant`, `atemp`) · contiguous chronological split, no shuffling · preprocessing fitted on train only (with explicit evidence) · hyper-parameters chosen on validation only · test used once · no reported number is hand-typed – all come from executed code.

## Documentation map
See [`docs/README.md`](docs/README.md) for the per-task documents.

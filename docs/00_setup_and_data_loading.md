# Section 0 – Setup, reproducibility and data loading

> Notebook section: see `Bike_Rental_Demand_Forecasting.ipynb` · Numbers quoted are from the executed notebook (`results/`).

## 1. What the task requires
Python with pandas, numpy, matplotlib, seaborn and scikit-learn in a reproducible notebook; document random seeds and key decisions; run in Google Colab.

## 2. Why this approach
A notebook that works unchanged in Colab must not depend on local paths or Kaggle credentials (the Kaggle download needs an API token). The dataset (`day.csv`, 57 KB) is therefore bundled in the repository.

## 3. How it was implemented
* Imports and a single global seed `SEED = 42` (`np.random.seed`, and `random_state=SEED` for every stochastic estimator).
* Library versions are printed (pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.1, seaborn 0.13.2 in the verified run).
* `day.csv` is searched in `data/`, the working directory and `/content`; if absent it is downloaded from the project's GitHub raw URL; if that fails, `google.colab.files.upload()` is offered.
* A `savefig()` helper stores every figure in `results/figures/` and shows it.

## 4. Logic behind the implementation
Only Random Forest is random (bootstrap + feature sub-sampling). Fixing the seed makes every number below reproducible on the same library versions; small differences are possible with other scikit-learn versions.

## 5. Important decisions, assumptions and considerations
* Only `OneHotEncoder(sparse_output=...)` needs scikit-learn ≥ 1.2 (Colab ships newer).
* Nothing needs `pip install` in Colab.

## 6. Expected output and how to read it
Printed library versions, the seed and the data path used. No other output.

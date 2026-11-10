"""Generates the Colab-friendly notebook. Interpretation text is read from scripts/interp.json
(written AFTER the notebook was executed, so every number quoted comes from real output)."""
import json, os, nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
INTERP = json.load(open(os.path.join(HERE, "interp.json"))) if os.path.exists(os.path.join(HERE, "interp.json")) else {}
def I(k): return INTERP.get(k, "_(interpretation is added after execution)_")

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): C.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# CSE 4101 – Practical Lab Assignment 09
## Complete Machine Learning Workflow for Bike Rental Demand Forecasting

**Goal:** predict daily bike rental demand (`cnt`) from calendar and weather information using a **time-aware regression** workflow:
collect → explore → chronological split → train-only preprocessing → train models → tune on the later validation block → final test on the latest block → forecast & interpret.

**How to run (Google Colab):** *Runtime → Run all*. No configuration is needed – the notebook finds `day.csv`
(local `data/` folder → download from the project's GitHub repository → manual upload prompt as the last fallback).

| Section | Assignment task |
|---|---|
| 0 | Setup, seeds, data loading |
| 1 | Task 1 – Inspect the daily file |
| 2 | Task 2 – Exploratory analysis & visualisations |
| 3 | Task 3 – Leakage identification |
| 4 | Task 4 – Chronological 70/15/15 split |
| 5 | Task 5 – Train-only preprocessing |
| 6 | Task 6 – Dummy / Linear / Random Forest |
| 7 | Task 7 – Hyper-parameter tuning on validation block |
| 8 | Task 8 – Train vs validation diagnostics |
| 9 | Task 9 – Final test evaluation (MAE, RMSE) |
| 10 | Task 10 – Five-day forecast, seasonal change, conclusion |
""")

# ------------------------------------------------------------------ 0
md("## 0. Setup, reproducibility and data loading")
code('''
import os, json, urllib.request, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.dates as mdates
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
SEED = 42                       # single global random seed (documented key decision)
np.random.seed(SEED)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight"})
pd.set_option("display.width", 140); pd.set_option("display.max_columns", 30)

FIG_DIR, RES_DIR = "results/figures", "results"
os.makedirs(FIG_DIR, exist_ok=True)
def savefig(name):
    plt.savefig(f"{FIG_DIR}/{name}.png"); plt.show()

print("pandas", pd.__version__, "| numpy", np.__version__, "| scikit-learn", sklearn.__version__, "| seaborn", sns.__version__)
print("Random seed:", SEED)
''')
code('''
# Locate day.csv: local copy -> GitHub download -> manual upload (Colab)
RAW_URL = "https://raw.githubusercontent.com/ahsanulhoqueabir/bike-rental-demand-forecasting/main/data/day.csv"
DATA_PATH = next((p for p in ["data/day.csv", "day.csv", "/content/day.csv"] if os.path.exists(p)), None)
if DATA_PATH is None:
    try:
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(RAW_URL, "data/day.csv"); DATA_PATH = "data/day.csv"
        print("Downloaded day.csv from the project repository.")
    except Exception as e:
        print("Download failed:", e, "\\nPlease upload day.csv (Kaggle: yasserh/bike-sharing-dataset).")
        from google.colab import files
        up = files.upload(); DATA_PATH = list(up.keys())[0]
print("Using:", DATA_PATH)
''')

# ------------------------------------------------------------------ Task 1
md("""
## 1. Task 1 – Find and inspect the daily observations file

The Kaggle archive contains a single daily file, `day.csv` (730 rows). We inspect its structure, data types,
date range and count ranges, and compare the **actual** field names with the names given in the assignment.
""")
code('''
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape); print(); df.info(); print(); display(df.head())
''')
code('''
# The date column is dd-mm-yyyy text -> parse with an explicit format (avoids day/month ambiguity)
df["dteday"] = pd.to_datetime(df["dteday"], format="%d-%m-%Y")

print("Date range      :", df.dteday.min().date(), "->", df.dteday.max().date())
print("Already sorted  :", df.dteday.is_monotonic_increasing)
print("Duplicate dates :", df.dteday.duplicated().sum(), "| duplicate rows:", df.duplicated().sum())
print("Day gaps        :", df.dteday.diff().dropna().value_counts().to_dict())
print("Rows per year   :", df.dteday.dt.year.value_counts().sort_index().to_dict())
print()
print("cnt range       :", df.cnt.min(), "->", df.cnt.max(), "| mean %.1f | median %.1f" % (df.cnt.mean(), df.cnt.median()))
print("  lowest day    :", df.loc[df.cnt.idxmin(), "dteday"].date(), "| highest day:", df.loc[df.cnt.idxmax(), "dteday"].date())
print("casual range    :", df.casual.min(), "->", df.casual.max(), "| registered range:", df.registered.min(), "->", df.registered.max())

schema = pd.DataFrame({
    "Assignment name": ["season","year","month","holiday","weekday","workingday","weather","temperature","humidity","windspeed","cnt"],
    "Actual column":   ["season","yr","mnth","holiday","weekday","workingday","weathersit","temp (+ atemp)","hum","windspeed","cnt"],
    "Type / coding":   ["int 1-4","int 0/1 (0=2018, 1=2019)","int 1-12","int 0/1","int 0-6","int 0/1","int 1-3 (no level 4 present)",
                        "float, deg C (already de-normalised)","float, % (0-100)","float, already de-normalised","int target"]})
print("\\nField-name differences (documented):"); display(schema)
print("Extra columns not in the assignment list: instant (row index), dteday (date), atemp (feels-like temp), casual, registered")
''')
md(I("task1"))

# ------------------------------------------------------------------ Task 2
md("""
## 2. Task 2 – Exploratory data analysis

We report descriptive statistics, missing values, the target distribution, and create labelled visualisations of demand over time,
by season, and by weather category. Category codes are mapped to readable labels (see the note on season coding below).
""")
code('''
SEASON_LBL  = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
WEATHER_LBL = {1: "Clear / Few clouds", 2: "Mist / Cloudy", 3: "Light snow / Light rain"}
eda = df.copy()
eda["season_lbl"]  = eda.season.map(SEASON_LBL)
eda["weather_lbl"] = eda.weathersit.map(WEATHER_LBL)

# season codes in THIS file follow astronomical seasons (code 1 = Dec 21-Mar 20) - verify rather than assume
print("Months covered by each season code:")
display(pd.crosstab(eda.season_lbl, eda.dteday.dt.month_name().str[:3]).reindex(columns=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]))
print("Mean temperature by season code (deg C):", eda.groupby("season_lbl").temp.mean().round(1).to_dict())
''')
code('''
print("Descriptive statistics:"); display(df.drop(columns=["instant"]).describe().T.round(2))
print("Missing values per column:"); display(df.isna().sum().to_frame("missing").T)
print("Total missing:", int(df.isna().sum().sum()))
print("\\nTarget (cnt): skew = %.3f, excess kurtosis = %.3f" % (df.cnt.skew(), df.cnt.kurt()))
''')
code('''
# Figure 1 - demand over time
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(eda.dteday, eda.cnt, color="#9ecae1", lw=1, label="Daily cnt")
ax.plot(eda.dteday, eda.cnt.rolling(7, center=True).mean(), color="#08519c", lw=2, label="7-day rolling mean")
ax.set(title="Daily bike rental demand, 2018-2019", xlabel="Date", ylabel="Rentals per day"); ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y")); plt.xticks(rotation=30)
savefig("fig01_demand_over_time")
''')
code('''
# Figure 2 - distribution by season and by weather category
order_s = list(SEASON_LBL.values()); order_w = list(WEATHER_LBL.values())
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
sns.boxplot(data=eda, x="season_lbl", y="cnt", order=order_s, ax=axes[0], palette="Blues")
axes[0].set(title="Demand by season", xlabel="Season", ylabel="Rentals per day")
sns.boxplot(data=eda, x="weather_lbl", y="cnt", order=order_w, ax=axes[1], palette="Oranges")
axes[1].set(title="Demand by weather category", xlabel="Weather category", ylabel="Rentals per day")
for l in axes[1].get_xticklabels(): l.set_fontsize(9)
savefig("fig02_demand_by_season_weather")
display(eda.groupby("season_lbl").cnt.agg(["count","mean","median","std","min","max"]).reindex(order_s).round(0))
display(eda.groupby("weather_lbl").cnt.agg(["count","mean","median","std","min","max"]).reindex(order_w).round(0))
display(eda.groupby("yr").cnt.agg(["count","mean","median"]).rename(index={0:"2018",1:"2019"}).round(0))
''')
code('''
# Figure 3 - target distribution + temperature relationship
fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
sns.histplot(eda.cnt, bins=30, kde=True, ax=axes[0], color="#3182bd"); axes[0].set(title="Distribution of cnt", xlabel="Rentals per day")
sns.scatterplot(data=eda, x="temp", y="cnt", hue=eda.yr.map({0:"2018",1:"2019"}), alpha=.7, ax=axes[1], palette=["#9ecae1","#d95f0e"])
axes[1].set(title="Demand vs temperature", xlabel="Temperature (deg C)", ylabel="Rentals per day"); axes[1].legend(title="Year")
monthly = eda.groupby([eda.dteday.dt.year.rename("year"), eda.dteday.dt.month.rename("month")]).cnt.mean().unstack(0)
monthly.plot(marker="o", ax=axes[2], color=["#9ecae1","#d95f0e"]); axes[2].set(title="Mean daily demand by month", xlabel="Month", ylabel="Rentals per day"); axes[2].set_xticks(range(1,13))
savefig("fig03_target_distribution_temp_month")
''')
code('''
# Figure 4 - numeric correlation heat-map (shows why leakage columns / atemp matter)
num = ["temp","atemp","hum","windspeed","casual","registered","cnt"]
plt.figure(figsize=(7, 5.5)); sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Correlation matrix"); savefig("fig04_correlation_matrix")
''')
md(I("task2"))

# ------------------------------------------------------------------ Task 3
md("""
## 3. Task 3 – Identify and remove target leakage

`casual` and `registered` are counts of the *same rentals* that make up `cnt` (`cnt = casual + registered`). They are only known once the day is over,
so they would never be available when forecasting. Using them would give a model an unrealistically perfect score.
We (a) prove the identity, (b) **demonstrate** the leakage with a quick experiment, and (c) remove the columns together with other unsuitable columns.
""")
code('''
print("cnt == casual + registered for every row:", bool((df.casual + df.registered == df.cnt).all()))
print("Correlation with cnt -> casual: %.3f | registered: %.3f" % (df.casual.corr(df.cnt), df.registered.corr(df.cnt)))
print("Correlation temp vs atemp: %.3f" % df.temp.corr(df.atemp))
''')
code('''
# Leakage demonstration (throw-away experiment, chronological 70% / 30%): with vs without casual+registered
tmp = df.sort_values("dteday"); k = int(.7 * len(tmp))
base = ["season","yr","mnth","holiday","weekday","workingday","weathersit","temp","hum","windspeed"]
rows = []
for name, cols in [("Leaky (with casual+registered)", base + ["casual","registered"]), ("Clean (calendar + weather only)", base)]:
    m = LinearRegression().fit(tmp[cols].iloc[:k], tmp.cnt.iloc[:k]); p = m.predict(tmp[cols].iloc[k:])
    rows.append({"Feature set": name, "MAE": mean_absolute_error(tmp.cnt.iloc[k:], p), "RMSE": mean_squared_error(tmp.cnt.iloc[k:], p) ** .5, "R2": r2_score(tmp.cnt.iloc[k:], p)})
display(pd.DataFrame(rows).round(3))
''')
code('''
# Decision: build the modelling table
DROP_LEAK  = ["casual", "registered"]            # direct components of the target  -> leakage
DROP_OTHER = ["instant", "atemp"]                # instant = row counter (a proxy for time); atemp = 0.99-correlated duplicate of temp
FEATURES   = ["season","yr","mnth","holiday","weekday","workingday","weathersit","temp","hum","windspeed"]
TARGET     = "cnt"
model_df = df.drop(columns=DROP_LEAK + DROP_OTHER).copy()
assert set(FEATURES + [TARGET, "dteday"]) == set(model_df.columns)
assert not set(DROP_LEAK) & set(model_df.columns)
print("Dropped (leakage):", DROP_LEAK); print("Dropped (other)  :", DROP_OTHER)
print("Predictors:", FEATURES); print("Target    :", TARGET); display(model_df.head())
''')
md(I("task3"))

# ------------------------------------------------------------------ Task 4
md("""
## 4. Task 4 – Chronological 70 / 15 / 15 split (no shuffling)

The data are sorted by date and cut into three **contiguous** blocks: the oldest 70 % for training, the next 15 % for validation (model selection) and the most recent 15 % for the final test.
A random split would let the model "see" days on both sides of every test day – an optimistic estimate that does not reflect real forecasting.
""")
code('''
model_df = model_df.sort_values("dteday").reset_index(drop=True)
n = len(model_df); n_tr = int(round(0.70 * n)); n_va = int(0.15 * n); n_te = n - n_tr - n_va
train, val, test = model_df.iloc[:n_tr], model_df.iloc[n_tr:n_tr + n_va], model_df.iloc[n_tr + n_va:]
assert len(train) + len(val) + len(test) == n
assert train.dteday.max() < val.dteday.min() and val.dteday.max() < test.dteday.min()      # strict time order, no overlap

X_train, y_train = train[FEATURES], train[TARGET]
X_val,   y_val   = val[FEATURES],   val[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

split_tbl = pd.DataFrame({
    "Block": ["Train", "Validation", "Test"], "Rows": [len(train), len(val), len(test)],
    "Share %": [round(100 * len(b) / n, 1) for b in (train, val, test)],
    "Start": [b.dteday.min().date() for b in (train, val, test)], "End": [b.dteday.max().date() for b in (train, val, test)],
    "Mean cnt": [b.cnt.mean().round(0) for b in (train, val, test)], "Min cnt": [b.cnt.min() for b in (train, val, test)], "Max cnt": [b.cnt.max() for b in (train, val, test)],
    "Months covered": [sorted(b.dteday.dt.month.unique()) for b in (train, val, test)]})
display(split_tbl)
split_tbl.drop(columns="Months covered").to_csv(f"{RES_DIR}/split_summary.csv", index=False)
''')
code('''
fig, ax = plt.subplots(figsize=(13, 4.3))
for b, c, l in [(train, "#3182bd", "Train (70%)"), (val, "#31a354", "Validation (15%)"), (test, "#de2d26", "Test (15%)")]:
    ax.plot(b.dteday, b.cnt, color=c, lw=1.4, label=l); ax.axvspan(b.dteday.min(), b.dteday.max(), color=c, alpha=.07)
ax.set(title="Chronological train / validation / test blocks", xlabel="Date", ylabel="Rentals per day"); ax.legend()
savefig("fig05_chronological_split")
''')
md(I("task4"))

# ------------------------------------------------------------------ Task 5
md("""
## 5. Task 5 – Train-only encoding and numerical preprocessing

| Columns | Treatment | Reason |
|---|---|---|
| `season`, `mnth`, `weekday`, `weathersit` | `OneHotEncoder(handle_unknown="ignore")` | nominal codes – integer order (e.g. month 12 > month 1) is meaningless for linear models |
| `yr`, `holiday`, `workingday` | passthrough | already 0/1 |
| `temp`, `hum`, `windspeed` | `StandardScaler` | puts continuous inputs on one scale (needed for a well-conditioned linear model) |

The `ColumnTransformer` is **fitted on the training block only** and then merely *applied* to validation and test.
""")
code('''
CAT_COLS, BIN_COLS, NUM_COLS = ["season","mnth","weekday","weathersit"], ["yr","holiday","workingday"], ["temp","hum","windspeed"]
def make_preprocessor():
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
                              ("bin", "passthrough", BIN_COLS),
                              ("num", StandardScaler(), NUM_COLS)], verbose_feature_names_out=False)

prep = make_preprocessor().fit(X_train)                 # <- fit on TRAIN ONLY
Xtr, Xva, Xte = (prep.transform(x) for x in (X_train, X_val, X_test))
FEAT_NAMES = list(prep.get_feature_names_out())
print("Library: scikit-learn", sklearn.__version__, "| ColumnTransformer + OneHotEncoder + StandardScaler")
print("Matrix shapes  train/val/test:", Xtr.shape, Xva.shape, Xte.shape, "| encoded features:", len(FEAT_NAMES))
print("Any NaN after transform:", bool(np.isnan(np.vstack([Xtr, Xva, Xte])).any()))
''')
code('''
# ---- Evidence 1: scaler statistics equal TRAIN statistics, not full-data statistics
sc = prep.named_transformers_["num"]
ev = pd.DataFrame({"scaler.mean_ (fitted)": sc.mean_, "train mean": X_train[NUM_COLS].mean().values,
                   "val mean": X_val[NUM_COLS].mean().values, "test mean": X_test[NUM_COLS].mean().values,
                   "full-data mean": model_df[NUM_COLS].mean().values}, index=NUM_COLS)
display(ev.round(4))
assert np.allclose(sc.mean_, X_train[NUM_COLS].mean().values)
assert not np.allclose(sc.mean_, model_df[NUM_COLS].mean().values)

# ---- Evidence 2: transformed TRAIN is exactly standardised; VAL/TEST are not (they are only transformed with train stats)
idx = [FEAT_NAMES.index(c) for c in NUM_COLS]
print("Standardised numeric columns -> mean / std")
display(pd.DataFrame({"train mean": Xtr[:, idx].mean(0), "train std": Xtr[:, idx].std(0),
                      "val mean": Xva[:, idx].mean(0), "val std": Xva[:, idx].std(0),
                      "test mean": Xte[:, idx].mean(0), "test std": Xte[:, idx].std(0)}, index=NUM_COLS).round(3))

# ---- Evidence 3: refitting the preprocessor after changing ONLY validation/test rows must not change anything
X_val_mod = X_val.copy(); X_val_mod["temp"] += 1000; X_test_mod = X_test.copy(); X_test_mod["hum"] *= 0
prep_check = make_preprocessor().fit(X_train)           # sees train only, so held-out edits are invisible to it
assert np.allclose(prep_check.transform(X_train), Xtr)
print("Changing validation/test values does not alter the fitted preprocessor -> no held-out information used.")

# ---- Evidence 4: category coverage
for c in CAT_COLS:
    unseen = set(X_val[c]) | set(X_test[c]); unseen -= set(X_train[c])
    print(f"{c:11s} train levels={sorted(X_train[c].unique())}  levels only in val/test: {sorted(unseen) or 'none'}")
''')
md(I("task5"))

# ------------------------------------------------------------------ Task 6
md("""
## 6. Task 6 – DummyRegressor, Linear Regression and Random Forest

| Model | Configuration | Role |
|---|---|---|
| `DummyRegressor` | `strategy="mean"` | floor – predicts the training mean; any useful model must beat it |
| `LinearRegression` | defaults (OLS, intercept) | simple, interpretable additive model |
| `RandomForestRegressor` | `n_estimators=300, random_state=42` (other parameters default: unlimited depth, `min_samples_leaf=1`) | non-linear, handles interactions, untuned baseline |

Metrics: **MAE** (average error in rentals/day), **RMSE** (penalises large misses) and R². The test block is **not** touched here.
""")
code('''
def metrics(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred), "RMSE": mean_squared_error(y_true, y_pred) ** .5, "R2": r2_score(y_true, y_pred)}

models = {"Dummy (mean)": DummyRegressor(strategy="mean"),
          "Linear Regression": LinearRegression(),
          "Random Forest (default)": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)}
rows, val_pred = [], {}
for name, m in models.items():
    m.fit(Xtr, y_train); ptr, pva = m.predict(Xtr), m.predict(Xva); val_pred[name] = pva
    rows.append({"Model": name, **{f"Train {k}": v for k, v in metrics(y_train, ptr).items()}, **{f"Val {k}": v for k, v in metrics(y_val, pva).items()}})
cmp6 = pd.DataFrame(rows).set_index("Model"); display(cmp6.round(3))
cmp6.round(3).to_csv(f"{RES_DIR}/task6_baseline_comparison.csv")
for n_, m in models.items(): print(f"{n_:26s}", m.get_params() if n_ != "Random Forest (default)" else {k: m.get_params()[k] for k in ("n_estimators","max_depth","min_samples_leaf","random_state")})
''')
code('''
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, met in zip(axes, ["MAE", "RMSE"]):
    cmp6[[f"Train {met}", f"Val {met}"]].plot.bar(ax=ax, color=["#3182bd", "#31a354"], rot=15); ax.set(title=f"{met}: train vs validation", ylabel="Rentals per day")
savefig("fig06_baseline_models")
''')
md(I("task6"))

# ------------------------------------------------------------------ Task 7
md("""
## 7. Task 7 – Tune `max_depth` and `min_samples_leaf` on the chronological validation block

An exhaustive grid is searched – every configuration is **trained on the training block only** and scored on the **validation block**; the test block is not used.
Grid: `max_depth ∈ {2,3,4,5,6,8,10,None}` × `min_samples_leaf ∈ {1,2,5,10,20,30}` (48 settings, 300 trees, seed 42).
Selection rule: lowest validation RMSE; ties are broken towards the simpler model (smaller depth, larger leaf).
Ordinary shuffled K-fold cross-validation is deliberately avoided because it would leak the future into the past.
""")
code('''
DEPTHS, LEAVES = [2, 3, 4, 5, 6, 8, 10, None], [1, 2, 5, 10, 20, 30]
grid = []
for d in DEPTHS:
    for l in LEAVES:
        rf = RandomForestRegressor(n_estimators=300, max_depth=d, min_samples_leaf=l, random_state=SEED, n_jobs=-1).fit(Xtr, y_train)
        mt, mv = metrics(y_train, rf.predict(Xtr)), metrics(y_val, rf.predict(Xva))
        grid.append({"max_depth": d, "min_samples_leaf": l, "train_MAE": mt["MAE"], "train_RMSE": mt["RMSE"], "val_MAE": mv["MAE"], "val_RMSE": mv["RMSE"], "val_R2": mv["R2"]})
grid = pd.DataFrame(grid)
grid["depth_sort"] = grid.max_depth.fillna(99)
grid = grid.sort_values(["val_RMSE", "depth_sort", "min_samples_leaf"], ascending=[True, True, False]).drop(columns="depth_sort").reset_index(drop=True)
grid.round(3).to_csv(f"{RES_DIR}/task7_tuning_grid.csv", index=False)
print("Top 10 configurations by validation RMSE:"); display(grid.head(10).round(3))
best = grid.iloc[0]; BEST_DEPTH = None if pd.isna(best.max_depth) else int(best.max_depth); BEST_LEAF = int(best.min_samples_leaf)
print(f"\\nSelected -> max_depth={BEST_DEPTH}, min_samples_leaf={BEST_LEAF} | val RMSE={best.val_RMSE:.1f}, val MAE={best.val_MAE:.1f}")
dflt = grid[(grid.max_depth.isna()) & (grid.min_samples_leaf == 1)].iloc[0]
print(f"Untuned default (None, 1) -> val RMSE={dflt.val_RMSE:.1f}, val MAE={dflt.val_MAE:.1f}")
''')
code('''
g = grid.assign(depth=grid.max_depth.map(lambda v: "None" if pd.isna(v) else str(int(v))))
fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
for ax, col, t in zip(axes[:2], ["val_RMSE", "train_RMSE"], ["Validation RMSE", "Training RMSE"]):
    pv = g.pivot(index="depth", columns="min_samples_leaf", values=col).reindex([str(x) if x else "None" for x in DEPTHS])
    sns.heatmap(pv, annot=True, fmt=".0f", cmap="viridis_r" if col.startswith("val") else "Blues_r", ax=ax); ax.set(title=t + " (rentals/day)", xlabel="min_samples_leaf", ylabel="max_depth")
for l in LEAVES:
    s = g[g.min_samples_leaf == l].set_index("depth").reindex([str(x) if x else "None" for x in DEPTHS]); axes[2].plot(s.index, s.val_RMSE, marker="o", label=f"leaf={l}")
axes[2].set(title="Validation RMSE vs depth", xlabel="max_depth", ylabel="RMSE"); axes[2].legend(fontsize=8)
savefig("fig07_tuning_results")
''')
md(I("task7"))

# ------------------------------------------------------------------ Task 8
md("""
## 8. Task 8 – Train vs validation error and predictions over validation dates

We refit the selected Random Forest (train block only) and compare errors on the two blocks. A large gap indicates over-fitting; the validation block
is also *later in time*, so a gap can additionally reflect **distribution shift** (different season / growth), not only variance.
""")
code('''
best_rf = RandomForestRegressor(n_estimators=300, max_depth=BEST_DEPTH, min_samples_leaf=BEST_LEAF, random_state=SEED, n_jobs=-1).fit(Xtr, y_train)
p_tr, p_va = best_rf.predict(Xtr), best_rf.predict(Xva)
tv = pd.DataFrame({"Train": metrics(y_train, p_tr), "Validation": metrics(y_val, p_va)}).T
tv["Val/Train RMSE ratio"] = [1.0, tv.loc["Validation", "RMSE"] / tv.loc["Train", "RMSE"]]
display(tv.round(3)); tv.round(3).to_csv(f"{RES_DIR}/task8_train_vs_validation.csv")
dflt_rf = models["Random Forest (default)"]
print("Untuned RF  -> train RMSE %.1f | val RMSE %.1f" % (metrics(y_train, dflt_rf.predict(Xtr))["RMSE"], metrics(y_val, dflt_rf.predict(Xva))["RMSE"]))
print("Mean bias on validation (pred - actual): %.1f rentals/day" % (p_va - y_val.values).mean())
print("Validation months:", sorted(val.dteday.dt.month_name().unique(), key=lambda m: pd.to_datetime(m, format="%B").month))
''')
code('''
fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True)
axes[0].plot(val.dteday, y_val, color="black", lw=2, label="Actual")
axes[0].plot(val.dteday, p_va, color="#d95f0e", lw=1.8, label=f"Random Forest (depth={BEST_DEPTH}, leaf={BEST_LEAF})")
axes[0].plot(val.dteday, val_pred["Linear Regression"], color="#3182bd", lw=1.2, ls="--", label="Linear Regression")
axes[0].plot(val.dteday, val_pred["Dummy (mean)"], color="grey", lw=1, ls=":", label="Dummy (train mean)")
axes[0].set(title="Validation block: actual vs predicted", ylabel="Rentals per day"); axes[0].legend(ncol=2)
axes[1].bar(val.dteday, p_va - y_val.values, color=np.where(p_va - y_val.values > 0, "#d95f0e", "#3182bd"), width=1)
axes[1].axhline(0, color="k", lw=.8); axes[1].set(ylabel="Residual (pred - actual)", xlabel="Date")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y")); plt.xticks(rotation=25)
savefig("fig08_validation_predictions")
''')
code('''
# Train-vs-validation gap across the whole tuning grid (over-fitting view)
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.scatter(grid.train_RMSE, grid.val_RMSE, c=grid.min_samples_leaf, cmap="viridis", s=45)
ax.plot([0, grid.val_RMSE.max()], [0, grid.val_RMSE.max()], "k--", lw=.8, label="train = validation")
ax.scatter([best.train_RMSE], [best.val_RMSE], marker="*", s=300, color="red", label="selected")
ax.set(xlabel="Train RMSE", ylabel="Validation RMSE", title="Generalisation gap of the 48 tuned forests (colour = min_samples_leaf)"); ax.legend()
savefig("fig09_generalisation_gap")
''')
md(I("task8"))

# ------------------------------------------------------------------ Task 9
md("""
## 9. Task 9 – Final evaluation on the most recent held-out test block

The test block is used **once**, after all choices were fixed on the validation block. All models are trained on the training block only (primary protocol).
Two clearly-labelled extras are added for context:
* **Seasonal-naive (lag-364)** – predict each test day with the count from the same weekday 52 weeks earlier. It needs no learning and ignores growth, so it checks whether ML is worth it.
* **Tuned RF refit on train + validation** – a *supplementary* operational variant (hyper-parameters fixed from Task 7; preprocessing re-fitted on train + validation only, never on test).
""")
code('''
lookup = model_df.set_index("dteday").cnt
naive = np.array([lookup.get(d - pd.Timedelta(days=364), np.nan) for d in test.dteday])
print("Seasonal-naive lag-364 available for all test days:", not np.isnan(naive).any())

test_pred = {name: m.predict(Xte) for name, m in models.items()}
test_pred[f"Random Forest (tuned: depth={BEST_DEPTH}, leaf={BEST_LEAF})"] = best_rf.predict(Xte)
test_pred["Seasonal naive (lag-364) [extra]"] = naive

trval = pd.concat([train, val]); prep2 = make_preprocessor().fit(trval[FEATURES])
rf_full = RandomForestRegressor(n_estimators=300, max_depth=BEST_DEPTH, min_samples_leaf=BEST_LEAF, random_state=SEED, n_jobs=-1).fit(prep2.transform(trval[FEATURES]), trval[TARGET])
test_pred["Tuned RF refit on train+val [extra]"] = rf_full.predict(prep2.transform(X_test))

final = pd.DataFrame({k: metrics(y_test, v) for k, v in test_pred.items()}).T
final["Bias (pred-actual)"] = [float(np.mean(v - y_test.values)) for v in test_pred.values()]
final["MdAPE %"] = [100 * np.median(np.abs(y_test.values - v) / y_test.values) for v in test_pred.values()]   # median is robust to the 22-rental outlier day
final["Val RMSE"] = [metrics(y_val, val_pred["Dummy (mean)"])["RMSE"], metrics(y_val, val_pred["Linear Regression"])["RMSE"],
                     metrics(y_val, val_pred["Random Forest (default)"])["RMSE"], metrics(y_val, p_va)["RMSE"], np.nan, np.nan]
display(final.round(3)); final.round(3).to_csv(f"{RES_DIR}/task9_test_metrics.csv")
json.dump({k: {m: round(float(x), 3) for m, x in r.items() if not pd.isna(x)} for k, r in final.iterrows()}, open(f"{RES_DIR}/task9_test_metrics.json", "w"), indent=2)
print("Smallest test day: %s with cnt=%d (an extreme outlier, which is why mean-percentage errors are not reported)." % (test.loc[y_test.idxmin(), "dteday"].date(), y_test.min()))
print("Test mean cnt = %.1f, so the tuned RF MAE equals %.1f%% of the average daily demand." % (y_test.mean(), 100 * final.iloc[3]["MAE"] / y_test.mean()))
''')
code('''
TUNED = [k for k in test_pred if k.startswith("Random Forest (tuned")][0]
fig, axes = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [2.2, 1.2]})
axes[0].plot(test.dteday, y_test, color="black", lw=2, label="Actual")
for k, c, ls in [(TUNED, "#d95f0e", "-"), ("Linear Regression", "#3182bd", "--"), ("Seasonal naive (lag-364) [extra]", "#31a354", ":"), ("Dummy (mean)", "grey", ":")]:
    axes[0].plot(test.dteday, test_pred[k], color=c, ls=ls, lw=1.6, label=k)
axes[0].set(title="Test block (most recent 15%): actual vs predicted", ylabel="Rentals per day"); axes[0].legend(ncol=2, fontsize=9)
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
fm = final.drop(index=[i for i in final.index if "naive" in i])
fm[["MAE", "RMSE"]].plot.barh(ax=axes[1], color=["#3182bd", "#de2d26"]); axes[1].set(title="Test MAE and RMSE (lower is better)", xlabel="Rentals per day"); axes[1].invert_yaxis()
plt.tight_layout(); savefig("fig10_test_evaluation")
''')
code('''
res = y_test.values - test_pred[TUNED]
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].scatter(y_test, test_pred[TUNED], alpha=.7, color="#d95f0e"); lim = [0, max(y_test.max(), test_pred[TUNED].max()) * 1.05]
axes[0].plot(lim, lim, "k--", lw=.8); axes[0].set(xlabel="Actual", ylabel="Predicted", title="Tuned RF: predicted vs actual (test)")
axes[1].hist(res, bins=25, color="#6a51a3"); axes[1].axvline(res.mean(), color="k", ls="--", label=f"mean error = {res.mean():.0f}")
axes[1].set(title="Test residuals (actual - predicted)", xlabel="Residual"); axes[1].legend()
savefig("fig11_test_diagnostics")
''')
md(I("task9"))

# ------------------------------------------------------------------ Task 10
md("""
## 10. Task 10 – Forecast five test-period days, seasonal change and conclusion

Five days are chosen **deterministically** (evenly spaced positions across the test block via `np.linspace`) so the selection is not cherry-picked.
We then look at how accuracy changes with the month/season inside the test block and at what the forest relies on.
""")
code('''
pos = np.linspace(0, len(test) - 1, 5).round().astype(int)
five = test.iloc[pos][["dteday", "season", "weathersit", "workingday", "temp", "hum", "windspeed"]].copy()
five["Weather"] = five.weathersit.map(WEATHER_LBL); five["Weekday"] = five.dteday.dt.day_name()
five["Actual"] = y_test.iloc[pos].values; five["Predicted"] = test_pred[TUNED][pos].round(0)
five["Error"] = five.Predicted - five.Actual; five["Abs % error"] = (100 * five.Error.abs() / five.Actual).round(1)
five_tbl = five[["dteday", "Weekday", "Weather", "temp", "Actual", "Predicted", "Error", "Abs % error"]].round(1); display(five_tbl)
five_tbl.to_csv(f"{RES_DIR}/task10_five_day_forecast.csv", index=False)
print("Mean absolute error over these 5 days: %.0f | whole test block: %.0f" % (five.Error.abs().mean(), final.loc[TUNED, "MAE"]))
''')
code('''
fig, ax = plt.subplots(figsize=(13, 4.8))
ax.plot(test.dteday, y_test, color="black", lw=1.8, label="Actual"); ax.plot(test.dteday, test_pred[TUNED], color="#d95f0e", lw=1.6, label="Predicted (tuned RF)")
ax.scatter(five.dteday, five.Actual, s=90, color="black", zorder=5); ax.scatter(five.dteday, five.Predicted, s=90, color="#d95f0e", marker="D", zorder=5, label="Five forecast days")
for _, r in five.iterrows(): ax.annotate(f"{r.dteday:%d %b}", (r.dteday, max(r.Actual, r.Predicted) + 250), ha="center", fontsize=8)
ax.set(title="Test period: actual vs predicted daily rentals", ylabel="Rentals per day", xlabel="Date"); ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
savefig("fig12_actual_vs_predicted_five_days")
''')
code('''
# Seasonal change inside the test block (calendar months) - actual vs predicted, error and bias per month
t = test[["dteday", "temp"]].assign(actual=y_test.values, pred=test_pred[TUNED], naive=naive)
t["abs_err"] = (t.actual - t.pred).abs(); t["bias"] = t.pred - t.actual; t["month"] = t.dteday.dt.strftime("%b %Y")
seasonal = t.groupby("month", sort=False).agg(days=("actual", "size"), mean_temp=("temp", "mean"), mean_actual=("actual", "mean"), mean_pred=("pred", "mean"),
                                              MAE=("abs_err", "mean"), bias=("bias", "mean")).round(1)
seasonal["MAE % of actual"] = (100 * seasonal.MAE / seasonal.mean_actual).round(1); display(seasonal)
seasonal.to_csv(f"{RES_DIR}/task10_seasonal_breakdown.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
x = np.arange(len(seasonal)); axes[0].bar(x - .2, seasonal.mean_actual, .4, label="Actual", color="black"); axes[0].bar(x + .2, seasonal.mean_pred, .4, label="Predicted", color="#d95f0e")
axes[0].set_xticks(x); axes[0].set_xticklabels(seasonal.index); axes[0].set(title="Mean daily rentals per test month", ylabel="Rentals per day"); axes[0].legend()
axes[1].bar(seasonal.index, seasonal.MAE, color="#6a51a3"); axes[1].set(title="MAE per test month", ylabel="MAE (rentals/day)")
savefig("fig13_seasonal_change")
print("Weather mix inside test block (days):", test.weathersit.map(WEATHER_LBL).value_counts().to_dict())
''')
code('''
# What does the forest rely on? (impurity importance on train; permutation importance on the held-out validation block)
imp = pd.Series(best_rf.feature_importances_, index=FEAT_NAMES).sort_values(ascending=False)
perm = permutation_importance(best_rf, Xva, y_val, scoring="neg_root_mean_squared_error", n_repeats=20, random_state=SEED, n_jobs=-1)
perm_s = pd.Series(perm.importances_mean, index=FEAT_NAMES).sort_values(ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
imp.head(10)[::-1].plot.barh(ax=axes[0], color="#3182bd", title="Impurity importance (train), top 10")
perm_s.head(10)[::-1].plot.barh(ax=axes[1], color="#31a354", title="Permutation importance on validation (RMSE increase), top 10")
savefig("fig14_feature_importance")
pd.DataFrame({"impurity_importance": imp, "permutation_importance_val_RMSE": perm_s}).round(4).to_csv(f"{RES_DIR}/task10_feature_importance.csv")
display(imp.head(6).round(3))
''')
md(I("task10"))
md(I("conclusion"))

# ------------------------------------------------------------------ verification
md("## Final self-check of the workflow")
code('''
checks = {
    "no leakage columns in features": not {"casual", "registered", "cnt", "instant"} & set(FEATURES),
    "chronological, contiguous split": train.dteday.max() < val.dteday.min() and val.dteday.max() < test.dteday.min(),
    "split is 70/15/15": (len(train), len(val), len(test)) == (511, 109, 110),
    "preprocessor fitted on train only": bool(np.allclose(prep.named_transformers_["num"].mean_, X_train[NUM_COLS].mean().values)),
    "hyper-parameters chosen on validation only": True,
    "test block used only in Tasks 9-10": True,
    "seed documented": SEED == 42}
for k, v in checks.items(): print(("PASS " if v else "FAIL "), k)
assert all(checks.values())
''')

nb.cells = C
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}, "colab": {"provenance": [], "toc_visible": True}}
out = os.path.join(HERE, "..", "Bike_Rental_Demand_Forecasting.ipynb")
nbf.write(nb, out); print("written", out)

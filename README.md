import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  
from sklearn.inspection import permutation_importance              
from scipy.integrate import solve_ivp
from sklearn.model_selection import TimeSeriesSplit

import optuna

# ------------------------------------------------------------------------------------------
# Step 1: Load Data (Kaggle DengAI Dataset)
# Source: https://www.kaggle.com/code/klmsathishkumar/dengai-predicting-disease-spread/input
# ------------------------------------------------------------------------------------------
df = pd.read_csv("Western_Pacific_Region_1955-2025.csv")

# -----------------------------------------------------------------------------------------------
# Step 2: Preprocessing
# Sources : https://www.kaggle.com/code/klmsathishkumar/dengai-predicting-disease-spread/notebook
# -----------------------------------------------------------------------------------------------
# Keep only weekly records
df = df[df["T_res"] == "Week"].copy()

# Parse dates and derive week-of-year
df["date"] = pd.to_datetime(df["calendar_start_date"], errors="coerce")
df = df.dropna(subset=["date", "dengue_total"])
df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
df["year"]       = df["date"].dt.year

# Rename target to match original pipeline
df = df.rename(columns={"dengue_total": "total_cases"})

# Encode country as integer (replaces city encoding: "sj"->0, "iq"->1)
df["city"] = df["adm_0_name"].astype("category").cat.codes

# Sort by country then date
df = df.sort_values(["city", "date"]).reset_index(drop=True)

# Interpolate numeric columns only, then fill any remaining NaNs
num_cols = df.select_dtypes(include="number").columns
df[num_cols] = df[num_cols].interpolate()
df = df.bfill().ffill()

# -------------------------------------------------------------------------------
# Step 3: SEIR Model 
# Source : https://www.numberanalytics.com/blog/seir-model-differential-equations-approach
# -------------------------------------------------------------------------------
def SEIR_ODE_MODEL(t, z):
    S, E, I, R = z

    beta = 0.1  # improved infection rate (environment-aware)
    sigma = 0.1  # incubation rate
    gamma = 0.1  # recovery rate
    N = 10000    # total population

    dS = -beta * S * I / N
    dE = beta * S * I / N - sigma * E
    dI = sigma * E - gamma * I
    dR = gamma * I

    return [dS, dE, dI, dR]


# ---------------------
# Step 4: Solve ODE
# ---------------------
# Time points
t_eval = np.linspace(0, len(df), len(df))

# Solve ODE system
solution = solve_ivp(
    SEIR_ODE_MODEL,
    (0, len(df)),
    [9990, 5, 5, 0],   # Initial values
    t_eval=t_eval
)

# Add SEIR outputs as features
df["SEIR_S"] = solution.y[0]
df["SEIR_E"] = solution.y[1]
df["SEIR_I"] = solution.y[2]
df["SEIR_R"] = solution.y[3]

# -----------------------------
# Step 5: Feature Engineering 
#------------------------------

# Time-based features
# Using lag features because dengue cases depend heavily on previous weeks
for lag in range(1, 13):
   df[f"lag_{lag}"] = df.groupby("city")["total_cases"].shift(lag)

# Rolling mean
# Smooth out short-term fluctuations 
df["roll_mean_3"] = df.groupby("city")["total_cases"].transform(lambda x: x.rolling(3, min_periods=1).mean())
df["roll_mean_6"] = df.groupby("city")["total_cases"].transform(lambda x: x.rolling(6, min_periods=1).mean())
df["roll_mean_12"] = df.groupby("city")["total_cases"].transform(lambda x: x.rolling(12, min_periods=1).mean())

df["roll_std_3"] = df.groupby("city")["total_cases"].transform(lambda x: x.rolling(3, min_periods=1).std())
df["roll_std_6"] = df.groupby("city")["total_cases"].transform(lambda x: x.rolling(6, min_periods=1).std())
df["roll_std_12"] = df.groupby("city")["total_cases"].transform(lambda x: x.rolling(12, min_periods=1).std())

# Exponential moving average
df["ema_3"] = df.groupby("city")["total_cases"].transform(lambda x: x.ewm(span=3).mean())
df["ema_6"] = df.groupby("city")["total_cases"].transform(lambda x: x.ewm(span=6).mean())
df["ema_12"] = df.groupby("city")["total_cases"].transform(lambda x: x.ewm(span=12).mean())

# Seasonal encoding
df["week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
df["week_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52)

# Remove rows with missing values after creating lag features
df = df.dropna(subset=[f"lag_{i}" for i in range(1, 13)] + ["roll_mean_3", "roll_mean_6", "roll_mean_12", "roll_std_3", "roll_std_6", "roll_std_12", "ema_3", "ema_6", "ema_12"]).reset_index(drop=True)

# ----------------------------
# Step 6: Feature Selection
# ----------------------------
# Features used for prediction 
features_cols = [
    "city",
    "week_sin",
    "week_cos",
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_5","lag_6","lag_7","lag_8", "lag_9", "lag_10", "lag_11", "lag_12",
    "roll_mean_3",
    "roll_mean_6",
    "roll_mean_12",
    "roll_std_3",
    "roll_std_6",
    "roll_std_12",
    "ema_3",
    "ema_6",
    "ema_12"
]

X = df[features_cols]
y = np.log1p(df["total_cases"])

# -----------------------------------------------------------------------------------------
# Step 7: Time Series Split (return first k folds as train set and (k+1)th fold as test set
# Source: https://scikit-learn.org/stable/modules/cross_validation.html
# -----------------------------------------------------------------------------------------
tscv = TimeSeriesSplit(n_splits=10)

# ----------------------------------------------------------
# Step 8: Histogram Based Gradient Boosting Regressor Model
# Source:https://scikit-learn.org/stable/modules/ensemble.html
#        https://optuna.org/
# ------------------------------------------------------------
#bayesian optimization
#define an objective function to be maximized
def objective(trial): 
    
    #suggest the values for the hyperparameters using a trial object
    max_depth = trial.suggest_int("max_depth", 4, 15)
    learning_rate = trial.suggest_float("learning_rate", 0.03, 0.3,log=True)
    max_iter = trial.suggest_int("max_iter", 300, 1500)
    min_samples_leaf = trial.suggest_int("min_samples_leaf",5,50)
    l2_regularization = trial.suggest_float("l2_regularization",0.0,2.0)
    max_bins = trial.suggest_int("max_bins", 64, 255)
    
    mae_scores = []

    for train, test in tscv.split(X): 

        X_train, X_val = X.iloc[train], X.iloc[test] #in pandas
        y_train, y_val = y.iloc[train], y.iloc[test] #select data using row and column position but not name
        
        model = HistGradientBoostingRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=max_iter,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_bins=max_bins,
            random_state=42
        ).fit(X_train, y_train)
        pred = np.expm1(model.predict(X_val))
        y_true = np.expm1(y_val)

        mae_scores.append(mean_absolute_error(y_true, pred))

    return np.mean(mae_scores)
#optuna training
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
print("Best parameters:", study.best_params)
print("Best MAE:", study.best_value)

# ------------------------------------------------------------------------------------------
# Step 9: Final Model Evaluation
# ------------------------------------------------------------------------------------------
best_params = study.best_params

all_actual = []
all_pred = []
all_index = []

for train, test in tscv.split(X):

    X_train, X_val = X.iloc[train], X.iloc[test]
    y_train, y_val = y.iloc[train], y.iloc[test]

    model = HistGradientBoostingRegressor(
        **best_params,
        random_state=42
    )

    model.fit(X_train, y_train)
    pred = np.expm1(model.predict(X_val))
    actual = np.expm1(y_val)

    pred = np.maximum(pred, 0)

    all_actual.extend(actual)
    all_pred.extend(pred)
    all_index.extend(X_val.index)

# -------------------
# Final Evaluation
# -------------------
mae = mean_absolute_error(all_actual, all_pred)
rmse = np.sqrt(mean_squared_error(all_actual, all_pred))
r2 = r2_score(all_actual, all_pred)

print("Final MAE:", mae)
print("Final RMSE:", rmse)
print("R2 Score:", r2)

# ------------------------
# Feature Importance
# ------------------------
result = permutation_importance(
    model,
    X_val,
    y_val,
    n_repeats=10,
    random_state=42
)

importances = result.importances_mean

print("\nFeature Importance:")
for name, score in zip(features_cols, importances):
    print(f"{name}: {score:.4f}")

# ------------------------
# Step 10: Visualisation
# ------------------------
df_plot = pd.DataFrame({"index": all_index,"actual": all_actual,"pred": all_pred}).sort_values("index")
plt.figure(figsize=(12,5))
plt.plot(df_plot["actual"].values[-200:], label="Actual")
plt.plot(df_plot["pred"].values[-200:], label="Predicted")
plt.title("Actual vs. HGB Predicted Cases for Dengue Prediction")
plt.xlabel("Time (Weeks)")
plt.ylabel("Number of Dengue Cases")
plt.legend()
plt.grid()
plt.show()

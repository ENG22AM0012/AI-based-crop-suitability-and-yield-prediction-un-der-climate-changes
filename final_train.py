# PYTHON 3.8
# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor, BaggingRegressor

from xgboost import XGBClassifier, XGBRegressor


# 2. LOAD DATA
df = pd.read_csv("dataset.csv")
df.columns = df.columns.str.strip()

print("Dataset Shape:", df.shape)


# 3. CLEANING
df = df.dropna()

df = df[(df['Temperature_K'] > 260) & (df['Temperature_K'] < 320)]
df = df[(df['Rainfall_mm'] > 100) & (df['Rainfall_mm'] < 1500)]
df = df[(df['Area_hectare'] > 10)]


# 4. TARGET CLEANING 
lower = df['Yield_tph'].quantile(0.02)
upper = df['Yield_tph'].quantile(0.98)
df['Yield_tph'] = df['Yield_tph'].clip(lower, upper)


# 5. FEATURE ENGINEERING
df['Rainfall_per_area'] = df['Rainfall_mm'] / (df['Area_hectare'] + 1)
df['NPK_sum'] = df['N'] + df['P'] + df['K']
df['NPK_ratio'] = df['N'] / (df['P'] + df['K'] + 1)
df['Temp_Rainfall'] = df['Temperature_K'] * df['Rainfall_mm']
df['Year_trend'] = df['Year'] - df['Year'].min()


# 6. ENCODING
label_encoders = {}

for col in ['District', 'Season', 'Crop']:
    df[col] = df[col].astype(str).str.strip()
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

joblib.dump(label_encoders, "label_encoders.pkl")


# 7. FEATURES & TARGETS
X = df[[
    'Year', 'District', 'Season',
    'Rainfall_mm', 'Temperature_K', 'Area_hectare',
    'N', 'P', 'K',
    'Rainfall_per_area', 'NPK_sum', 'NPK_ratio',
    'Temp_Rainfall', 'Year_trend'
]]

# Classification target
y_class = df['Crop']

# Regression target 
y_reg = np.log1p(df['Yield_tph'])


# 8. SPLIT
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42
)

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)


# 9. CLASSIFICATION MODEL
clf = RandomForestClassifier(n_estimators=400, random_state=42)
clf.fit(Xc_train, yc_train)

preds_c = clf.predict(Xc_test)
acc = accuracy_score(yc_test, preds_c)

print("\n===== CLASSIFICATION =====")
print(f"Accuracy: {acc:.4f}")


# 10. REGRESSION MODELS
rf = RandomForestRegressor(n_estimators=500, max_depth=12, random_state=42)

gb = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5
)

xgb = XGBRegressor(
    n_estimators=1500,
    learning_rate=0.015,
    max_depth=8,
    min_child_weight=2,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    reg_alpha=0.3,
    reg_lambda=2,
    random_state=42
)

# Bagging XGBoost
xgb_bag = BaggingRegressor(
    estimator=xgb,
    n_estimators=5,
    random_state=42
)


# 11. HYBRID STACK MODEL
stack = StackingRegressor(
    estimators=[
        ('rf', rf),
        ('gb', gb),
        ('xgb', xgb_bag)
    ],
    final_estimator=RandomForestRegressor(n_estimators=300),
    passthrough=True
)

models = {
    "RandomForest": rf,
    "GradientBoosting": gb,
    "XGB_Bagging": xgb_bag,
    "HybridStack": stack
}


# 12. TRAIN & SELECT BEST
print("\n===== REGRESSION =====")

best_model = None
best_r2 = -999

for name, model in models.items():
    model.fit(Xr_train, yr_train)

    preds = model.predict(Xr_test)

    # Reverse log transform
    preds = np.expm1(preds)
    actual = np.expm1(yr_test)

    r2 = r2_score(actual, preds)
    rmse = np.sqrt(mean_squared_error(actual, preds))
    mae = mean_absolute_error(actual, preds)

    print(f"{name} -> R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name

print(f"\n Best Regression Model: {best_name}")


# 13. SAVE MODELS
joblib.dump(clf, "best_crop_model.pkl")
joblib.dump(best_model, "best_yield_model.pkl")

print("\n Training Completed Successfully!")

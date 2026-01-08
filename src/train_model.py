# =====================================
# TRAIN MODEL FILE
# Random Forest + XGBoost with
# RandomizedSearchCV
# =====================================

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

# -------------------------------
# Create models directory
# -------------------------------
os.makedirs("models", exist_ok=True)

# -------------------------------
# Load cleaned dataset
# -------------------------------
df = pd.read_csv("data/cleaned_housing.csv")

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# 1️⃣ LINEAR REGRESSION
# =====================================================
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
print("Linear Regression R2:", r2_score(y_test, y_pred_lr))

# ✅ SAVE MODEL
joblib.dump(lr, "models/linear_model.pkl")

# =====================================================
# 2️⃣ RANDOM FOREST REGRESSOR
# =====================================================
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest R2:", r2_score(y_test, y_pred_rf))

# ✅ SAVE MODEL
joblib.dump(rf, "models/random_forest_model.pkl")

# =====================================================
# 3️⃣ XGBOOST REGRESSOR
# =====================================================
xgb = XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=8,              # 👈 fixed
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_train_xgb = xgb.predict(X_train)

print("XGBoost Train R2:", r2_score(y_train, y_train_xgb))
print("XGBoost Test  R2:", r2_score(y_test, y_pred_xgb))

# ✅ SAVE MODEL
joblib.dump(xgb, "models/xgboost_model.pkl")

print("\n✅ Training completed & all models saved successfully!")

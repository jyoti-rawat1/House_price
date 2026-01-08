# ================================
# EDA FILE
# ================================

# =====================================
# COMPLETE EDA FILE
# =====================================

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/cleaned_housing.csv")

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("models/xgboost_model.pkl")

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# ==================================================
# 1️⃣ TARGET VARIABLE DISTRIBUTION
# ==================================================
plt.figure()
sns.histplot(y, kde=True)
plt.title("Target Variable Distribution (House Price)")
plt.xlabel("House Price")
plt.show()

# ==================================================
# 2️⃣ ACTUAL vs PREDICTED
# ==================================================
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle="--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.show()

# ==================================================
# 3️⃣ RESIDUAL PLOT
# ==================================================
plt.figure()
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# ==================================================
# 4️⃣ ERROR DISTRIBUTION
# ==================================================
plt.figure()
sns.histplot(residuals, kde=True)
plt.xlabel("Prediction Error")
plt.title("Error Distribution")
plt.show()

# ==================================================
# 5️⃣ ACTUAL vs PREDICTED DISTRIBUTION
# ==================================================
plt.figure()
sns.kdeplot(y_test, label="Actual")
sns.kdeplot(y_pred, label="Predicted")
plt.title("Actual vs Predicted Distribution")
plt.legend()
plt.show()

# ==================================================
# 6️⃣ CORRELATION HEATMAP
# ==================================================
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# ==================================================
# 7️⃣ FEATURE IMPORTANCE (XGBOOST)
# ==================================================
importances = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(
    x="Importance",
    y="Feature",
    data=importance_df
)
plt.title("Feature Importance (XGBoost)")
plt.show()

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    layout="centered"
)

st.title("🏠 House Price Prediction App")

# ---------------------------------
# Load dataset
# ---------------------------------
df = pd.read_csv("data/cleaned_housing.csv")

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------
# Load trained models
# ---------------------------------
models = {
    "Linear Regression": joblib.load("models/linear_model.pkl"),
    "Random Forest": joblib.load("models/random_forest_model.pkl"),
    "XGBoost": joblib.load("models/xgboost_model.pkl")
}

# ---------------------------------
# Model selection
# ---------------------------------
model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]

# ---------------------------------
# User input section
# ---------------------------------
st.subheader("Enter House Features")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(
        label=col,
        value=float(X[col].mean())
    )

input_df = pd.DataFrame([input_data])

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("Predict House Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted House Price: ₹ {prediction:,.2f}")

# ---------------------------------
# Model Evaluation
# ---------------------------------
st.subheader("📊 Model Performance")

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

st.write(f"**R² Score ({model_name}) :** {r2:.3f}")

# ---------------------------------
# Actual vs Predicted plot
# ---------------------------------
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title(f"Actual vs Predicted ({model_name})")

st.pyplot(fig)

# Canada GDP Prediction with Streamlit
# Requirements:
# pip install streamlit scikit-learn matplotlib pandas numpy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("Canada_GDP.csv")   # <-- apna CSV file yaha rakho
x = df[['Year']]
y = df['GDP-Per']

# Lookup dictionary for exact values
gdp_lookup = dict(zip(df['Year'], df['GDP-Per']))

# -------------------------
# Polynomial Regression Model
# -------------------------
poly = PolynomialFeatures(degree=4)   # Degree change kar sakte ho
x_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Canada GDP Prediction", layout="centered")

st.title("📈 Canada GDP Per Capita Prediction")
st.markdown("This app uses **Polynomial Regression** to predict GDP per capita of Canada.")

# Show dataset
if st.checkbox("📂 Show Raw Data"):
    st.write(df)

# Year input from user
year_input = st.number_input("Enter a Year to get GDP:", min_value=int(df['Year'].min()), 
                             max_value=2100, step=1)

# Prediction function
def predict_gdp(year):
    if year in gdp_lookup:
        return f"✅ Year {year}: Actual GDP = {gdp_lookup[year]}"
    else:
        year_poly = poly.transform([[year]])
        pred = model.predict(year_poly)[0]
        return f"📌 Year {year}: Predicted GDP (by model) = {pred:.2f}"

if st.button("🔮 Predict GDP"):
    result = predict_gdp(int(year_input))
    st.success(result)

# -------------------------
# Plotting Graph
# -------------------------
fig, ax = plt.subplots(figsize=(10, 5))

# Scatter plot (actual values)
ax.scatter(x, y, color='blue', label="Actual GDP", marker='*')

# Prediction curve
x_range = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
y_pred = model.predict(poly.transform(x_range))
ax.plot(x_range, y_pred, color='red', label="Polynomial Prediction")

ax.set_xlabel("Year")
ax.set_ylabel("GDP Per Capita")
ax.set_title("Canada GDP Per Capita Prediction")
ax.legend()
ax.grid(True)

st.pyplot(fig)

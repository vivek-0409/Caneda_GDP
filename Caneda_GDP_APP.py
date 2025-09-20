# Canada GDP Prediction with Streamlit + Plotly
# Requirements:
# pip install streamlit scikit-learn pandas numpy plotly

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
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
year_input = st.number_input("Enter a Year to get GDP:", 
                             min_value=int(df['Year'].min()), 
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
# Interactive Plotly Chart
# -------------------------
x_range = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
y_pred = model.predict(poly.transform(x_range))

# Create Plotly figure
fig = px.scatter(df, x="Year", y="GDP-Per", 
                 title="Canada GDP Per Capita (Actual vs Predicted)",
                 labels={"Year": "Year", "GDP-Per": "GDP Per Capita (US $)"},
                 hover_data={"Year": True, "GDP-Per": ":.2f"},
                 color_discrete_sequence=["blue"])

# Add polynomial regression line
fig.add_traces(px.line(x=x_range.flatten(), y=y_pred, 
                       labels={"x": "Year", "y": "Predicted GDP"},
                       color_discrete_sequence=["red"]).data)

fig.update_traces(marker=dict(size=8, symbol="star"))  # Actual data stars
fig.update_layout(hovermode="x unified")               # Hover effect

# Show Plotly figure in Streamlit
st.plotly_chart(fig, use_container_width=True)

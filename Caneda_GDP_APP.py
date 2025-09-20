import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(page_title="Canada GDP Prediction", layout="centered")

# Custom CSS for animated heading
st.markdown(
    """
    <style>
    @keyframes colorchange {
      0%   {color: #FF5733;}
      25%  {color: #33FF57;}
      50%  {color: #3357FF;}
      75%  {color: #F333FF;}
      100% {color: #FFBD33;}
    }
    .animated-title {
      font-size: 40px;
      font-weight: bold;
      text-align: center;
      animation: colorchange 1s infinite;
    }
    </style>
    <h1 class="animated-title">📈 Canada GDP Per Capita Prediction</h1>
    """,
    unsafe_allow_html=True
)

st.markdown("This app uses **Polynomial Regression** to predict GDP per capita of Canada, with an animated and attractive UI. ✨")


uploaded_file = st.file_uploader("Canada_GDP_Dataset.csv file", type="csv")

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

x_range = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
y_pred = model.predict(poly.transform(x_range))


fig = px.scatter(df, x="Year", y="GDP-Per", 
                 title="Canada GDP Per Capita (Actual vs Predicted)",
                 labels={"Year": "Year", "GDP-Per": "GDP Per Capita (US $)"},
                 hover_data={"Year": True, "GDP-Per": ":.2f"},
                 color_discrete_sequence=["blue"])


fig.add_traces(px.line(x=x_range.flatten(), y=y_pred, 
                       labels={"x": "Year", "y": "Predicted GDP"},
                       color_discrete_sequence=["red"]).data)

fig.update_traces(marker=dict(size=8, symbol="star"))  # Actual data stars
fig.update_layout(hovermode="x unified")               # Hover effect

st.plotly_chart(fig, use_container_width=True)

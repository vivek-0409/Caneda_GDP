import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("Canada_GDP_Dataset.csv")  
x = df[['Year']]
y = df['GDP-Per']

# Lookup dictionary for exact values
gdp_lookup = dict(zip(df['Year'], df['GDP-Per']))


poly = PolynomialFeatures(degree=4)   # Degree change kar sakte ho
x_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)


st.set_page_config(page_title="Canada GDP Prediction", layout="centered")

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



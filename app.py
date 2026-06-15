import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Housing.csv")

# Features and target
X = df[['area', 'bedrooms', 'bathrooms', 'stories', 'parking',]]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("🏠 House Price Prediction")

area = st.number_input("Area", min_value=0)
bedrooms = st.number_input("Bedrooms", min_value=0)
bathrooms = st.number_input("Bathrooms", min_value=0)
stories = st.number_input("stories", min_value=0)
parking = st.number_input("Parking", min_value=0)

if st.button("Predict Price"):
    prediction = model.predict(
        [[area, bedrooms, bathrooms, stories, parking]]
    )

    st.success(f"Predicted House Price: ₹{prediction[0]:,.2f}")
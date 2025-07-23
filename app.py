

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import pyttsx3
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
import joblib

st.set_page_config(page_title="Turbulence Prediction Dashboard", layout="wide")
st.title("✈️ Flight Turbulence Safety Dashboard")

# User input for custom flight data
st.sidebar.header("✍️ User Input Parameters")

latitude = st.sidebar.number_input("Latitude", value=37.77, format="%.6f")
longitude = st.sidebar.number_input("Longitude", value=-122.42, format="%.6f")
weight = st.sidebar.slider("Aircraft Weight (lbs)", 1000, 10000, 5000)
arm = st.sidebar.slider("Arm Length (inches)", 10, 60, 35)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 50.0, 15.0)
altitude = st.sidebar.slider("Altitude (feet)", 0, 20000, 10000)

# Single-row DataFrame for prediction and display
now = datetime.datetime.now()
data = {
    "Time": [now],
    "Latitude": [latitude],
    "Longitude": [longitude],
    "Weight": [weight],
    "Arm": [arm],
    "WindSpeed": [wind_speed],
    "Altitude": [altitude]
}
df = pd.DataFrame(data)
df["Moment"] = df["Weight"] * df["Arm"]
df["COG"] = df["Moment"] / df["Weight"]
df["TurbulenceScore"] = df["WindSpeed"] / 45.0
df["TurbulenceClass"] = df["TurbulenceScore"].apply(
    lambda x: "Low" if x < 0.3 else "Medium" if x < 0.7 else "High"
)

st.subheader("📋 Flight Snapshot")
st.dataframe(df)

# Speak turbulence level
def speak_turbulence_level(level):
    engine = pyttsx3.init()
    engine.say(f"Current turbulence level is {level}")
    engine.runAndWait()

if st.button("🔊 Speak Turbulence Level"):
    speak_turbulence_level(df["TurbulenceClass"].iloc[0])

# Live Wind Fetch Option
def fetch_live_weather(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        wind_speed = data["wind"]["speed"]
        score = min(wind_speed / 45, 1.0)
        return wind_speed, score
    except:
        return None, None

st.subheader("🌐 Live Wind Option")
if st.checkbox("Use Live Wind Speed"):
    api_key = "e9e42833dd2e06259a55b7ea59429ab1"
    live_wind, score = fetch_live_weather(latitude, longitude, api_key)
    if live_wind:
        st.metric("Live Wind Speed (m/s)", f"{live_wind:.2f}")
        st.metric("Turbulence Score", f"{score:.2f}")
        df["WindSpeed"] = live_wind
        df["TurbulenceScore"] = score
        df["TurbulenceClass"] = df["TurbulenceScore"].apply(
            lambda x: "Low" if x < 0.3 else "Medium" if x < 0.7 else "High"
        )
    else:
        st.warning("Failed to fetch live data.")

# Heatmap
st.subheader("🗺️ Location Map")
m = folium.Map(location=[latitude, longitude], zoom_start=6)
HeatMap([[latitude, longitude, df["TurbulenceScore"].iloc[0]]]).add_to(m)
st_data = st_folium(m, width=700)

# Plot
st.subheader("📈 Center of Gravity and Altitude")
fig = px.line(df, x="Time", y=["COG", "Altitude"], title="COG and Altitude")
st.plotly_chart(fig)

# Prediction
st.subheader("🔮 Turbulence Prediction")
model = joblib.load("model_turbulence.pkl")
features = df[["Weight", "Arm", "WindSpeed", "Altitude"]].values
prediction = model.predict(features)[0]
st.success(f"Predicted Turbulence Class: {prediction}")

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import folium
from folium.plugins import HeatMap, AntPath
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
import joblib
from gtts import gTTS
from tempfile import NamedTemporaryFile
from sklearn.ensemble import IsolationForest

# Prophet for trend forecasting
from prophet import Prophet

# --- API Key from secrets ---
api_key = "e9e42833dd2e06259a55b7ea59429ab1"

# --- Sidebar: Inputs ---
st.sidebar.header("🧮 Aircraft Parameters")
latitude = st.sidebar.number_input("Latitude", value=37.77, format="%.6f")
longitude = st.sidebar.number_input("Longitude", value=-122.42, format="%.6f")
weight = st.sidebar.slider("Aircraft Weight (lbs)", 1000, 10000, 5000)
arm = st.sidebar.slider("Arm Length (inches)", 10, 60, 35)

st.sidebar.header("🌦️ Environmental Conditions")
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 50.0, 15.0)
altitude = st.sidebar.slider("Altitude (feet)", 0, 20000, 10000)
use_live_wind = st.sidebar.checkbox("Use Live Wind Speed")

# --- Fetch live wind speed if selected ---
@st.cache_data(ttl=600)
def fetch_live_weather(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()["wind"]["speed"]
    except:
        return None

if use_live_wind:
    live_ws = fetch_live_weather(latitude, longitude, api_key)
    if live_ws:
        wind_speed = live_ws
        st.sidebar.success(f"Live Wind Speed: {wind_speed:.2f} m/s")

# --- Data Preparation ---
now = datetime.datetime.now()
df = pd.DataFrame([{
    "Time": now, "Latitude": latitude, "Longitude": longitude,
    "Weight": weight, "Arm": arm, "WindSpeed": wind_speed, "Altitude": altitude
}])

df["Moment"] = df["Weight"] * df["Arm"]
df["COG"] = df["Moment"] / df["Weight"]
df["TurbulenceScore"] = df["WindSpeed"] / 45.0
df["TurbulenceClass"] = df["TurbulenceScore"].apply(
    lambda x: "Low" if x < 0.3 else "Medium" if x < 0.7 else "High"
)

# --- Store and Update Historical Data ---
if "historical_data" not in st.session_state:
    st.session_state["historical_data"] = df.copy()
else:
    st.session_state["historical_data"] = pd.concat(
        [st.session_state["historical_data"], df], ignore_index=True
    ).drop_duplicates(subset=["Time"], keep="last")

historical_data = st.session_state["historical_data"]

# --- Dashboard Layout ---
st.set_page_config(page_title="Flight Turbulence Dashboard", layout="wide")
st.title("✈️ Flight Turbulence Safety Dashboard")
st.markdown("---")

# --- Cockpit Panel ---
st.markdown("## 🛩️ Cockpit Panel: Dials & Instruments")
col1, col2, col3 = st.columns(3)
with col1:
    fig_cog = go.Figure(go.Indicator(
        mode="gauge+number", value=df["COG"].iloc[0],
        title={'text': "COG (in)"},
        gauge={'axis': {'range': [10, 60]}}
    ))
    st.plotly_chart(fig_cog, use_container_width=True)
with col2:
    fig_alt = go.Figure(go.Indicator(
        mode="gauge+number", value=df["Altitude"].iloc[0],
        title={'text': "Altitude (ft)"},
        gauge={'axis': {'range': [0, 20000]}}
    ))
    st.plotly_chart(fig_alt, use_container_width=True)
with col3:
    fig_wind = go.Figure(go.Indicator(
        mode="gauge+number", value=df["WindSpeed"].iloc[0],
        title={'text': "Wind Speed (m/s)"},
        gauge={'axis': {'range': [0, 50]}}
    ))
    st.plotly_chart(fig_wind, use_container_width=True)

# --- Flight Snapshot Table ---
st.markdown("## 📋 Flight Snapshot")
st.dataframe(df, use_container_width=True)

# --- Voice Output (gTTS) ---
def speak_turbulence_level(level):
    text = f"Current turbulence level is {level}."
    tts = gTTS(text=text, lang='en')
    with NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")
if st.button("🔊 Speak Turbulence Level"):
    speak_turbulence_level(df["TurbulenceClass"].iloc[0])

# --- Risk Summary ---
st.markdown("## 📋 Flight Risk Summary")
turb = df["TurbulenceClass"].iloc[0]
color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[turb]
note = {"Low": "Safe to proceed.", "Medium": "Proceed with caution.", "High": "Delay or reroute suggested."}[turb]
st.info(f"**Turbulence Level:** {turb} {color} | **Recommendation:** {note}")

# --- Location Map & Heatmap ---
st.markdown("## 🗺️ Location Map")
m = folium.Map(location=[latitude, longitude], zoom_start=6)
HeatMap([[latitude, longitude, df["TurbulenceScore"].iloc[0]]]).add_to(m)
folium.Marker([latitude, longitude], tooltip="Current Location").add_to(m)
st_folium(m, width=700)

# --- Historical Trends Chart ---
st.markdown("## 📈 Historical Trends")
fig_hist = px.line(historical_data, x="Time", y=["COG", "Altitude", "WindSpeed"], markers=True)
st.plotly_chart(fig_hist, use_container_width=True)

# --- Anomaly Detection (Isolation Forest) ---
st.markdown("## 🚨 Anomaly Detection")
if len(historical_data) >= 10:  # Minimum 10 samples for IF
    clf = IsolationForest(contamination=0.1, random_state=42)
    feat_cols = ["WindSpeed", "COG", "Altitude"]
    X = historical_data[feat_cols]
    anomaly_pred = clf.fit_predict(X)
    historical_data["Anomaly"] = anomaly_pred
    anomalies = historical_data[historical_data["Anomaly"] == -1]
    if not anomalies.empty:
        st.warning("⚠️ Anomaly detected in flight conditions!")
        st.dataframe(anomalies)
else:
    st.info("Add more historical points to enable anomaly detection (minimum 10).")

# --- Route Optimization & Visualization ---
st.markdown("## 🌍 Route Visualization")
with st.expander("Show Route Input"):
    origin_lat = st.number_input("Origin Latitude", value=37.77)
    origin_lon = st.number_input("Origin Longitude", value=-122.42)
    dest_lat = st.number_input("Destination Latitude", value=34.05)
    dest_lon = st.number_input("Destination Longitude", value=-118.25)

def generate_route(lat1, lon1, lat2, lon2, points=10):
    lats = np.linspace(lat1, lat2, points)
    lons = np.linspace(lon1, lon2, points)
    return list(zip(lats, lons))

route_coords = generate_route(origin_lat, origin_lon, dest_lat, dest_lon)

route_map = folium.Map(location=[(origin_lat+dest_lat)/2, (origin_lon+dest_lon)/2], zoom_start=6)
AntPath(route_coords, color="blue").add_to(route_map)
folium.Marker([origin_lat, origin_lon], tooltip="Origin", icon=folium.Icon(color='green')).add_to(route_map)
folium.Marker([dest_lat, dest_lon], tooltip="Destination", icon=folium.Icon(color='red')).add_to(route_map)
st_folium(route_map, width=700)

# --- Trend Forecasting (Prophet) ---
st.markdown("## 📉 Wind Speed Trend Forecasting")
try:
    if len(historical_data) > 12:
        # Prophet expects 'ds' and 'y' columns
        forecast_data = historical_data[["Time", "WindSpeed"]].rename(columns={"Time": "ds", "WindSpeed": "y"})
        model = Prophet()
        model.fit(forecast_data)
        future = model.make_future_dataframe(periods=24, freq='H')
        forecast = model.predict(future)
        fig_forecast = px.line(forecast, x='ds', y='yhat', title="Wind Speed Forecast")
        fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Conf.')
        fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Conf.')
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.info("Add at least 12 points for wind speed forecasting.")
except Exception as e:
    st.error(f"Forecasting error: {e}")

# --- Data Export ---
csv = historical_data.to_csv(index=False)
st.download_button("Export Data", csv, "flight_data.csv")

# --- Turbulence Prediction (ML Model) ---
st.markdown("## 🔮 Turbulence Prediction")
try:
    model = joblib.load("model_turbulence.pkl")
    features = df[["Weight", "Arm", "WindSpeed", "Altitude"]].values
    prediction = model.predict(features)[0]
    st.success(f"Predicted Turbulence Class: {prediction}")
except Exception as e:
    st.error(f"Prediction error: {e}")

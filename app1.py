
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

# Theme: Refined Blue Steel
st.set_page_config(page_title="UMA Aviation Club Dashboard", page_icon="✈️", layout="wide")

st.markdown("""
<style>
/* Refined Theme: Blue Steel */
.reportview-container, .stApp {
    background: linear-gradient(135deg, #d3e2f8 0%, #43688b 100%) !important;
    color: #13344d !important;
}
.sidebar .sidebar-content {
    background: #e3e9f3 !important;
    padding: 1rem !important;
}
h1 { font-size: 2.4rem !important; font-weight: 900 !important; color: #205080 !important;}
h2 { font-size: 2rem !important; font-weight: bold !important; color: #205080 !important;}
h3, h4 { font-size: 1.5rem !important; color: #205080 !important;}
.stMarkdown p, .stMarkdown li { font-size: 1.1rem !important;}
button[role="tab"] > div > span { font-size: 1.6rem !important;}
button[role="tab"][aria-selected="true"] > div > span { font-size: 1.8rem !important; color: #f5b942 !important;}
.stButton > button, .stDownloadButton > button {
    padding: 0.5rem 1rem !important;
    background-color: #205080 !important;
    box-shadow: 0px 4px 6px rgba(32, 80, 128, 0.2) !important;
    color: #fff !important;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# App Title and Logo
st.image("logo.png", width=150)
st.markdown("<h1>UMA Aviation Club Dashboard</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.info("🛩️ Enter aircraft and weather data, analyze risk, forecast turbulence, and export reports.")
    st.markdown("---")
    user_wind_alert = st.slider("Wind Speed Alert Threshold (m/s)", 0.0, 50.0, 30.0)
    st.markdown("---")
    latitude = st.number_input("Latitude", value=37.77, format="%.6f")
    longitude = st.number_input("Longitude", value=-122.42, format="%.6f")
    wind_speed = st.slider("Wind Speed (m/s)", 0.0, 50.0, 15.0)

# Data Processing
df = pd.DataFrame([{"Latitude": latitude, "Longitude": longitude, "WindSpeed": wind_speed}])
df["TurbulenceScore"] = df["WindSpeed"] / 45.0
df["TurbulenceClass"] = df["TurbulenceScore"].apply(lambda x: "Low" if x < 0.3 else "Medium" if x < 0.7 else "High")

# Alerts and Recommendations
turb = df["TurbulenceClass"].iloc[0]
color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[turb]
note = {"Low": "Safe to proceed.", "Medium": "Proceed with caution.", "High": "Delay or reroute suggested."}[turb]

st.markdown(f"""
<div style="padding: 10px; border-radius: 8px; background-color: {'#28a745' if turb == 'Low' else '#f5b942' if turb == 'Medium' else '#dc3545'}20;">
  <strong>Turbulence Level:</strong> {turb} {color}<br>
  <strong>Recommendation:</strong> {note}
</div>
""", unsafe_allow_html=True)

# Enhanced Map
st.markdown("<h2>🗺️ Location Map</h2>", unsafe_allow_html=True)
m = folium.Map(location=[latitude, longitude], zoom_start=6, tiles="CartoDB Positron")
HeatMap([[latitude, longitude, df["TurbulenceScore"].iloc[0]]],
        gradient={0.3: '#28a745', 0.6: '#f5b942', 0.9: '#dc3545'}).add_to(m)
folium.Marker([latitude, longitude], tooltip="Current Location",
              icon=folium.Icon(color='green' if turb=="Low" else 'orange' if turb=="Medium" else 'red')).add_to(m)
st_folium(m, width=900, height=500)

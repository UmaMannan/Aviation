import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
import joblib
from gtts import gTTS
from tempfile import NamedTemporaryFile

# ----- Custom Dark Theme -----
st.set_page_config(page_title="Flight Turbulence Dashboard", layout="wide")
st.markdown("""
    <style>
        .reportview-container {
            background: #101820;
            color: #F2F3F4;
        }
        .sidebar .sidebar-content {
            background: #17181c;
        }
        h1, h2, h3, h4, h5 {
            color: #FFD700;
        }
        .stButton > button {
            background-color: #444654;
            color: #FFD700;
        }
        .st-cm, .st-bf {
            color: #FFD700;
        }
    </style>
""", unsafe_allow_html=True)

st.title("✈️ Flight Turbulence Safety Dashboard")
st.markdown("---")

# ----- Sidebar: Aircraft Inputs -----
st.sidebar.header("🧮 Aircraft Input")
latitude = st.sidebar.number_input("Latitude", value=37.77, format="%.6f")
longitude = st.sidebar.number_input("Longitude", value=-122.42, format="%.6f")
weight = st.sidebar.slider("Aircraft Weight (lbs)", 1000, 10000, 5000)
arm = st.sidebar.slider("Arm Length (inches)", 10, 60, 35)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 50.0, 15.0)
altitude = st.sidebar.slider("Altitude (feet)", 0, 20000, 10000)

now = datetime.datetime.now()
df = pd.DataFrame([{
    "Time": now,
    "Latitude": latitude,
    "Longitude": longitude,
    "Weight": weight,
    "Arm": arm,
    "WindSpeed": wind_speed,
    "Altitude": altitude
}])

df["Moment"] = df["Weight"] * df["Arm"]
df["COG"] = df["Moment"] / df["Weight"]
df["TurbulenceScore"] = df["WindSpeed"] / 45.0
df["TurbulenceClass"] = df["TurbulenceScore"].apply(
    lambda x: "Low" if x < 0.3 else "Medium" if x < 0.7 else "High"
)

# ----- Visual Cockpit Panel (COG & Altitude Dials) -----
st.markdown("## 🛩️ Cockpit Panel: Dials & Instruments")
col1, col2, col3 = st.columns([2,2,2])

with col1:
    fig_cog = go.Figure(go.Indicator(
        mode="gauge+number",
        value=df["COG"].iloc[0],
        title={'text': "COG (in)", 'font': {'color': '#FFD700'}},
        gauge={
            'axis': {'range': [10, 60], 'tickcolor':'#FFD700'},
            'bar': {'color': "#FFD700"},
            'bgcolor': "#23272f",
            'borderwidth': 2,
            'bordercolor': "#444654"
        },
        number={'suffix': " in", 'font': {'color': '#FFD700'}}
    ))
    fig_cog.update_layout(paper_bgcolor="#101820", font={'color':'#FFD700'})
    st.plotly_chart(fig_cog, use_container_width=True)

with col2:
    fig_alt = go.Figure(go.Indicator(
        mode="gauge+number",
        value=df["Altitude"].iloc[0],
        title={'text': "Altitude (ft)", 'font': {'color': '#FFD700'}},
        gauge={
            'axis': {'range': [0, 20000], 'tickcolor':'#FFD700'},
            'bar': {'color': "#FFD700"},
            'bgcolor': "#23272f",
            'borderwidth': 2,
            'bordercolor': "#444654"
        },
        number={'suffix': " ft", 'font': {'color': '#FFD700'}}
    ))
    fig_alt.update_layout(paper_bgcolor="#101820", font={'color':'#FFD700'})
    st.plotly_chart(fig_alt, use_container_width=True)

with col3:
    fig_wind = go.Figure(go.Indicator(
        mode="gauge+number",
        value=df["WindSpeed"].iloc[0],
        title={'text': "Wind Speed (m/s)", 'font': {'color': '#FFD700'}},
        gauge={
            'axis': {'range': [0, 50], 'tickcolor':'#FFD700'},
            'bar': {'color': "#FFD700"},
            'bgcolor': "#23272f",
            'borderwidth': 2,
            'bordercolor': "#444654"
        },
        number={'suffix': " m/s", 'font': {'color': '#FFD700'}}
    ))
    fig_wind.update_layout(paper_bgcolor="#101820", font={'color':'#FFD700'})
    st.plotly_chart(fig_wind, use_container_width=True)

# ----- Snapshot Table -----
st.markdown("## 📋 Flight Snapshot")
st.dataframe(df, use_container_width=True)

# ----- Voice Output (gTTS) -----
def speak_turbulence_level(level):
    try:
        level_str = str(level).strip().capitalize()
        if level_str not in ["Low", "Medium", "High"]:
            raise ValueError(f"Unexpected turbulence level: {level_str}")
        text = f"Current turbulence level is {level_str}."
        tts = gTTS(text=text, lang='en')
        with NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format="audio/mp3")
    except Exception as e:
        st.error(f"Speech synthesis failed: {e}")

if st.button("🔊 Speak Turbulence Level"):
    level = str(df["TurbulenceClass"].iloc[0])
    speak_turbulence_level(level)

# ----- Live Weather -----
st.markdown("## 🌐 Live Wind Option")
def fetch_live_weather(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        wind_speed = data["wind"]["speed"]
        score = min(wind_speed / 45.0, 1.0)
        return wind_speed, score
    except:
        return None, None

if st.checkbox("Use Live Wind Speed"):
    api_key = "e9e42833dd2e06259a55b7ea59429ab1"
    live_ws, score = fetch_live_weather(latitude, longitude, api_key)
    if live_ws:
        st.metric("Live Wind Speed (m/s)", f"{live_ws:.2f}")
        st.metric("Turbulence Score", f"{score:.2f}")
        df["WindSpeed"] = live_ws
        df["TurbulenceScore"] = score
        df["TurbulenceClass"] = df["TurbulenceScore"].apply(
            lambda x: "Low" if x < 0.3 else "Medium" if x < 0.7 else "High"
        )

# ----- Risk Summary -----
st.markdown("## 📋 Flight Risk Summary")
w, a, cog, alt, ws = df.iloc[0][["Weight", "Arm", "COG", "Altitude", "WindSpeed"]]
turb = str(df["TurbulenceClass"].iloc[0])
color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}

if turb == "Low":
    risk = "🟢 LOW"
    note = "Safe to proceed."
elif turb == "Medium":
    risk = "🟡 MODERATE"
    note = "Proceed with caution."
else:
    risk = "🔴 HIGH"
    note = "Delay or reroute suggested."

st.markdown(f"""
<div style="background-color:#23272f;padding:20px;border-radius:15px;">
<b>Weight:</b> <span style="color:#FFD700;">{w} lbs</span><br>
<b>Arm:</b> <span style="color:#FFD700;">{a} in</span><br>
<b>COG:</b> <span style="color:#FFD700;">{cog:.2f}</span><br>
<b>Altitude:</b> <span style="color:#FFD700;">{alt} ft</span><br>
<b>Wind Speed:</b> <span style="color:#FFD700;">{ws:.1f} m/s</span><br>
<b>Turbulence Level:</b> <span style="color:#FFD700;">{turb} {color[turb]}</span><br>
<h3 style="color:#FFD700;">{risk}</h3>
<b>Recommendation:</b> <span style="color:#FFD700;">{note}</span>
</div>
""", unsafe_allow_html=True)

# ----- Map -----
st.markdown("## 🗺️ Location Map")
m = folium.Map(location=[latitude, longitude], zoom_start=6, tiles="CartoDB dark_matter")
HeatMap([[latitude, longitude, df["TurbulenceScore"].iloc[0]]]).add_to(m)
st_folium(m, width=700)

# ----- Line Chart -----
st.markdown("## 📈 Center of Gravity and Altitude")
fig = px.line(df, x="Time", y=["COG", "Altitude"], title="COG and Altitude")
fig.update_layout(template='plotly_dark', font=dict(color='#FFD700'))
st.plotly_chart(fig, use_container_width=True)

# ----- Prediction -----
st.markdown("## 🔮 Turbulence Prediction")
try:
    model = joblib.load("model_turbulence.pkl")
    features = df[["Weight", "Arm", "WindSpeed", "Altitude"]].values
    prediction = model.predict(features)[0]
    st.success(f"Predicted Turbulence Class: {prediction}")
except Exception as e:
    st.error(f"Prediction error: {e}")

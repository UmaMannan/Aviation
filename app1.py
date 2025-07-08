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
from prophet import Prophet
from fpdf import FPDF
import base64

# --- Steel Grey & Blue Cockpit CSS ---
st.markdown("""
<style>
body, .stApp {background: #23282d;}
.reportview-container, .st-emotion-cache-1d391kg, .st-emotion-cache-1kyxreq {
    background: linear-gradient(120deg, #37404a 0%, #23282d 100%);
}
.dashboard-panel {
    background: #262b30;
    border-radius: 20px;
    box-shadow: 0 4px 24px #1e222566;
    padding: 2rem 1.5rem;
    border: 2px solid #3a4147;
    margin-bottom: 1.7rem;
}
.cockpit-header {
    background: linear-gradient(90deg, #4682b4 50%, #35393c 100%);
    color: #fff;
    border-radius: 20px;
    padding: 1.5rem 2rem 1.1rem 2rem;
    font-size: 1.7rem;
    margin-bottom: 1.7rem;
    border-bottom: 2px solid #5bc0eb;
    letter-spacing: 1px;
}
.stButton > button {
    background: #4682b4;
    color: #fff;
    border-radius: 1.1rem;
    border: 2px solid #3a4147;
    font-weight: bold;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #31546b;
    color: #FFDD57;
}
</style>
""", unsafe_allow_html=True)

# --- Cockpit Header ---
st.markdown(
    '<div class="cockpit-header">✈️ <b>IN-FLIGHT COCKPIT DASHBOARD</b> &nbsp;|&nbsp; Flight #AB123 &nbsp;|&nbsp; Operator: SkyWing</div>',
    unsafe_allow_html=True
)

# --- Sidebar Branding & Help ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2917/2917991.png", width=80)
    st.markdown("### Flight Turbulence Dashboard")
    st.info(
        "🛩️ Enter your aircraft and weather data, analyze risk, "
        "forecast turbulence, and download safety reports."
    )
    st.markdown("---")
    st.markdown("**Instructions:**\n- Enter flight parameters\n- Analyze & visualize results\n- Forecast turbulence\n- Download/export session")
    st.markdown("---")
    user_wind_alert = st.slider("Wind Speed Alert Threshold (m/s)", 0.0, 50.0, 30.0, help="Set a custom alert for high wind speed.")
    st.markdown("---")
    st.markdown("#### Data Management")
    uploaded = st.file_uploader("Upload Previous Session", type="csv", help="Restore a saved flight session.")

# --- API Key ---
api_key = "e9e42833dd2e06259a55b7ea59429ab1"

# --- Inputs ---
st.sidebar.header("🧮 Aircraft Parameters")
latitude = st.sidebar.number_input("Latitude", value=37.77, format="%.6f")
longitude = st.sidebar.number_input("Longitude", value=-122.42, format="%.6f")
weight = st.sidebar.slider("Aircraft Weight (lbs)", 1000, 10000, 5000)
arm = st.sidebar.slider("Arm Length (inches)", 10, 60, 35)

st.sidebar.header("🌦️ Environmental Conditions")
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 50.0, 15.0)
altitude = st.sidebar.slider("Altitude (feet)", 0, 20000, 10000)
use_live_wind = st.sidebar.checkbox("Use Live Wind Speed")

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

if "historical_data" not in st.session_state or uploaded:
    if uploaded:
        st.session_state["historical_data"] = pd.read_csv(uploaded)
        st.success("Session loaded from file!")
    else:
        st.session_state["historical_data"] = df.copy()
else:
    st.session_state["historical_data"] = pd.concat(
        [st.session_state["historical_data"], df], ignore_index=True
    ).drop_duplicates(subset=["Time"], keep="last")

historical_data = st.session_state["historical_data"]

if df["WindSpeed"].iloc[0] > user_wind_alert:
    st.error(f"🚨 ALERT: Wind speed ({df['WindSpeed'].iloc[0]:.1f} m/s) exceeds your safe threshold!")

# --- Main Dashboard Panel ---
st.markdown('<div class="dashboard-panel">', unsafe_allow_html=True)

# --- Cockpit Instrument Panel (4 gauges) ---
col1, col2, col3, col4 = st.columns([1.7,1.7,1.7,2])
with col1:
    fig1 = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=df["COG"].iloc[0],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "<b>COG</b><br><span style='font-size:15px'>inches</span>", 'font': {'color': '#FFDD57'}},
        gauge={
            'axis': {'range': [10, 60], 'tickcolor': '#ABB8C3', 'tickwidth': 2},
            'bar': {'color': "#FFDD57"},
            'bgcolor': "#23282d",
            'borderwidth': 3,
            'bordercolor': "#5bc0eb",
            'steps': [
                {'range': [10, 25], 'color': "#39424a"},
                {'range': [25, 50], 'color': "#3a4147"},
                {'range': [50, 60], 'color': "#39424a"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 55
            }
        },
        number={'suffix': " in", 'font': {'color': '#FFDD57'}}
    ))
    fig1.update_layout(paper_bgcolor="#262b30", font={'color':'#FFDD57'})
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    fig2 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=df["Altitude"].iloc[0],
        title={'text': "<b>ALTITUDE</b><br><span style='font-size:15px'>feet</span>", 'font': {'color': '#5bc0eb'}},
        gauge={
            'axis': {'range': [0, 20000], 'tickcolor': '#ABB8C3'},
            'bar': {'color': "#5bc0eb"},
            'bgcolor': "#23282d",
            'borderwidth': 3,
            'bordercolor': "#5bc0eb"
        },
        number={'suffix': " ft", 'font': {'color': '#5bc0eb'}}
    ))
    fig2.update_layout(paper_bgcolor="#262b30", font={'color':'#5bc0eb'})
    st.plotly_chart(fig2, use_container_width=True)
with col3:
    fig3 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=df["WindSpeed"].iloc[0],
        title={'text': "<b>WIND</b><br><span style='font-size:15px'>m/s</span>", 'font': {'color': '#FFDD57'}},
        gauge={
            'axis': {'range': [0, 50], 'tickcolor': '#ABB8C3'},
            'bar': {'color': "#4682b4"},
            'bgcolor': "#23282d",
            'borderwidth': 3,
            'bordercolor': "#4682b4"
        },
        number={'suffix': " m/s", 'font': {'color': '#4682b4'}}
    ))
    fig3.update_layout(paper_bgcolor="#262b30", font={'color':'#4682b4'})
    st.plotly_chart(fig3, use_container_width=True)
with col4:
    # Airspeed: example or you could add another metric (or a logo)
    st.markdown(
        '<div style="background:#1a1d22;padding:2.6rem 1.5rem;border-radius:1.4rem;box-shadow:0 3px 10px #4682b455;">'
        '<div style="color:#5bc0eb;font-size:2.5rem;font-family:monospace;font-weight:bold;text-align:center;">'
        f'IAS<br>{np.random.randint(130,270)} <span style="font-size:1.1rem;color:#fff;">knots</span></div></div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True) # close dashboard panel

# --- Tabs for rest of dashboard ---
tab1, tab2, tab3, tab4 = st.tabs(["Snapshot", "Trends & Anomaly", "3D Path", "Settings & Export"])

with tab1:
    st.markdown("### 📋 Flight Snapshot")
    st.dataframe(df, use_container_width=True)

    def speak_turbulence_level(level):
        text = f"Current turbulence level is {level}."
        tts = gTTS(text=text, lang='en')
        with NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format="audio/mp3")
    if st.button("🔊 Speak Turbulence Level"):
        speak_turbulence_level(df["TurbulenceClass"].iloc[0])

    st.markdown("### 🗺️ Location Map")
    m = folium.Map(location=[latitude, longitude], zoom_start=6, tiles="CartoDB dark_matter")
    HeatMap([[latitude, longitude, df["TurbulenceScore"].iloc[0]]]).add_to(m)
    folium.Marker([latitude, longitude], tooltip="Current Location").add_to(m)
    st_folium(m, width=700)

    st.markdown("### 🌍 Route Visualization")
    with st.expander("Show Route Input"):
        origin_lat = st.number_input("Origin Latitude", value=37.77, key="route1")
        origin_lon = st.number_input("Origin Longitude", value=-122.42, key="route2")
        dest_lat = st.number_input("Destination Latitude", value=34.05, key="route3")
        dest_lon = st.number_input("Destination Longitude", value=-118.25, key="route4")
    def generate_route(lat1, lon1, lat2, lon2, points=10):
        lats = np.linspace(lat1, lat2, points)
        lons = np.linspace(lon1, lon2, points)
        return list(zip(lats, lons))
    route_coords = generate_route(origin_lat, origin_lon, dest_lat, dest_lon)
    route_map = folium.Map(location=[(origin_lat+dest_lat)/2, (origin_lon+dest_lon)/2], zoom_start=6, tiles="CartoDB dark_matter")
    AntPath(route_coords, color="blue").add_to(route_map)
    folium.Marker([origin_lat, origin_lon], tooltip="Origin", icon=folium.Icon(color='green')).add_to(route_map)
    folium.Marker([dest_lat, dest_lon], tooltip="Destination", icon=folium.Icon(color='red')).add_to(route_map)
    st_folium(route_map, width=700)

    st.markdown("### 🔮 Turbulence Prediction")
    try:
        model = joblib.load("model_turbulence.pkl")
        features = df[["Weight", "Arm", "WindSpeed", "Altitude"]].values
        prediction = model.predict(features)[0]
        st.success(f"Predicted Turbulence Class: {prediction}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

with tab2:
    st.markdown("### 📈 Historical Trends")
    fig_hist = px.line(historical_data, x="Time", y=["COG", "Altitude", "WindSpeed"], markers=True, color_discrete_sequence=['#5bc0eb','#4682b4','#ff5c57'])
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("### 🚨 Anomaly Detection")
    if len(historical_data) >= 10:
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

    st.markdown("### 📉 Wind Speed Trend Forecasting")
    try:
        if len(historical_data) > 12:
            forecast_data = historical_data[["Time", "WindSpeed"]].rename(columns={"Time": "ds", "WindSpeed": "y"})
            model = Prophet()
            model.fit(forecast_data)
            future = model.make_future_dataframe(periods=24, freq='H')
            forecast = model.predict(future)
            fig_forecast = px.line(forecast, x='ds', y='yhat', title="Wind Speed Forecast", color_discrete_sequence=['#4682b4'])
            fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Conf.', line=dict(dash='dot',color='#90caf9'))
            fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Conf.', line=dict(dash='dot',color='#4682b4'))
            st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            st.info("Add at least 12 points for wind speed forecasting.")
    except Exception as e:
        st.error(f"Forecasting error: {e}")

with tab3:
    st.markdown("### 🛰️ 3D Flight Path Visualization")
    try:
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=historical_data['Longitude'],
            y=historical_data['Latitude'],
            z=historical_data['Altitude'],
            mode='markers+lines',
            marker=dict(size=6, color=historical_data['WindSpeed'], colorscale='Blues', colorbar=dict(title='WindSpeed', tickcolor='#4682b4')),
            line=dict(color='#4682b4', width=3)
        )])
        fig_3d.update_layout(
            scene = dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Altitude (ft)',
                bgcolor="#23282d"
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            title="3D Flight Path: Altitude and Turbulence",
            paper_bgcolor="#262b30",
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    except Exception as e:
        st.error(f"3D Visualization error: {e}")

with tab4:
    st.markdown("### ⚙️ Settings & Data Export")
    csv = historical_data.to_csv(index=False).encode()
    st.download_button("Download Session CSV", csv, "session.csv")
    def create_pdf_report(data):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Flight Turbulence Report", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Generated on: {datetime.datetime.now()}", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=10)
        for i, row in data.tail(10).iterrows():
            text = f"Time: {row['Time']} | Lat: {row['Latitude']} | Lon: {row['Longitude']} | Wind: {row['WindSpeed']} | Alt: {row['Altitude']} | COG: {row['COG']:.2f} | Turb: {row['TurbulenceClass']}"
            pdf.multi_cell(0, 8, txt=text)
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            pdf.output(tmp_pdf.name)
            return tmp_pdf.name
    if st.button("Download PDF Report"):
        pdf_file = create_pdf_report(historical_data)
        with open(pdf_file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="flight_report.pdf">Download Report</a>'
            st.markdown(href, unsafe_allow_html=True)
    st.markdown("### App Info")
    st.markdown("""
    - Built with :blue[Streamlit] and :blue[Plotly]
    - [Prophet](https://facebook.github.io/prophet/) for forecasting
    - [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) for anomaly detection
    - [Folium](https://python-visualization.github.io/folium/) for interactive maps
    - Powered by OpenWeatherMap API
    """)
    st.markdown("---")
    st.markdown("**Developer:** Your Name Here | **Contact:** you@example.com")

# End

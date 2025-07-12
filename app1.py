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
import openai

# ----------- THEME: Blue Steel, Everything Transparent -----------
st.set_page_config(
    page_title="UMA Aviation Club Dashboard",
    page_icon="✈️",
    layout="wide"
)
st.markdown("""
<style>
/* 1. General App Styling */
.reportview-container, .stApp {
    background: linear-gradient(135deg, #d3e2f8 0%, #43688b 100%) !important;
    color: #233145 !important;
}
.sidebar .sidebar-content {
    background: #e3e9f3 !important;
}
html, body, [class*="css"] {
    font-size: 1rem !important;
}
h1, h2, h3, h4, h5 {
    color: #205080 !important;
    font-size: 2rem !important;
    font-weight: bold;
}
.stButton > button, .stDownloadButton > button {
    background-color: #205080 !important;
    color: #fff !important;
    border-radius: 8px;
}
.stDataFrame, .stTable, .element-container, .st-cg, .st-ag, .st-emotion-cache-1h9usn3 {
    background-color: transparent !important;
    color: #205080 !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    font-size: 1.12rem !important;
}
.stMarkdown p, .stMarkdown li, .stMarkdown span {
    font-size: 1.2rem !important;
}
.stAlert, .st-info, .element-container {
    font-size: 1.12rem !important;
}
.stComponent > div {
    background: transparent !important;
    box-shadow: none !important;
    border-radius: 10px !important;
}
iframe {
    background: transparent !important;
}
thead tr th {
    background: #e3e9f3 !important;
    color: #205080 !important;
}
tbody tr {
    background: rgba(67,104,139,0.09) !important;
}

/* 2. ***** THE TABS: Make font-size bigger ***** */
button[role="tab"] {
    min-height: 60px !important;
    padding: 1.2rem 2.6rem !important;
    border-radius: 18px 18px 0 0 !important;
    background: transparent !important;
}
/* The LABEL INSIDE each tab */
button[role="tab"] > div > span {
    font-size: 2.1rem !important;    /* Try 2.4rem for even larger text */
    font-weight: bold !important;
    color: #205080 !important;
    line-height: 2.2rem !important;
}
button[role="tab"][aria-selected="true"] > div > span {
    color: #f5b942 !important;
    font-size: 5rem !important;
}
button[role="tab"]:hover > div > span {
    color: #f5b942 !important;
}
button[role="tab"]:hover {
    background: #e3e9f3 !important;
    cursor: pointer;
}
button[role="tab"][aria-selected="true"] {
    border-bottom: 5px solid #f5b942 !important;
    background: #e9eef8 !important;
}
</style>
""", unsafe_allow_html=True)





# --- LOGO AND CLUB NAME ON MAIN PAGE ---
st.image("logo.png", width=150)
st.markdown("<h1 style='text-align:left; color:#205080; font-size:2rem; font-weight:900; letter-spacing:2px;'>UMA Aviation Club Dashboard</h1>", unsafe_allow_html=True)

# ------------ SIDEBAR (NO LOGO!) -------------
with st.sidebar:
    st.info(
        "🛩️ Enter your aircraft and weather data, analyze risk, "
        "forecast turbulence, and download safety reports."
    )
    st.markdown("---")
    st.markdown("""
    **Instructions:**
    - Enter flight parameters
    - Analyze & visualize results
    - Forecast turbulence
    - Download/export session
    """)
    st.markdown("---")
    user_wind_alert = st.slider("Wind Speed Alert Threshold (m/s)", 0.0, 50.0, 30.0, help="Set a custom alert for high wind speed.")
    st.markdown("---")
    st.markdown("#### Data Management")
    uploaded = st.file_uploader("Upload Previous Session", type="csv", help="Restore a saved flight session.")

    st.markdown("---")
    st.header("🤖 AI Copilot (Beta)")

    user_ai_q = st.text_area("Ask the AI Copilot anything about aviation, safety, weather, or flight planning...", key="aiq")

    if st.button("Ask AI"):
        with st.spinner("AI Copilot is thinking..."):
            try:
                openai.api_key = st.secrets["OPENAI"]["api_key"]
            except Exception:
                openai.api_key = "sk-...yourkeyhere..."
            try:
                # For best results, give the AI current session context:
                system_prompt = (
                    "You are an aviation safety assistant for pilots and dispatchers. "
                    "Answer user questions about flight planning, aviation weather, NOTAMs, safety, regulations, and club operations in a concise, helpful way. "
                    "If a question relates to the user's entered ICAO code or current weather, use that context."
                )
                # We'll pass dummy values for ICAO, wind, altitude as fallback (they update after user input)
                _icao = st.session_state.get("icao_code", "KSFO") if "icao_code" in st.session_state else "KSFO"
                _wind = st.session_state.get("wind_speed", 15.0) if "wind_speed" in st.session_state else 15.0
                _alt = st.session_state.get("altitude", 10000) if "altitude" in st.session_state else 10000
                user_context = f"ICAO code: {_icao}\nWind: {_wind} m/s\nAltitude: {_alt} ft"
                messages = [
                    {"role": "system", "content": system_prompt + "\n" + user_context},
                    {"role": "user", "content": user_ai_q},
                ]
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=400,
                    temperature=0.3
                )
                ai_answer = resp.choices[0].message.content
                st.success(ai_answer)
            except Exception as e:
                st.error(f"AI error: {e}")

# ------------ API KEY -------------
api_key = "e9e42833dd2e06259a55b7ea59429ab1"

# -------------- Inputs --------------
st.sidebar.header("🧮 Aircraft Parameters")
latitude = st.sidebar.number_input("Latitude", value=37.77, format="%.6f")
longitude = st.sidebar.number_input("Longitude", value=-122.42, format="%.6f")
weight = st.sidebar.slider("Aircraft Weight (lbs)", 1000, 10000, 5000)
arm = st.sidebar.slider("Arm Length (inches)", 10, 60, 35)

st.sidebar.header("🌦️ Environmental Conditions")
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 50.0, 15.0)
altitude = st.sidebar.slider("Altitude (feet)", 0, 20000, 10000)
use_live_wind = st.sidebar.checkbox("Use Live Wind Speed")

# ----------- LIVE AVIATION DATA INPUTS ----------
st.sidebar.header("🛩️ Airport/Route Data (Realtime)")
icao_code = st.sidebar.text_input("Airport ICAO Code", value="KSFO", max_chars=4, help="Enter ICAO (e.g., KSFO, EGLL)")

# Save context for AI Copilot (so it's always current)
st.session_state["icao_code"] = icao_code
st.session_state["wind_speed"] = wind_speed
st.session_state["altitude"] = altitude

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

# --- Delay Risk Calculation (NEW FEATURE) ---
def calculate_delay_risk(wind_speed, altitude):
    risk_score = 0.5 * (wind_speed / 50.0) + 0.5 * (1 - altitude / 20000)
    risk_score = max(0.0, min(1.0, risk_score))
    if risk_score < 0.3:
        level = "Low"
    elif risk_score < 0.7:
        level = "Medium"
    else:
        level = "High"
    return risk_score, level

df["DelayRiskScore"], df["DelayRiskLevel"] = zip(*df.apply(lambda row: calculate_delay_risk(row["WindSpeed"], row["Altitude"]), axis=1))

# --- Session data management ---
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

# --- Wind Speed Alert ---
if df["WindSpeed"].iloc[0] > user_wind_alert:
    st.error(f"🚨 ALERT: Wind speed ({df['WindSpeed'].iloc[0]:.1f} m/s) exceeds your safe threshold!")

# --- Main Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Flight Analysis", "Trends & Forecast", "3D Flight Path", "Settings & Export", "Live Aviation Data"
])

# --- Tab 1: Flight Analysis ---
with tab1:
    st.markdown("<h2 style='color:#205080;font-size:2rem;'><span style='font-size:1.5rem;'>✈️</span> Cockpit Panel</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        fig_cog = go.Figure(go.Indicator(
            mode="gauge+number", value=df["COG"].iloc[0],
            title={'text': "COG (in)"},
            gauge={
                'axis': {'range': [10, 60]},
                'bar': {'color': '#205080'},
                'bgcolor': 'rgba(67,104,139,0.10)',
                'borderwidth': 2,
                'bordercolor': "#205080",
            }
        ))
        fig_cog.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#205080"
        )
        st.plotly_chart(fig_cog, use_container_width=True)
    with col2:
        fig_alt = go.Figure(go.Indicator(
            mode="gauge+number", value=df["Altitude"].iloc[0],
            title={'text': "Altitude (ft)"},
            gauge={
                'axis': {'range': [0, 20000]},
                'bar': {'color': '#205080'},
                'bgcolor': 'rgba(67,104,139,0.10)',
                'borderwidth': 2,
                'bordercolor': "#205080",
            }
        ))
        fig_alt.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#205080"
        )
        st.plotly_chart(fig_alt, use_container_width=True)
    with col3:
        fig_wind = go.Figure(go.Indicator(
            mode="gauge+number", value=df["WindSpeed"].iloc[0],
            title={'text': "Wind Speed (m/s)"},
            gauge={
                'axis': {'range': [0, 50]},
                'bar': {'color': '#205080'},
                'bgcolor': 'rgba(67,104,139,0.10)',
                'borderwidth': 2,
                'bordercolor': "#205080",
            }
        ))
        fig_wind.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#205080"
        )
        st.plotly_chart(fig_wind, use_container_width=True)
    with col4:
        fig_delay = go.Figure(go.Indicator(
            mode="gauge+number", value=df["DelayRiskScore"].iloc[0],
            title={'text': "Delay Risk"},
            gauge={
                'axis': {'range': [0, 1], 'tickvals': [0, 0.5, 1], 'ticktext': ['Low', 'Medium', 'High']},
                'bar': {'color': '#f5b942' if df["DelayRiskLevel"].iloc[0]=="Medium" else "#d62728" if df["DelayRiskLevel"].iloc[0]=="High" else "#2ca02c"},
                'bgcolor': 'rgba(67,104,139,0.10)',
                'borderwidth': 2,
                'bordercolor': "#205080",
            }
        ))
        fig_delay.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#205080"
        )
        st.plotly_chart(fig_delay, use_container_width=True)

    # Voice Output
    def speak_turbulence_level(level):
        text = f"Current turbulence level is {level}."
        tts = gTTS(text=text, lang='en')
        with NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format="audio/mp3")
    if st.button("🔊 Speak Turbulence Level"):
        speak_turbulence_level(df["TurbulenceClass"].iloc[0])

    st.markdown("<h2 style='color:#205080;font-size:2rem;'><span style='font-size:1.3rem;'>🗂️</span> Flight Snapshot</h2>", unsafe_allow_html=True)
    st.dataframe(df[["Time","Latitude","Longitude","Weight","Arm","WindSpeed","Altitude","COG","TurbulenceScore","TurbulenceClass","DelayRiskScore","DelayRiskLevel"]], use_container_width=True)

    st.markdown("<h2 style='color:#205080;font-size:2rem;'><span style='font-size:1.3rem;'>📝</span> Risk Summary</h2>", unsafe_allow_html=True)
    turb = df["TurbulenceClass"].iloc[0]
    color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[turb]
    note = {"Low": "Safe to proceed.", "Medium": "Proceed with caution.", "High": "Delay or reroute suggested."}[turb]
    st.info(f"**Turbulence Level:** {turb} {color} | **Recommendation:** {note}")

    # Delay Risk (NEW)
    delay_level = df["DelayRiskLevel"].iloc[0]
    delay_note = {"Low": "Low delay risk.", "Medium": "Some risk of delay. Check for weather or ATC updates.", "High": "High risk of flight delay or disruption."}[delay_level]
    delay_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[delay_level]
    st.info(f"**Delay Risk:** {delay_level} {delay_color} | **Recommendation:** {delay_note}")

    st.markdown("<h2 style='color:#205080;font-size:2rem;'><span style='font-size:1.3rem;'>🗺️</span> Location Map</h2>", unsafe_allow_html=True)
    m = folium.Map(location=[latitude, longitude], zoom_start=6, tiles="CartoDB dark_matter")
    HeatMap([[latitude, longitude, df["TurbulenceScore"].iloc[0]]]).add_to(m)
    folium.Marker([latitude, longitude], tooltip="Current Location").add_to(m)
    st_folium(m, width=900, height=500)

    # Route Input & Visualization
    st.markdown("<h2 style='color:#205080;font-size:2rem;'><span style='font-size:1.3rem;'>🌍</span> Route Visualization</h2>", unsafe_allow_html=True)
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
    route_map = folium.Map(location=[(origin_lat+dest_lat)/2, (origin_lon+dest_lon)/2], zoom_start=6, tiles="CartoDB dark_matter")
    AntPath(route_coords, color="blue").add_to(route_map)
    folium.Marker([origin_lat, origin_lon], tooltip="Origin", icon=folium.Icon(color='green')).add_to(route_map)
    folium.Marker([dest_lat, dest_lon], tooltip="Destination", icon=folium.Icon(color='red')).add_to(route_map)
    st_folium(route_map, width=900, height=500)

    # --- Turbulence Prediction (ML Model) ---
    st.markdown("<h2 style='color:#205080;font-size:2rem;'><span style='font-size:1.3rem;'>🔮</span> Turbulence Prediction</h2>", unsafe_allow_html=True)
    try:
        model = joblib.load("model_turbulence.pkl")
        features = df[["Weight", "Arm", "WindSpeed", "Altitude"]].values
        prediction = model.predict(features)[0]
        st.success(f"Predicted Turbulence Class: {prediction}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# --- Tab 2: Trends & Forecast ---
with tab2:
    st.markdown("<h2 style='color:#205080;font-size:2rem;'>📈 Historical Trends</h2>", unsafe_allow_html=True)
    fig_hist = px.line(historical_data, x="Time", y=["COG", "Altitude", "WindSpeed"], markers=True,
                       color_discrete_map={"COG": "#205080", "Altitude": "#5e7fa5", "WindSpeed": "#f5b942"})
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#205080"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("<h2 style='color:#205080;font-size:2rem;'>🚨 Anomaly Detection</h2>", unsafe_allow_html=True)
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

    st.markdown("<h2 style='color:#205080;font-size:2rem;'>📉 Wind Speed Trend Forecasting</h2>", unsafe_allow_html=True)
    try:
        if len(historical_data) > 12:
            forecast_data = historical_data[["Time", "WindSpeed"]].rename(columns={"Time": "ds", "WindSpeed": "y"})
            model = Prophet()
            model.fit(forecast_data)
            future = model.make_future_dataframe(periods=24, freq='H')
            forecast = model.predict(future)
            fig_forecast = px.line(forecast, x='ds', y='yhat', title="Wind Speed Forecast", color_discrete_sequence=["#205080"])
            fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Conf.', line=dict(color="#f5b942"))
            fig_forecast.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Conf.', line=dict(color="#5e7fa5"))
            fig_forecast.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#205080"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            st.info("Add at least 12 points for wind speed forecasting.")
    except Exception as e:
        st.error(f"Forecasting error: {e}")

# --- Tab 3: 3D Flight Path ---
with tab3:
    st.markdown("<h2 style='color:#205080;font-size:2rem;'>🛰️ 3D Flight Path Visualization</h2>", unsafe_allow_html=True)
    try:
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=historical_data['Longitude'],
            y=historical_data['Latitude'],
            z=historical_data['Altitude'],
            mode='markers+lines',
            marker=dict(size=5, color=historical_data['WindSpeed'], colorscale='Blues', colorbar=dict(title='WindSpeed'))
        )])
        fig_3d.update_layout(
            scene = dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Altitude (ft)'
            ),
            title="3D Flight Path: Altitude and Turbulence",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#205080"
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    except Exception as e:
        st.error(f"3D Visualization error: {e}")

# --- Tab 4: Settings & Export ---
with tab4:
    st.markdown("<h2 style='color:#205080;font-size:2rem;'>⚙️ Settings & Data Export</h2>", unsafe_allow_html=True)
    # Save session
    csv = historical_data.to_csv(index=False).encode()
    st.download_button("Download Session CSV", csv, "session.csv")
    # PDF report
    def create_pdf_report(data):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Flight Turbulence Report", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Generated on: {datetime.datetime.now()}", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=10)
        for i, row in data.tail(10).iterrows():
            text = f"Time: {row['Time']} | Lat: {row['Latitude']} | Lon: {row['Longitude']} | Wind: {row['WindSpeed']} | Alt: {row['Altitude']} | COG: {row['COG']:.2f} | Turb: {row['TurbulenceClass']} | DelayRisk: {row['DelayRiskLevel']}"
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
    - Built with :blue[Streamlit] and :orange[Plotly]
    - [Prophet](https://facebook.github.io/prophet/) for forecasting
    - [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) for anomaly detection
    - [Folium](https://python-visualization.github.io/folium/) for interactive maps
    - Powered by OpenWeatherMap API
    """)
    st.markdown("---")
    st.markdown("**Developer:** Uma Mannan | **Contact:** umamannan16@gmail.com")

# --- Tab 5: Live Aviation Data ---
def get_metar_taf(icao):
    headers = {"Accept": "application/json"}
    metar_url = f"https://avwx.rest/api/metar/{icao}?options=info,translate"
    taf_url = f"https://avwx.rest/api/taf/{icao}?options=info,translate"
    try:
        metar = requests.get(metar_url, headers=headers, timeout=5).json()
        taf = requests.get(taf_url, headers=headers, timeout=5).json()
        return metar, taf
    except Exception as e:
        return None, None

def get_notams(icao):
    url = f"https://notaminfo.com/api/airport/{icao}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            return data.get("notams", [])[:5]
        else:
            return []
    except Exception as e:
        return []

with tab5:
    st.markdown("<h2 style='color:#205080;font-size:2rem;'>🛰️ Live Aviation Data</h2>", unsafe_allow_html=True)
    st.info(f"Showing METAR/TAF and NOTAMs for airport: **{icao_code.upper()}**")
    metar, taf = get_metar_taf(icao_code)
    notams = get_notams(icao_code)

    # METAR
    if metar and "raw" in metar:
        st.subheader("METAR (Current Weather)")
        st.write(metar["raw"])
        if "translate" in metar and metar["translate"]:
            st.success(metar["translate"]["summary"])
    else:
        st.warning("No METAR data found for this ICAO code.")

    # TAF
    if taf and "raw" in taf:
        st.subheader("TAF (Forecast)")
        st.write(taf["raw"])
        if "translate" in taf and taf["translate"]:
            st.info(taf["translate"]["summary"])
    else:
        st.warning("No TAF data found for this ICAO code.")

    # NOTAMs
    st.subheader("Recent NOTAMs")
    if notams:
        for i, n in enumerate(notams, 1):
            st.write(f"{i}. {n.get('text', '')}")
    else:
        st.info("No NOTAMs found or this is not a US airport.")

# --- End ---

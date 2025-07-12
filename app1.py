import streamlit as st
import pandas as pd
import numpy as np
import datetime
import requests
import folium
from folium.plugins import AntPath
from folium.features import CustomIcon
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
import base64

# --- MODERN CUSTOM THEME ---
st.set_page_config(
    page_title="UMA Aviation Club Dashboard",
    page_icon="✈️",
    layout="wide"
)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
        }
        .reportview-container, .stApp {
            background: linear-gradient(135deg, #dde6f8 0%, #233145 100%) !important;
            color: #233145 !important;
        }
        .sidebar .sidebar-content {
            background: rgba(255,255,255,0.85) !important;
            border-radius: 1.2rem;
            box-shadow: 0 4px 24px rgba(67,104,139,0.08);
        }
        h1, h2, h3, h4, h5, .stTabs [data-baseweb="tab"] {
            color: #133B5C !important;
            font-weight: 700;
            letter-spacing: 0.5px;
        }
        .stButton > button, .stDownloadButton > button {
            background: linear-gradient(90deg, #205080 60%, #5e7fa5 100%);
            color: #fff !important;
            border-radius: 12px;
            padding: 0.6rem 1.5rem;
            transition: box-shadow 0.2s;
            box-shadow: 0 4px 16px rgba(33,64,104,0.13);
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            box-shadow: 0 6px 24px rgba(33,64,104,0.17);
            filter: brightness(1.08);
        }
        .stDataFrame, .stTable, .element-container, .st-cg, .st-ag, .st-emotion-cache-1h9usn3 {
            background: rgba(255,255,255,0.93) !important;
            color: #205080 !important;
            border-radius: 20px !important;
            box-shadow: 0 2px 12px rgba(33,64,104,0.08);
            padding: 1rem;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: #e9eef8 !important;
            color: #f5b942 !important;
            font-weight: bold;
            border-bottom: 3px solid #f5b942;
            border-radius: 1.5rem 1.5rem 0 0;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 1.5rem 1.5rem 0 0;
            padding: 0.6rem 2rem;
            margin-right: 0.2rem;
            background: rgba(255,255,255,0.75);
        }
        .stAlert, .st-info {
            background: linear-gradient(90deg,#d6e7fa 80%, #e9eef8 100%) !important;
            color: #205080 !important;
            border-radius: 16px !important;
            box-shadow: 0 1px 8px rgba(33,64,104,0.10);
        }
        .stCard, .st-cg, .stContainer, .stMarkdown, .stComponent > div {
            border-radius: 22px !important;
            box-shadow: 0 4px 20px rgba(33,64,104,0.09);
        }
    </style>
""", unsafe_allow_html=True)

# --- CENTERED LOGO & TITLE ---
col_logo, col_title = st.columns([1,6])
with col_logo:
    st.image("logo.png", width=82)
with col_title:
    st.markdown("<h1 style='color:#205080; font-size:2.2rem; font-weight:900; letter-spacing:1.2px; margin-top:1rem'>UMA Aviation Club Dashboard</h1>", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.info(
        "🛩️ Enter aircraft/weather data, analyze risk, visualize flights, and download safety reports."
    )
    st.markdown("---")
    st.markdown("**Instructions:**\n- Enter flight parameters\n- Analyze & visualize results\n- Download/export session\n")
    st.markdown("---")
    user_wind_alert = st.slider("Wind Speed Alert Threshold (m/s)", 0.0, 50.0, 30.0)
    st.markdown("---")
    st.markdown("#### Data Management")
    uploaded = st.file_uploader("Upload Previous Session", type="csv", help="Restore a saved flight session.")

# --- INPUTS ---
st.sidebar.header("🧮 Aircraft Parameters")
latitude = st.sidebar.number_input("Latitude", value=37.77, format="%.6f")
longitude = st.sidebar.number_input("Longitude", value=-122.42, format="%.6f")
weight = st.sidebar.slider("Aircraft Weight (lbs)", 1000, 10000, 5000)
arm = st.sidebar.slider("Arm Length (inches)", 10, 60, 35)

st.sidebar.header("🌦️ Environmental Conditions")
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 50.0, 15.0)
altitude = st.sidebar.slider("Altitude (feet)", 0, 20000, 10000)
use_live_wind = st.sidebar.checkbox("Use Live Wind Speed")

st.sidebar.header("🛩️ Airport/Route Data (Realtime)")
icao_code = st.sidebar.text_input("Airport ICAO Code", value="KSFO", max_chars=4)

# --- LIVE WEATHER (optional) ---
api_key = "e9e42833dd2e06259a55b7ea59429ab1"
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

# --- DATA PREP ---
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

# --- WIND SPEED ALERT ---
if df["WindSpeed"].iloc[0] > user_wind_alert:
    st.error(f"🚨 ALERT: Wind speed ({df['WindSpeed'].iloc[0]:.1f} m/s) exceeds your safe threshold!")

# --- MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Flight Analysis", "Trends & Maps", "Route & 3D", "Export & Info"
])

# --- TAB 1: FLIGHT ANALYSIS ---
with tab1:
    st.markdown("<h2 style='color:#205080;font-size:2rem;'><span style='font-size:2rem;'>✈️</span> Cockpit Panel</h2>", unsafe_allow_html=True)
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

    st.markdown("<h2 style='color:#205080;font-size:2rem;'><span style='font-size:1.3rem;'>🗂️</span> Flight Snapshot</h2>", unsafe_allow_html=True)
    st.dataframe(df[["Time","Latitude","Longitude","Weight","Arm","WindSpeed","Altitude","COG","TurbulenceScore","TurbulenceClass","DelayRiskScore","DelayRiskLevel"]], use_container_width=True)

    st.markdown("<h2 style='color:#205080;font-size:2rem;'><span style='font-size:1.3rem;'>📝</span> Risk Summary</h2>", unsafe_allow_html=True)
    turb = df["TurbulenceClass"].iloc[0]
    color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[turb]
    note = {"Low": "Safe to proceed.", "Medium": "Proceed with caution.", "High": "Delay or reroute suggested."}[turb]
    st.info(f"**Turbulence Level:** {turb} {color} | **Recommendation:** {note}")

    delay_level = df["DelayRiskLevel"].iloc[0]
    delay_note = {"Low": "Low delay risk.", "Medium": "Some risk of delay. Check for weather or ATC updates.", "High": "High risk of flight delay or disruption."}[delay_level]
    delay_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}[delay_level]
    st.info(f"**Delay Risk:** {delay_level} {delay_color} | **Recommendation:** {delay_note}")

# --- TAB 2: TRENDS & MAPS ---
with tab2:
    st.markdown("<h2 style='color:#205080;font-size:2rem;'>📈 Historical Trends</h2>", unsafe_allow_html=True)
    fig_hist = px.line(
        historical_data, 
        x="Time", 
        y=["COG", "Altitude", "WindSpeed"], 
        markers=True,
        color_discrete_map={"COG": "#205080", "Altitude": "#5e7fa5", "WindSpeed": "#f5b942"}
    )
    fig_hist.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#205080"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("<h2 style='color:#205080;font-size:2rem;'><span style='font-size:1.3rem;'>🗺️</span> Location Map</h2>", unsafe_allow_html=True)
    m = folium.Map(
        location=[latitude, longitude], 
        zoom_start=7, 
        tiles="Stamen Terrain",
        control_scale=True
    )
    folium.CircleMarker(
        [latitude, longitude],
        radius=18,
        color="#133B5C",
        fill=True,
        fill_color="#f5b942" if df["TurbulenceClass"].iloc[0]=="High" else "#5e7fa5" if df["TurbulenceClass"].iloc[0]=="Medium" else "#41d67c",
        fill_opacity=0.7,
        tooltip=f"Turbulence: {df['TurbulenceClass'].iloc[0]}"
    ).add_to(m)
    try:
        plane_icon = CustomIcon("plane-icon.png", icon_size=(32,32))
        folium.Marker([latitude, longitude], icon=plane_icon, tooltip="Current Location").add_to(m)
    except Exception:
        folium.Marker([latitude, longitude], tooltip="Current Location").add_to(m)
    st_folium(m, width=880, height=420)

# --- TAB 3: ROUTE & 3D MAP ---
with tab3:
    st.markdown("<h2 style='color:#205080;font-size:2rem;'><span style='font-size:1.3rem;'>🌍</span> Route Visualization</h2>", unsafe_allow_html=True)
    with st.expander("Show Route Input"):
        origin_lat = st.number_input("Origin Latitude", value=37.77, key='orig_lat')
        origin_lon = st.number_input("Origin Longitude", value=-122.42, key='orig_lon')
        dest_lat = st.number_input("Destination Latitude", value=34.05, key='dest_lat')
        dest_lon = st.number_input("Destination Longitude", value=-118.25, key='dest_lon')
    def generate_route(lat1, lon1, lat2, lon2, points=10):
        lats = np.linspace(lat1, lat2, points)
        lons = np.linspace(lon1, lon2, points)
        return list(zip(lats, lons))
    route_coords = generate_route(origin_lat, origin_lon, dest_lat, dest_lon)
    route_map = folium.Map(
        location=[(origin_lat+dest_lat)/2, (origin_lon+dest_lon)/2], 
        zoom_start=6, 
        tiles="Stamen Terrain",
        control_scale=True
    )
    AntPath(route_coords, color="#205080", weight=6, dash_array=[15,10], delay=800).add_to(route_map)
    folium.Marker([origin_lat, origin_lon], tooltip="Origin", icon=folium.Icon(color='green', icon='play')).add_to(route_map)
    folium.Marker([dest_lat, dest_lon], tooltip="Destination", icon=folium.Icon(color='red', icon='flag')).add_to(route_map)
    st_folium(route_map, width=880, height=420)

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
            title="3D Flight Path: Altitude and Wind",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#205080"
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    except Exception as e:
        st.error(f"3D Visualization error: {e}")

# --- TAB 4: EXPORT & INFO ---
with tab4:
    st.markdown("<h2 style='color:#205080;font-size:2rem;'>⚙️ Data Export & App Info</h2>", unsafe_allow_html=True)
    # Save session
    csv = historical_data.to_csv(index=False).encode()
    st.download_button("Download Session CSV", csv, "session.csv")
    # PDF export (simple, just for recent records)
    def create_pdf_report(data):
        from fpdf import FPDF
        from tempfile import NamedTemporaryFile
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
    - :earth_africa: Interactive maps via Folium
    - Powered by OpenWeatherMap API
    """)
    st.markdown("---")
    st.markdown("**Developer:** Uma Mannan | **Contact:** umamannan16@gmail.com")

# --- END ---

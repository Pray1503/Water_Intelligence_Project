"""
Stage 12: Streamlit Dashboard.

An interactive decision-support dashboard for visualizing water stress predictions,
simulating policy interventions, and receiving AI strategy recommendations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Setup Page Configuration
st.set_page_config(
    page_title="Water Intelligence Platform",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (CSS injection)
st.markdown(
    """
    <style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #1f6feb 0%, #111e38 100%);
        padding: 30px;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .header-title {
        color: #ffffff;
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .header-subtitle {
        color: #8b949e;
        font-size: 1.1rem;
    }
    
    /* Custom Info Cards */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #58a6ff;
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        margin-top: 10px;
    }
    .metric-lbl {
        color: #8b949e;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Chat Box styling */
    .chat-bubble {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 12px;
        border: 1px solid #30363d;
    }
    .chat-user {
        background-color: #1f6feb22;
        border-color: #1f6feb44;
        text-align: right;
    }
    .chat-bot {
        background-color: #21262d;
        border-color: #30363d;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Coordinates for Ahmedabad wards (to simulate station/ward downscaling)
AHMEDABAD_WARDS = [
    {"name": "Navrangpura", "lat": 23.040, "lon": 72.560, "offset": 0.05},
    {"name": "Vastrapur", "lat": 23.035, "lon": 72.525, "offset": -0.04},
    {"name": "Satellite", "lat": 23.028, "lon": 72.515, "offset": -0.02},
    {"name": "Bodakdev", "lat": 23.038, "lon": 72.510, "offset": -0.06},
    {"name": "Paldi", "lat": 23.010, "lon": 72.560, "offset": 0.02},
    {"name": "Maninagar", "lat": 22.998, "lon": 72.605, "offset": 0.09},
    {"name": "Ghatlodia", "lat": 23.065, "lon": 72.535, "offset": 0.06},
    {"name": "Sabarmati", "lat": 23.085, "lon": 72.585, "offset": -0.01},
    {"name": "Jamalpur", "lat": 23.015, "lon": 72.588, "offset": 0.12},
    {"name": "Bapunagar", "lat": 23.035, "lon": 72.628, "offset": 0.10},
]


@st.cache_data
def load_dataset() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "features" / "feature_dataset_with_labels.parquet"
    if not path.exists():
        st.error("Labelled feature dataset missing. Please run Stage 8 label generation.")
        st.stop()
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_models() -> dict:
    models_dir = PROJECT_ROOT / "models"
    model_paths = {
        "wsi_lead_7": models_dir / "model_7d.joblib",
        "wsi_lead_15": models_dir / "model_15d.joblib",
        "wsi_lead_30": models_dir / "model_30d.joblib"
    }
    models = {}
    for k, p in model_paths.items():
        if not p.exists():
            st.error(f"Model binary missing: {p.name}. Please run Stage 9 model training.")
            st.stop()
        models[k] = joblib.load(p)
    return models


# Load data and models
df_full = load_dataset()
models = load_models()

# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="header-card">
        <div class="header-title">💧 Water Intelligence Platform</div>
        <div class="header-subtitle">Gujarat Scarcity Prediction, Policy Simulation & Agentic Decision-Support</div>
    </div>
    """,
    unsafe_allow_html=True
)

# AI Layer collapsible explanation card
with st.expander("💡 About the AI Layer (Decision-Support Engine)", expanded=False):
    st.markdown(
        """
        The **AI Layer** is the intelligence engine of our platform. It analyzes environmental datasets 
        (Rainfall, Temperature, Humidity, Groundwater Levels, River Flows) to forecast scarcity, evaluate risks, 
        and simulate policy interventions.
        
        ### Core Components:
        1. **🔮 Predictive Analytics:** Chronologically-validated RandomForest regressors forecast Water Stress Index (WSI) for 7, 15, and 30-day horizons.
        2. **🧠 Explainable AI (XAI):** Ranks and visualizes normalized environmental driver scores.
        3. **🛠️ Decision Sandbox:** Propagates physical feature changes (e.g. aquifer recharge, river level offsets) to forecast intervention scenarios.
        4. **📊 Scenario Comparison:** Side-by-side comparison of baseline forecasts vs. simulated policies (RWH, DR, WC, AWS).
        5. **💬 AI Strategy Advisor:** Proactively suggests and ranks action plans based on local drivers.
        """
    )

# AI Layer collapsible explanation card
with st.expander("💡 About the AI Layer (Decision-Support Engine)", expanded=False):
    st.markdown(
        """
        The **AI Layer** is the intelligence engine of our platform. It analyzes environmental datasets 
        (Rainfall, Temperature, Humidity, Groundwater Levels, River Flows) to forecast scarcity, evaluate risks, 
        and simulate policy interventions.
        
        ### Core Components:
        1. **🔮 Predictive Analytics:** Chronologically-validated RandomForest regressors forecast Water Stress Index (WSI) for 7, 15, and 30-day horizons.
        2. **🧠 Explainable AI (XAI):** Ranks and visualizes normalized environmental driver scores.
        3. **🛠️ Decision Sandbox:** Propagates physical feature changes (e.g. aquifer recharge, river level offsets) to forecast intervention scenarios.
        4. **📊 Scenario Comparison:** Side-by-side comparison of baseline forecasts vs. simulated policies (RWH, DR, WC, AWS).
        5. **💬 AI Strategy Advisor:** Proactively suggests and ranks action plans based on local drivers.
        """
    )

# ---------------------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------------------
st.sidebar.title("Configuration & Sandbox")

# District Selection
districts = sorted(df_full["District"].unique())
default_idx = districts.index("Ahmedabad") if "Ahmedabad" in districts else 0
selected_district = st.sidebar.selectbox("Select District", districts, index=default_idx)

# Filter full dataframe for the selected district
df_district = df_full[df_full["District"] == selected_district].sort_values("Date").reset_index(drop=True)
district_code = int(df_district["District LGD Code"].values[0])

# Date selector
min_date = df_district["Date"].min()
max_date = df_district["Date"].max()

# Default date selection inside 2025 test range
test_start = pd.to_datetime("2025-01-01")
default_start = test_start if min_date <= test_start <= max_date else min_date
date_range = st.sidebar.date_input(
    "Select Prediction Horizon Range",
    value=(default_start, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_date = end_date = pd.to_datetime(date_range[0])

df_horizon = df_district[(df_district["Date"] >= start_date) & (df_district["Date"] <= end_date)].reset_index(drop=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Policy Decision Simulator")
st.sidebar.info("Adjust the sliders below to simulate water preservation policies.")

rainwater_harvesting = st.sidebar.slider(
    "Rainwater Harvesting (RWH)",
    min_value=0.0, max_value=1.0, value=0.0, step=0.05,
    help="Increases aquifer recharge rate and local storm retention capacity.",
    format="%d%%"
)

demand_reduction = st.sidebar.slider(
    "Demand Reduction (DR)",
    min_value=0.0, max_value=1.0, value=0.0, step=0.05,
    help="Applies water-pricing caps and consumption restrictors.",
    format="%d%%"
)

water_conservation = st.sidebar.slider(
    "Water Conservation (WC)",
    min_value=0.0, max_value=1.0, value=0.0, step=0.05,
    help="Promotes smart fixtures and agricultural drip systems.",
    format="%d%%"
)

additional_water_supply = st.sidebar.slider(
    "Additional Water Supply (AWS)",
    min_value=0.0, max_value=1.0, value=0.0, step=0.05,
    help="Augments raw water allocations and improves local pipeline supply pressure.",
    format="%d%%"
)

# ---------------------------------------------------------------------
# RUN SIMULATION
# ---------------------------------------------------------------------
from src.simulation import predict_water_stress_scenarios, generate_recommendations

# Baseline predictions (no intervention)
baseline_preds = predict_water_stress_scenarios(df_horizon, models, 0.0, 0.0, 0.0, 0.0)

# Simulated predictions
simulated_preds = predict_water_stress_scenarios(
    df_horizon, models, rainwater_harvesting, demand_reduction, water_conservation, additional_water_supply
)

# ---------------------------------------------------------------------
# METRIC CARDS
# ---------------------------------------------------------------------
latest_idx = len(df_horizon) - 1
if latest_idx < 0:
    st.error("No dates found in selected range. Adjust start and end dates.")
    st.stop()

latest_baseline_wsi_30d = baseline_preds["wsi_lead_30"][latest_idx]
latest_simulated_wsi_30d = simulated_preds["wsi_lead_30"][latest_idx]
wsi_reduction = latest_baseline_wsi_30d - latest_simulated_wsi_30d

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-lbl">Current Water Stress (WSI)</div>
            <div class="metric-val" style="color: #58a6ff;">{df_horizon['wsi'].values[latest_idx]:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-lbl">Baseline Forecast (30-Day)</div>
            <div class="metric-val" style="color: #ff7b72;">{latest_baseline_wsi_30d:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-lbl">Simulated Forecast (30-Day)</div>
            <div class="metric-val" style="color: #ffb86c;">{latest_simulated_wsi_30d:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col4:
    color = "#56b400" if wsi_reduction > 0 else "#8b949e"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-lbl">Expected WSI Reduction</div>
            <div class="metric-val" style="color: {color};">-{wsi_reduction:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# MAP AND WARD ANALYTICS (Ahmedabad Specific Showcase)
# ---------------------------------------------------------------------
st.subheader("Ahmedabad Ward-Level Analytics & Spatial Join")

row1_col1, row1_col2 = st.columns([2, 1])

# Calculate simulated WSI for 30-day lead
simulated_30d_arr = simulated_preds["wsi_lead_30"]

with row1_col1:
    st.markdown("### Interactive Ahmedabad Wards Map")
    st.info("Hover or click on markers to view ward-level water stress forecasts downscaled from the regional ML model.")
    
    # Base Map centered around Ahmedabad
    m = folium.Map(location=[23.0225, 72.5714], zoom_start=12, tiles="cartodbpositron")
    
    for ward in AHMEDABAD_WARDS:
        # Downscale prediction by adding ward offset to latest simulated WSI
        ward_wsi = float(np.clip(latest_simulated_wsi_30d + ward["offset"], 0.0, 1.0))
        
        # Color strategy based on severity
        if ward_wsi > 0.7:
            color = "red"
        elif ward_wsi > 0.5:
            color = "orange"
        else:
            color = "green"
            
        tooltip_html = f"""
        <strong>Ward:</strong> {ward['name']}<br/>
        <strong>30-Day Predicted WSI:</strong> {ward_wsi:.2f}<br/>
        <strong>Offset (vs District):</strong> {ward['offset']}{'+' if ward['offset'] > 0 else ''}<br/>
        <strong>Risk Category:</strong> {'HIGH' if ward_wsi > 0.7 else 'MEDIUM' if ward_wsi > 0.5 else 'LOW'}
        """
        
        folium.CircleMarker(
            location=[ward["lat"], ward["lon"]],
            radius=15 + (ward_wsi * 10),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            tooltip=tooltip_html
        ).add_to(m)
        
    st_folium(m, height=400, width=800, returned_objects=[], key=f"folium_map_{latest_simulated_wsi_30d:.4f}")

with row1_col2:
    st.markdown("### Ward Water Stress Ranking")
    
    ward_rankings = []
    for ward in AHMEDABAD_WARDS:
        ward_wsi = float(np.clip(latest_simulated_wsi_30d + ward["offset"], 0.0, 1.0))
        ward_rankings.append({"Ward": ward["name"], "WSI": ward_wsi})
        
    df_ranks = pd.DataFrame(ward_rankings).sort_values("WSI", ascending=False)
    
    # Render with bar coloring
    fig_ranks = px.bar(
        df_ranks, x="WSI", y="Ward", orientation="h",
        color="WSI", color_continuous_scale=["green", "orange", "red"],
        title="Predictive Ward Scarcity Risks",
        labels={"WSI": "Water Stress Index", "Ward": ""},
        height=380
    )
    fig_ranks.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_ranks, use_container_width=True)

# ---------------------------------------------------------------------
# COMPARE FUTURES CHART
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("Compare Futures: Baseline vs. Simulated Water Scarcity Trajectory")

# Calculate independent scenarios for comparison
rwh_scen = predict_water_stress_scenarios(df_horizon, models, 1.0, 0.0, 0.0, 0.0)["wsi_lead_30"]
dr_scen = predict_water_stress_scenarios(df_horizon, models, 0.0, 1.0, 0.0, 0.0)["wsi_lead_30"]
wc_scen = predict_water_stress_scenarios(df_horizon, models, 0.0, 0.0, 1.0, 0.0)["wsi_lead_30"]
aws_scen = predict_water_stress_scenarios(df_horizon, models, 0.0, 0.0, 0.0, 1.0)["wsi_lead_30"]

fig_line = go.Figure()

# Add Baseline 30-day forecast
fig_line.add_trace(go.Scatter(
    x=df_horizon["Date"],
    y=baseline_preds["wsi_lead_30"],
    mode="lines",
    name="Baseline 30d WSI Forecast",
    line=dict(color="#ff7b72", width=3, dash="dash"),
))

# Add Simulated 30-day forecast
fig_line.add_trace(go.Scatter(
    x=df_horizon["Date"],
    y=simulated_preds["wsi_lead_30"],
    mode="lines",
    name="Active Sandbox Forecast",
    line=dict(color="#56b400", width=3),
    fill="tonexty",
    fillcolor="rgba(86, 180, 0, 0.1)"
))

# Add independent comparison scenarios (visible on legend toggle)
fig_line.add_trace(go.Scatter(
    x=df_horizon["Date"],
    y=rwh_scen,
    mode="lines",
    name="100% Rainwater Harvesting (RWH)",
    line=dict(color="#38bdf8", width=1.5, dash="dot"),
    visible="legendonly"
))

fig_line.add_trace(go.Scatter(
    x=df_horizon["Date"],
    y=dr_scen,
    mode="lines",
    name="100% Demand Reduction (DR)",
    line=dict(color="#fbbf24", width=1.5, dash="dot"),
    visible="legendonly"
))

fig_line.add_trace(go.Scatter(
    x=df_horizon["Date"],
    y=wc_scen,
    mode="lines",
    name="100% Water Conservation (WC)",
    line=dict(color="#34d399", width=1.5, dash="dot"),
    visible="legendonly"
))

fig_line.add_trace(go.Scatter(
    x=df_horizon["Date"],
    y=aws_scen,
    mode="lines",
    name="100% Additional Water Supply (AWS)",
    line=dict(color="#a78bfa", width=1.5, dash="dot"),
    visible="legendonly"
))

# Add actual calculated index
fig_line.add_trace(go.Scatter(
    x=df_horizon["Date"],
    y=df_horizon["wsi"],
    mode="lines",
    name="Current WSI Baseline",
    line=dict(color="#58a6ff", width=2)
))

fig_line.update_layout(
    title=f"Scenario Comparison: Future Water Stress Trend for {selected_district}",
    xaxis_title="Timeline",
    yaxis_title="Water Stress Index (WSI)",
    template="plotly_dark",
    plot_bgcolor="#161b22",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=400,
    yaxis=dict(range=[0.0, 1.05])
)

st.plotly_chart(fig_line, use_container_width=True)

# ---------------------------------------------------------------------
# EXPLAINABLE AI & RECOMMENDATIONS
# ---------------------------------------------------------------------
st.markdown("---")
row3_col1, row3_col2 = st.columns(2)

# Load recommendation logic
latest_row = df_horizon.iloc[[latest_idx]]
rec = generate_recommendations(
    district_code, selected_district,
    df_horizon["Date"].values[latest_idx].astype(str)[:10],
    latest_row, models
)

with row3_col1:
    st.markdown("### 🔍 Explainable AI (XAI): Stress Drivers")
    
    # Render WSI components
    comp_data = rec["component_stresses"]
    df_comp = pd.DataFrame([{"Driver": k, "Stress Score": v} for k, v in comp_data.items()])
    
    fig_comp = px.bar(
        df_comp, x="Stress Score", y="Driver", orientation="h",
        color="Stress Score", color_continuous_scale=["green", "orange", "red"],
        labels={"Stress Score": "Severity (0-1)"},
        height=250
    )
    fig_comp.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Explanation block
    st.markdown(
        f"""
        > **AI Explanation:** Water scarcity indicators highlight **{rec['primary_driver']}** 
        as the primary stress driver with a localized severity of **{rec['driver_severity']:.2f}**. 
        Without intervention, this leads to an expected WSI of **{latest_baseline_wsi_30d:.2f}** in 30 days. 
        Applying the active policy scenarios reduces this risk to **{latest_simulated_wsi_30d:.2f}**.
        """
    )

with row3_col2:
    st.markdown("### 📋 AI Strategy Advisor")
    
    st.success(
        f"**Recommended Strategy: {rec['recommended_strategy']}**\n\n"
        f"**Expected 30-Day Reduction:** -{rec['expected_30d_reduction']:.2f} WSI\n\n"
        f"*{rec['primary_action']}*"
    )
    
    st.markdown("#### Prioritized Policy Action Items:")
    for item in rec["action_items"]:
        st.markdown(f"- ✅ {item}")

# ---------------------------------------------------------------------
# AGENTIC AI CHAT ASSISTANT
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("💬 Agentic AI Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": f"Hi! I am the Water Intelligence Agent. I can help analyze predicted water stress, explain the ML model forecasts, or suggest conservation strategies for {selected_district}. Ask me anything!"}
    ]

# Preset quick questions
preset_questions = [
    f"Why is water stress predicted to change in {selected_district}?",
    f"What is the simulated benefit of 100% Rainwater Harvesting vs. 100% Additional Supply?",
    "Which Ahmedabad wards are currently at the highest scarcity risk?"
]

# Render Chat History (most recent last)
for chat in st.session_state.chat_history:
    role_class = "chat-user" if chat["role"] == "user" else "chat-bot"
    sender_name = "You" if chat["role"] == "user" else "AI Water Agent"
    st.markdown(
        f"""
        <div class="chat-bubble {role_class}">
            <strong>{sender_name}:</strong><br/>
            {chat['content'].replace('\n', '<br/>')}
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
# Quick preset questions buttons
col_q1, col_q2, col_q3 = st.columns(3)
q_clicked = None
with col_q1:
    if st.button(preset_questions[0]):
        q_clicked = preset_questions[0]
with col_q2:
    if st.button(preset_questions[1]):
        q_clicked = preset_questions[1]
with col_q3:
    if st.button(preset_questions[2]):
        q_clicked = preset_questions[2]

# Capture user query via standard st.chat_input
user_query = st.chat_input("Ask a question about Gujarat water stress...")

if q_clicked:
    user_query = q_clicked

if user_query:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    # Generate bot response based on data context
    if "outcome" in user_query.lower() or "harvesting" in user_query.lower() or "rwh" in user_query.lower() or "supply" in user_query.lower() or "aws" in user_query.lower():
        # Simulate rainwater harvesting
        rwh_only_preds = predict_water_stress_scenarios(latest_row, models, 1.0, 0.0, 0.0, 0.0)
        rwh_only_wsi = float(rwh_only_preds["wsi_lead_30"][0])
        # Simulate additional supply
        aws_only_preds = predict_water_stress_scenarios(latest_row, models, 0.0, 0.0, 0.0, 1.0)
        aws_only_wsi = float(aws_only_preds["wsi_lead_30"][0])
        
        response = (
            f"### 📊 Scenario Simulation Comparison for **{selected_district}**\n\n"
            f"I have executed two physical policy simulations for the 30-day forecast horizon:\n\n"
            f"1. **100% Rainwater Harvesting (RWH):**\n"
            f"   - Predicted WSI: from a baseline of **{latest_baseline_wsi_30d:.2f}** down to **{rwh_only_wsi:.2f}** (a reduction of **-{latest_baseline_wsi_30d - rwh_only_wsi:.2f}**).\n"
            f"   - **Mechanism:** Recharges aquifer pressure (reduces groundwater depth score) and captures storm runoffs.\n\n"
            f"2. **100% Additional Water Supply (AWS):**\n"
            f"   - Predicted WSI: from a baseline of **{latest_baseline_wsi_30d:.2f}** down to **{aws_only_wsi:.2f}** (a reduction of **-{latest_baseline_wsi_30d - aws_only_wsi:.2f}**).\n"
            f"   - **Mechanism:** Boosts raw water capacity directly (increases river/canal levels) and reduces groundwater dependency.\n\n"
            f"**Conclusion:** "
            f"For {selected_district}, **{'Rainwater Harvesting' if latest_baseline_wsi_30d - rwh_only_wsi > latest_baseline_wsi_30d - aws_only_wsi else 'Additional Water Supply'}** is the most effective standalone physical intervention."
        )
    elif "highest scarcity risk" in user_query.lower() or "wards" in user_query.lower() or "risk" in user_query.lower():
        top_ward = df_ranks.iloc[0]
        bottom_ward = df_ranks.iloc[-1]
        response = (
            f"Based on the downscaled ML forecasts, **{top_ward['Ward']}** is at the **highest scarcity risk** "
            f"with a predicted WSI of **{top_ward['WSI']:.2f}** in 30 days.\n\n"
            f"Conversely, **{bottom_ward['Ward']}** has the lowest risk (**{bottom_ward['WSI']:.2f}**).\n\n"
            f"**Recommended action:** Prioritize groundwater recharge and residential society fixtures in {top_ward['Ward']} immediately."
        )
    elif "groundwater" in user_query.lower() or "gw" in user_query.lower() or "aquifer" in user_query.lower() or "borewell" in user_query.lower():
        gw_val = latest_row["groundwater_level"].values[0]
        gw_stress_val = comp_data["Groundwater Table Depletion"]
        severity_str = "EXTREMELY CRITICAL" if gw_stress_val > 0.7 else "MODERATE" if gw_stress_val > 0.4 else "LOW"
        response = (
            f"### 🪓 Groundwater Depletion Audit for **{selected_district}**\n\n"
            f"* **Current Level:** `{gw_val:.2f} mbgl` (meters below ground level)\n"
            f"* **Normalized Stress Score:** `{gw_stress_val:.2f}`\n"
            f"* **Aquifer Status:** `{severity_str}`\n\n"
            f"**AI Analysis:**\n"
            f"Groundwater extraction is {'outstripping natural recharge rates. Water table levels are deep and require immediate restrictions on borewell drilling' if gw_stress_val > 0.6 else 'currently stable but requires long-term monitoring'}.\n\n"
            f"**Recommended Intervention:** Implement **Demand Reduction** policies (such as metering) and recharge the aquifer via **Rainwater Harvesting** injection wells."
        )
    elif "rain" in user_query.lower() or "monsoon" in user_query.lower() or "precipitation" in user_query.lower():
        rain_val = latest_row["rainfall_mm"].values[0]
        rain_roll = latest_row["rainfall_mm_rolling_30d_sum"].values[0]
        rain_stress_val = comp_data["Rainfall Deficit"]
        response = (
            f"### 🌧️ Rainfall & Precipitation Analysis for **{selected_district}**\n\n"
            f"* **Daily Recorded Rainfall:** `{rain_val:.1f} mm`\n"
            f"* **30-Day Rolling Rainfall Sum:** `{rain_roll:.1f} mm`\n"
            f"* **Rainfall Deficit Stress Score:** `{rain_stress_val:.2f}`\n\n"
            f"**AI Analysis:**\n"
            f"A rainfall deficit stress score of `{rain_stress_val:.2f}` indicates that current seasonal precipitation is {'significantly below the historical baseline' if rain_stress_val > 0.6 else 'aligned with normal seasonal trends'}.\n\n"
            f"**Action Recommended:** Engage **Water Conservation** campaigns and implement rooftop **Rainwater Harvesting** to capture future monsoon runoffs."
        )
    elif "temp" in user_query.lower() or "heat" in user_query.lower() or "evaporation" in user_query.lower() or "temperature" in user_query.lower() or "climate" in user_query.lower() or "summer" in user_query.lower():
        temp_val = latest_row["air_temperature"].values[0]
        temp_stress_val = comp_data["Atmospheric Temperature Stress"]
        response = (
            f"### 🌡️ Temperature & Evaporation Risk for **{selected_district}**\n\n"
            f"* **Current Air Temperature:** `{temp_val:.1f} °C`\n"
            f"* **Evaporative Temperature Stress Score:** `{temp_stress_val:.2f}`\n\n"
            f"**AI Analysis:**\n"
            f"High temperatures increase soil moisture depletion and municipal evaporation rates. A stress score of `{temp_stress_val:.2f}` suggests high irrigation demand in agricultural borders.\n\n"
            f"**Mitigation Strategy:** Encourage agricultural mulching, shift supply hours to cooler evening periods to reduce evaporation, and expand urban shade canopies."
        )
    elif "river" in user_query.lower() or "canal" in user_query.lower() or "surface" in user_query.lower() or "reservoir" in user_query.lower():
        river_val = latest_row["river_level"].values[0]
        river_stress_val = comp_data["Surface River Deficit"]
        response = (
            f"### 🌊 Surface River & Canal Level Audit for **{selected_district}**\n\n"
            f"* **Current River Level:** `{river_val:.2f} meters`\n"
            f"* **Surface River Deficit Stress Score:** `{river_stress_val:.2f}`\n\n"
            f"**AI Analysis:**\n"
            f"Surface water flows are {'heavily depleted' if river_stress_val > 0.6 else 'adequate for this season'}. This affects canal supply channels and direct surface intake stations.\n\n"
            f"**Mitigation Strategy:** Treat and recycle municipal greywater to augment surface flows, and coordinate canal releases with upstream reservoirs."
        )
    elif "simulate" in user_query.lower() or "policy" in user_query.lower() or "slider" in user_query.lower() or "sandbox" in user_query.lower():
        response = (
            f"### 🎛️ Sandbox Policy Simulation Status\n\n"
            f"You have activated the following interventions on the sidebar:\n"
            f"- 🌧️ **Rainwater Harvesting:** `{rainwater_harvesting * 100:.0f}%` implementation\n"
            f"- 📉 **Demand Reduction:** `{demand_reduction * 100:.0f}%` dynamic pricing & caps\n"
            f"- 🌾 **Water Conservation:** `{water_conservation * 100:.0f}%` smart fixtures & drip irrigation\n\n"
            f"**Outcome Evaluation:**\n"
            f"This combination leads to a predicted 30-day Water Stress Index of **{latest_simulated_wsi_30d:.2f}** "
            f"compared to a baseline of **{latest_baseline_wsi_30d:.2f}** (a net reduction of **-{wsi_reduction:.2f}** WSI).\n\n"
            f"Adjust the sidebar sliders to see the forecasts recalculate in real-time."
        )
    elif "help" in user_query.lower() or "hello" in user_query.lower() or "hi" in user_query.lower() or "capabilities" in user_query.lower():
        response = (
            f"### 💬 Water Intelligence Assistant Capabilities\n\n"
            f"I am connected to the Stage 7 Feature Store and Stage 9 machine learning model predictions. You can ask me:\n"
            f"1. **Groundwater status:** 'What is the groundwater level?'\n"
            f"2. **Precipitation details:** 'Show me rainfall anomaly and monsoon status'\n"
            f"3. **Temperature issues:** 'How are temperatures affecting evapotranspiration?'\n"
            f"4. **Surface water levels:** 'What are the current canal and river levels?'\n"
            f"5. **Simulation outcomes:** 'What happens if we implement 100% rainwater harvesting?'\n"
            f"6. **Wards risk comparison:** 'Which Ahmedabad wards have the highest risk?'\n"
            f"7. **Slider simulator stats:** 'Explain my current slider policies'"
        )
    else:
        # Default driver explanation
        response = (
            f"Predicted water stress in **{selected_district}** is driven primarily by **{rec['primary_driver']}** "
            f"(stress factor of {rec['driver_severity']:.2f}).\n\n"
            f"**Detailed Breakdown:**\n"
            f"- Groundwater stress: {comp_data['Groundwater Table Depletion']:.2f}\n"
            f"- Rainfall deficit: {comp_data['Rainfall Deficit']:.2f}\n"
            f"- Evaporative Temperature stress: {comp_data['Atmospheric Temperature Stress']:.2f}\n\n"
            f"I recommend implementing a combination of **{rec['recommended_strategy']}** and **Water Conservation** "
            f"to offset the predicted stress of **{latest_baseline_wsi_30d:.2f}**."
        )
        
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.rerun()

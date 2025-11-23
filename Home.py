# -*- coding: utf-8 -*-
"""
CHEMISCO ENTERPRISE: Integrated Biorefinery Simulation Platform
---------------------------------------------------------------
Version: 11.0 (Final Stable Release)
Description: Advanced Torrefaction Simulation with Physics, Economics, PDF Reporting, and Gamification.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.integrate import odeint
from io import BytesIO
import base64
import os
import random
import time
import matplotlib.pyplot as plt # Used for static plots in PDF

# --- Libraries for PDF Report ---
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

# ==============================================================================
# 1. SYSTEM CONFIGURATION & CONSTANTS
# ==============================================================================

st.set_page_config(
    page_title="Chemisco Enterprise", 
    layout="wide", 
    page_icon="🏭",
    initial_sidebar_state="expanded"
)

# Thermodynamics Constants
R_GAS = 8.314

# Biomass Database (Pre-defined properties)
EMPIRICAL_DATA = {
    "Wood": {"A": 2.5e10, "Ea": 135000, "k_drying_base": 0.05, "Ash": 0.02, "Gas_Factor": 0.35, "HHV": 19.0},
    "Agricultural Waste": {"A": 5.0e11, "Ea": 150000, "k_drying_base": 0.07, "Ash": 0.08, "Gas_Factor": 0.45, "HHV": 17.0},
    "Municipal Waste": {"A": 1.0e12, "Ea": 165000, "k_drying_base": 0.10, "Ash": 0.15, "Gas_Factor": 0.55, "HHV": 15.0},
    "Olive Pits": {"A": 3.0e10, "Ea": 132000, "k_drying_base": 0.04, "Ash": 0.03, "Gas_Factor": 0.30, "HHV": 20.5}
}

SIZE_FACTOR = {"Fine (<1mm)": 1.0, "Medium (1-5mm)": 0.85, "Coarse (>5mm)": 0.65}

# Image Helper
LOGO_PATH = "chemisco_logo.png"

@st.cache_data
def get_img_as_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return None

logo_b64 = get_img_as_base64(LOGO_PATH)

# ==============================================================================
# 2. ADVANCED STYLING (CSS)
# ==============================================================================

st.markdown("""
<style>
    /* Main App Background */
    .stApp {
        background-color: #F8F9FA;
    }
    
    /* Cards & Containers */
    .css-1r6slb0 {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Hero Banner */
    .hero-section {
        background: linear-gradient(135deg, #00695c 0%, #004d40 100%);
        color: white;
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
    }
    .hero-section h1 { color: white !important; font-size: 3em; font-weight: 700; }
    .hero-section p { font-size: 1.2em; opacity: 0.9; }
    
    /* KPI Metrics */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-left: 5px solid #00695c;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
    }
    
    /* BFD Diagram */
    .bfd-container {
        display: flex; justify-content: space-between; align-items: center;
        padding: 20px; overflow-x: auto;
    }
    .bfd-box {
        background: white; color: #333; padding: 15px; border-radius: 8px;
        border: 2px solid #004d40; text-align: center; min-width: 120px;
        font-weight: bold; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .bfd-arrow { font-size: 24px; color: #00695c; margin: 0 10px; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 5px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap;
        background-color: #FFFFFF; border-radius: 5px;
        color: #333; border: 1px solid #DDD;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00695c; color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. PHYSICS ENGINE
# ==============================================================================

@st.cache_data
def run_simulation_engine(biomass_type, moisture, temp_C, duration_min, size, initial_mass):
    # Parameters
    data = EMPIRICAL_DATA[biomass_type]
    temp_K = temp_C + 273.15
    
    # Severity Factor (R0)
    severity = np.log10(duration_min * np.exp((temp_C - 100) / 14.75))
    
    # Kinetic Rates
    k_rxn = data["A"] * np.exp(-data["Ea"] / (R_GAS * temp_K)) * SIZE_FACTOR[size]
    k_dry = data["k_drying_base"]
    
    # Initial State
    f_moist = moisture / 100.0
    f_ash = data["Ash"]
    f_vol = 1.0 - f_moist - f_ash
    
    # ODE Solver
    def model(y, t):
        m, v = y
        dm = -k_dry * m if m > 0.001 else 0
        dv = -k_rxn * v
        return [dm, dv]
    
    t = np.linspace(0, duration_min, 100)
    y0 = [f_moist, f_vol]
    sol = odeint(model, y0, t)
    
    # Mass Balance
    final_moist = sol[-1, 0]
    final_vol = sol[-1, 1]
    
    mass_lost_water = (f_moist - final_moist) * initial_mass
    mass_lost_vol = (f_vol - final_vol) * initial_mass
    mass_solid_out = initial_mass - mass_lost_water - mass_lost_vol
    
    # Yields
    mass_yield = (mass_solid_out / initial_mass) * 100
    
    # Energy Balance
    # Energy Densification Ratio (EDR) approx correlation
    edr = 1 + (1 - (mass_yield/100)) * 0.6
    hhv_out = data["HHV"] * edr
    energy_yield = mass_yield * edr
    
    total_energy_in = initial_mass * data["HHV"]
    total_energy_out = mass_solid_out * hhv_out
    
    # Thermal Load (Input Heat)
    q_sensible = initial_mass * 1.4 * (temp_C - 25)
    q_latent = mass_lost_water * 2260
    q_rxn = (initial_mass - mass_solid_out) * 200 # Endothermic heat
    q_total = (q_sensible + q_latent + q_rxn) / 1000 # MJ
    
    return {
        "t": t,
        "profile_moist": sol[:, 0] * initial_mass,
        "profile_vol": sol[:, 1] * initial_mass,
        "mass_out": mass_solid_out,
        "mass_yield": mass_yield,
        "energy_yield": energy_yield,
        "hhv_out": hhv_out,
        "severity": severity,
        "energy_data": {
            "Q_Sensible": q_sensible/1000,
            "Q_Latent": q_latent/1000,
            "Q_Reaction": q_rxn/1000,
            "Total_MJ": q_total,
            "Energy_In": total_energy_in,
            "Energy_Out": total_energy_out
        },
        "params": {
            "bio": biomass_type, "temp": temp_C, "time": duration_min, "mass": initial_mass
        }
    }

# ==============================================================================
# 4. PDF REPORTING ENGINE
# ==============================================================================

def generate_pdf(res):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Header
    story.append(Paragraph("<b>CHEMISCO TECHNICAL REPORT</b>", styles['Title']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Date: {pd.Timestamp.now()}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # 1. Summary Table
    story.append(Paragraph("1. Process Summary", styles['Heading2']))
    data = [
        ["Parameter", "Value"],
        ["Feedstock", res['params']['bio']],
        ["Temperature", f"{res['params']['temp']} °C"],
        ["Duration", f"{res['params']['time']} min"],
        ["Mass Yield", f"{res['mass_yield']:.2f}%"],
        ["Energy Yield", f"{res['energy_yield']:.2f}%"],
        ["Product HHV", f"{res['hhv_out']:.2f} MJ/kg"]
    ]
    t = Table(data, colWidths=[3*inch, 2.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#00695c")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*inch))
    
    # 2. Chart Image (Matplotlib)
    story.append(Paragraph("2. Kinetic Profile", styles['Heading2']))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(res['t'], res['profile_vol'], label='Volatiles', color='orange')
    ax.plot(res['t'], res['profile_moist'], label='Moisture', color='blue')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Mass (kg)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png', dpi=150)
    img_buf.seek(0)
    story.append(ReportImage(img_buf, width=5*inch, height=3*inch))
    plt.close(fig)
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# ==============================================================================
# 5. MAIN APPLICATION CONTROLLER
# ==============================================================================

def main():
    # --- Session State Init ---
    if 'target_yield' not in st.session_state: st.session_state.target_yield = 75
    if 'budget' not in st.session_state: st.session_state.budget = 50000
    
    # --- Sidebar ---
    with st.sidebar:
        if logo_b64:
            st.markdown(f'<img src="data:image/png;base64,{logo_b64}" style="width:80px; margin:0 auto; display:block">', unsafe_allow_html=True)
        
        st.title("Control Center")
        st.markdown("---")
        
        st.subheader("1. Feedstock")
        b_type = st.selectbox("Biomass", list(EMPIRICAL_DATA.keys()))
        mass = st.number_input("Mass (kg)", 100.0, 10000.0, 1000.0)
        moist = st.slider("Moisture %", 0, 60, 15)
        size = st.select_slider("Size", options=list(SIZE_FACTOR.keys()))
        
        st.subheader("2. Reactor")
        temp = st.slider("Temp (°C)", 200, 350, 275)
        time_min = st.slider("Time (min)", 15, 180, 45)
        
        st.subheader("3. Economics")
        price = st.number_input("Price ($/kg)", 0.5, 5.0, 1.5)
        capex = st.number_input("CAPEX ($)", 100000, 5000000, 1000000)
        
        run = st.button("🚀 RUN SIMULATION", type="primary")

    # --- Hero Section ---
    st.markdown("""
    <div class="hero-section">
        <h1>CHEMISCO ENTERPRISE</h1>
        <p>High-Fidelity Process Simulation & Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Simulation Trigger ---
    res = run_simulation_engine(b_type, moist, temp, time_min, size, mass)
    
    # --- BFD Diagram ---
    st.markdown(f"""
    <div class="bfd-container">
        <div class="bfd-box" style="border-color:#2ecc71">FEED<br>{mass:.0f} kg</div>
        <div class="bfd-arrow">➜</div>
        <div class="bfd-box" style="border-color:#f1c40f">DRYER<br>-H₂O</div>
        <div class="bfd-arrow">➜</div>
        <div class="bfd-box" style="border-color:#e74c3c">REACTOR<br>{temp}°C</div>
        <div class="bfd-arrow">➜</div>
        <div class="bfd-box" style="border-color:#3498db">CHAR<br>{res['mass_out']:.1f} kg</div>
    </div>
    """, unsafe_allow_html=True)

    # --- KPIs ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mass Yield", f"{res['mass_yield']:.1f}%", delta="- Loss")
    c2.metric("Energy Yield", f"{res['energy_yield']:.1f}%", delta="+ HHV")
    c3.metric("Severity Index", f"{res['severity']:.2f}")
    c4.metric("Heat Req", f"{res['energy_data']['Total_MJ']:.1f} MJ")

    # --- Advanced Tabs ---
    tabs = st.tabs(["📈 Dynamics", "🔥 Thermal", "💰 Economics", "🎮 Game Mode", "📄 Report"])

    # 1. Dynamics
    with tabs[0]:
        st.subheader("Reaction Kinetics")
        df = pd.DataFrame({
            "Time": res['t'],
            "Moisture": res['profile_moist'],
            "Volatiles": res['profile_vol']
        })
        fig = px.line(df, x="Time", y=["Moisture", "Volatiles"], title="Component Mass Loss")
        st.plotly_chart(fig, use_container_width=True)

    # 2. Thermal (Energy Balance)
    with tabs[1]:
        st.subheader("Energy Waterfall")
        e = res['energy_data']
        
        # Waterfall Chart
        fig_w = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Sensible Heat", "Latent Heat", "Reaction Heat", "Total Required"],
            y=[e['Q_Sensible'], e['Q_Latent'], e['Q_Reaction'], 0],
            text=[f"{v:.1f}" for v in [e['Q_Sensible'], e['Q_Latent'], e['Q_Reaction'], e['Total_MJ']]],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        # Manually set the total bar value
        fig_w.data[0].y[3] = e['Total_MJ']
        fig_w.update_layout(title="Process Heat Breakdown (MJ)", showlegend=False)
        st.plotly_chart(fig_w, use_container_width=True)

    # 3. Economics
    with tabs[2]:
        st.subheader("Cash Flow Analysis")
        
        revenue = res['mass_out'] * price
        opex = (res['params']['time']/60) * 20 + (mass/1000)*30 # Fake OPEX model
        profit = revenue - opex
        
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.metric("Batch Revenue", f"${revenue:,.2f}")
            st.metric("Batch OPEX", f"${opex:,.2f}")
            st.metric("Net Profit", f"${profit:,.2f}", delta_color="normal")
        
        with col_e2:
            # ROI calculation
            annual_profit = profit * (24*300 / (time_min/60)) # 300 days
            payback = capex / annual_profit if annual_profit > 0 else 99
            st.info(f"Estimated Payback Period: {payback:.1f} Years")
            st.progress(min(1.0, max(0.0, 1/payback if payback > 0 else 0)))

    # 4. Game Mode
    with tabs[3]:
        st.subheader("🎯 Factory Manager Challenge")
        target = st.session_state.target_yield
        st.info(f"**Mission:** Optimize the reactor to hit exactly **{target}% Mass Yield** (+/- 1%).")
        
        curr = res['mass_yield']
        diff = abs(curr - target)
        
        c_g1, c_g2 = st.columns(2)
        c_g1.metric("Current", f"{curr:.1f}%")
        c_g2.metric("Target", f"{target}%")
        
        if diff <= 1.0:
            st.balloons()
            st.success("🎉 PERFECT MATCH! You are a Master Engineer.")
            if st.button("Next Level"):
                st.session_state.target_yield = random.randint(50, 85)
                st.experimental_rerun()
        elif diff <= 5.0:
            st.warning("So close! Adjust temperature slightly.")
        else:
            st.error("Way off target. Try changing residence time.")

    # 5. Report
    with tabs[4]:
        st.subheader("Professional Documentation")
        st.markdown("Generate a compliant PDF report for stakeholders.")
        
        if st.button("📄 Generate PDF Report"):
            pdf_file = generate_pdf(res)
            st.download_button(
                label="Download PDF",
                data=pdf_file,
                file_name="Chemisco_Report.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()

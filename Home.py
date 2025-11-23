# -*- coding: utf-8 -*-
"""
CHEMISCO ENTERPRISE: Integrated Biorefinery Simulation Platform
---------------------------------------------------------------
Version: 10.0 (Stable Final Release)
Author: Chemisco Development Team
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
import matplotlib.pyplot as plt

# --- Libraries for PDF Report ---
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

# ==============================================================================
# 1. CONFIGURATION & CONSTANTS
# ==============================================================================

st.set_page_config(page_title="Chemisco Enterprise", layout="wide", page_icon="🏭")

# Physical Constants
R_GAS = 8.314

# Empirical Database for Biomass
EMPIRICAL_DATA = {
    "Wood Chips": {"A": 2.5e10, "Ea": 135000, "k_drying_base": 0.05, "Ash": 0.01, "Gas_Factor": 0.35, "HHV_raw": 19.0},
    "Wheat Straw": {"A": 5.0e11, "Ea": 150000, "k_drying_base": 0.07, "Ash": 0.08, "Gas_Factor": 0.45, "HHV_raw": 17.0},
    "Sewage Sludge": {"A": 1.0e12, "Ea": 165000, "k_drying_base": 0.10, "Ash": 0.25, "Gas_Factor": 0.55, "HHV_raw": 14.0},
    "Olive Pits": {"A": 3.0e10, "Ea": 130000, "k_drying_base": 0.04, "Ash": 0.03, "Gas_Factor": 0.30, "HHV_raw": 20.0}
}

SIZE_FACTOR = {"Fine (<1mm)": 1.0, "Medium (1-5mm)": 0.85, "Coarse (>5mm)": 0.65}

# --- Helper: Image Handling ---
LOGO_PATH = "chemisco_logo.png" # Ensure this file exists or code handles it gracefully

@st.cache_data
def get_image_base64(path):
    """Safe image loading to Base64"""
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except:
        return None
    return None

LOGO_B64 = get_image_base64(LOGO_PATH)

# ==============================================================================
# 2. CSS STYLING (High-End UI)
# ==============================================================================

STYLING = """
<style>
    /* Main Layout */
    .stApp { background-color: #F4F7F6; color: #333; }
    
    /* Headers */
    h1, h2, h3 { color: #2C3E50; font-family: 'Segoe UI', sans-serif; }
    
    /* Custom Banner */
    .hero-banner {
        background: linear-gradient(120deg, #16a085, #2980b9);
        padding: 40px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        margin-bottom: 30px;
    }
    .hero-banner h1 { color: white !important; margin: 0; font-size: 3em; font-weight: 800; }
    .hero-banner p { font-size: 1.2em; opacity: 0.9; }

    /* Metrics Cards */
    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #E0E0E0;
        padding: 15px 25px;
        border-radius: 10px;
        border-left: 5px solid #2980b9;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover { transform: translateY(-5px); }
    
    /* BFD Diagram Container */
    .bfd-wrap {
        display: flex; justify-content: center; align-items: center;
        margin: 30px 0; font-family: monospace;
    }
    .bfd-box {
        background: white; padding: 15px; border: 2px solid #2980b9;
        border-radius: 8px; text-align: center; min-width: 140px;
        box-shadow: 0 4px 10px rgba(41, 128, 185, 0.1);
        color: #2c3e50; font-weight: bold;
    }
    .bfd-arrow { color: #2980b9; font-size: 24px; margin: 0 10px; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px; font-weight: bold;
    }
</style>
"""
st.markdown(STYLING, unsafe_allow_html=True)

# ==============================================================================
# 3. CORE SIMULATION ENGINE
# ==============================================================================

@st.cache_data
def run_simulation(biomass, moisture, temp_C, duration_min, size, initial_mass):
    # 1. Setup Parameters
    props = EMPIRICAL_DATA[biomass]
    temp_K = temp_C + 273.15
    
    # Severity Index Calculation (R0)
    # Log(R0) = Log(t * exp((T-100)/14.75))
    severity = np.log10(duration_min * np.exp((temp_C - 100) / 14.75))
    
    # 2. Kinetic Model (Arrhenius)
    k_base = props["A"] * np.exp(-props["Ea"] / (R_GAS * temp_K))
    k_eff = k_base * SIZE_FACTOR[size]
    k_dry = props["k_drying_base"]
    
    # Initial Fractions
    f_moist = moisture / 100.0
    f_ash = props["Ash"]
    f_vol = 1.0 - f_moist - f_ash
    
    # ODE System
    def model(y, t):
        m, v = y
        dm = -k_dry * m if m > 0.001 else 0
        dv = -k_eff * v
        return [dm, dv]
    
    t = np.linspace(0, duration_min, 100)
    sol = odeint(model, [f_moist, f_vol], t)
    
    # 3. Mass Balance Results
    final_moist = sol[-1, 0]
    final_vol = sol[-1, 1]
    
    # Fixed Carbon is assumed relatively stable in Torrefaction range, 
    # simplified assumption: FC stays, Volatiles leave.
    # More accurately: Solid = FC + Ash + Remaining Volatiles + Remaining Moisture
    
    # Calculate losses
    lost_moist = f_moist - final_moist
    lost_vol = f_vol - final_vol
    
    mass_solid_frac = 1.0 - lost_moist - lost_vol
    mass_solid_kg = mass_solid_frac * initial_mass
    
    # Ash Concentration Effect
    final_ash_frac = (f_ash * initial_mass) / mass_solid_kg
    
    # 4. Energy Balance (Advanced)
    # Energy Densification Ratio (EDR) = HHV_char / HHV_raw
    # Mass Yield (MY) = Solid_Out / Feed_In
    # Energy Yield (EY) = MY * EDR
    
    my = mass_solid_frac
    # Empirical correlation for EDR based on Mass Yield (MY)
    edr = 1 + (1 - my)**0.5 # EDR increases as mass is lost
    edr = min(edr, 1.4) # Physical cap
    
    hhv_in = props["HHV_raw"]
    hhv_out = hhv_in * edr
    ey = my * edr
    
    total_energy_in_mj = initial_mass * hhv_in
    total_energy_out_mj = mass_solid_kg * hhv_out
    
    # Thermal Load (Input Heat Required)
    q_sensible = initial_mass * 1.5 * (temp_C - 25) # Heating biomass
    q_latent = (lost_moist * initial_mass) * 2260 # Evaporating water
    q_reaction = (1-my) * initial_mass * 200 # Endothermic estimate
    total_heat_req_mj = (q_sensible + q_latent + q_reaction) / 1000
    
    return {
        "t": t,
        "profiles": sol * initial_mass, # Scale to kg
        "mass_yield_pct": my * 100,
        "energy_yield_pct": ey * 100,
        "final_mass_kg": mass_solid_kg,
        "final_ash_pct": final_ash_frac * 100,
        "severity": severity,
        "hhv_out": hhv_out,
        "energy_data": {
            "Input": total_energy_in_mj,
            "Output": total_energy_out_mj,
            "Process_Heat": total_heat_req_mj,
            "Loss_Volatiles": total_energy_in_mj - total_energy_out_mj
        },
        "params": {
            "bio": biomass, "temp": temp_C, "time": duration_min, "mass": initial_mass
        }
    }

# ==============================================================================
# 4. PDF REPORT GENERATOR
# ==============================================================================

def create_pdf(res):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    elements.append(Paragraph("CHEMISCO SIMULATION REPORT", styles['Title']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Summary Table
    data = [
        ["Parameter", "Value"],
        ["Biomass Type", res['params']['bio']],
        ["Temperature", f"{res['params']['temp']} °C"],
        ["Duration", f"{res['params']['time']} min"],
        ["Mass Yield", f"{res['mass_yield_pct']:.2f}%"],
        ["Energy Yield", f"{res['energy_yield_pct']:.2f}%"],
        ["Final HHV", f"{res['hhv_out']:.2f} MJ/kg"]
    ]
    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#2980b9")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.5*inch))
    
    # Graph Image (Mass Profile)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res['t'], res['profiles'][:, 1], label='Volatiles')
    ax.set_title("Devolatilization Profile")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Mass (kg)")
    ax.grid(True)
    
    img_buf = BytesIO()
    plt.savefig(img_buf, format='png', dpi=100)
    img_buf.seek(0)
    elements.append(ReportImage(img_buf, width=5*inch, height=3.5*inch))
    plt.close(fig)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ==============================================================================
# 5. MAIN APP UI
# ==============================================================================

def main():
    # --- Session State ---
    if 'baseline' not in st.session_state: st.session_state.baseline = None
    if 'price' not in st.session_state: st.session_state.price = 1.5
    if 'capex' not in st.session_state: st.session_state.capex = 1000000.0

    # --- Sidebar ---
    with st.sidebar:
        if LOGO_B64:
            st.markdown(f'<img src="data:image/png;base64,{LOGO_B64}" style="width:100px; display:block; margin:auto;">', unsafe_allow_html=True)
        
        st.title("⚙️ Control Panel")
        
        # 1. Feedstock
        st.subheader("1. Feedstock")
        b_type = st.selectbox("Biomass", list(EMPIRICAL_DATA.keys()))
        mass_in = st.number_input("Batch Mass (kg)", 100.0, 10000.0, 1000.0)
        moist = st.slider("Moisture %", 0, 60, 15)
        size = st.select_slider("Particle Size", options=list(SIZE_FACTOR.keys()))
        
        # 2. Process
        st.subheader("2. Process")
        temp = st.slider("Temperature (°C)", 200, 350, 275)
        dur = st.slider("Duration (min)", 10, 180, 45)
        
        # 3. Economics
        st.subheader("3. Economics")
        price = st.number_input("Biochar Price ($/kg)", value=st.session_state.price)
        capex = st.number_input("CAPEX ($)", value=st.session_state.capex)
        
        st.divider()
        if st.button("💾 Save as Baseline"):
            st.session_state.calc_flag = True # Trigger calc to save
            st.toast("Baseline Scenario Saved!", icon="✅")

    # --- Main Hero Section ---
    st.markdown("""
    <div class="hero-banner">
        <h1>CHEMISCO ENTERPRISE</h1>
        <p>Advanced Torrefaction & Biorefinery Simulation Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Calculation ---
    res = run_simulation(b_type, moist, temp, dur, size, mass_in)
    
    # Save baseline logic
    if st.session_state.get('calc_flag', False):
        st.session_state.baseline = res
        st.session_state.calc_flag = False

    # --- BFD ---
    st.markdown(f"""
    <div class="bfd-wrap">
        <div class="bfd-box" style="border-color:#27ae60">INPUT<br>{mass_in} kg</div>
        <div class="bfd-arrow">➜</div>
        <div class="bfd-box" style="border-color:#e67e22">DRYER<br>- H₂O</div>
        <div class="bfd-arrow">➜</div>
        <div class="bfd-box" style="border-color:#c0392b">REACTOR<br>{temp}°C</div>
        <div class="bfd-arrow">➜</div>
        <div class="bfd-box" style="border-color:#2980b9">PRODUCT<br>{res['final_mass_kg']:.1f} kg</div>
    </div>
    """, unsafe_allow_html=True)

    # --- KPIs with Delta ---
    st.subheader("📊 Key Performance Indicators")
    
    d_mass = None
    d_energy = None
    if st.session_state.baseline:
        d_mass = res['mass_yield_pct'] - st.session_state.baseline['mass_yield_pct']
        d_energy = res['energy_yield_pct'] - st.session_state.baseline['energy_yield_pct']

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mass Yield", f"{res['mass_yield_pct']:.1f}%", delta=f"{d_mass:.1f}%" if d_mass is not None else None)
    k2.metric("Energy Yield", f"{res['energy_yield_pct']:.1f}%", delta=f"{d_energy:.1f}%" if d_energy is not None else None)
    k3.metric("Severity Index", f"{res['severity']:.2f}", help="Reaction Severity R0")
    k4.metric("Biochar HHV", f"{res['hhv_out']:.1f} MJ/kg", delta="Enhanced")

    # --- Tabs ---
    t1, t2, t3, t4, t5 = st.tabs(["📈 Dynamics", "⚡ Energy", "💰 Economics", "🤖 AI Expert", "📑 Report"])

    with t1:
        st.subheader("Reaction Kinetics Profile")
        df_prof = pd.DataFrame(res['profiles'], columns=['Moisture', 'Volatiles'])
        df_prof['Time'] = res['t']
        df_prof['Solid'] = (res['final_mass_kg'] / mass_in) * mass_in # Simplified line for solid
        
        fig = px.line(df_prof, x='Time', y=['Moisture', 'Volatiles'], 
                      title="Component Loss Over Time",
                      labels={"value": "Mass (kg)"})
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        st.subheader("Energy Balance Waterfall")
        e = res['energy_data']
        
        fig_water = go.Figure(go.Waterfall(
            name = "Energy", orientation = "v",
            measure = ["relative", "relative", "relative", "total"],
            x = ["Input Feedstock", "Volatile Loss", "Heat Loss", "Biochar Energy"],
            textposition = "outside",
            y = [e['Input'], -e['Loss_Volatiles'], -e['Input']*0.05, e['Output']],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig_water.update_layout(title = "Energy Flow (MJ)", showlegend = False)
        st.plotly_chart(fig_water, use_container_width=True)
        
        st.info(f"Process Heat Required: {e['Process_Heat']:.1f} MJ (Ideally supplied by burning volatiles)")

    with t3:
        st.subheader("Financial Analysis")
        revenue = res['final_mass_kg'] * price
        ops_cost = (res['params']['time']/60) * 50 # Assuming $50/hr ops cost
        feed_cost = (mass_in/1000) * 30 # Assuming $30/ton
        profit = revenue - (ops_cost + feed_cost)
        
        col_eco1, col_eco2 = st.columns(2)
        with col_eco1:
            st.write(f"**Revenue:** ${revenue:,.2f}")
            st.write(f"**OPEX:** ${ops_cost+feed_cost:,.2f}")
            st.metric("Net Profit (Batch)", f"${profit:,.2f}")
        
        with col_eco2:
            # Simple ROI chart
            fig_pie = px.pie(names=['Feedstock', 'Operations', 'Profit'], 
                             values=[feed_cost, ops_cost, max(0, profit)], 
                             title="Cost vs Profit Breakdown")
            st.plotly_chart(fig_pie, use_container_width=True)

    with t4:
        st.subheader("🤖 Intelligent Process Consultant")
        
        my = res['mass_yield_pct']
        temp_curr = res['params']['temp']
        
        # Rule-based Logic
        if my < 60:
            st.error("⚠️ **Yield Alert:** Mass yield is critically low (<60%).")
            st.write(f"**Diagnosis:** Temperature ({temp_curr}°C) is too high for standard torrefaction.")
            st.write("**Recommendation:** Reduce temperature by 10-20°C to preserve solid mass.")
        elif my > 90:
            st.warning("⚠️ **Quality Alert:** Mass yield is very high (>90%).")
            st.write("**Diagnosis:** Biomass is likely under-torrefied (Raw).")
            st.write("**Recommendation:** Increase residence time or temperature to ensure hydrophobicity.")
        else:
            st.success("✅ **Optimal Operation:** Process parameters are within the industrial sweet spot.")
            st.write("The balance between mass loss and energy densification is ideal.")

    with t5:
        st.subheader("Download Professional Report")
        st.write("Generate a PDF document containing all simulation parameters, results, and charts.")
        
        if st.button("📄 Generate PDF"):
            pdf_bytes = create_pdf(res)
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name="Chemisco_Report.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()

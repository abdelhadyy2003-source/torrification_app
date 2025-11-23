import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.integrate import odeint
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from reportlab.lib.units import inch
from reportlab.lib import colors
import matplotlib.pyplot as plt # Essential for PDF generation
import base64
import os # Import os for path checking
import random # Import random for the game logic

# --- 1. Chemical and Empirical Constants (UNCHANGED) ---
R_GAS = 8.314
EMPIRICAL_DATA = {
    "Wood": {"A": 2.5e10, "Ea": 135000, "k_drying_base": 0.05, "Ash": 0.02, "Gas_Factor": 0.35, "HHV_raw": 19.5},
    "Agricultural Waste": {"A": 5.0e11, "Ea": 150000, "k_drying_base": 0.07, "Ash": 0.08, "Gas_Factor": 0.45, "HHV_raw": 17.0},
    "Municipal Waste": {"A": 1.0e12, "Ea": 165000, "k_drying_base": 0.10, "Ash": 0.15, "Gas_Factor": 0.55, "HHV_raw": 15.0}
}
SIZE_FACTOR = {"Fine (<1mm)": 1.0, "Medium (1-5mm)": 0.85, "Coarse (>5mm)": 0.65}

# --- Base64 Utility for Robust Image Embedding ---
LOGO_PATH = "chemisco_logo.png"

@st.cache_data  # Added caching for performance
def _get_image_base64(image_path):
    """Encodes an image to Base64 string for safe embedding in HTML/Markdown."""
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()
        else:
            return None
    except Exception:
        return None

LOGO_BASE64_STRING = _get_image_base64(LOGO_PATH)


# --- 2. Global CSS (ENHANCED & MODERNIZED) ---
GLOBAL_CSS = """
<style>
    .stApp { padding-top: 20px; background-color: #f9fcf9; }
    
    /* MODERN CARD STYLE */
    .css-1r6slb0, .stMarkdown, .stMetric {
        border-radius: 10px;
    }
    
    /* MAIN BANNER WITH GLASSMORPHISM */
    .main-banner {
        background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 100%);
        padding: 40px 30px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 35px;
        box-shadow: 0 10px 30px rgba(46, 125, 50, 0.3);
        position: relative;
        overflow: hidden;
    }
    .main-banner::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
        pointer-events: none;
    }
    .main-banner-content { position: relative; z-index: 1; }
    
    .main-banner h1 { color: #FFFFFF; margin: 0; font-size: 3.5em; font-weight: 900; letter-spacing: 3px; text-shadow: 0 2px 4px rgba(0,0,0,0.2); }
    
    .banner-tagline { 
        color: #E8F5E9; 
        font-size: 1.3em; 
        font-weight: 400;
        margin: 15px auto;
        max-width: 800px;
        padding: 8px 0;
        border-top: 1px solid rgba(255, 255, 255, 0.3);
        border-bottom: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .dedication { 
        color: #FFEB3B; 
        font-weight: bold; 
        font-size: 1.2em; 
        margin-top: 20px; 
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* METRICS STYLE (Professional Cards) */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #4CAF50;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    }
    [data-testid="stMetricValue"] { font-size: 32px; color: #1B5E20; font-weight: 800; }
    [data-testid="stMetricLabel"] { color: #555; font-weight: 600; font-size: 14px; text-transform: uppercase; }
    
    /* BFD ENHANCEMENTS */
    .bfd-container { display: flex; justify-content: center; align-items: center; margin: 40px 0 70px 0; gap: 0; }
    .bfd-block { 
        padding: 20px; 
        border: 2px solid #4CAF50; 
        border-radius: 10px; 
        text-align: center; 
        background: white; 
        box-shadow: 0 8px 20px rgba(76, 175, 80, 0.15); 
        font-weight: bold; 
        color: #2E7D32; 
        min-width: 160px; 
        z-index: 2;
        transition: all 0.3s;
    }
    .bfd-block:hover { transform: scale(1.05); border-color: #2E7D32; }
    .bfd-stream { 
        flex-grow: 1; 
        height: 4px; 
        background: linear-gradient(90deg, #4CAF50 0%, #81C784 100%); 
        position: relative; 
        min-width: 50px;
    }
    .bfd-stream::after { 
        content: '▶'; 
        position: absolute; 
        right: -8px; top: -9px; 
        color: #81C784; 
        font-size: 14px;
    }
    
    .side-stream { position: absolute; left: 50%; height: 40px; width: 4px; background-color: #FF9800; bottom: -40px; margin-left: -2px; }
    .side-stream::after { content: '▼'; position: absolute; bottom: -15px; left: -5px; color: #FF9800; font-size: 12px;}
    .side-stream-label { position: absolute; bottom: -75px; left: 50%; transform: translateX(-50%); font-size: 11px; font-weight: bold; color: #E65100; background: #FFF3E0; padding: 2px 6px; border-radius: 4px;}

    /* SIDEBAR POLISH */
    .sidebar-logo-container { text-align: center; padding: 20px 0; }
    .stSidebar { border-right: 1px solid #eee; }
</style>
"""

# --- 3. Simulation Core Logic (OPTIMIZED) ---
@st.cache_data # Caching prevents recalculation on simple UI interactions
def simulate_torrefaction(biomass, moisture, temp_C, duration_min, size, initial_mass_kg, reactor_type="N/A"): 
    temp_K = temp_C + 273.15
    data = EMPIRICAL_DATA.get(biomass)
    R_GAS_LOCAL = R_GAS 
    
    # Severity Factor Calculation (R0) for Torrefaction Analysis
    # R0 = t * exp((T - T_ref)/14.75)
    t_ref_severe = 100 # Reference temp usually 100C for Hydrolysis but used as index here
    severity_factor = np.log10(duration_min * np.exp((temp_C - t_ref_severe) / 14.75))
    
    k_devol_arrhenius = data["A"] * np.exp(-data["Ea"] / (R_GAS_LOCAL * temp_K))
    k_devol_eff = k_devol_arrhenius * SIZE_FACTOR.get(size)
    k_drying = data["k_drying_base"] 
    
    initial_moisture_frac = moisture / 100
    initial_ash_frac = data["Ash"]
    initial_volatiles_frac = 1.0 - initial_moisture_frac - initial_ash_frac
    
    mass_ash_kg = initial_mass_kg * initial_ash_frac
    fixed_carbon_frac_initial = 1.0 - initial_moisture_frac - initial_volatiles_frac - initial_ash_frac
    
    def model(y, t, k1, k2):
        m_moist, m_vol = y
        d_moist = -k1 * m_moist if m_moist > 0.001 else 0
        d_vol = -k2 * m_vol
        return [d_moist, d_vol]
    
    t = np.linspace(0, duration_min, 100)
    y0 = [initial_moisture_frac, initial_volatiles_frac]
    sol = odeint(model, y0, t, args=(k_drying, k_devol_eff))
    sol[sol < 0] = 0
    
    moisture_curve = sol[:, 0] 
    volatiles_curve = sol[:, 1]
    
    current_total_mass_fraction = moisture_curve + volatiles_curve + fixed_carbon_frac_initial + initial_ash_frac
    
    # Safe division for Ash concentration
    solid_mass_fraction = volatiles_curve + fixed_carbon_frac_initial + initial_ash_frac
    ash_concentration_percent = np.divide(initial_ash_frac, solid_mass_fraction, out=np.zeros_like(solid_mass_fraction), where=solid_mass_fraction!=0) * 100
    
    final_moisture_loss = initial_moisture_frac 
    final_volatiles_remaining = volatiles_curve[-1]
    final_volatiles_lost = initial_volatiles_frac - final_volatiles_remaining
    
    final_solid_fraction = 1.0 - final_moisture_loss - final_volatiles_lost
    mass_biochar_total = final_solid_fraction * initial_mass_kg
    final_ash_percent = (mass_ash_kg / mass_biochar_total) * 100 if mass_biochar_total > 0 else 0

    yields_percent = pd.DataFrame({
        "Yield (%)": [final_solid_fraction * 100, final_volatiles_lost * 100, final_moisture_loss * 100, initial_ash_frac * 100]},
        index=["Biochar (Solid Product)", "Non-Condensable Gases", "Moisture Loss (Water Vapor)", "Original Ash Content"]
    )
    yields_mass = yields_percent.copy()
    yields_mass["Mass (kg)"] = yields_percent["Yield (%)"] * initial_mass_kg / 100
    yields_mass.drop(columns=["Yield (%)"], inplace=True)

    mass_volatiles_remaining = final_volatiles_remaining * initial_mass_kg
    mass_fixed_carbon = fixed_carbon_frac_initial * initial_mass_kg
    
    solid_composition = pd.DataFrame({
        "Mass (kg)": [mass_fixed_carbon, mass_volatiles_remaining, mass_ash_kg]
    }, index=["Fixed Carbon", "Remaining Volatiles", "Ash"])

    gas_comp_mass_fractions = {"CO2": 0.45, "CO": 0.35, "CH4": 0.15, "H2": 0.05}
    gas_composition_molar_data = {}
    total_molar_sum = sum(gas_comp_mass_fractions.values())
    
    for gas, fraction in gas_comp_mass_fractions.items():
        gas_composition_molar_data[gas] = (fraction / total_molar_sum) * 100 
            
    gas_composition_molar = pd.DataFrame.from_dict(
        gas_composition_molar_data, 
        orient="index", columns=["Molar % in Dry Gas"]
    ).fillna(0)
    
    mass_profile = pd.DataFrame({
        "Time (min)": t,
        "Total Mass Yield (%)": current_total_mass_fraction * 100,
        "Ash Concentration in Solid (%)": ash_concentration_percent
    }).set_index("Time (min)")
    
    # --- Energy Calculation (New Feature) ---
    # Simplified HHV Estimation based on mass loss (Van Krevelen logic simplified)
    mass_yield_ratio = final_solid_fraction
    energy_yield_factor = 1.0 / mass_yield_ratio if mass_yield_ratio > 0 else 0 # Energy densification
    # Cap reasonable densification
    energy_densification = min(1.4, 1 + (1 - mass_yield_ratio)) 
    
    hhv_original = data.get("HHV_raw", 18.0)
    hhv_biochar = hhv_original * energy_densification
    
    total_energy_in = initial_mass_kg * hhv_original
    total_energy_out_char = mass_biochar_total * hhv_biochar
    energy_yield_percent = (total_energy_out_char / total_energy_in) * 100 if total_energy_in > 0 else 0

    return {
        "yields_percent": yields_percent,
        "yields_mass": yields_mass,
        "solid_composition": solid_composition,
        "final_ash_percent": final_ash_percent,
        "gas_composition_molar": gas_composition_molar,
        "mass_profile": mass_profile,
        "k_devol_eff": k_devol_eff,
        "severity_factor": severity_factor,
        "energy_data": {
            "HHV_in": hhv_original,
            "HHV_out": hhv_biochar,
            "Energy_Yield": energy_yield_percent,
            "Energy_Densification": energy_densification
        },
        "parameters": {
            "biomass": biomass, "moisture": moisture, "temperature": temp_C, 
            "duration": duration_min, "size": size, "initial_mass": initial_mass_kg,
            "reactor": reactor_type
        }
    }

# --- 4. PDF Report Generation Function (UNCHANGED) ---
def generate_pdf_report(results):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    elements = []
    
    # -- Styles --
    title_style = styles["Title"]
    title_style.textColor = colors.HexColor("#388E3C")
    heading_style = styles["Heading2"]
    heading_style.textColor = colors.HexColor("#2E7D32")
    normal_style = styles["Normal"]
    
    elements.append(Paragraph("CHEMISCO TORREFACTION REPORT", title_style))
    elements.append(Paragraph("Project presented to: د. عمرو الرفاعي", styles["Heading3"])) 
    
    try:
        img_path = LOGO_PATH 
        logo_pdf = ReportImage(img_path, width=1.5*inch, height=1.5*inch)
        logo_pdf.hAlign = 'CENTER'
        elements.append(logo_pdf)
    except FileNotFoundError:
        elements.append(Paragraph("CHEMISCO", title_style)) 
        
    elements.append(Paragraph(f"Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # -- Parameters --
    elements.append(Paragraph("1. Simulation Parameters & Kinetics", heading_style))
    p = results["parameters"]
    param_data = [
        ["Parameter", "Value"],
        ["Biomass Type", p['biomass']],
        ["Reactor Type", p['reactor']],
        ["Initial Mass", f"{p['initial_mass']} kg"],
        ["Temperature", f"{p['temperature']} °C"],
        ["Duration", f"{p['duration']} min"],
        ["Severity Index", f"{results.get('severity_factor', 0):.2f}"]
    ]
    
    t_param = Table(param_data, colWidths=[3*inch, 3*inch])
    t_param.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E8F5E9")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
    ]))
    elements.append(t_param)
    elements.append(Spacer(1, 0.2*inch))
    
    # -- Yields --
    elements.append(Paragraph("2. Product Yields", heading_style))
    yield_data = [["Component", "Mass (kg)", "Yield (%)"]]
    for idx, row in results["yields_percent"].iterrows():
        mass = results["yields_mass"].loc[idx, "Mass (kg)"]
        yield_data.append([idx, f"{mass:.2f}", f"{row['Yield (%)']:.2f}"])
        
    t_yield = Table(yield_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
    t_yield.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E8F5E9")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ]))
    elements.append(t_yield)
    elements.append(Spacer(1, 0.2*inch))
    
    # -- Visuals --
    elements.append(Paragraph("3. Results Visualization", heading_style))
    def get_image_bytes(fig):
        img_buf = BytesIO()
        fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=150, transparent=True)
        img_buf.seek(0)
        return img_buf

    fig_pie1, ax_pie1 = plt.subplots(figsize=(4.5, 4.5)) 
    colors_solid_pdf = ['#6A1B9A', '#AB47BC', '#BDBDBD']
    ax_pie1.pie(results["solid_composition"]["Mass (kg)"], labels=results["solid_composition"].index, autopct='%1.1f%%', colors=colors_solid_pdf)
    ax_pie1.set_title("Solid Composition")
    img1 = ReportImage(get_image_bytes(fig_pie1), width=3*inch, height=3*inch) 
    plt.close(fig_pie1)
    elements.append(img1)

    # Build
    doc.build(elements)
    buffer.seek(0)
    return buffer

# --- 5. Main Streamlit App ---
def main():
    if 'cost_biomass_per_ton' not in st.session_state:
        st.session_state.cost_biomass_per_ton = 30.0
        st.session_state.cost_energy_per_hour = 5.0
        st.session_state.price_biochar_per_kg = 1.20
        st.session_state.target_yield = random.randint(60, 85)
        st.session_state.target_ash = 0 
        st.session_state.has_won = False
        st.session_state.baseline_results = None # NEW: For scenario comparison

    st.set_page_config(page_title="Chemisco Enterprise", layout="wide", initial_sidebar_state="expanded")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    
    # --- Sidebar ---
    with st.sidebar:
        if LOGO_BASE64_STRING:
            st.markdown(f"""
                <div class="sidebar-logo-container">
                    <img src="data:image/png;base64,{LOGO_BASE64_STRING}" style="width: 70%; margin: 0 auto; display: block;">
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🕹️ Control Panel")
        
        reactor_type = st.selectbox("🏭 Reactor Configuration", 
            ["Rotary Drum Reactor", "Fluidized Bed Reactor", "Auger/Screw Reactor", "Fixed Bed Reactor"])
        
        with st.expander("🌱 Feedstock Settings", expanded=True):
            initial_mass_kg = st.number_input("Batch Mass (kg)", 1.0, 10000.0, 100.0, step=10.0)
            biomass_type = st.selectbox("Biomass Type", list(EMPIRICAL_DATA.keys()))
            moisture_content = st.slider("Moisture (%)", 0.0, 60.0, 10.0)
            particle_size = st.select_slider("Particle Granularity", list(SIZE_FACTOR.keys()))
            
        with st.expander("🔥 Thermal Settings", expanded=True):
            temperature = st.slider("Temperature (°C)", 200, 350, 275, step=5)
            duration = st.slider("Residence Time (min)", 10, 180, 45, step=5)
            
        with st.expander("📊 Economic Parameters"):
            st.session_state.cost_biomass_per_ton = st.number_input("Feedstock ($/ton)", value=st.session_state.cost_biomass_per_ton)
            st.session_state.cost_energy_per_hour = st.number_input("Ops Cost ($/hr)", value=st.session_state.cost_energy_per_hour)
            st.session_state.price_biochar_per_kg = st.number_input("Biochar Price ($/kg)", value=st.session_state.price_biochar_per_kg)
            
        # Feature: Save Baseline
        st.markdown("---")
        col_b1, col_b2 = st.columns(2)
        if col_b1.button("💾 Save Baseline"):
            st.session_state.save_baseline = True
        if col_b2.button("🔄 Reset"):
            st.session_state.baseline_results = None
            st.rerun()

        game_mode = st.checkbox("🏆 Engineer Challenge Mode", value=False)


    # --- Main Content ---
    if LOGO_BASE64_STRING:
        st.markdown(f"""
            <div class="main-banner">
                <div class="main-banner-content">
                    <img src="data:image/png;base64,{LOGO_BASE64_STRING}" style="width: 120px; margin-bottom: 10px;">
                    <h1>CHEMISCO ENTERPRISE</h1> 
                    <p class="banner-tagline">High-Fidelity Torrefaction & Pyrolysis Simulation Platform</p>
                    <div class="dedication">Special Release for Dr. Amr El-Rifai</div> 
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""<div class="main-banner"><h1>CHEMISCO</h1><p>Advanced Simulation Platform</p></div>""", unsafe_allow_html=True)
    
    # Run Simulation
    results = simulate_torrefaction(biomass_type, moisture_content, temperature, duration, particle_size, initial_mass_kg, reactor_type)
    
    # Handle Baseline Logic
    if st.session_state.get("save_baseline"):
        st.session_state.baseline_results = results
        st.session_state.save_baseline = False
        st.toast("Baseline Scenario Saved!", icon="💾")

    # BFD Visualization
    st.markdown("#### 🔄 Process Block Flow Diagram (BFD)")
    bfd_html = f"""
    <div class="bfd-container">
        <div class="bfd-block" style="border-color: #1976D2; color: #0D47A1;">
            FEED PREPARATION
            <p>Input: {initial_mass_kg:.0f} kg</p>
            <p>H₂O: {moisture_content:.1f}%</p>
        </div>
        <div class="bfd-stream"></div>
        <div class="bfd-block">
            DRYING ZONE
            <p>Pre-heating</p>
            <div class="side-stream" style="background: #90CAF9;"></div>
            <div class="side-stream-label" style="color: #1565C0;">Steam</div>
        </div>
        <div class="bfd-stream"></div>
        <div class="bfd-block" style="border-color: #D32F2F; background-color: #FFEBEE; color: #B71C1C;">
            {reactor_type.upper()}
            <p>T: {temperature}°C | t: {duration}min</p>
            <p>Severity: {results['severity_factor']:.2f}</p>
            <div class="side-stream" style="background-color: #FFC107;"></div>
            <div class="side-stream-label" style="color: #FF6F00;">Volatiles</div>
        </div>
        <div class="bfd-stream"></div>
        <div class="bfd-block" style="border-color: #388E3C; background-color: #E8F5E9; color: #1B5E20;">
            PRODUCT COOLING
            <p>Biochar Output</p>
        </div>
    </div>
    """
    st.markdown(bfd_html, unsafe_allow_html=True)
    
    # --- Game Logic (Existing) ---
    if game_mode:
        if st.session_state.get('target_ash', 0) == 0:
            st.session_state.target_ash = round(random.uniform(EMPIRICAL_DATA[biomass_type]["Ash"]*100 + 1.0, EMPIRICAL_DATA[biomass_type]["Ash"]*100 + 5.0), 1)

        st.info(f"🎯 **CHALLENGE:** Target Yield: {st.session_state.target_yield}% | Max Ash: {st.session_state.target_ash}%")
        curr_yield = results["yields_percent"].loc["Biochar (Solid Product)", "Yield (%)"]
        score = max(0, 100 - (abs(curr_yield - st.session_state.target_yield) * 2 + max(0, results["final_ash_percent"] - st.session_state.target_ash) * 10))
        st.progress(int(score), text=f"Efficiency Score: {score:.1f}/100")
        if score >= 90: st.success("🌟 OPTIMAL PROCESS REACHED!")

    # --- RESULTS DISPLAY ---
    st.markdown("### 📈 Key Performance Indicators")
    
    # Calculate Deltas if baseline exists
    d_mass, d_ash, d_hhv = None, None, None
    if st.session_state.baseline_results:
        base = st.session_state.baseline_results
        d_mass = results["yields_mass"].loc["Biochar (Solid Product)", "Mass (kg)"] - base["yields_mass"].loc["Biochar (Solid Product)", "Mass (kg)"]
        d_ash = results["final_ash_percent"] - base["final_ash_percent"]
        d_hhv = results["energy_data"]["HHV_out"] - base["energy_data"]["HHV_out"]

    c1, c2, c3, c4 = st.columns(4)
    biochar_mass = results["yields_mass"].loc["Biochar (Solid Product)", "Mass (kg)"]
    c1.metric("Biochar Mass", f"{biochar_mass:.2f} kg", delta=f"{d_mass:.2f} kg" if d_mass is not None else None)
    c2.metric("Final Ash Content", f"{results['final_ash_percent']:.2f} %", delta=f"{d_ash:.2f} %" if d_ash is not None else None, delta_color="inverse")
    c3.metric("Biochar HHV", f"{results['energy_data']['HHV_out']:.2f} MJ/kg", delta=f"{d_hhv:.2f}" if d_hhv is not None else None)
    c4.metric("Severity Index", f"{results['severity_factor']:.2f}", help="Logarithmic measure of reaction severity")

    # --- TABS ---
    tab_mass, tab_dyn, tab_energy, tab_econ, tab_rep = st.tabs([
        "⚖️ Mass Balance", "📉 Process Dynamics", "⚡ Energy Analysis", "💰 Economics", "📄 Export Report"
    ])
    
    with tab_mass:
        col_pie1, col_pie2 = st.columns(2)
        with col_pie1:
            fig1 = px.pie(results["solid_composition"].reset_index(), values='Mass (kg)', names='index', 
                          title="Solid Product Composition", color_discrete_sequence=px.colors.sequential.Purples_r)
            st.plotly_chart(fig1, use_container_width=True)
        with col_pie2:
            df_glob = results["yields_percent"].iloc[[0,1,2]].reset_index()
            fig2 = px.pie(df_glob, values='Yield (%)', names='index', title="Global Mass Distribution", 
                          color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig2, use_container_width=True)

    with tab_dyn:
        # Dual Axis Chart for Mass vs Ash
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dual.add_trace(go.Scatter(x=results["mass_profile"].index, y=results["mass_profile"]["Total Mass Yield (%)"], 
                                      name="Mass Remaining (%)", line=dict(color='#4CAF50', width=3)), secondary_y=False)
        fig_dual.add_trace(go.Scatter(x=results["mass_profile"].index, y=results["mass_profile"]["Ash Concentration in Solid (%)"], 
                                      name="Ash Conc. (%)", line=dict(color='#FF5252', dash='dot')), secondary_y=True)
        fig_dual.update_layout(title="Reaction Kinetics Profile", hovermode="x unified")
        fig_dual.update_yaxes(title_text="Mass %", secondary_y=False)
        fig_dual.update_yaxes(title_text="Ash %", secondary_y=True)
        st.plotly_chart(fig_dual, use_container_width=True)

    with tab_energy:
        st.subheader("Energy Balance & Densification")
        
        e_data = results["energy_data"]
        col_e1, col_e2 = st.columns(2)
        
        # Waterfall Chart for Energy
        energy_in = initial_mass_kg * e_data["HHV_in"]
        energy_char = biochar_mass * e_data["HHV_out"]
        energy_vol = energy_in - energy_char # Simplified
        
        fig_water = go.Figure(go.Waterfall(
            name = "Energy", orientation = "v",
            measure = ["absolute", "relative", "relative", "total"],
            x = ["Input Feedstock", "Volatiles/Gas Loss", "Process Efficiency Loss", "Biochar Product"],
            textposition = "outside",
            y = [energy_in, -energy_vol*0.8, -energy_vol*0.2, energy_char],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig_water.update_layout(title = "Energy Flow (MJ)", showlegend = False)
        col_e1.plotly_chart(fig_water, use_container_width=True)
        
        with col_e2:
            st.markdown("#### Densification Metrics")
            st.write(f"**Energy Densification Ratio:** {e_data['Energy_Densification']:.2f}")
            st.write(f"**Energy Yield:** {e_data['Energy_Yield']:.1f}%")
            st.progress(e_data['Energy_Yield']/100)
            st.info("Note: Energy yield represents the percentage of original chemical energy retained in the solid biochar.")

    with tab_econ:
        st.subheader("Economic Feasibility")
        # Recalc economics
        cost_feed = (initial_mass_kg/1000) * st.session_state.cost_biomass_per_ton
        cost_ops = (duration/60) * st.session_state.cost_energy_per_hour
        rev = biochar_mass * st.session_state.price_biochar_per_kg
        profit = rev - (cost_feed + cost_ops)
        
        eco_df = pd.DataFrame({
            "Category": ["Revenue", "Feedstock Cost", "Ops Cost", "Net Profit"],
            "Value ($)": [rev, -cost_feed, -cost_ops, profit]
        })
        
        fig_eco = px.bar(eco_df, x="Category", y="Value ($)", color="Value ($)", 
                         color_continuous_scale="RdYlGn", text_auto='.2f')
        st.plotly_chart(fig_eco, use_container_width=True)

    with tab_rep:
        st.markdown("### 📄 Professional Reporting")
        st.write("Generate a standardized engineering report for documentation.")
        if st.button("Generate PDF Report"):
            pdf_data = generate_pdf_report(results)
            st.download_button("⬇️ Download Report", pdf_data, "Chemisco_Report.pdf", "application/pdf")

# --- Execution Entry Point ---
if __name__ == "__main__":
    main()

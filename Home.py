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
import matplotlib.pyplot as plt
import random

# --- 1. Chemical and Empirical Constants ---
R_GAS = 8.314
EMPIRICAL_DATA = {
    "Wood": {"A": 2.5e10, "Ea": 135000, "k_drying_base": 0.05, "Ash": 0.02, "Gas_Factor": 0.35},
    "Agricultural Waste": {"A": 5.0e11, "Ea": 150000, "k_drying_base": 0.07, "Ash": 0.08, "Gas_Factor": 0.45},
    "Municipal Waste": {"A": 1.0e12, "Ea": 165000, "k_drying_base": 0.10, "Ash": 0.15, "Gas_Factor": 0.55}
}
SIZE_FACTOR = {"Fine (<1mm)": 1.0, "Medium (1-5mm)": 0.85, "Coarse (>5mm)": 0.65}

# --- 2. Global CSS for Aesthetic and BFD (No need for Base64 classes) ---
GLOBAL_CSS = """
<style>
    .stApp { padding-top: 20px; }
    .main-banner {
        background-color: #388E3C; /* Dark Green */
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .main-banner h1 { color: #FFFFFF; margin: 0; font-size: 3em; font-weight: 800; letter-spacing: 2px;}
    .main-banner p { color: #C8E6C9; margin-top: 5px; font-size: 1.2em; }
    .dedication { 
        color: #FFEB3B; /* Yellow */
        font-weight: bold; 
        font-size: 1.1em; 
        margin-top: 15px; 
        padding-top: 10px;
        border-top: 1px solid rgba(255,255,255,0.3);
    }
    
    /* Metrics Style */
    [data-testid="stMetricValue"] { font-size: 28px; color: #388E3C; }
    
    /* BFD Styles */
    .bfd-container { display: flex; justify-content: center; align-items: center; margin: 30px 0 60px 0; position: relative; }
    .bfd-block { padding: 15px 25px; border: 3px solid #4CAF50; border-radius: 6px; text-align: center; background-color: #E8F5E9; box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15); font-weight: bold; color: #1B5E20; position: relative; min-width: 180px; }
    .bfd-block p { margin: 5px 0 0; font-size: 12px; font-weight: normal; }
    .bfd-stream { width: 70px; height: 3px; background-color: #4CAF50; position: relative; }
    .bfd-stream::before { content: ''; position: absolute; right: -10px; top: -5px; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-left: 10px solid #4CAF50; }
    .side-stream { position: absolute; left: 50%; transform: translateX(-50%); width: 3px; height: 40px; background-color: #FF9800; bottom: -40px; }
    .side-stream-label { position: absolute; bottom: -65px; left: 50%; transform: translateX(-50%); font-size: 11px; white-space: nowrap; color: #FF9800; }
</style>
"""

# --- 3. Simulation Core Logic (UNCHANGED) ---
def simulate_torrefaction(biomass, moisture, temp_C, duration_min, size, initial_mass_kg, reactor_type):
    temp_K = temp_C + 273.15
    data = EMPIRICAL_DATA.get(biomass)
    R_GAS_LOCAL = R_GAS 
    
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
    ash_concentration_percent = (initial_ash_frac / current_total_mass_fraction) * 100
    
    final_moisture_loss = initial_moisture_frac
    final_volatiles_remaining = volatiles_curve[-1]
    final_volatiles_lost = initial_volatiles_frac - final_volatiles_remaining
    
    final_solid_fraction = 1.0 - final_moisture_loss - final_volatiles_lost
    mass_biochar_total = final_solid_fraction * initial_mass_kg
    final_ash_percent = (mass_ash_kg / mass_biochar_total) * 100

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

    gas_fraction = final_volatiles_lost * data["Gas_Factor"]
    gas_comp_mass_fractions = {"CO2": 0.45, "CO": 0.35, "CH4": 0.15, "H2": 0.05}
    
    total_volatiles_lost_mass = final_volatiles_lost * initial_mass_kg
    
    gas_composition_molar_data = {}
    if total_volatiles_lost_mass > 0.001:
        for gas, fraction in gas_comp_mass_fractions.items():
            gas_composition_molar_data[gas] = (fraction * 100) 
    
    gas_composition_molar = pd.DataFrame.from_dict(
        gas_composition_molar_data, 
        orient="index", columns=["Molar % in Dry Gas"]
    ).fillna(0)
    if not gas_composition_molar.empty and gas_composition_molar["Molar % in Dry Gas"].sum() > 0:
        gas_composition_molar = gas_composition_molar / gas_composition_molar["Molar % in Dry Gas"].sum() * 100 
    else:
        gas_composition_molar = pd.DataFrame(0, index=["CO2", "CO", "CH4", "H2"], columns=["Molar % in Dry Gas"])


    mass_profile = pd.DataFrame({
        "Time (min)": t,
        "Total Mass Yield (%)": current_total_mass_fraction * 100,
        "Ash Concentration in Solid (%)": ash_concentration_percent
    }).set_index("Time (min)")
    
    return {
        "yields_percent": yields_percent,
        "yields_mass": yields_mass,
        "solid_composition": solid_composition,
        "final_ash_percent": final_ash_percent,
        "gas_composition_molar": gas_composition_molar,
        "mass_profile": mass_profile,
        "k_devol_eff": k_devol_eff,
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
    
    # -- 1. Header with Logo (for PDF) --
    try:
        # Load the logo for PDF
        img_path = "chemisco_logo.png" 
        logo_pdf = ReportImage(img_path, width=1.5*inch, height=1.5*inch)
        logo_pdf.hAlign = 'CENTER'
        elements.append(logo_pdf)
    except FileNotFoundError:
        elements.append(Paragraph("CHEMISCO", title_style)) # Fallback text
        
    elements.append(Paragraph("CHEMISCO REPORT", title_style))
    elements.append(Paragraph("Project presented to: Dr. Amr El-Rifai", styles["Heading3"]))
    elements.append(Paragraph(f"Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # -- 2. Parameters Table --
    elements.append(Paragraph("1. Simulation Parameters & Config", heading_style))
    p = results["parameters"]
    param_data = [
        ["Parameter", "Value"],
        ["Biomass Type", p['biomass']],
        ["Reactor Type", p['reactor']], 
        ["Initial Mass", f"{p['initial_mass']} kg"],
        ["Moisture Content", f"{p['moisture']}%"],
        ["Temperature", f"{p['temperature']} °C"],
        ["Duration", f"{p['duration']} min"],
        ["Particle Size", p["size"]]
    ]
    
    t_param = Table(param_data, colWidths=[3*inch, 3*inch])
    t_param.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E8F5E9")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
    ]))
    elements.append(t_param)
    elements.append(Spacer(1, 0.2*inch))
    
    # -- 3. Yields Table --
    elements.append(Paragraph("2. Product Yields (Mass Balance)", heading_style))
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
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph(f"<b>Final Ash Concentration:</b> {results['final_ash_percent']:.2f}%", normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # -- 4. Visualizations (Code remains the same) --
    elements.append(Paragraph("3. Results Visualization", heading_style))
    
    def get_image_bytes(fig):
        img_buf = BytesIO()
        fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=150, transparent=True)
        img_buf.seek(0)
        return img_buf

    # Pie Charts (matplotlib for reportlab compatibility)
    fig_pie1, ax_pie1 = plt.subplots(figsize=(4.5, 4.5)) 
    colors_solid_pdf = ['#6A1B9A', '#AB47BC', '#BDBDBD']
    ax_pie1.pie(results["solid_composition"]["Mass (kg)"], labels=results["solid_composition"].index, 
                autopct='%1.1f%%', colors=colors_solid_pdf, startangle=140, pctdistance=0.85, 
                textprops={'fontsize': 8})
    ax_pie1.set_title("Solid Product Composition", fontsize=10, weight='bold')
    img1 = ReportImage(get_image_bytes(fig_pie1), width=3.25*inch, height=3.25*inch) 
    
    fig_pie2, ax_pie2 = plt.subplots(figsize=(4.5, 4.5))
    filtered_yields = results["yields_percent"].iloc[[0, 1, 2]]
    colors_global_pdf = ['#388E3C', '#7CB342', '#C5E1A5']
    ax_pie2.pie(filtered_yields["Yield (%)"], labels=filtered_yields.index, 
                autopct='%1.1f%%', colors=colors_global_pdf, startangle=90, pctdistance=0.85,
                textprops={'fontsize': 8})
    ax_pie2.set_title("Global Mass Balance", fontsize=10, weight='bold')
    img2 = ReportImage(get_image_bytes(fig_pie2), width=3.25*inch, height=3.25*inch)
    
    t_pies = Table([[img1, img2]], colWidths=[3.7*inch, 3.7*inch])
    elements.append(t_pies)
    plt.close(fig_pie1)
    plt.close(fig_pie2)
    elements.append(Spacer(1, 0.2*inch)) 
    
    # Line Chart
    fig_line, ax1 = plt.subplots(figsize=(8, 4))
    color_mass = '#388E3C'
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Total Mass Remaining (%)', color=color_mass, weight='bold')
    ax1.plot(results["mass_profile"].index, results["mass_profile"]["Total Mass Yield (%)"], color=color_mass, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color_mass)
    ax1.grid(True, alpha=0.4, linestyle='--', color='lightgrey') 
    
    ax2 = ax1.twinx()
    color_ash = '#D32F2F'
    ax2.set_ylabel('Ash Concentration (%)', color=color_ash, weight='bold')
    ax2.plot(results["mass_profile"].index, results["mass_profile"]["Ash Concentration in Solid (%)"], color=color_ash, linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color_ash)
    
    plt.title("Mass Depletion vs. Ash Enrichment", fontsize=12)
    img_line = ReportImage(get_image_bytes(fig_line), width=6.5*inch, height=3.25*inch)
    elements.append(img_line)
    elements.append(Spacer(1, 0.1*inch))
    plt.close(fig_line)

    # Bar Chart
    fig_bar, ax_bar = plt.subplots(figsize=(8, 3))
    results["gas_composition_molar"].plot(kind='bar', ax=ax_bar, legend=False, color='#1565C0')
    ax_bar.set_title("Dry Gas Composition (Molar %)", fontsize=12)
    ax_bar.set_ylabel("Molar %")
    plt.xticks(rotation=0)
    ax_bar.grid(axis='y', alpha=0.4, linestyle='--', color='lightgrey')
    img_bar = ReportImage(get_image_bytes(fig_bar), width=6.5*inch, height=2.5*inch)
    elements.append(img_bar)
    plt.close(fig_bar)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# --- 5. Main Streamlit App (Simplified Logo Display) ---
def main():
    st.set_page_config(page_title="Chemisco", layout="wide", initial_sidebar_state="expanded")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    
    LOGO_PATH = "chemisco_logo.png"

    # --- Sidebar ---
    with st.sidebar:
        # 1. SIMPLE AND ROBUST LOGO DISPLAY (Sidebar)
        try:
            st.image(LOGO_PATH, use_column_width=True) 
        except FileNotFoundError:
            st.info("Logo not found. Please ensure 'chemisco_logo.png' is in the same folder.")

        st.markdown("""
            <div style='text-align: center; padding: 15px; border-radius: 8px; background-color: #1B5E20; margin-top: 15px;'>
                <h1 style='color: white; margin: 0; font-size: 2.2em; letter-spacing: 1px;'>CHEMISCO</h1>
                <p style='color: #A5D6A7; margin: 0; font-size: 0.9em;'>Torrefaction Process Simulator</p>
                <hr style='margin: 10px 0; border-color: #4CAF50;'>
                <p style='color: #C8E6C9; font-size: 0.85em;'>Project presented to:</p>
                <h3 style='color: #FFF176; margin: 0;'>Dr. Amr El-Rifai</h3>
            </div>
            """, unsafe_allow_html=True)
        
        st.header("⚙️ Input Parameters")
        
        # Group 1: Material
        with st.expander("🌲 Biomass Properties", expanded=True):
            initial_mass_kg = st.number_input("Initial Biomass Mass (kg)", min_value=1.0, value=100.0, step=10.0)
            biomass_type = st.selectbox("Biomass Type", list(EMPIRICAL_DATA.keys()))
            moisture_content = st.slider("Initial Moisture Content (%)", 0.0, 50.0, 10.0, step=1.0)
            particle_size = st.selectbox("Particle Size", list(SIZE_FACTOR.keys()))
            
        # Group 2: Process 
        with st.expander("🌡️ Process Conditions", expanded=True):
            reactor_type = st.selectbox("Reactor Type", 
                ["Rotary Drum Reactor", "Fluidized Bed Reactor", "Auger/Screw Reactor", "Fixed Bed Reactor"])
            
            temperature = st.slider("Torrefaction Temperature (°C)", 200, 350, 275, step=5)
            duration = st.slider("Process Duration (min)", 10, 120, 45, step=5)
            ash_percent_init = EMPIRICAL_DATA[biomass_type]["Ash"] * 100
            st.info(f"Initial Ash Content: **{ash_percent_init:.1f}%**")
            
        # Group 3: Cost Management
        with st.expander("💰 Cost Management", expanded=False):
            st.caption("Economic Feasibility Parameters")
            cost_biomass_per_ton = st.number_input("Biomass Feedstock Cost ($/ton)", min_value=0.0, value=30.0, step=5.0)
            cost_energy_per_hour = st.number_input("Operational/Energy Cost ($/hour)", min_value=0.0, value=5.0, step=0.5, help="Total cost of electricity + labor per hour of operation")
            price_biochar_per_kg = st.number_input("Biochar Selling Price ($/kg)", min_value=0.0, value=1.20, step=0.1)
        
        # Game Mode Toggle
        st.markdown("---")
        st.subheader("🎮 Gamification")
        game_mode = st.checkbox("Activate 'Plant Manager Challenge'", value=False)

    # --- Main Banner (Using st.columns and st.image for robust display) ---
    col_logo, col_text = st.columns([1, 4])
    
    with col_logo:
        # 2. SIMPLE AND ROBUST LOGO DISPLAY (Main Banner)
        try:
            st.image(LOGO_PATH, width=150)
        except FileNotFoundError:
            st.write(" ") # Space placeholder if logo fails

    with col_text:
        st.markdown(f"""
            <div style="background-color: #388E3C; padding: 15px 15px 15px 0; border-radius: 12px; margin-left: -50px; text-align: left;">
                <h1 style="color: #FFFFFF; margin: 0; font-size: 2.5em; font-weight: 800; letter-spacing: 2px;">CHEMISCO</h1>
                <p style="color: #C8E6C9; margin-top: 5px; font-size: 1em;">Advanced Torrefaction Process Simulator</p>
                <div style="color: #FFEB3B; font-weight: bold; font-size: 0.9em; margin-top: 10px;">Project presented to Dr. Amr El-Rifai</div>
            </div>
            """, unsafe_allow_html=True)
    
    # BFD
    st.subheader("Process Flow Block Diagram (BFD)")
    # ... (BFD HTML remains the same)
    bfd_html = f"""
    <div class="bfd-container">
        <div class="bfd-block">
            FEED PREPARATION
            <p style="color: #1565C0;">Mass: {initial_mass_kg:.0f} kg</p>
            <p style="color: #0277BD;">Moist: {moisture_content:.1f}%</p>
        </div>
        <div class="bfd-stream"></div>
        <div class="bfd-block">
            DRYING
            <p>100-200 °C</p>
            <div class="side-stream"></div>
            <div class="side-stream-label">Water Vapor</div>
        </div>
        <div class="bfd-stream"></div>
        <div class="bfd-block" style="border-color: #D32F2F; background-color: #FFCDD2; color: #B71C1C;">
            {reactor_type.upper()}
            <p style="color: #B71C1C;">{temperature} °C | {duration} min</p>
            <div class="side-stream" style="background-color: #FFC107;"></div>
            <div class="side-stream-label" style="color: #FFC107;">Volatile Gases</div>
        </div>
        <div class="bfd-stream"></div>
        <div class="bfd-block" style="border-color: #388E3C; background-color: #C8E6C9; color: #1B5E20;">
            PRODUCT COOLING
            <p>Biochar</p>
        </div>
    </div>
    <div style="height: 40px;"></div>
    """
    st.markdown(bfd_html, unsafe_allow_html=True)
    
    # Input validation
    if moisture_content / 100 + EMPIRICAL_DATA[biomass_type]["Ash"] > 1:
        st.error("**Input Error:** Initial Moisture and Ash content exceed 100%. Please adjust input parameters.")
        return 
        
    # Run Simulation
    results = simulate_torrefaction(biomass_type, moisture_content, temperature, duration, particle_size, initial_mass_kg, reactor_type)
    
    # --- GAME LOGIC SECTION ---
    if game_mode:
        st.markdown("---")
        st.markdown("""
        <div style="background-color: #E8F5E9; padding: 20px; border-radius: 10px; border-left: 6px solid #2E7D32; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style="color: #1B5E20; margin-top:0;">🏭 Plant Manager Challenge</h3>
            <p style="color: #388E3C; font-size: 1.1em;">The client has sent specific requirements for the Biochar. Adjust <b>Temperature</b> and <b>Duration</b> to match them!</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'target_yield' not in st.session_state:
            st.session_state.target_yield = random.randint(60, 85)
            st.session_state.target_ash = round(random.uniform(ash_percent_init + 1.0, ash_percent_init + 5.0), 1)
            st.session_state.has_won = False

        col_g1, col_g2, col_g3 = st.columns([1.5, 2, 1])
        
        with col_g1:
            st.info(f"📋 **CLIENT ORDER:**\n\n🎯 Target Yield: **{st.session_state.target_yield}%**\n\n🎯 Max Ash: **{st.session_state.target_ash}%**")
            
        with col_g2:
            curr_yield = results["yields_percent"].loc["Biochar (Solid Product)", "Yield (%)"]
            curr_ash = results["final_ash_percent"]
            diff_yield = abs(curr_yield - st.session_state.target_yield)
            diff_ash = abs(curr_ash - st.session_state.target_ash)
            score = max(0, 100 - (diff_yield * 2 + diff_ash * 5))
            st.metric("🏆 Your Efficiency Score", f"{score:.1f} / 100")
            
            if score >= 90:
                st.success("🌟 PERFECT MATCH! Order fulfilled successfully.")
                if not st.session_state.has_won:
                    st.balloons()
                    st.session_state.has_won = True
            elif score >= 70:
                st.warning("⚠️ Acceptable, but try to optimize further.")
            else:
                st.error("❌ Specification mismatch. Quality too low.")
                
        with col_g3:
            st.write("###") 
            if st.button("🔄 New Client Order"):
                del st.session_state['target_yield']
                del st.session_state['target_ash']
                st.session_state.has_won = False
                st.rerun()
        st.markdown("---")
    # --------------------------

    # --- Display Results ---
    st.header("📊 Simulation Results & Analysis")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Yields & Ash", "Kinetics", "Gas Analysis", "💰 Cost Analysis", "PDF Report"])
    
    with tab1:
        st.subheader(f"Product Yields & Ash Enrichment")
        col_m1, col_m2, col_m3 = st.columns(3)
        
        biochar_mass = results["yields_mass"].loc["Biochar (Solid Product)", "Mass (kg)"]
        col_m1.metric("⚖️ Total Biochar Mass", f"{biochar_mass:.2f} kg")
        
        final_ash = results["final_ash_percent"]
        ash_increase = final_ash - ash_percent_init
        col_m2.metric("⚗️ Final Ash Concentration", f"{final_ash:.2f} %", delta=f"+{ash_increase:.2f}% (Enrichment)")
        
        moisture_loss = results["yields_mass"].loc["Moisture Loss (Water Vapor)", "Mass (kg)"]
        col_m3.metric("💧 Moisture Removed", f"{moisture_loss:.2f} kg")

        st.markdown("---")
        
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            st.markdown("##### Final Biochar Composition")
            df_solid = results["solid_composition"].reset_index()
            df_solid.columns = ["Component", "Mass (kg)"]
            fig1 = px.pie(df_solid, values='Mass (kg)', names='Component', hole=0.5,
                          color='Component',
                          color_discrete_map={"Fixed Carbon": "#6A1B9A", "Remaining Volatiles": "#AB47BC", "Ash": "#BDBDBD"})
            fig1.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.2), margin=dict(t=20, b=50))
            st.plotly_chart(fig1, use_container_width=True)

        with col_t2:
            st.markdown("##### Global Mass Balance")
            filtered_yields = results["yields_percent"].iloc[[0, 1, 2]].reset_index()
            filtered_yields.columns = ["Component", "Yield (%)"]
            fig2 = px.pie(filtered_yields, values='Yield (%)', names='Component', hole=0.5,
                          color='Component',
                          color_discrete_map={"Biochar (Solid Product)": "#388E3C", "Non-Condensable Gases": "#7CB342", "Moisture Loss (Water Vapor)": "#C5E1A5"})
            fig2.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.2), margin=dict(t=20, b=50))
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Mass Depletion Kinetics")
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Scatter(x=results["mass_profile"].index, y=results["mass_profile"]["Total Mass Yield (%)"], name="Total Mass %", line=dict(color="#4CAF50", width=3), yaxis="y1"))
        fig_dual.add_trace(go.Scatter(x=results["mass_profile"].index, y=results["mass_profile"]["Ash Concentration in Solid (%)"], name="Ash Concentration %", line=dict(color="#D32F2F", width=3, dash='dot'), yaxis="y2"))
        fig_dual.update_layout(
            xaxis=dict(title="Time (min)"),
            yaxis=dict(title="Total Mass Remaining (%)", title_font=dict(color="#4CAF50"), tickfont=dict(color="#4CAF50")),
            yaxis2=dict(title="Ash Concentration (%)", title_font=dict(color="#D32F2F"), tickfont=dict(color="#D32F2F"), overlaying="y", side="right"),
            legend=dict(x=0.1, y=1.1, orientation="h"), height=450
        )
        st.plotly_chart(fig_dual, use_container_width=True)

    with tab3:
        st.subheader("Gas Composition")
        st.bar_chart(results["gas_composition_molar"])

    with tab4:
        st.subheader("💰 Economic Feasibility Analysis")
        cost_feedstock_total = (initial_mass_kg / 1000) * cost_biomass_per_ton
        hours = duration / 60
        cost_operations_total = hours * cost_energy_per_hour
        total_cost = cost_feedstock_total + cost_operations_total
        biochar_produced_kg = results["yields_mass"].loc["Biochar (Solid Product)", "Mass (kg)"]
        revenue_total = biochar_produced_kg * price_biochar_per_kg
        net_profit = revenue_total - total_cost
        roi = (net_profit / total_cost) * 100 if total_cost > 0 else 0
        
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        col_c1.metric("📉 Total Cost", f"${total_cost:.2f}")
        col_c2.metric("📈 Total Revenue", f"${revenue_total:.2f}")
        col_c3.metric("💵 Net Profit", f"${net_profit:.2f}", delta=f"${abs(net_profit):.2f}", delta_color="normal" if net_profit > 0 else "inverse")
        col_c4.metric("📊 ROI", f"{roi:.1f}%", delta=f"{abs(roi):.1f}%", delta_color="normal" if roi > 0 else "inverse")
        
        st.markdown("---")
        
        # Waterfall Chart 
        fig_waterfall = go.Figure(go.Waterfall(
            name = "Cash Flow", orientation = "v",
            measure = ["relative", "relative", "relative", "total"],
            x = ["Gross Revenue", "Feedstock Cost", "Operational Cost", "Net Profit"],
            textposition = "outside",
            text = [f"${revenue_total:.1f}", f"${-cost_feedstock_total:.1f}", f"${-cost_operations_total:.1f}", f"${net_profit:.1f}"],
            y = [revenue_total, -cost_feedstock_total, -cost_operations_total, net_profit],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
            decreasing = {"marker":{"color":"#EF5350"}},
            increasing = {"marker":{"color":"#66BB6A"}},
            totals = {"marker":{"color":"#42A5F5"}}
        ))
        fig_waterfall.update_layout(title = "Cash Flow Waterfall Chart (USD)", showlegend = False, height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        st.info(f"""
        **Analysis:**
        Producing **{biochar_produced_kg:.1f} kg** of biochar from **{initial_mass_kg} kg** biomass.
        Break-even selling price: **${(total_cost/biochar_produced_kg):.2f} / kg**.
        """)

    with tab5:
        st.subheader("Download Professional Report")
        if st.button("⬇️ Generate PDF Report"):
            pdf_buffer = generate_pdf_report(results)
            st.download_button(
                label="Download Report",
                data=pdf_buffer,
                file_name=f"Chemisco_Torrefaction_Report.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main()

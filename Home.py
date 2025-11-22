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

# --- 1. Constants ---
R_GAS = 8.314
EMPIRICAL_DATA = {
    "Wood": {"A": 2.5e10, "Ea": 135000, "k_drying_base": 0.05, "Ash": 0.02, "Gas_Factor": 0.35},
    "Agricultural Waste": {"A": 5.0e11, "Ea": 150000, "k_drying_base": 0.07, "Ash": 0.08, "Gas_Factor": 0.45},
    "Municipal Waste": {"A": 1.0e12, "Ea": 165000, "k_drying_base": 0.10, "Ash": 0.15, "Gas_Factor": 0.55}
}
SIZE_FACTOR = {"Fine (<1mm)": 1.0, "Medium (1-5mm)": 0.85, "Coarse (>5mm)": 0.65}

# --- 2. CSS (Safe Mode) ---
GLOBAL_CSS = """
<style>
    .main-header {
        background-color: #2E7D32;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
    }
    .sub-text { color: #A5D6A7; font-size: 14px; }
    .dedication-text { color: #FFEB3B; font-weight: bold; margin-top: 10px; }
</style>
"""

# --- 3. Logic ---
def simulate_torrefaction(biomass, moisture, temp_C, duration_min, size, initial_mass_kg, reactor_type):
    temp_K = temp_C + 273.15
    data = EMPIRICAL_DATA.get(biomass)
    k_devol_arrhenius = data["A"] * np.exp(-data["Ea"] / (R_GAS * temp_K))
    k_devol_eff = k_devol_arrhenius * SIZE_FACTOR.get(size)
    k_drying = data["k_drying_base"] 
    initial_moisture_frac = moisture / 100
    initial_ash_frac = data["Ash"]
    initial_volatiles_frac = 1.0 - initial_moisture_frac - initial_ash_frac
    mass_ash_kg = initial_mass_kg * initial_ash_frac
    
    def model(y, t, k1, k2):
        m_moist, m_vol = y
        d_moist = -k1 * m_moist if m_moist > 0.001 else 0
        d_vol = -k2 * m_vol
        return [d_moist, d_vol]
    
    t = np.linspace(0, duration_min, 100)
    y0 = [initial_moisture_frac, initial_volatiles_frac]
    sol = odeint(model, y0, t, args=(k_drying, k_devol_eff))
    sol[sol < 0] = 0 # Prevent negative mass
    
    moisture_curve = sol[:, 0] 
    volatiles_curve = sol[:, 1]
    fixed_carbon_frac_initial = 1.0 - initial_moisture_frac - initial_volatiles_frac - initial_ash_frac
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
    gas_comp_mass = {"CO2": 0.45, "CO": 0.35, "CH4": 0.15, "H2": 0.05}
    
    # Safe division
    div = final_volatiles_lost if final_volatiles_lost > 0.001 else 1.0
    
    gas_composition_molar = pd.DataFrame.from_dict(
        {k: (v * gas_fraction * initial_mass_kg) * 100 / div for k, v in gas_comp_mass.items()}, 
        orient="index", columns=["Molar %"]
    ).fillna(0)

    mass_profile = pd.DataFrame({
        "Time (min)": t,
        "Total Mass Yield (%)": current_total_mass_fraction * 100,
        "Ash Concentration in Solid (%)": ash_concentration_percent
    }).set_index("Time (min)")
    
    return {
        "yields_percent": yields_percent, "yields_mass": yields_mass,
        "solid_composition": solid_composition, "final_ash_percent": final_ash_percent,
        "gas_composition_molar": gas_composition_molar, "mass_profile": mass_profile,
        "k_devol_eff": k_devol_eff,
        "parameters": {"biomass": biomass, "moisture": moisture, "temperature": temp_C, "duration": duration_min, "size": size, "initial_mass": initial_mass_kg, "reactor": reactor_type}
    }

def generate_pdf_report(results):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("CHEMISCO REPORT", styles["Title"]))
    elements.append(Paragraph("Presented to: Dr. Amr El-Rifai", styles["Heading3"]))
    elements.append(Spacer(1, 0.2*inch))
    
    # Simple Data Table
    data = [["Parameter", "Value"]]
    for k, v in results['parameters'].items():
        data.append([k, str(v)])
    t = Table(data)
    t.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
    elements.append(t)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# --- 4. Main App ---
def main():
    st.set_page_config(page_title="Chemisco", layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # --- Banner ---
    st.markdown("""
        <div class="main-header">
            <h1>CHEMISCO</h1>
            <p class="sub-text">Advanced Torrefaction Simulator</p>
            <p class="dedication-text">Project presented to Dr. Amr El-Rifai</p>
        </div>
    """, unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Inputs")
        initial_mass = st.number_input("Mass (kg)", 10.0, 1000.0, 100.0)
        biomass_type = st.selectbox("Biomass", list(EMPIRICAL_DATA.keys()))
        reactor_type = st.selectbox("Reactor", ["Rotary Drum", "Fluidized Bed", "Screw Reactor"])
        moisture = st.slider("Moisture %", 0, 50, 10)
        temp = st.slider("Temp (°C)", 200, 350, 275)
        duration = st.slider("Duration (min)", 10, 120, 45)
        size = st.selectbox("Size", list(SIZE_FACTOR.keys()))
        
        st.markdown("---")
        st.subheader("💰 Economics")
        cost_feed = st.number_input("Feed Cost ($/ton)", value=30.0)
        cost_ops = st.number_input("Ops Cost ($/hr)", value=5.0)
        price_char = st.number_input("Selling Price ($/kg)", value=1.2)
        
        st.markdown("---")
        game_mode = st.checkbox("🎮 Plant Manager Game")

    # --- Game Logic ---
    if game_mode:
        if 'target_yield' not in st.session_state:
             st.session_state.target_yield = random.randint(65, 80)
        
        st.info(f"🎯 **Mission:** Achieve a Yield close to **{st.session_state.target_yield}%**")

    # --- Simulation ---
    results = simulate_torrefaction(biomass_type, moisture, temp, duration, size, initial_mass, reactor_type)
    
    # --- Results ---
    t1, t2, t3, t4 = st.tabs(["Yields", "Kinetics", "Economics", "Report"])
    
    with t1:
        col1, col2, col3 = st.columns(3)
        mass_char = results["yields_mass"].loc["Biochar (Solid Product)", "Mass (kg)"]
        ash_final = results["final_ash_percent"]
        
        col1.metric("Biochar Mass", f"{mass_char:.2f} kg")
        col2.metric("Final Ash", f"{ash_final:.2f} %")
        col3.metric("Yield", f"{results['yields_percent'].iloc[0,0]:.1f} %")
        
        # Game Feedback
        if game_mode:
            current_yield = results['yields_percent'].iloc[0,0]
            diff = abs(current_yield - st.session_state.target_yield)
            if diff < 2:
                st.success("🏆 You Won! Target Achieved.")
            else:
                st.warning(f"Try adjusting Temp/Duration. You are {diff:.1f}% away.")

        st.plotly_chart(px.pie(results["solid_composition"].reset_index(), values="Mass (kg)", names="index", title="Composition"))

    with t2:
        st.plotly_chart(px.line(results["mass_profile"], y=["Total Mass Yield (%)", "Ash Concentration in Solid (%)"], title="Kinetics"))

    with t3:
        cost_total = (initial_mass/1000)*cost_feed + (duration/60)*cost_ops
        revenue = mass_char * price_char
        profit = revenue - cost_total
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Cost", f"${cost_total:.2f}")
        c2.metric("Revenue", f"${revenue:.2f}")
        c3.metric("Profit", f"${profit:.2f}", delta_color="normal" if profit > 0 else "inverse")
        
        st.caption(f"Break-even price: ${(cost_total/mass_char):.2f}/kg")

    with t4:
        if st.button("Download PDF"):
            pdf = generate_pdf_report(results)
            st.download_button("Download", pdf, "report.pdf", "application/pdf")

if __name__ == "__main__":
    main()

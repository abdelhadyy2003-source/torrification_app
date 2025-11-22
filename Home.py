import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportImage, Table
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from reportlab.lib.units import inch
from reportlab.lib import colors

# --- 1. Chemical and Empirical Constants ---
R_GAS = 8.314  # Universal Gas Constant (J/mol·K)

EMPIRICAL_DATA = {
    "Wood": {
        "A": 2.5e10, "Ea": 135000, "k_drying_base": 0.05, 
        "Ash": 0.02, "Gas_Factor": 0.35
    },
    "Agricultural Waste": {
        "A": 5.0e11, "Ea": 150000, "k_drying_base": 0.07, 
        "Ash": 0.08, "Gas_Factor": 0.45
    },
    "Municipal Waste": {
        "A": 1.0e12, "Ea": 165000, "k_drying_base": 0.10, 
        "Ash": 0.15, "Gas_Factor": 0.55
    }
}

SIZE_FACTOR = {
    "Fine (<1mm)": 1.0,
    "Medium (1-5mm)": 0.85,
    "Coarse (>5mm)": 0.65
}

# --- 2. Simulation Function (simulate_torrefaction) ---
def simulate_torrefaction(biomass, moisture, temp_C, duration_min, size, initial_mass_kg):
    """Core torrefaction simulation logic using Arrhenius and particle size correction."""
    temp_K = temp_C + 273.15
    data = EMPIRICAL_DATA.get(biomass)
    
    # 1. Calculate effective devolatilization rate
    k_devol_arrhenius = data["A"] * np.exp(-data["Ea"] / (R_GAS * temp_K))
    k_devol_eff = k_devol_arrhenius * SIZE_FACTOR.get(size)
    k_drying = data["k_drying_base"]
    ash_content = data["Ash"]

    # 2. ODE Model (mass fractions)
    def model(y, t, k1, k2):
        moisture, volatiles = y
        d_moisture = -k1 * moisture if moisture > 0.001 else 0
        d_volatiles = -k2 * volatiles
        return [d_moisture, d_volatiles]
    
    t = np.linspace(0, duration_min, 100)
    initial_moisture_fraction = moisture / 100
    initial_volatiles_fraction = 1 - initial_moisture_fraction - ash_content
    y0 = [initial_moisture_fraction, initial_volatiles_fraction]
        
    sol = odeint(model, y0, t, args=(k_drying, k_devol_eff))
    sol[sol < 0] = 0

    # 3. Calculate final results (Fractions and Mass in kg)
    final_moisture = sol[-1, 0]
    final_volatiles_remaining = sol[-1, 1]
    
    # Fractions
    final_biochar_fraction = (1 - final_moisture - final_volatiles_remaining - ash_content)
    final_volatiles_lost_fraction = initial_volatiles_fraction - final_volatiles_remaining
    moisture_lost_fraction = initial_moisture_fraction - final_moisture
    
    # Yields (%)
    yields_percent = pd.DataFrame({
        "Yield (%)": [
            (final_biochar_fraction + ash_content) * 100,
            final_volatiles_lost_fraction * 100,
            moisture_lost_fraction * 100,
            ash_content * 100
        ]},
        index=["Biochar (Solid) & Ash", "Non-Condensable Gases", "Moisture Loss (Water Vapor)", "Initial Ash Content"]
    )
    
    # Yields (Mass in kg)
    yields_mass = yields_percent.copy()
    yields_mass["Mass (kg)"] = yields_percent["Yield (%)"] * initial_mass_kg / 100
    yields_mass.drop(columns=["Yield (%)"], inplace=True)

    # Gas Composition
    gas_fraction = final_volatiles_lost_fraction * data["Gas_Factor"]
    
    gas_comp_mass = {
        "CO2": 0.45 * gas_fraction * initial_mass_kg,
        "CO": 0.35 * gas_fraction * initial_mass_kg,
        "CH4": 0.15 * gas_fraction * initial_mass_kg,
        "H2": 0.05 * gas_fraction * initial_mass_kg
    }
    
    gas_composition_molar = pd.DataFrame.from_dict(
        {k: v * 100 / final_volatiles_lost_fraction for k, v in gas_comp_mass.items() if final_volatiles_lost_fraction > 0.001}, 
        orient="index", columns=["Molar % in Dry Gas"]
    ).fillna(0)

    # Mass Profile
    mass_profile = pd.DataFrame({
        "Time (min)": t,
        "Moisture Fraction": sol[:, 0],
        "Volatiles Fraction": sol[:, 1],
        "Biochar Fraction": 1 - sol[:, 0] - sol[:, 1] - ash_content,
    }).set_index("Time (min)")
    
    return {
        "yields_percent": yields_percent,
        "yields_mass": yields_mass,
        "temp_profile": pd.DataFrame({"Temperature (°C)": temp_C * np.ones_like(t)}, index=t),
        "gas_composition_molar": gas_composition_molar,
        "mass_profile": mass_profile,
        "k_devol_eff": k_devol_eff,
        "parameters": {
            "biomass": biomass, "moisture": moisture, "temperature": temp_C, 
            "duration": duration_min, "size": size, "initial_mass": initial_mass_kg
        }
    }

# --- 3. Streamlit Main App (main) ---
def main():
    # Streamlit Config: (Supports dark/light mode based on user's system/browser settings)
    st.set_page_config(page_title="Chemisco Pro Torrefaction Simulator", layout="wide", initial_sidebar_state="expanded")
    
    # 3.1. Sidebar (Logo and Inputs)
    with st.sidebar:
        # Logo Placeholder (Stylized Banner)
        st.markdown(
            """
            <div style='text-align: center; padding: 10px; border-radius: 5px; background-color: #4CAF50;'>
                <h1 style='color: white; margin: 0;'>CHEMISCO PRO</h1>
                <p style='color: white; margin: 0;'>Torrefaction Analytics</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.header("⚙️ Input Parameters")
        
        # Input Sections
        with st.expander("Biomass Properties", expanded=True):
            initial_mass_kg = st.number_input("Initial Biomass Mass (kg)", min_value=1.0, value=100.0, step=10.0)
            biomass_type = st.selectbox("Biomass Type", list(EMPIRICAL_DATA.keys()))
            moisture_content = st.slider("Initial Moisture Content (%)", 0.0, 50.0, 10.0, step=1.0)
            particle_size = st.selectbox("Particle Size", list(SIZE_FACTOR.keys()))
        
        with st.expander("Process Conditions", expanded=True):
            temperature = st.slider("Torrefaction Temperature (°C)", 200, 350, 275, step=5)
            duration = st.slider("Process Duration (min)", 10, 120, 45, step=5)
            
            ash_percent = EMPIRICAL_DATA[biomass_type]["Ash"] * 100
            st.info(f"Assumed Initial Ash Content: **{ash_percent:.1f}%**")
            
    # 3.2. Main Content (Banner and Flow Sheet)
    
    # Banner
    st.markdown(
        """
        <div style='background-color: #4CAF50; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;'>
            <h1 style='color: white; margin: 0;'>🔥 Advanced Torrefaction Simulator</h1>
            <p style='color: white; margin: 0;'>Enhanced Kinetic Model for Process Optimization</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Process Flow Sheet (Block Flow Diagram - BFD Style)
    st.subheader("Process Flow Block Diagram (BFD)")
    
    # Define CSS styles for the BFD
    bfd_style = """
    <style>
        .bfd-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 30px 0 60px 0;
            position: relative;
        }
        .bfd-block {
            padding: 15px 25px;
            border: 3px solid #4CAF50; /* Green border for blocks */
            border-radius: 6px;
            text-align: center;
            background-color: #E8F5E9; /* Light green background */
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            font-weight: bold;
            color: #1B5E20; /* Dark green text */
            position: relative;
            min-width: 180px;
        }
        .bfd-stream {
            width: 70px;
            height: 3px;
            background-color: #4CAF50;
            position: relative;
        }
        .bfd-stream::before { /* Arrowhead for main streams */
            content: '';
            position: absolute;
            right: -10px;
            top: -5px;
            border-top: 6px solid transparent;
            border-bottom: 6px solid transparent;
            border-left: 10px solid #4CAF50;
        }
        .side-stream {
            position: absolute;
            left: 50%;
            transform: translateX(-50%);
            width: 3px;
            height: 40px;
            background-color: #FF9800; /* Orange for side streams */
            bottom: -40px;
        }
        .side-stream-label {
            position: absolute;
            bottom: -65px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 11px;
            white-space: nowrap;
            color: #FF9800;
        }
    </style>
    """
    st.markdown(bfd_style, unsafe_allow_html=True)

    # HTML structure for the BFD
    bfd_html = f"""
    <div class="bfd-container">
        
        <div class="bfd-block">
            FEED PREPARATION
            <p style="font-size: 12px; margin: 5px 0 0;">Raw Biomass (M={moisture_content}%)</p>
        </div>

        <div class="bfd-stream"></div>

        <div class="bfd-block">
            DRYING & PREHEATING
            <p style="font-size: 12px; margin: 5px 0 0;">100-200 °C</p>
            <div class="side-stream"></div>
            <div class="side-stream-label">Water Vapor</div>
        </div>

        <div class="bfd-stream"></div>

        <div class="bfd-block" style="border-color: #D32F2F; background-color: #FFCDD2; color: #B71C1C;">
            TORREFACTION REACTOR
            <p style="font-size: 12px; margin: 5px 0 0;">{temperature} °C / {duration} min</p>
            <div class="side-stream" style="background-color: #FFC107;"></div>
            <div class="side-stream-label" style="color: #FFC107;">Volatile Gases</div>
        </div>

        <div class="bfd-stream"></div>

        <div class="bfd-block" style="border-color: #388E3C; background-color: #C8E6C9; color: #1B5E20;">
            COOLING & PRODUCT
            <p style="font-size: 12px; margin: 5px 0 0;">Torrefied Biochar</p>
        </div>
    </div>
    <div style="height: 40px;"></div> """
    st.markdown(bfd_html, unsafe_allow_html=True)
    
    # --- Run Simulation ---
    if moisture_content / 100 + EMPIRICAL_DATA[biomass_type]["Ash"] > 1:
        st.error("**Input Error:** Initial Moisture and Ash content exceed 100%. Please adjust the parameters.")
        return 
        
    results = simulate_torrefaction(biomass_type, moisture_content, temperature, duration, particle_size, initial_mass_kg)
    
    # --- Display Results ---
    st.header("📊 Simulation Results & Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["Yields & Mass Balance", "Mass Conversion Kinetics", "Gas Composition", "PDF Report"])
    
    with tab1:
        st.subheader(f"Product Yields (Based on {initial_mass_kg:.0f} kg Input)")
        col_m1, col_m2, col_m3 = st.columns(3)
        
        # Display Metrics
        biochar_mass_metric = results["yields_mass"].loc["Biochar (Solid) & Ash", "Mass (kg)"]
        col_m1.metric("⚖️ Total Solid Product (kg)", f"{biochar_mass_metric:.2f} kg", delta=f"{results['k_devol_eff']:.3f} min⁻¹ (Rate)")
        
        gas_mass_metric = results["yields_mass"].loc["Non-Condensable Gases", "Mass (kg)"]
        col_m2.metric("💨 Non-Condensable Gas Mass (kg)", f"{gas_mass_metric:.2f} kg")
        
        moisture_mass_metric = results["yields_mass"].loc["Moisture Loss (Water Vapor)", "Mass (kg)"]
        col_m3.metric("💧 Water Vapor Loss (kg)", f"{moisture_mass_metric:.2f} kg")

        st.markdown("---")
        
        # Tables and Pie Chart
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("Yield Distribution Tables")
            st.markdown("##### 1. Mass Yields (kg)")
            st.dataframe(results["yields_mass"].style.format("{:.2f}"), use_container_width=True)
            st.markdown("##### 2. Mass Fractions (%)")
            st.dataframe(results["yields_percent"].style.format("{:.2f}"), use_container_width=True)
        
        with col_t2:
            st.subheader("Mass Balance Pie Chart")
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            filtered_yields = results["yields_percent"].iloc[[0, 1, 2]] 
            ax1.pie(filtered_yields["Yield (%)"].values, labels=filtered_yields.index, autopct='%1.1f%%', startangle=90, colors=['#8B4513', '#A9A9A9', '#ADD8E6'])
            ax1.axis('equal')
            st.pyplot(fig1)

    with tab2:
        st.subheader("Mass Component Conversion Over Time")

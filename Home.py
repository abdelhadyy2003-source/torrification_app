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
import base64
import os 
import random 
import time 

# --- 1. Chemical and Kinetic Constants ---
R_GAS = 8.314  # J/(mol.K)
# Constants for Thermal Calculation (J/kg.K or kJ/kg.K)
CP_BIOMASS = 1300.0  # J/kg.K (Specific Heat of Biomass)
CP_WATER = 4180.0    # J/kg.K (Specific Heat of Water)
CP_BIOCHAR = 1000.0  # J/kg.K (Specific Heat of Biochar)
H_VAPOR = 2260000.0  # J/kg (Latent Heat of Vaporization of Water)
TEMP_REF_K = 298.15  # K (Reference Temperature: 25 C)

HHV_INITIAL = {"Wood": 18.0, "Agricultural Waste": 16.5, "Municipal Waste": 15.0}
HHV_ENRICHMENT_FACTOR = 1.3

KINETICS = {
    "Hemicellulose": [1.5e10, 110000],
    "Cellulose": [1.0e12, 130000],
    "Lignin": [2.0e9, 100000]
}
BIOMASS_COMPOSITION = {
    "Wood": {"Hemicellulose": 0.35, "Cellulose": 0.45, "Lignin": 0.20, "Ash": 0.02, "Gas_Factor": 0.40},
    "Agricultural Waste": {"Hemicellulose": 0.45, "Cellulose": 0.35, "Lignin": 0.20, "Ash": 0.08, "Gas_Factor": 0.50},
    "Municipal Waste": {"Hemicellulose": 0.30, "Cellulose": 0.40, "Lignin": 0.30, "Ash": 0.15, "Gas_Factor": 0.60}
}
DRYING_RATE_CONST = 0.05
SIZE_FACTOR = {"Fine (<1mm)": 1.0, "Medium (1-5mm)": 0.85, "Coarse (>5mm)": 0.65}
BASE_FC_FACTOR = 0.20

# --- Utility Functions ---
LOGO_PATH = "chemisco_logo.png"

def _get_image_base64(image_path):
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()
        else:
            return None
    except Exception:
        return None

LOGO_BASE64_STRING = _get_image_base64(LOGO_PATH)


# --- 2. Global CSS (Professional DARK Mode) ---
GLOBAL_CSS = """
<style>
    /* ------------------- DARK MODE BASE STYLING ------------------- */
    .stApp {
        padding-top: 10px;
        background-color: #1E1E1E; /* Deep Dark Background */
        color: #F5F5F5; /* Light Text */
    }

    /* Global Text and Headers */
    h1, h2, h3, p, label, .stMarkdown, .stText {
        color: #F5F5F5 !important;
        font-family: 'Tahoma', sans-serif;
    }

    /* Sidebar Input Text Overrides (Important for visibility in Dark Mode) */
    .st-emotion-cache-1wvlc34 { /* Targetting specific Streamlit widgets */
        color: #F5F5F5 !important;
    }

    /* Metrics Style - Ultimate Clean Look (Adapted for Dark) */
    [data-testid="stMetric"] {
        background-color: #2D2D2D; /* Darker container background */
        padding: 15px 20px;
        border-radius: 12px;
        border-left: 5px solid #4CAF50; /* Brighter Green accent */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    [data-testid="stMetricValue"] {
        font-size: 38px;
        color: #FFFFFF; /* White value */
        font-weight: 900;
    }
    [data-testid="stMetricLabel"] {
        font-size: 15px;
        color: #81C784; /* Lighter Green Label */
        font-weight: 700;
        text-transform: uppercase;
    }
    [data-testid="stMetricDelta"] {
        font-size: 16px;
        font-weight: bold;
        color: #FFC107 !important; /* Gold Delta for contrast */
    }

    /* Sidebar Styling */
    .sidebar-header-box {
        background: linear-gradient(135deg, #1A4D2E, #2EAF6C); /* Darker Green Gradient */
        padding: 25px;
        border-radius: 15px;
        margin-top: 20px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.5);
        border: 1px solid #FFC107;
    }
    .sidebar-header-box h1 { color: #FFFFFF; margin: 0; font-size: 3.0em; letter-spacing: 3px; font-weight: 900; }
    .sidebar-header-box p { color: #C8E6C9; margin: 0; font-size: 1.1em; font-weight: 500;}
    .sidebar-header-box h3 { color: #FFC107; margin: 5px 0 0; font-size: 1.5em; font-family: 'GE SS Unique', Arial, sans-serif;} /* GOLD */

    /* Tabs Styling (High Contrast) */
    div[data-testid="stTabs"] button {
        color: #FFC107 !important; /* Gold tab text */
        background-color: #2D2D2D !important; /* Dark tab background */
        font-weight: bold !important;
        border-bottom: 4px solid #4CAF50 !important; /* Bright Green underline */
        padding: 10px 15px;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #FFFFFF !important; /* White for active tab */
    }

    /* Block Flow Diagram (BFD) - Adapted for Dark */
    .bfd-block {
        padding: 20px 35px;
        border: none;
        border-radius: 15px;
        text-align: center;
        background: #2D2D2D; /* Dark block background */
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4), inset 0 0 10px rgba(255, 255, 255, 0.05);
        font-weight: bold;
        color: #F5F5F5;
        min-width: 220px;
        transition: transform 0.3s;
    }
    .bfd-stream { background-color: #FFC107; height: 5px; } /* Gold Stream */
    .bfd-stream::before { border-left-color: #FFC107; }

    /* Chatbot Styling */
    .stChatMessage {
        border-radius: 20px 20px 20px 5px;
        background-color: #2D2D2D; /* Darker chat bubble */
        color: #F5F5F5;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.3);
        padding: 15px;
    }

    /* Info and Warning Boxes */
    div.stAlert.info {
        background-color: #2D3A3B; /* Dark teal background */
        border-left: 5px solid #00BCD4; /* Cyan accent */
        color: #F5F5F5;
    }
    div.stAlert.warning {
        background-color: #3B3020; /* Dark yellow background */
        border-left: 5px solid #FFC107; /* Gold accent */
        color: #F5F5F5;
    }

    /* Main Content Containers (Tables/Charts background) */
    .st-emotion-cache-1c7v0s {
        background-color: #2D2D2D; /* Dark container background */
        border: 1px solid #444444;
        border-radius: 15px;
        padding: 15px;
    }

    /* Streamlit specific elements fix for dark mode */
    .st-emotion-cache-1v0x1p5 { /* Target Streamlit's main content area */
        color: #F5F5F5;
    }
    .st-emotion-cache-1fv9t6m { /* Fix for markdown blockquotes/infos */
        background-color: #2D2D2D;
        color: #F5F5F5;
    }
    .st-emotion-cache-1wvlc34, .st-emotion-cache-1o6g6gh, .st-emotion-cache-q172k {
        background-color: #2D2D2D;
        color: #F5F5F5;
    }

</style>
"""

# --- 3. Simulation Core Logic (Functions defined first) ---

def simulate_torrefaction(biomass, moisture, temp_C, duration_min, size, initial_mass_kg, reactor_type="N/A"):
    temp_K = temp_C + 273.15
    comp = BIOMASS_COMPOSITION.get(biomass)
    R_GAS_LOCAL = R_GAS 
    
    # Initial Fractions & Masses
    initial_moisture_frac = moisture / 100
    initial_ash_frac = comp["Ash"]
    daf_frac = 1.0 - initial_moisture_frac - initial_ash_frac
    
    m_h_init = comp["Hemicellulose"] * daf_frac
    m_c_init = comp["Cellulose"] * daf_frac
    m_l_init = comp["Lignin"] * daf_frac
    initial_mass_fixed_carbon_daf = daf_frac * BASE_FC_FACTOR 
    
    # Rate Constants
    k_drying = DRYING_RATE_CONST * SIZE_FACTOR.get(size)
    size_factor_val = SIZE_FACTOR.get(size)
    
    k_h_eff = KINETICS["Hemicellulose"][0] * np.exp(-KINETICS["Hemicellulose"][1] / (R_GAS_LOCAL * temp_K)) * size_factor_val
    k_c_eff = KINETICS["Cellulose"][0] * np.exp(-KINETICS["Cellulose"][1] / (R_GAS_LOCAL * temp_K)) * size_factor_val
    k_l_eff = KINETICS["Lignin"][0] * np.exp(-KINETICS["Lignin"][1] / (R_GAS_LOCAL * temp_K)) * size_factor_val

    # ODE System
    def model(y, t, k_dry, kh, kc, kl):
        m_moist, m_h, m_c, m_l = y
        # Drying rate only applies if moisture is present
        d_moist = -k_dry * m_moist if m_moist > 0.001 else 0 
        d_h = -kh * m_h
        d_c = -kc * m_c
        d_l = -kl * m_l
        return [d_moist, d_h, d_c, d_l]
    
    t = np.linspace(0, duration_min, 100)
    y0 = [initial_moisture_frac, m_h_init, m_c_init, m_l_init]
    
    sol = odeint(model, y0, t, args=(k_drying, k_h_eff, k_c_eff, k_l_eff))
    sol[sol < 0] = 0
    
    final_moisture_remaining = sol[:, 0][-1]
    final_h_remaining = sol[:, 1][-1]
    final_c_remaining = sol[:, 2][-1]
    final_l_remaining = sol[:, 3][-1]
    
    lost_h_frac = m_h_init - final_h_remaining
    lost_c_frac = m_c_init - final_c_remaining
    lost_l_frac = m_l_init - final_l_remaining
    total_volatiles_lost_frac = lost_h_frac + lost_c_frac + lost_l_frac

    # Final Mass Balance
    mass_ash_kg = initial_mass_kg * initial_ash_frac
    mass_fixed_carbon_kg = initial_mass_kg * initial_mass_fixed_carbon_daf
    mass_remaining_components = (final_h_remaining + final_c_remaining + final_l_remaining) * initial_mass_kg

    mass_biochar_total = mass_fixed_carbon_kg + mass_remaining_components + mass_ash_kg
    final_solid_yield_percent = (mass_biochar_total / initial_mass_kg) * 100
    
    mass_moisture_loss_kg = (initial_moisture_frac - final_moisture_remaining) * initial_mass_kg
    mass_non_condensable_gas_kg = total_volatiles_lost_frac * initial_mass_kg * comp["Gas_Factor"] 
    mass_bio_oil_kg = total_volatiles_lost_frac * initial_mass_kg * (1 - comp["Gas_Factor"]) 

    # DataFrames for Display
    yields_percent = pd.DataFrame({
        "Yield (%)": [
            final_solid_yield_percent,
            (mass_moisture_loss_kg / initial_mass_kg) * 100,
            (mass_bio_oil_kg / initial_mass_kg) * 100,
            (mass_non_condensable_gas_kg / initial_mass_kg) * 100
        ]
    }, index=["Biochar (Solid Product)", "Water Vapor", "Bio-Oil (Condensable)", "Non-Condensable Gas"])
    
    yields_mass = pd.DataFrame({
        "Mass (kg)": [
            mass_biochar_total,
            mass_moisture_loss_kg,
            mass_bio_oil_kg,
            mass_non_condensable_gas_kg
        ]
    }, index=["Biochar (Solid Product)", "Water Vapor", "Bio-Oil (Condensable)", "Non-Condensable Gas"])

    solid_composition = pd.DataFrame({
        "Mass (kg)": [
            mass_fixed_carbon_kg,
            mass_remaining_components,
            mass_ash_kg
        ]
    }, index=["Fixed Carbon", "Volatile Matter Remaining", "Ash"])


    # Final Ash Concentration
    final_ash_percent = (mass_ash_kg / mass_biochar_total) * 100

    # Energy & Sustainability Metrics
    initial_hhv_mj_kg = HHV_INITIAL.get(biomass, 17.0) 
    # Simplified assumption for biochar HHV enrichment
    biochar_hhv_mj_kg = initial_hhv_mj_kg * HHV_ENRICHMENT_FACTOR 
    
    initial_energy_mj = initial_mass_kg * initial_hhv_mj_kg * (1 - initial_moisture_frac)
    final_biochar_energy_mj = mass_biochar_total * biochar_hhv_mj_kg
    energy_yield_percent = (final_biochar_energy_mj / initial_energy_mj) * 100
    
    carbon_efficiency = final_solid_yield_percent * (biochar_hhv_mj_kg / initial_hhv_mj_kg) / 100 
    
    avg_devol_rate = (k_h_eff + k_c_eff + k_l_eff) / 3
    
    # Final Results Dictionary
    return {
        "yields_percent": yields_percent, "yields_mass": yields_mass, "solid_composition": solid_composition,
        "final_ash_percent": final_ash_percent, "initial_hhv": initial_hhv_mj_kg,
        "biochar_hhv": biochar_hhv_mj_kg, "energy_yield_percent": energy_yield_percent,
        "carbon_efficiency": carbon_efficiency, "avg_devol_rate": avg_devol_rate,
        "parameters": {
            "biomass": biomass, "moisture": moisture, "temperature": temp_C, 
            "duration": duration_min, "size": size, "initial_mass": initial_mass_kg,
            "reactor": reactor_type
        },
        "mass_profile_final": sol[:, 1] + sol[:, 2] + sol[:, 3] + initial_mass_fixed_carbon_daf + initial_ash_frac,
        "mass_moisture_loss_kg": mass_moisture_loss_kg,
        "mass_dry_biomass_kg": initial_mass_kg * daf_frac, 
        "mass_biochar_total": mass_biochar_total
    }


# --- 4. Thermal Balance Calculation ---
def calculate_thermal_balance(p, results):
    # p: parameters dictionary from results
    # results: full results dictionary from simulate_torrefaction
    
    T_C = p['temperature']
    T_K = T_C + 273.15
    initial_mass_kg = p['initial_mass']
    moisture_content = p['moisture'] / 100
    
    # 1. Sensible Heat (kJ)
    # Q_sensible = m * Cp * (T_final - T_initial)
    Q_sensible_biomass = (initial_mass_kg * (1 - moisture_content) * CP_BIOMASS * (T_K - TEMP_REF_K)) / 1000
    Q_sensible_water = (initial_mass_kg * moisture_content * CP_WATER * (T_K - TEMP_REF_K)) / 1000
    
    # 2. Latent Heat (kJ)
    # Q_latent = m_water_lost * H_vaporization
    m_water_lost = results.get('mass_moisture_loss_kg', 0.0)
    Q_latent_water = (m_water_lost * H_VAPOR) / 1000
    
    # 3. Torrefaction Heat (kJ) - Simplified Endothermic Estimate (approx 100 kJ/kg dry biomass)
    mass_dry_biomass_kg = results.get('mass_dry_biomass_kg', 0.0)
    Q_torrefaction = mass_dry_biomass_kg * 100.0
    
    # 4. Total Heat Input Required (kJ/kg initial biomass)
    Q_total_required_kJ = Q_sensible_biomass + Q_sensible_water + Q_latent_water + Q_torrefaction
    Q_total_per_kg = Q_total_required_kJ / initial_mass_kg
    
    # Energy Yield from simulation (used for comparison)
    energy_yield = results.get('energy_yield_percent', 0.0)
    
    return {
        'Q_sensible_biomass': Q_sensible_biomass,
        'Q_sensible_water': Q_sensible_water,
        'Q_latent_water': Q_latent_water,
        'Q_torrefaction': Q_torrefaction,
        'Q_total_required_kJ': Q_total_required_kJ,
        'Q_total_per_kg': Q_total_per_kg,
        'thermal_efficiency': energy_yield 
    }

# --- 5. Sensitivity Analysis (Fixed NameError and Logic) ---
@st.cache_data
def run_sensitivity_analysis(biomass, moisture, size, initial_mass_kg, reactor_type):
    T_range = np.linspace(220, 320, 10)
    D_range = np.linspace(30, 90, 10)

    # 1. Sensitivity to Temperature
    results_T = []
    for T in T_range:
        res = simulate_torrefaction(biomass, moisture, T, 60, size, initial_mass_kg, reactor_type) 
        # Calculate the thermal balance for the temperature points too
        thermal_res = calculate_thermal_balance(res['parameters'], res) 
        results_T.append((
            T, 
            res["yields_percent"].loc["Biochar (Solid Product)", "Yield (%)"],
            thermal_res.get('thermal_efficiency', 0.0)
        ))

    # 2. Sensitivity to Duration
    results_D = []
    for D in D_range:
        res = simulate_torrefaction(biomass, moisture, 275, D, size, initial_mass_kg, reactor_type) 
        thermal_res = calculate_thermal_balance(res['parameters'], res) 
        results_D.append((
            D, 
            res["yields_percent"].loc["Biochar (Solid Product)", "Yield (%)"],
            thermal_res.get('thermal_efficiency', 0.0)
        ))
        
    return pd.DataFrame(results_T, columns=["Temperature (°C)", "Yield (%)", "Energy Yield (%)"]), \
           pd.DataFrame(results_D, columns=["Duration (min)", "Yield (%)", "Energy Yield (%)"])

# --- 6. AI Chatbot Logic ---
def mock_ai_response(prompt, results, thermal_results, finance_results): 
    p = results.get("parameters", {})
    prompt_lower = prompt.lower()
    
    # Safety checks for default values
    mass_yield = results.get('yields_percent', pd.DataFrame({"Yield (%)": [0]}, index=["Biochar (Solid Product)"])).loc["Biochar (Solid Product)", "Yield (%)"]
    energy_yield = results.get('energy_yield_percent', 0.0)
    final_ash = results.get('final_ash_percent', 0.0)
    initial_hhv = results.get('initial_hhv', 0.0)
    biochar_hhv = results.get('biochar_hhv', 0.0)
    net_profit = finance_results.get('Net Profit', 0.0)


    # --- EXECUTIVE SUMMARY (Default Answer) ---
    summary = f"""
    ## 🎯 ملخص تنفيذي لنتائج المحاكاة (وضع العرض الداكن)

    بناءً على الشروط المدخلة (**{p.get('biomass', 'N/A')}**، عند **{p.get('temperature', 0)}°C** لمدة **{p.get('duration', 0)} min**):

    * **مردود الكتلة (Mass Yield):** **{mass_yield:.1f}\\%** (هدف جيد هو 75-85%).
    * **كفاءة الطاقة (Energy Yield):** **{energy_yield:.1f}\\%** (قيمة ممتازة تضمن عملية فعالة).
    * **تركيز الرماد النهائي:** **{final_ash:.2f}\\%** (يجب مقارنته بالحدود المطلوبة).
    * **جودة الوقود (HHV):** تحسن من **{initial_hhv:.2f} $\\text{{MJ/kg}}$** إلى **{biochar_hhv:.2f} $\\text{{MJ/kg}}$**.

    هل تود المزيد من التحليل حول **الربحية**، **الحركية الكيميائية**، أو **تحسين الظروف**؟
    """
    
    if "optimize" in prompt_lower or "تحسين" in prompt_lower or "ربحية" in prompt_lower:
        # Simplified optimization feedback (needs optimization logic in a real app)
        recommendation = "لتحسين الربحية، قم بموازنة **HHV** (حرارة أعلى) مع **المردود الكتلي** (حرارة أقل)."
        return f"""
        ## 📈 تحليل الربحية والتحسين

        * **الربح الصافي:** ${net_profit:.2f}
        * **الطاقة الحرارية المطلوبة:** {thermal_results.get('Q_total_per_kg', 0.0):.2f} $\\text{{kJ/kg}}$
        * **توصية التحسين:** {recommendation}
        """
    return summary


# --- 7. Main Streamlit App (Now defined after all dependencies) ---
def main():
    st.set_page_config(page_title="Chemisco Torrefaction Simulator", layout="wide", initial_sidebar_state="expanded")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    
    # Initialize session state 
    if "messages" not in st.session_state:
        # Initializing data structure for AI expert
        dummy_results = {"parameters": {}, "yields_percent": pd.DataFrame({"Yield (%)": [0]}, index=["Biochar (Solid Product)"]), "energy_yield_percent": 0, "final_ash_percent": 0, "initial_hhv": 0, "biochar_hhv": 0}
        dummy_thermal = {'Q_total_per_kg': 0.0}
        dummy_finance = {'Net Profit': 0.0}
        st.session_state["messages"] = [{"role": "assistant", "content": mock_ai_response("summary", dummy_results, dummy_thermal, dummy_finance)}]
    
    if 'target_yield' not in st.session_state:
        st.session_state['target_yield'] = 75
        st.session_state['target_ash'] = 8.0
        st.session_state['has_won'] = False
        st.session_state['cost_biomass_per_ton'] = 30.0
        st.session_state['cost_energy_per_hour'] = 5.0
        st.session_state['price_biochar_per_kg'] = 1.20

    # --- Sidebar (Inputs) ---
    with st.sidebar:
        # Header (Styled by CSS)
        st.markdown(f"""
            <div class="sidebar-header-box">
                <h1>CHEMISCO</h1>
                <p>Torrefaction Process Simulator</p>
                <hr style='margin: 10px 0; border-color: #388E3C;'>
                <p style='color: #C8E6C9;'>Project presented to:</p>
                <h3>د. عمرو الرفاعي</h3>
            </div>
            """, unsafe_allow_html=True)
        
        st.header("⚙️ Simulation Inputs")
        
        # Reactor Type (Kept as is)
        reactor_type = st.selectbox("🏭 Reactor Type", 
            ["Rotary Drum Reactor", "Fluidized Bed Reactor", "Auger/Screw Reactor", "Fixed Bed Reactor"])
        
        # 1. Biomass & Feedstock (Improved Expander)
        with st.expander("🌲 1. خصائص المواد الخام", expanded=True):
            initial_mass_kg = st.number_input("⚖️ Initial Mass (kg)", min_value=1.0, value=100.0, step=10.0)
            biomass_type = st.selectbox("🌿 Biomass Type", list(BIOMASS_COMPOSITION.keys()))
            moisture_content = st.slider("💧 Initial Moisture (%)", 0.0, 50.0, 10.0, step=1.0, help="Moisture must be dried before torrefaction.")
            particle_size = st.selectbox("📏 Particle Size", list(SIZE_FACTOR.keys()))
            ash_percent_init = BIOMASS_COMPOSITION[biomass_type]["Ash"] * 100
            st.info(f"Initial Ash Content: **{ash_percent_init:.1f}%**")
            
        # 2. Process Conditions (Improved Expander)
        with st.expander("🌡️ 2. ظروف عملية التحميص", expanded=True):
            temperature = st.slider("🔥 Temperature (°C)", 200, 350, 275, step=5, help="Higher temp increases HHV but reduces yield.")
            duration = st.slider("⏳ Duration (min)", 10, 120, 45, step=5, help="Longer duration increases the degree of torrefaction.")
            
        # 3. Economic Factors (Improved Expander)
        with st.expander("💰 3. المدخلات الاقتصادية", expanded=False):
            st.session_state.cost_biomass_per_ton = st.number_input("Feedstock Cost ($/ton)", min_value=0.0, value=st.session_state.cost_biomass_per_ton, step=5.0)
            st.session_state.cost_energy_per_hour = st.number_input("Operational Cost ($/hour)", min_value=0.0, value=st.session_state.cost_energy_per_hour, step=0.5, help="Represents energy/utility cost for running the reactor.")
            st.session_state.price_biochar_per_kg = st.number_input("Biochar Price ($/kg)", min_value=0.0, value=st.session_state.price_biochar_per_kg, step=0.1)
            
        st.markdown("---")
        st.subheader("🎮 Challenge Mode")
        game_mode = st.checkbox("Activate Plant Manager Challenge", value=False)


    # Input validation (Kept as is)
    if moisture_content / 100 + BIOMASS_COMPOSITION[biomass_type]["Ash"] > 1:
        st.error("**Input Error:** Moisture and Ash content exceed 100%. Adjust inputs.")
        return 
        
    # --- Run Simulation and Thermal Analysis ---
    with st.spinner("Running complex kinetic and thermal simulations..."):
        # The simulate_torrefaction function MUST return a dictionary with a 'parameters' key
        results = simulate_torrefaction(biomass_type, moisture_content, temperature, duration, particle_size, initial_mass_kg, reactor_type)
        # thermal_results depends on results, and calculates thermal balance
        thermal_results = calculate_thermal_balance(results['parameters'], results) 

    # --- Main Content ---
    st.title("CHEMISCO: Advanced Torrefaction Dashboard 🌙")
    st.subheader("Integrated Simulation, Analysis, and Optimization Platform")
    
    # 1. Block Flow Diagram (BFD) - Added Diagram for context 
    st.markdown("---")
    st.subheader("Process Flow Overview")
    
    # --- BFD Structure (Improved for clarity) ---
    bfd_html = f"""
    <div class="bfd-container" style="display: flex; justify-content: center; align-items: center; gap: 10px; flex-wrap: wrap;">
        <div class="bfd-block" style="border-left: 5px solid #00BCD4; transform: scale(0.95);">
            <p style="color: #00BCD4; margin: 0; font-size: 1.1em;">FEED PREP</p>
            <p style="color: #00BCD4; margin: 0; font-size: 0.9em;">{initial_mass_kg:.0f} kg</p>
        </div>
        <div class="bfd-stream">➡️</div>
        <div class="bfd-block" style="border-left: 5px solid #FFC107; transform: scale(0.95);">
            <p style="color: #FFC107; margin: 0; font-size: 1.1em;">DRYING</p>
            <p style="color: #FFC107; margin: 0; font-size: 0.9em;">Q_Latent: {thermal_results.get('Q_latent_water', 0.0):.0f} kJ</p>
        </div>
        <div class="bfd-stream">➡️</div>
        <div class="bfd-block" style="background: linear-gradient(135deg, #790000, #B71C1C); border-left: 5px solid #D32F2F; transform: scale(1.05);">
            <p style="color: #F5F5F5; margin: 0; font-size: 1.2em;">{reactor_type.upper()}</p>
            <p style="color: #F5F5F5; margin: 0; font-size: 1.0em;">T: {temperature}°C, t: {duration}min</p>
        </div>
        <div class="bfd-stream">➡️</div>
        <div class="bfd-block" style="background: linear-gradient(135deg, #388E3C, #4CAF50); border-left: 5px solid #4CAF50; transform: scale(1.10);">
            <p style="color: #FFFFFF; margin: 0; font-size: 1.3em;">PRODUCT</p>
            <p style="color: #FFFFFF; margin: 0; font-size: 1.1em;">Biochar: {results['yields_mass'].loc["Biochar (Solid Product)", "Mass (kg)"]:.2f} kg</p>
        </div>
    </div>
    """
    st.markdown(bfd_html, unsafe_allow_html=True)
    

[Image of Block Flow Diagram symbols]

    st.markdown("---")

    # 2. Results Dashboard (KPIs)
    st.header("🔑 Key Performance Indicators (KPIs)")
    
    col_kpi_1, col_kpi_2, col_kpi_3, col_kpi_4 = st.columns(4)
    
    # Financial data calculation for KPI
    cost_feedstock_total = (initial_mass_kg / 1000) * st.session_state.cost_biomass_per_ton
    hours = duration / 60
    # Adjusted calculation to include thermal cost approximation
    cost_operations_total = hours * st.session_state.cost_energy_per_hour
    total_cost = cost_feedstock_total + cost_operations_total
    biochar_produced_kg = results["yields_mass"].loc["Biochar (Solid Product)", "Mass (kg)"]
    revenue_total = biochar_produced_kg * st.session_state.price_biochar_per_kg
    profit_delta = revenue_total - total_cost

    
    col_kpi_1.metric("⚖️ Mass Yield", 
        f"{results['yields_percent'].loc['Biochar (Solid Product)', 'Yield (%)']:.1f} %", 
        delta=f"Product: {biochar_produced_kg:.2f} kg") # Used the mass as the delta
        
    col_kpi_2.metric("⚡ Energy Yield", 
        f"{results['energy_yield_percent']:.1f} %",
        delta=f"HHV: {results['biochar_hhv']:.2f} MJ/kg")
        
    col_kpi_3.metric("🔥 Heat Required", 
        f"{thermal_results.get('Q_total_per_kg', 0.0):.1f} kJ/kg",
        delta="Total Input Est.") # Added the new thermal metric
        
    col_kpi_4.metric("📈 Net Profit (Sim.)", 
        f"${profit_delta:.2f}",
        delta="Pre-Op Est.", delta_color="normal" if profit_delta > 0 else "inverse")

    st.markdown("---")

    # 3. Detailed Tabs
    tab1, tab2, tab3, tab4, tab_ai, tab_game = st.tabs(["Mass Balance & Quality", "Advanced Kinetics & Sensitivity", "Energy & Economics", "PDF Report", "🤖 AI Expert Analysis", "🎮 Plant Manager Challenge"])
    
    # --- Tab 1: Mass Balance & Quality (Kept as is) ---
    with tab1:
        st.subheader("Mass Distribution and Product Quality")
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            st.markdown("##### Overall Mass Distribution")
            df_global = results["yields_percent"].iloc[[0, 1, 2, 3]].reset_index()
            fig2 = px.pie(df_global, values='Yield (%)', names='index', hole=0.5, color_discrete_sequence=px.colors.sequential.Plotly3)
            fig2.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=1)))
            st.plotly_chart(fig2, use_container_width=True)

        with col_t2:
            st.markdown("##### Biochar Quality Metrics")
            df_solid = results["solid_composition"].reset_index()
            fig1 = px.pie(df_solid, values='Mass (kg)', names='index', hole=0.5, 
                            color_discrete_map={"Fixed Carbon": "#B39DDB", "Volatile Matter Remaining": "#FFEB3B", "Ash": "#9E9E9E"})
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig1, use_container_width=True)
            
            # Ash Metric Box
            st.markdown(f"""
                <div style="background-color: #3B3020; padding: 15px; border-radius: 8px; border-left: 5px solid #FFC107;">
                    <p style='margin: 0; font-weight: bold; color: #FFC107;'>⚗️ Final Ash Concentration:</p>
                    <h3 style='margin: 5px 0 0; color: #FFFFFF;'>{results['final_ash_percent']:.2f} %</h3>
                    <p style='margin: 0; font-size: 12px; color: #FFC107;'>Factor increase: {(results['final_ash_percent'] / ash_percent_init):.2f}x</p>
                </div>
            """, unsafe_allow_html=True)
            
    # --- Tab 2: Advanced Kinetics & Sensitivity (Now the function is defined above main) ---
    with tab2:
        st.subheader("Dynamic Simulation and Sensitivity Analysis")
        col_t2_1, col_t2_2 = st.columns(2)

        with st.spinner("Running Sensitivity Analysis..."):
            # Now run_sensitivity_analysis is defined before main()
            df_T, df_D = run_sensitivity_analysis(biomass_type, moisture_content, particle_size, initial_mass_kg, reactor_type)

        with col_t2_1:
            st.markdown("##### Sensitivity to Temperature and Duration")

            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(x=df_T["Temperature (°C)"], y=df_T["Yield (%)"], name='Yield vs. Temp.', mode='lines+markers', line=dict(color='#FF5252')))
            fig_sens.add_trace(go.Scatter(x=df_D["Duration (min)"], y=df_D["Yield (%)"], name='Yield vs. Duration', mode='lines+markers', line=dict(color='#00BCD4', dash='dot')))
            
            fig_sens.update_layout(
                title='Mass Yield Sensitivity Analysis', height=350,
                xaxis_title='Variable Value', yaxis_title='Biochar Mass Yield (%)',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                paper_bgcolor='#2D2D2D', plot_bgcolor='#2D2D2D', font=dict(color='#F5F5F5')
            )
            st.plotly_chart(fig_sens, use_container_width=True)

        with col_t2_2:
            st.markdown("##### Multi-Component Kinetic Rates")
            # Adjusted calculation to use temperature + 273.15 
            T_K_local = temperature + 273.15
            
            kinetics_data = {
                "Hemicellulose": KINETICS["Hemicellulose"][0] * np.exp(-KINETICS["Hemicellulose"][1] / (T_K_local * R_GAS)) * SIZE_FACTOR.get(particle_size) * 1000, 
                "Cellulose": KINETICS["Cellulose"][0] * np.exp(-KINETICS["Cellulose"][1] / (T_K_local * R_GAS)) * SIZE_FACTOR.get(particle_size) * 1000,
                "Lignin": KINETICS["Lignin"][0] * np.exp(-KINETICS["Lignin"][1] / (T_K_local * R_GAS)) * SIZE_FACTOR.get(particle_size) * 1000,
            }
            df_kinetics = pd.DataFrame(kinetics_data, index=["Rate Factor (a.u.)"]).T
            
            fig_rates = px.bar(df_kinetics, y='Rate Factor (a.u.)', color=df_kinetics.index, 
                               color_discrete_map={"Hemicellulose": "#00BCD4", "Cellulose": "#80DEEA", "Lignin": "#B2EBF2"})
            fig_rates.update_layout(height=350, title="Devolatilization Rate Factors (Scaled)", showlegend=False,
                                    paper_bgcolor='#2D2D2D', plot_bgcolor='#2D2D2D', font=dict(color='#F5F5F5'))
            st.plotly_chart(fig_rates, use_container_width=True)

            st.info(f"Avg. Devol Rate: **{results['avg_devol_rate']:.4f} $\\text{{min}}^{-1}$**. Particle size **{particle_size}** acts as a physical barrier to reaction.")

    # --- Tab 3: Energy & Economics (Fixed KeyError) ---
    with tab3:
        st.subheader("Energy and Economic Performance")
        col_e1, col_e2 = st.columns(2)
        
        with col_e1:
            st.markdown("##### Energy Performance Metrics")
            
            # --- Thermal Breakdown Table (New) ---
            st.markdown("##### Heat Input Breakdown (kJ)")
            df_thermal_breakdown = pd.DataFrame({
                "Heat Component": ["Q_Sensible_Biomass", "Q_Sensible_Water", "Q_Latent_Water", "Q_Torrefaction"],
                "Value (kJ)": [
                    thermal_results.get('Q_sensible_biomass', 0.0),
                    thermal_results.get('Q_sensible_water', 0.0),    
                    thermal_results.get('Q_latent_water', 0.0),
                    thermal_results.get('Q_torrefaction', 0.0)
                ]
            })
            st.dataframe(df_thermal_breakdown.set_index("Heat Component").style.format({"Value (kJ)": "{:,.0f}"}))
            
            st.markdown(f"""
                <div style="background-color: #2D2D2D; padding: 10px; border-radius: 8px; border: 1px solid #4CAF50;">
                    <p style='margin: 0; font-weight: bold; color: #4CAF50;'>💡 Total Required Heat Input:</p>
                    <h4 style='margin: 0; color: #FFFFFF;'>{thermal_results.get('Q_total_required_kJ', 0.0):,.0f} kJ</h4>
                </div>
            """, unsafe_allow_html=True)
            
        with col_e2:
            st.markdown("##### Economic Breakdown (Per Batch)")
            
            # Use previously calculated financial data
            finance_results = {
                'Feedstock Cost': cost_feedstock_total,
                'Operational Cost': cost_operations_total,
                'Total Cost': total_cost,
                'Revenue (Biochar)': revenue_total,
                'Net Profit': profit_delta
            }
            
            fig_waterfall = go.Figure(go.Waterfall(
                x = ["Feedstock Cost", "Operational Cost", "Revenue (Biochar)", "Net Profit"],
                y = [-finance_results['Feedstock Cost'], -finance_results['Operational Cost'], finance_results['Revenue (Biochar)'], finance_results['Net Profit']],
                decreasing = {"marker":{"color": "#FF5252"}}, 
                increasing = {"marker":{"color": "#4CAF50"}}, 
                totals = {"marker":{"color": "#FFC107"}}
            ))
            fig_waterfall.update_layout(title = "Economic Flow", showlegend = False, height=400,
                                        paper_bgcolor='#2D2D2D', plot_bgcolor='#2D2D2D', font=dict(color='#F5F5F5'))
            st.plotly_chart(fig_waterfall, use_container_width=True)
            
            st.metric("📊 Return on Investment (ROI)", f"{(finance_results['Net Profit'] / finance_results['Total Cost']) * 100:.1f} %" if finance_results['Total Cost'] > 0 else "N/A")

    # --- Tab 4: PDF Report (Placeholder) ---
    with tab4:
        st.subheader("📥 Generate Professional Report (Dark Mode Ready)")
        st.info("The full PDF report includes all charts, tables, and a summary of the KPIs.")
        
        st.download_button(
            label="⬇️ Download PDF Report",
            data=b"Placeholder PDF content", 
            file_name=f"Chemisco_Pro_Report_Dark_{biomass_type}_{temperature}C.pdf",
            mime="application/pdf"
        )
        
    # --- Tab 5: AI Expert Analysis ---
    with tab_ai:
        st.header("🤖 AI Expert: Strategic Analysis")
        st.info("الذكاء الاصطناعي يقدم ملخصاً تنفيذياً وتحليلاً معمقاً لنتائجك. اسأل عن التحسين، الحركية، أو الربحية.")

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input handling
        if prompt := st.chat_input("Ask a question to the Expert..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing data and generating strategic report..."):
                    time.sleep(2) 
                    # Passed the calculated thermal and finance results
                    ai_response = mock_ai_response(prompt, results, thermal_results, finance_results)
                
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
    
    # --- Tab 6: Game Mode (Kept as is) ---
    with tab_game:
        if game_mode:
            st.header("🎮 Plant Manager Challenge")
            st.markdown("""
            <div style="background-color: #3B3020; padding: 20px; border-radius: 10px; border-left: 6px solid #FFC107;">
                <h3 style="color: #FFC107; margin-top:0;">Fulfill the Client Order!</h3>
                <p style='color: #F5F5F5;'>Adjust Temperature and Duration in the sidebar to match the specifications below.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col_g1, col_g2, col_g3 = st.columns([1.5, 2, 1])
            
            if col_g3.button("🔄 New Client Order", key="new_order_btn_game"):
                st.session_state.target_yield = random.randint(60, 85)
                st.session_state.target_ash = round(random.uniform(ash_percent_init + 1.0, ash_percent_init + 5.0), 1)
                st.session_state.has_won = False
                st.experimental_rerun()
            
            with col_g1:
                st.markdown(f"**🎯 Target Yield:** `{st.session_state.target_yield}%`")
                st.markdown(f"**🎯 Max Ash:** `{st.session_state.target_ash}%`")
                
            with col_g2:
                curr_yield = results["yields_percent"].loc["Biochar (Solid Product)", "Yield (%)"]
                curr_ash = results["final_ash_percent"]
                diff_yield = abs(curr_yield - st.session_state.target_yield)
                diff_ash = abs(curr_ash - st.session_state.target_ash)
                
                score = max(0, 100 - (diff_yield * 10 + diff_ash * 20))
                st.metric("🏆 Your Efficiency Score", f"{score:.1f} / 100")
                
                if score >= 90:
                    st.success("🎉 **PERFECT MATCH! Order fulfilled successfully.**")
                    if not st.session_state.has_won:
                        st.session_state.has_won = True
                        st.balloons() 
                elif score >= 70:
                    st.warning("⚠️ Acceptable, but try to optimize further.")
                else:
                    st.error("❌ Specification mismatch. Adjust conditions!")
        else:
            st.info("Activate the 'Plant Manager Challenge' in the sidebar to test your optimization skills!")


# --- Execution Entry Point ---
if __name__ == "__main__":
    main()

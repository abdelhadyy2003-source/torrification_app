import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.integrate import odeint
from scipy.optimize import minimize
from io import BytesIO
import base64
import os 
import random 
import time 

# --- A. GLOBAL CONSTANTS AND KINETICS ---
R_GAS = 8.314             # J/(mol.K)
SPECIFIC_HEAT_BIOMASS = 1.3 # kJ/(kg.K) (Average)
SPECIFIC_HEAT_BIOCHAR = 1.0 # kJ/(kg.K)
SPECIFIC_HEAT_WATER = 4.18  # kJ/(kg.K)
ENTHALPY_WATER_VAP = 2500 # kJ/kg (Latent Heat of Vaporization)

# KINETIC PARAMETERS (Parallel First-Order Reactions)
KINETICS = {
    "Hemicellulose": [1.5e10, 110000],
    "Cellulose":     [1.0e12, 130000],
    "Lignin":        [2.0e9, 100000]
}

BIOMASS_COMPOSITION = {
    "Wood": {"Hemicellulose": 0.35, "Cellulose": 0.45, "Lignin": 0.20, "Ash": 0.02, "Gas_Factor": 0.40, "Density": 550},
    "Agricultural Waste": {"Hemicellulose": 0.45, "Cellulose": 0.35, "Lignin": 0.20, "Ash": 0.08, "Gas_Factor": 0.50, "Density": 400},
    "Municipal Waste": {"Hemicellulose": 0.30, "Cellulose": 0.40, "Lignin": 0.30, "Ash": 0.15, "Gas_Factor": 0.60, "Density": 300}
}

HHV_INITIAL = { "Wood": 18.0, "Agricultural Waste": 16.5, "Municipal Waste": 15.0 }
HHV_ENRICHMENT_FACTOR = 1.3 

DRYING_RATE_CONST = 0.05 
SIZE_FACTOR = {"Fine (<1mm)": 1.0, "Medium (1-5mm)": 0.85, "Coarse (>5mm)": 0.65}
BASE_FC_FACTOR = 0.20 

# --- Utility Function for Logo ---
LOGO_PATH = "chemisco_logo.png"

@st.cache_data(show_spinner=False)
def _get_image_base64(image_path):
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()
        return None
    except Exception:
        return None

LOGO_BASE64_STRING = _get_image_base64(LOGO_PATH)


# --- B. ADVANCED PROFESSIONAL DARK MODE CSS ---
GLOBAL_CSS = f"""
<style>
    /* ------------------- DARK MODE BASE STYLING ------------------- */
    .stApp {{ 
        padding-top: 10px; 
        background-color: #121212; 
        color: #E0E0E0; 
    }}
    
    /* Global Text and Headers */
    h1, h2, h3, p, label, .stMarkdown, .stText, .st-emotion-cache-1v0x1p5 {{ 
        color: #E0E0E0 !important; 
        font-family: 'Tahoma', sans-serif; 
    }}
    
    /* Metrics Style - High Contrast Dark (Enhanced for better visual hierarchy) */
    [data-testid="stMetric"] {{
        background-color: #1E1E1E; 
        padding: 20px 25px;
        border-radius: 15px;
        border-left: 6px solid #FFC107; /* Gold Accent for main KPIs */
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.6);
        transition: transform 0.2s; /* Hover effect */
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-3px);
    }}
    [data-testid="stMetricValue"] {{ 
        font-size: 40px; 
        color: #FFFFFF; 
        font-weight: 900; 
    }}
    [data-testid="stMetricLabel"] {{ 
        font-size: 16px; 
        color: #4CAF50; /* Green Label */
        font-weight: 700;
        text-transform: uppercase;
    }}
    [data-testid="stMetricDelta"] {{ 
        font-size: 18px; 
        font-weight: bold;
        color: #00BCD4 !important; /* Cyan Delta */
    }}
    
    /* Sidebar Styling */
    .sidebar-header-box {{
        background: linear-gradient(135deg, #004D40, #00897B); 
        padding: 25px;
        border-radius: 20px;
        margin-top: 25px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.7);
        border: 2px solid #FFC107;
    }}
    .sidebar-header-box h1 {{ color: #FFFFFF; font-size: 2.5em; letter-spacing: 4px; margin-bottom: 5px; }}
    .sidebar-header-box p {{ color: #C8E6C9; margin: 0; font-size: 1.0em; font-weight: 500;}}
    
    /* Tabs Styling (High Contrast) */
    div[data-testid="stTabs"] button {{
        color: #00BCD4 !important; 
        background-color: #1E1E1E !important; 
        font-weight: bold !important;
        border-bottom: 4px solid #00BCD4 !important; /* Changed accent to Cyan */
        padding: 12px 18px;
        font-size: 1.1em;
    }}
    div[data-testid="stTabs"] button[aria-selected="true"] {{
        color: #FFFFFF !important; 
        border-bottom: 4px solid #FFC107 !important; /* Selected tab accent Gold */
    }}
    
    /* General Containers */
    .st-emotion-cache-1c7v0s, .st-emotion-cache-1fv9t6m, .st-emotion-cache-q8b7t8 {{ 
        background-color: #1E1E1E; 
        border: 1px solid #333333;
        border-radius: 15px;
        padding: 20px;
    }}

    /* BFD Styling */
    .bfd-block {{ 
        padding: 25px 40px; 
        background: #282828; 
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5); 
        color: #E0E0E0; 
        border-radius: 15px;
        text-align: center;
    }}
    /* Enhanced Info/Success Boxes */
    .custom-info-box {{
        background-color: #1A3440; 
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00BCD4;
        color: #E0E0E0;
        font-weight: bold;
    }}
    .custom-success-box {{
        background-color: #1A3D24; 
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        color: #E0E0E0;
        font-weight: bold;
    }}
</style>
"""

# --- C. CORE SIMULATION AND KINETICS ---
def simulate_torrefaction(biomass, moisture, temp_C, duration_min, size, initial_mass_kg, reactor_type="N/A"): 
    temp_K = temp_C + 273.15
    comp = BIOMASS_COMPOSITION.get(biomass)
    R_GAS_LOCAL = R_GAS 
    
    initial_moisture_frac = moisture / 100
    initial_ash_frac = comp["Ash"]
    daf_frac = 1.0 - initial_moisture_frac - initial_ash_frac
    
    m_h_init = comp["Hemicellulose"] * daf_frac
    m_c_init = comp["Cellulose"] * daf_frac
    m_l_init = comp["Lignin"] * daf_frac
    initial_mass_fixed_carbon_daf = daf_frac * BASE_FC_FACTOR 
    
    k_drying = DRYING_RATE_CONST * SIZE_FACTOR.get(size)
    size_factor_val = SIZE_FACTOR.get(size)
    
    k_h_eff = KINETICS["Hemicellulose"][0] * np.exp(-KINETICS["Hemicellulose"][1] / (R_GAS_LOCAL * temp_K)) * size_factor_val
    k_c_eff = KINETICS["Cellulose"][0] * np.exp(-KINETICS["Cellulose"][1] / (R_GAS_LOCAL * temp_K)) * size_factor_val
    k_l_eff = KINETICS["Lignin"][0] * np.exp(-KINETICS["Lignin"][1] / (R_GAS_LOCAL * temp_K)) * size_factor_val

    def model(y, t, k_dry, kh, kc, kl):
        m_moist, m_h, m_c, m_l = y
        d_moist = -k_dry * m_moist if m_moist > 0.001 else 0 
        d_h = -kh * m_h
        d_c = -kc * m_c
        d_l = -kl * m_l
        return [d_moist, d_h, d_c, d_l]
    
    t = np.linspace(0, duration_min, 100)
    y0 = [initial_moisture_frac, m_h_init, m_c_init, m_l_init]
    
    sol = odeint(model, y0, t, args=(k_drying, k_h_eff, k_c_eff, k_l_eff))
    sol[sol < 0] = 0
    
    final_h_remaining = sol[:, 1][-1]
    final_c_remaining = sol[:, 2][-1]
    final_l_remaining = sol[:, 3][-1]
    
    lost_h_frac = m_h_init - final_h_remaining
    lost_c_frac = m_c_init - final_c_remaining
    lost_l_frac = m_l_init - final_l_remaining
    total_volatiles_lost_frac = lost_h_frac + lost_c_frac + lost_l_frac

    mass_ash_kg = initial_mass_kg * initial_ash_frac
    mass_fixed_carbon_kg = initial_mass_kg * initial_mass_fixed_carbon_daf
    mass_remaining_components = (final_h_remaining + final_c_remaining + final_l_remaining) * initial_mass_kg

    mass_biochar_total = mass_fixed_carbon_kg + mass_remaining_components + mass_ash_kg
    final_solid_yield_percent = (mass_biochar_total / initial_mass_kg) * 100
    
    mass_moisture_loss_kg = (initial_moisture_frac - sol[:, 0][-1]) * initial_mass_kg
    mass_non_condensable_gas_kg = total_volatiles_lost_frac * initial_mass_kg * comp["Gas_Factor"] 
    mass_bio_oil_kg = total_volatiles_lost_frac * initial_mass_kg * (1 - comp["Gas_Factor"]) 

    final_ash_percent = (mass_ash_kg / mass_biochar_total) * 100

    yields_percent = pd.DataFrame({
        "Yield (%)": [final_solid_yield_percent, (mass_bio_oil_kg / initial_mass_kg) * 100, (mass_non_condensable_gas_kg / initial_mass_kg) * 100, (mass_moisture_loss_kg / initial_mass_kg) * 100]},
        index=["Biochar (Solid Product)", "Bio-Oil (Condensable)", "Non-Condensable Gases", "Moisture Loss (Water Vapor)"]
    )
    
    yields_mass = yields_percent.copy()
    yields_mass["Mass (kg)"] = yields_percent["Yield (%)"] * initial_mass_kg / 100
    
    solid_composition = pd.DataFrame({
        "Mass (kg)": [mass_fixed_carbon_kg, mass_remaining_components, mass_ash_kg]
    }, index=["Fixed Carbon", "Volatile Matter Remaining", "Ash"])

    initial_hhv_mj_kg = HHV_INITIAL.get(biomass, 17.0) 
    biochar_hhv_mj_kg = initial_hhv_mj_kg * HHV_ENRICHMENT_FACTOR
    
    initial_energy_mj = initial_mass_kg * initial_hhv_mj_kg * (1 - initial_moisture_frac)
    final_biochar_energy_mj = mass_biochar_total * biochar_hhv_mj_kg
    energy_yield_percent = (final_biochar_energy_mj / initial_energy_mj) * 100
    
    carbon_efficiency = final_solid_yield_percent * (biochar_hhv_mj_kg / initial_hhv_mj_kg) / 100 

    df_mass_profile = pd.DataFrame({
        'Time (min)': t,
        'Hemicellulose': sol[:, 1] * initial_mass_kg,
        'Cellulose': sol[:, 2] * initial_mass_kg,
        'Lignin': sol[:, 3] * initial_mass_kg,
        'Total Mass (DAF)': (sol[:, 1] + sol[:, 2] + sol[:, 3]) * initial_mass_kg
    })
    
    return {
        "yields_percent": yields_percent, "yields_mass": yields_mass, "solid_composition": solid_composition,
        "final_ash_percent": final_ash_percent, "initial_hhv": initial_hhv_mj_kg,
        "biochar_hhv": biochar_hhv_mj_kg, "energy_yield_percent": energy_yield_percent,
        "carbon_efficiency": carbon_efficiency, 
        "parameters": {
            "biomass": biomass, "moisture": moisture, "temperature": temp_C, 
            "duration": duration_min, "size": size, "initial_mass": initial_mass_kg,
            "reactor": reactor_type
        },
        "df_mass_profile": df_mass_profile
    }

# --- D. SENSITIVITY ANALYSIS (Improved: Full 2D grid) ---
@st.cache_data
def run_sensitivity_analysis_2d(biomass, moisture, size, initial_mass_kg, reactor_type):
    # استخدام نطاقات أوسع لإنشاء مخطط حراري أكثر دقة
    T_points = 15
    D_points = 15
    T_range = np.linspace(220, 320, T_points)
    D_range = np.linspace(30, 120, D_points)
    
    yield_matrix = np.zeros((T_points, D_points))
    
    for i, T in enumerate(T_range):
        for j, D in enumerate(D_range):
            res = simulate_torrefaction(biomass, moisture, T, D, size, initial_mass_kg, reactor_type) 
            yield_matrix[i, j] = res["yields_percent"].loc["Biochar (Solid Product)", "Yield (%)"]
            
    return T_range, D_range, yield_matrix

# --- E. THERMAL MODELING UNIT ---
def calculate_thermal_loads(biomass_type, initial_mass_kg, temp_C, duration_min, moisture):
    T_ambient = 25.0 
    T_reactor = temp_C 
    
    delta_T = T_reactor - T_ambient
    mass_dry_biomass = initial_mass_kg * (1 - moisture / 100)
    
    Q_sensible_biomass = mass_dry_biomass * SPECIFIC_HEAT_BIOMASS * delta_T # kJ
    
    mass_water = initial_mass_kg * (moisture / 100)
    Q_sensible_water = mass_water * SPECIFIC_HEAT_WATER * delta_T # kJ
    
    Q_latent_vaporization = mass_water * ENTHALPY_WATER_VAP # kJ
    
    Q_reaction_kJ_kg = -150 # kJ/kg (Assumed net endothermic)
    Q_reaction = mass_dry_biomass * Q_reaction_kJ_kg # kJ
    
    reactor_volume_m3 = initial_mass_kg / (BIOMASS_COMPOSITION[biomass_type]["Density"] * 1.5)
    reactor_surface_area_m2 = 6 * (reactor_volume_m3**(2/3))
    
    U_overall = 0.5 # kW/(m2.K)
    delta_T_K = temp_C - 25.0
    duration_s = duration_min * 60
    
    Q_loss = U_overall * 1000 * reactor_surface_area_m2 * delta_T_K * duration_s # J
    Q_loss_kJ = Q_loss / 1000 
    
    Q_total_required_kJ = Q_sensible_biomass + Q_sensible_water + Q_latent_vaporization + abs(Q_reaction) + Q_loss_kJ
    
    Q_gas_potential_MJ = (initial_mass_kg * BIOMASS_COMPOSITION[biomass_type]["Gas_Factor"]) * 12 
    Q_gas_potential_kJ = Q_gas_potential_MJ * 1000
    Q_gas_utilized_kJ = Q_gas_potential_kJ * 0.50 
    
    Q_net_external_requirement_kJ = Q_total_required_kJ - Q_gas_utilized_kJ
    
    energy_required_kWh = Q_net_external_requirement_kJ / 3600
    
    return {
        "Q_total_kJ": Q_total_required_kJ,
        "Q_net_external_kJ": Q_net_external_requirement_kJ,
        "Q_sensible_biomass": Q_sensible_biomass,
        "Q_latent_vaporization": Q_latent_vaporization,
        "Q_reaction": Q_reaction,
        "Q_loss_kJ": Q_loss_kJ,
        "Energy_kWh": energy_required_kWh,
        "Gas_Self_Heating_kJ": Q_gas_utilized_kJ
    }

# --- F. OPTIMIZATION UNIT ---
def optimization_function(params, biomass, moisture, size, initial_mass_kg):
    T, D = params
    
    if not (220 <= T <= 320 and 30 <= D <= 120):
        return 1e10 
        
    results = simulate_torrefaction(biomass, moisture, T, D, size, initial_mass_kg)
    
    energy_yield = results['energy_yield_percent'] / 100
    mass_yield = results['yields_percent'].loc["Biochar (Solid Product)", "Yield (%)"] / 100
    
    cost_factor = (T / 320) + (D / 120) 
    
    objective = (energy_yield * mass_yield) - 0.2 * cost_factor
    
    return -objective

def run_optimization(biomass, moisture, size, initial_mass_kg):
    initial_guess = [275, 60]
    bnds = ((220, 320), (30, 120))
    
    result = minimize(
        optimization_function, 
        initial_guess, 
        args=(biomass, moisture, size, initial_mass_kg), 
        method='L-BFGS-B', 
        bounds=bnds,
        options={'disp': False, 'maxiter': 50}
    )
    
    if result.success:
        opt_T = result.x[0]
        opt_D = result.x[1]
        opt_value = -result.fun
        return opt_T, opt_D, opt_value
    else:
        return None, None, None

# --- G. MOCK AI RESPONSE (for Tab 5) ---
def mock_ai_response(prompt, results, annual_net_profit, payback_period_years):
    p = results["parameters"]
    prompt_lower = prompt.lower()
    
    summary = f"""
    ## 🎯 ملخص تنفيذي (محدث)

    **المردود الكتلي:** {results['yields_percent'].loc["Biochar (Solid Product)", "Yield (%)"]:.1f}\\%
    **كفاءة الطاقة:** {results['energy_yield_percent']:.1f}\\%
    **صافي الربح السنوي:** ${annual_net_profit:,.0f}
    """ 
    
    if "thermal load" in prompt_lower or "حرارة" in prompt_lower:
        thermal_results = calculate_thermal_loads(p["biomass"], p["initial_mass"], p["temperature"], p["duration"], p["moisture"])
        return f"""
        ## 🔥 تحليل الحمل الحراري (Thermal Load Analysis)
        
        * **الحرارة الكامنة للتبخير:** {thermal_results['Q_latent_vaporization']/1000:.1f} MJ
        * **صافي الطاقة الخارجية المطلوبة:** {thermal_results['Q_net_external_kJ']/1000:.1f} MJ/Batch.
        
        **توصية:** لتقليل التكلفة، يجب تخفيض محتوى الرطوبة الابتدائي.
        """
    elif "payback" in prompt_lower or "استرداد" in prompt_lower or "roi" in prompt_lower:
        return f"""
        ## 💰 تحليل الجدوى الاقتصادية (Payback Analysis)
        
        * **الربح السنوي الصافي:** ${annual_net_profit:,.0f}
        * **فترة استرداد رأس المال:** **{payback_period_years:.1f} سنة**.
        
        **ملاحظة:** يرجى مراجعة التكاليف الرأسمالية (CAPEX) لضمان دقة هذا الرقم.
        """
    elif "optimal" in prompt_lower or "تحسين" in prompt_lower:
         if 'opt_T' in st.session_state:
             return f"""
             ## ✨ نتائج التحسين (Optimization Results)
             
             * **الحرارة المثلى:** **{st.session_state['opt_T']:.1f}°C**
             * **المدة المثلى:** **{st.session_state['opt_D']:.1f} min**
             
             لتحقيق أعلى ربحية ممكنة ضمن نطاق الشروط، يوصى باستخدام هذه الإعدادات.
             """
         else:
            return "الرجاء تشغيل زر 'Run Optimization' في الشريط الجانبي أولاً للحصول على النتائج المثلى."
    
    return summary


# --- H. MAIN STREAMLIT APPLICATION ---
def main():
    st.set_page_config(page_title="Chemisco Torrefaction Simulator", layout="wide", initial_sidebar_state="expanded")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    
    # Initialize session state 
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Welcome to Chemisco. Run the simulation first to get an executive summary."}]
    if 'target_yield' not in st.session_state:
        st.session_state['target_yield'] = 75
        st.session_state['target_ash'] = 8.0
        st.session_state['has_won'] = False
        st.session_state['cost_biomass_per_ton'] = 30.0
        st.session_state['cost_energy_per_hour'] = 5.0
        st.session_state['price_biochar_per_kg'] = 1.20
        st.session_state['capex'] = 1500000 
        st.session_state['operating_days'] = 300 
        
    # --- Sidebar (Inputs) ---
    with st.sidebar:
        # Header with Logo
        header_html = f"""
            <div class="sidebar-header-box">
                {'<img src="data:image/png;base64,' + LOGO_BASE64_STRING + '" style="max-width: 80px; margin-bottom: 10px;">' if LOGO_BASE64_STRING else ''}
                <h1>CHEMISCO</h1>
                <p>Advanced Process Simulation</p>
                <hr style='margin: 10px 0; border-color: #00897B;'>
            </div>
            """
        st.markdown(header_html, unsafe_allow_html=True)
        
        st.header("⚙️ Simulation Inputs")
        
        with st.expander("🏭 Reactor & Feedstock Type", expanded=True):
            reactor_type = st.selectbox("🏭 Reactor Type", 
                ["Rotary Drum Reactor", "Fluidized Bed Reactor", "Auger/Screw Reactor", "Fixed Bed Reactor"])
            biomass_type = st.selectbox("🌿 Biomass Type", list(BIOMASS_COMPOSITION.keys()))
            
        with st.expander("🔬 Input Properties", expanded=True):
            initial_mass_kg = st.number_input("⚖️ Initial Mass (kg/Batch)", min_value=1.0, value=100.0, step=10.0)
            moisture_content = st.slider("💧 Initial Moisture (%)", 0.0, 50.0, 10.0, step=1.0)
            particle_size = st.selectbox("📏 Particle Size", list(SIZE_FACTOR.keys()))
            ash_percent_init = BIOMASS_COMPOSITION[biomass_type]["Ash"] * 100
            st.markdown(f"<div class='custom-info-box'>Initial Ash Content: <b>{ash_percent_init:.1f}%</b></div>", unsafe_allow_html=True)
            
        with st.expander("🌡️ Process Conditions", expanded=True):
            temperature = st.slider("🔥 Temperature (°C)", 200, 350, 275, step=5)
            duration = st.slider("⏳ Duration (min)", 10, 120, 45, step=5)
            
        with st.expander("💰 Economic Factors (CAPEX/OPEX)", expanded=False):
            st.session_state.capex = st.number_input("Total CAPEX ($)", min_value=100000, value=st.session_state.capex, step=50000)
            st.session_state.operating_days = st.number_input("Operating Days/Year", min_value=100, value=st.session_state.operating_days, step=10)
            st.session_state.cost_biomass_per_ton = st.number_input("Feedstock Cost ($/ton)", min_value=0.0, value=st.session_state.cost_biomass_per_ton, step=5.0)
            st.session_state.cost_energy_per_hour = st.number_input("Operational Cost ($/hour)", min_value=0.0, value=st.session_state.cost_energy_per_hour, step=0.5)
            st.session_state.price_biochar_per_kg = st.number_input("Biochar Price ($/kg)", min_value=0.0, value=st.session_state.price_biochar_per_kg, step=0.1)
            
        st.markdown("---")
        if st.button("✨ Run Optimization (Find Max Profit Conditions)"):
            opt_T, opt_D, opt_val = run_optimization(biomass_type, moisture_content, particle_size, initial_mass_kg)
            if opt_T and opt_D:
                st.session_state['opt_T'] = opt_T
                st.session_state['opt_D'] = opt_D
                st.session_state['opt_val'] = opt_val
                st.success(f"Optimal Conditions Found! T={opt_T:.1f}°C, D={opt_D:.1f} min")
            else:
                st.error("Optimization failed to converge.")

    # Input validation
    if moisture_content / 100 + BIOMASS_COMPOSITION[biomass_type]["Ash"] > 1:
        st.error("**Input Error:** Moisture and Ash content exceed 100%. Adjust inputs.")
        return 
        
    # Run Simulation & Thermal Calculations
    results = simulate_torrefaction(biomass_type, moisture_content, temperature, duration, particle_size, initial_mass_kg, reactor_type)
    thermal_results = calculate_thermal_loads(biomass_type, initial_mass_kg, temperature, duration, moisture_content)
    
    # Economic Calculations for KPIs
    profit_delta = ((results['yields_mass'].loc['Biochar (Solid Product)', 'Mass (kg)'] * st.session_state.price_biochar_per_kg) - (initial_mass_kg / 1000) * st.session_state.cost_biomass_per_ton)
    hourly_capex = st.session_state.capex / (5 * st.session_state.operating_days * (60/duration) * (duration/60)) if duration > 0 else 0 
    net_hourly_profit = (profit_delta * (60/duration)) - st.session_state.cost_energy_per_hour - hourly_capex 
    
    # Annual Economics for Tabs
    batch_per_day = 60 / duration if duration > 0 else 0
    annual_batches = st.session_state.operating_days * batch_per_day
    annual_feedstock_input_ton = (annual_batches * initial_mass_kg) / 1000
    annual_biochar_output_kg = annual_batches * results['yields_mass'].loc["Biochar (Solid Product)", "Mass (kg)"]
    annual_feedstock_cost = annual_feedstock_input_ton * st.session_state.cost_biomass_per_ton
    annual_operational_cost = st.session_state.operating_days * 24 * st.session_state.cost_energy_per_hour 
    total_annual_opex = annual_feedstock_cost + annual_operational_cost
    annual_revenue = annual_biochar_output_kg * st.session_state.price_biochar_per_kg
    annual_net_profit = annual_revenue - total_annual_opex
    payback_period_years = st.session_state.capex / annual_net_profit if annual_net_profit > 0 else 999 

    # --- Main Content ---
    st.title("CHEMISCO: Integrated Torrefaction Simulation Platform 🌌")
    st.subheader("Advanced Kinetic, Thermal, and Economic Analysis (V4.0)")
    
    # 1. Block Flow Diagram (BFD)
    st.markdown("---")
    st.subheader("Process Flow and Status")
    bfd_html = f"""
    <div class="bfd-container" style="display: flex; justify-content: space-around; align-items: center; padding: 20px;">
        <div class="bfd-block" style="border-left: 5px solid #4CAF50;">FEEDSTOCK<p style="color: #4CAF50; font-size: 1.1em;">{initial_mass_kg:.0f} kg @ {moisture_content}% H₂O</p></div>
        <span style="color: #E0E0E0; font-size: 24px; margin: 0 15px;">➡️</span>
        <div class="bfd-block" style="border-left: 5px solid #00BCD4;">DRYING<p style="color: #00BCD4; font-size: 1.1em;">Q_Latent: {thermal_results['Q_latent_vaporization']/1000:.1f} MJ</p></div>
        <span style="color: #E0E0E0; font-size: 24px; margin: 0 15px;">➡️</span>
        <div class="bfd-block" style="background: linear-gradient(135deg, #790000, #B71C1C); border-left: 5px solid #FFC107;">
            TORREFACTION ({reactor_type})
            <p style="color: #F5F5F5; font-size: 1.1em;">{temperature}°C for {duration} min</p>
        </div>
        <span style="color: #E0E0E0; font-size: 24px; margin: 0 15px;">➡️</span>
        <div class="bfd-block" style="background: linear-gradient(135deg, #388E3C, #4CAF50); border-left: 5px solid #FFC107;">
            BIOCHAR PRODUCT
            <p style="color: #FFC107; font-size: 1.1em;">Yield: {results['yields_percent'].loc["Biochar (Solid Product)", "Yield (%)"]:.1f} %</p>
        </div>
    </div>
    """
    st.markdown(bfd_html, unsafe_allow_html=True)
    st.markdown("---")

    # 2. Results Dashboard (KPIs)
    st.header("🔑 Key Performance Indicators (KPIs)")
    col_kpi_1, col_kpi_2, col_kpi_3, col_kpi_4 = st.columns(4)
    
    col_kpi_1.metric("⚖️ Mass Yield", 
        f"{results['yields_percent'].loc['Biochar (Solid Product)', 'Yield (%)']:.1f} %", 
        delta=f"{results['yields_mass'].loc['Biochar (Solid Product)', 'Mass (kg)']:.2f} kg/Batch")
        
    col_kpi_2.metric("⚡ Energy Yield", 
        f"{results['energy_yield_percent']:.1f} %",
        delta=f"HHV: {results['biochar_hhv']:.2f} MJ/kg")
        
    col_kpi_3.metric("💰 Hourly Net Profit", 
        f"${net_hourly_profit:.2f}",
        delta="Incl. Amortization", delta_color="normal" if net_hourly_profit > 0 else "inverse")
        
    col_kpi_4.metric("📊 Carbon Efficiency", 
        f"{results['carbon_efficiency'] * 100:.1f} %",
        delta=f"Final Ash: {results['final_ash_percent']:.1f}%")

    st.markdown("---")

    # 3. Detailed Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Mass Balance & Quality", 
        "🧪 Process Dynamics & Sensitivity", 
        "🔥 Thermal & Energy Analysis", 
        "💵 Project Economics", 
        "🤖 AI Expert Consultant", 
        "🎮 Manager Challenge"
    ])
    
    # --- Tab 1: Mass Balance & Quality ---
    with tab1:
        st.subheader("Mass Distribution and Product Quality")
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            st.markdown("##### Overall Mass Distribution")
            df_global = results["yields_percent"].iloc[[0, 1, 2, 3]].reset_index()
            df_global.rename(columns={'index': 'Component'}, inplace=True) 
            
            fig2 = px.pie(df_global, values='Yield (%)', names='Component', hole=0.5, 
                          color_discrete_sequence=['#4CAF50', '#00BCD4', '#FFC107', '#9E9E9E'], 
                          title="Total Mass Yield Breakdown")
            fig2.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#121212', width=1)))
            fig2.update_layout(paper_bgcolor='#1E1E1E', plot_bgcolor='#1E1E1E', font=dict(color='#E0E0E0'), height=450)
            st.plotly_chart(fig2, use_container_width=True)

        with col_t2:
            st.markdown("##### Biochar Solid Composition and Quality")
            df_solid = results["solid_composition"].reset_index()
            fig1 = px.bar(df_solid, x='index', y='Mass (kg)', 
                          color='index', 
                          color_discrete_map={"Fixed Carbon": "#00BCD4", "Volatile Matter Remaining": "#FFC107", "Ash": "#9E9E9E"},
                          title="Solid Product Composition (Dry Basis)")
            fig1.update_layout(paper_bgcolor='#1E1E1E', plot_bgcolor='#1E1E1E', font=dict(color='#E0E0E0'), height=450)
            st.plotly_chart(fig1, use_container_width=True)
            
            st.markdown(f"""
                <div class='custom-success-box'>
                    <p style='margin: 0; font-weight: bold; color: #4CAF50;'>💡 Product Quality Summary:</p>
                    <ul style='list-style-type: none; padding-left: 0;'>
                        <li>**HHV (Biochar):** <span style='color:#FFC107;'>{results['biochar_hhv']:.2f} MJ/kg</span></li>
                        <li>**Final Ash:** <span style='color:#FFC107;'>{results['final_ash_percent']:.2f} %</span></li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
    # --- Tab 2: Process Dynamics & Sensitivity ---
    with tab2:
        st.subheader("Kinetic Devolatilization and Process Sensitivity")
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.markdown("##### Component Mass Consumption Over Time (Dry Basis)")
            df_mass = results['df_mass_profile'].set_index('Time (min)').drop(columns=['Total Mass (DAF)'])
            
            fig_mass_loss = px.line(df_mass, 
                                    x=df_mass.index, 
                                    y=df_mass.columns, 
                                    title='Mass Remaining of Biomass Components',
                                    color_discrete_map={"Hemicellulose": "#FF5252", "Cellulose": "#00BCD4", "Lignin": "#FFC107"})
            fig_mass_loss.update_layout(paper_bgcolor='#1E1E1E', plot_bgcolor='#1E1E1E', font=dict(color='#E0E0E0'),
                                        yaxis_title='Mass Remaining (kg)', height=450)
            st.plotly_chart(fig_mass_loss, use_container_width=True)

        with col_d2:
            st.markdown("##### 2D Sensitivity Analysis: Yield vs. Temp. & Duration")
            T_range_sens, D_range_sens, yield_matrix = run_sensitivity_analysis_2d(biomass_type, moisture_content, particle_size, initial_mass_kg, reactor_type)
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=yield_matrix.T, # Transpose to match X (Duration) and Y (Temp)
                x=D_range_sens,
                y=T_range_sens,
                colorscale='Viridis',
                colorbar=dict(title='Yield (%)')
            ))
            fig_heatmap.update_layout(title='Biochar Yield (%) Heatmap', height=450,
                                    xaxis_title='Duration (min)', yaxis_title='Temperature (°C)',
                                    paper_bgcolor='#1E1E1E', plot_bgcolor='#1E1E1E', font=dict(color='#E0E0E0'))
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.markdown(f"""
            <div class='custom-info-box'>
            <p><strong>Note:</strong> Heatmap shows the yield trade-off. Higher yields (Yellow/White) are achieved at higher temperatures and longer durations, but at higher cost.</p>
            </div>
            """, unsafe_allow_html=True)


    # --- Tab 3: Thermal & Energy ---
    with tab3:
        st.subheader("Detailed Thermal Load and Energy Balance")
        
        col_e1, col_e2, col_e3 = st.columns(3)
        col_e1.metric("Heat Loss (Total)", f"{thermal_results['Q_loss_kJ']/1000:.1f} MJ", delta="From Reactor Surface")
        col_e2.metric("Latent Heat (Drying)", f"{thermal_results['Q_latent_vaporization']/1000:.1f} MJ", delta="Drying is key step")
        col_e3.metric("Net External Energy Req.", f"{thermal_results['Q_net_external_kJ']/1000:.1f} MJ", delta=f"{thermal_results['Energy_kWh']:.2f} kWh/Batch")

        st.markdown("##### Energy Balance Waterfall (Total kJ/Batch)")
        df_thermal = pd.DataFrame({
            'Category': ["Sensible Heat (Biomass)", "Sensible Heat (Water)", "Latent Heat (Vaporization)", 
                         "Reaction Heat (Endo)", "Heat Loss", "Gas Self-Heating (Credit)", "Net External Requirement"],
            'Value': [thermal_results['Q_sensible_biomass'], thermal_results['Q_sensible_water'], 
                      thermal_results['Q_latent_vaporization'], abs(thermal_results['Q_reaction']), 
                      thermal_results['Q_loss_kJ'], -thermal_results['Gas_Self_Heating_kJ'], 
                      thermal_results['Q_net_external_kJ']],
            'Type': ['Demand', 'Demand', 'Demand', 'Demand', 'Demand', 'Credit', 'Net']
        })

        fig_thermal = go.Figure(go.Waterfall(
            x=df_thermal['Category'],
            y=df_thermal['Value'],
            name="",
            connector = {"line": {"color": "rgb(63, 63, 63)"}},
            # Custom colors for better visibility
            decreasing = {"marker":{"color": "#FF5252"}}, 
            increasing = {"marker":{"color": "#4CAF50"}}, 
            totals = {"marker":{"color": "#FFC107"}}
        ))
        fig_thermal.update_layout(title="Thermal Energy Waterfall (Total kJ)", showlegend=False, height=500,
                                  paper_bgcolor='#1E1E1E', plot_bgcolor='#1E1E1E', font=dict(color='#E0E0E0'))
        st.plotly_chart(fig_thermal, use_container_width=True)
            
    # --- Tab 4: Project Economics ---
    with tab4:
        st.subheader("Integrated Project Economics and CAPEX/OPEX Analysis")

        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("Total Annual Revenue", f"${annual_revenue:.0f}", delta="Annual Biochar Sales")
        col_p2.metric("Total Annual OPEX", f"${total_annual_opex:.0f}", delta="Feedstock + Operational Costs")
        col_p3.metric("Payback Period (Years)", f"{payback_period_years:.1f}", delta=f"CAPEX: ${st.session_state.capex:,.0f}")
        
        st.markdown("##### Annual Financial Statement (Simplified)")
        df_finance = pd.DataFrame({
            "Metric": ["Annual Revenue (A)", "Annual Feedstock Cost (B)", "Annual Operational Cost (C)", "Total OPEX (B+C)", "Annual Net Profit (A - OPEX)", "CAPEX (D)"],
            "Value": [annual_revenue, annual_feedstock_cost, annual_operational_cost, total_annual_opex, annual_net_profit, st.session_state.capex]
        })
        st.table(df_finance.style.format({"Value": "${:,.0f}"}).hide(axis="index"))
        
        st.markdown("##### Project Timeline Overview")
        df_gantt = pd.DataFrame([
            dict(Task="Permitting & Design", Start='2025-01-01', Finish='2025-06-30', Resource='Phase 1'),
            dict(Task="Procurement & CAPEX", Start='2025-07-01', Finish='2026-03-31', Resource='Phase 2'),
            dict(Task="Construction", Start='2026-04-01', Finish='2026-10-31', Resource='Phase 2'),
            dict(Task="Commissioning", Start='2026-11-01', Finish='2027-01-31', Resource='Phase 3'),
            dict(Task="Full Operation", Start='2027-02-01', Finish='2028-02-01', Resource='Phase 4')
        ])
        
        fig_gantt = px.timeline(df_gantt, x_start="Start", x_end="Finish", y="Task", color="Resource")
        fig_gantt.update_layout(title="Project Schedule Overview (Gantt Chart)", paper_bgcolor='#1E1E1E', plot_bgcolor='#1E1E1E', font=dict(color='#E0E0E0'))
        st.plotly_chart(fig_gantt, use_container_width=True)

    # --- Tab 5: AI Expert Analysis ---
    with tab5:
        st.header("🤖 AI Expert: Strategic Consulting")
        st.markdown("""
        <div class='custom-info-box'>
        هذا المستشار الآلي (AI Expert) يوفر تحليلاً فورياً بناءً على نتائج المحاكاة.
        يمكنك سؤاله عن:
        <ul>
        <li>**"Thermal Load" (الحمل الحراري):** لتحليل استهلاك الطاقة.</li>
        <li>**"Payback Period" (فترة الاسترداد):** لتحليل الجدوى الاقتصادية.</li>
        <li>**"Optimal Conditions" (الظروف المثلى):** بعد تشغيل زر التحسين.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask a question to the Expert..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing data and generating strategic report..."):
                    time.sleep(2) 
                    ai_response = mock_ai_response(prompt, results, annual_net_profit, payback_period_years)
                    
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})

    # --- Tab 6: Game Mode ---
    with tab6:
        st.header("🎮 Plant Manager Challenge: Quality Control")
        game_mode = st.checkbox("Activate Plant Manager Challenge", value=False)
        
        if game_mode:
            st.markdown("""
            <div style="background-color: #3B3020; padding: 20px; border-radius: 10px; border-left: 6px solid #FFC107;">
                <h3 style="color: #FFC107; margin-top:0;">Fulfill the Client Order!</h3>
                <p style='color: #F5F5F5;'>Adjust **Temperature** and **Duration** in the sidebar to match the specifications below and maximize your score.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col_g1, col_g2, col_g3 = st.columns([1.5, 2, 1])
            
            with col_g3:
                if st.button("🔄 New Client Order", key="new_order_btn_game"):
                    st.session_state.target_yield = random.randint(60, 85)
                    st.session_state.target_ash = round(random.uniform(ash_percent_init + 1.0, ash_percent_init + 5.0), 1)
                    st.session_state.has_won = False
                    st.experimental_rerun()
            
            with col_g1:
                st.markdown(f"**🎯 Target Yield:** **`{st.session_state.target_yield}%`**")
                st.markdown(f"**🎯 Max Ash Content:** **`{st.session_state.target_ash}%`**")
                
            with col_g2:
                curr_yield = results["yields_percent"].loc["Biochar (Solid Product)", "Yield (%)"]
                curr_ash = results["final_ash_percent"]
                diff_yield = abs(curr_yield - st.session_state.target_yield)
                diff_ash_penalty = max(0, curr_ash - st.session_state.target_ash)
                
                score = max(0, 100 - (diff_yield * 5 + diff_ash_penalty * 20)) # Weighted scoring
                
                st.metric("🏆 Your Efficiency Score", f"{score:.1f} / 100")
                
                if score >= 90 and diff_ash_penalty == 0:
                    st.success("🎉 **PERFECT MATCH! Order fulfilled successfully.**")
                    if not st.session_state.has_won:
                        st.session_state.has_won = True
                        st.balloons() 
                elif score >= 70 and diff_ash_penalty == 0:
                    st.warning("⚠️ Acceptable result. Try to optimize yield further.")
                elif diff_ash_penalty > 0:
                    st.error("❌ High Ash Content! Specifications failed.")
                else:
                    st.error("❌ Yield mismatch. Adjust conditions!")
        else:
            st.info("Activate the 'Plant Manager Challenge' to test your optimization skills!")


# --- Execution Entry Point ---
if __name__ == "__main__":
    main()

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
import base64
import os 
import random 
import time 

# --- 1. Chemical and Empirical Constants ---
R_GAS = 8.314
# High Heating Value (HHV) in MJ/kg (Initial Biomass - Dry Basis)
HHV_INITIAL = {
    "Wood": 18.0, 
    "Agricultural Waste": 16.5, 
    "Municipal Waste": 15.0
}
# Torrefaction effect on HHV (Mass basis) - Rough factor
HHV_ENRICHMENT_FACTOR = 1.3 # Torrefied biomass HHV is roughly 1.3 times the initial on a mass basis

EMPIRICAL_DATA = {
    "Wood": {"A": 2.5e10, "Ea": 135000, "k_drying_base": 0.05, "Ash": 0.02, "Gas_Factor": 0.35},
    "Agricultural Waste": {"A": 5.0e11, "Ea": 150000, "k_drying_base": 0.07, "Ash": 0.08, "Gas_Factor": 0.45},
    "Municipal Waste": {"A": 1.0e12, "Ea": 165000, "k_drying_base": 0.10, "Ash": 0.15, "Gas_Factor": 0.55}
}
SIZE_FACTOR = {"Fine (<1mm)": 1.0, "Medium (1-5mm)": 0.85, "Coarse (>5mm)": 0.65}
BASE_FC_DRY_ASH_FREE = 0.20 

# --- Base64 Utility ---
LOGO_PATH = "chemisco_logo.png"

def _get_image_base64(image_path):
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()
        else:
            return None
    except Exception as e:
        return None

LOGO_BASE64_STRING = _get_image_base64(LOGO_PATH)


# --- 2. Global CSS (Updated for better aesthetics) ---
GLOBAL_CSS = """
<style>
    .stApp { padding-top: 10px; }
    
    /* Metrics Style */
    [data-testid="stMetricValue"] { font-size: 32px; color: #1D7948; font-weight: bold; }
    [data-testid="stMetricLabel"] { font-size: 14px; color: #388E3C; font-weight: 600;}
    [data-testid="stMetricDelta"] { font-size: 14px; }
    
    /* Sidebar Styling */
    .sidebar-logo-container {
        padding: 10px;
        border-radius: 8px;
        background-color: white;
        margin-bottom: 15px;
    }
    .sidebar-header-box {
        background-color: #1D7948; /* Dark Green */
        padding: 20px;
        border-radius: 10px;
        margin-top: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .sidebar-header-box h1 { color: #FFFFFF; margin: 0; font-size: 2.5em; letter-spacing: 1px; }
    .sidebar-header-box p { color: #81C784; margin: 0; font-size: 0.9em; }
    .sidebar-header-box h3 { color: #FFD700; margin: 0; font-size: 1.2em; font-family: 'Tahoma', Arial, sans-serif;} /* GOLD for Doctor's Name */

    /* BFD Styles */
    .bfd-container { display: flex; justify-content: center; align-items: center; margin: 30px 0 60px 0; position: relative; }
    .bfd-block { padding: 15px 25px; border: 3px solid #2EAF6C; border-radius: 8px; text-align: center; background-color: #F1F8E9; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); font-weight: bold; color: #1D7948; position: relative; min-width: 180px; transition: all 0.3s;}
    .bfd-block:hover { transform: translateY(-5px); box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2); }
    .bfd-block p { margin: 5px 0 0; font-size: 12px; font-weight: normal; }
    .bfd-stream { width: 70px; height: 3px; background-color: #2EAF6C; position: relative; }
    .bfd-stream::before { content: ''; position: absolute; right: -10px; top: -5px; border-top: 6px solid transparent; border-bottom: 6px solid transparent; border-left: 10px solid #2EAF6C; }
    .side-stream { position: absolute; left: 50%; transform: translateX(-50%); width: 3px; height: 40px; background-color: #FF9800; bottom: -40px; }
    .side-stream-label { position: absolute; bottom: -65px; left: 50%; transform: translateX(-50%); font-size: 11px; white-space: nowrap; color: #FF9800; }
    
    /* Chatbot Styling */
    .stChatMessage { border-radius: 15px; }
    .stChatMessage [data-testid="stMarkdownContainer"] { padding: 10px 15px; }
</style>
"""

# --- 3. Simulation Core Logic (HHV Calculation Added) ---
def simulate_torrefaction(biomass, moisture, temp_C, duration_min, size, initial_mass_kg, reactor_type="N/A"): 
    temp_K = temp_C + 273.15
    data = EMPIRICAL_DATA.get(biomass)
    R_GAS_LOCAL = R_GAS 
    
    # Arrhenius equation for Devolatilization Rate Constant
    k_devol_arrhenius = data["A"] * np.exp(-data["Ea"] / (R_GAS_LOCAL * temp_K))
    k_devol_eff = k_devol_arrhenius * SIZE_FACTOR.get(size)
    k_drying = data["k_drying_base"] 
    
    initial_moisture_frac = moisture / 100
    initial_ash_frac = data["Ash"]
    
    dry_ash_free_frac = 1.0 - initial_moisture_frac - initial_ash_frac
    fixed_carbon_frac_initial = dry_ash_free_frac * BASE_FC_DRY_ASH_FREE
    initial_volatiles_frac = dry_ash_free_frac * (1 - BASE_FC_DRY_ASH_FREE)

    mass_ash_kg = initial_mass_kg * initial_ash_frac
    mass_fixed_carbon_kg = initial_mass_kg * fixed_carbon_frac_initial
    
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
    
    final_moisture_remaining = moisture_curve[-1]
    final_volatiles_remaining = volatiles_curve[-1]
    
    final_moisture_loss = initial_moisture_frac - final_moisture_remaining
    final_volatiles_lost = initial_volatiles_frac - final_volatiles_remaining
    
    mass_volatiles_remaining = final_volatiles_remaining * initial_mass_kg
    mass_biochar_total = mass_fixed_carbon_kg + mass_volatiles_remaining + mass_ash_kg
    
    final_solid_fraction = mass_biochar_total / initial_mass_kg
    
    final_ash_percent = (mass_ash_kg / mass_biochar_total) * 100

    yields_percent = pd.DataFrame({
        "Yield (%)": [final_solid_fraction * 100, final_volatiles_lost * 100, final_moisture_loss * 100, initial_ash_frac * 100]},
        index=["Biochar (Solid Product)", "Non-Condensable Gases", "Moisture Loss (Water Vapor)", "Original Ash Content"]
    )
    
    yields_mass = yields_percent.copy()
    yields_mass["Mass (kg)"] = yields_percent["Yield (%)"] * initial_mass_kg / 100
    yields_mass.drop(columns=["Yield (%)"], inplace=True)
    
    solid_composition = pd.DataFrame({
        "Mass (kg)": [mass_fixed_carbon_kg, mass_volatiles_remaining, mass_ash_kg]
    }, index=["Fixed Carbon", "Remaining Volatiles", "Ash"])

    gas_comp_mass_fractions = {"CO2": 0.45, "CO": 0.35, "CH4": 0.15, "H2": 0.05}
    gas_composition_molar_data = {}
    total_volatiles_lost_mass = final_volatiles_lost * initial_mass_kg
    
    if total_volatiles_lost_mass > 0.001:
        for gas, fraction in gas_comp_mass_fractions.items():
            gas_composition_molar_data[gas] = fraction * 100 
    
    gas_composition_molar = pd.DataFrame.from_dict(
        gas_composition_molar_data, 
        orient="index", columns=["Molar % in Dry Gas"]
    ).fillna(0)
    if not gas_composition_molar.empty and gas_composition_molar["Molar % in Dry Gas"].sum() > 0:
        gas_composition_molar = gas_composition_molar / gas_composition_molar["Molar % in Dry Gas"].sum() * 100 
    else:
        gas_composition_molar = pd.DataFrame(0, index=["CO2", "CO", "CH4", "H2"], columns=["Molar % in Dry Gas"])


    current_solid_mass_fraction_curve = volatiles_curve + fixed_carbon_frac_initial + initial_ash_frac
    ash_concentration_percent = (initial_ash_frac / current_solid_mass_fraction_curve) * 100
    
    mass_profile = pd.DataFrame({
        "Time (min)": t,
        "Total Mass Yield (%)": current_solid_mass_fraction_curve * 100,
        "Ash Concentration in Solid (%)": ash_concentration_percent
    }).set_index("Time (min)")
    
    # --- New Energy Calculation ---
    initial_hhv_mj_kg = HHV_INITIAL.get(biomass, 17.0) 
    biochar_hhv_mj_kg = initial_hhv_mj_kg * HHV_ENRICHMENT_FACTOR
    
    initial_energy_mj = initial_mass_kg * initial_hhv_mj_kg * (1 - initial_moisture_frac)
    final_biochar_energy_mj = mass_biochar_total * biochar_hhv_mj_kg
    
    energy_yield_percent = (final_biochar_energy_mj / initial_energy_mj) * 100
    
    return {
        "yields_percent": yields_percent,
        "yields_mass": yields_mass,
        "solid_composition": solid_composition,
        "final_ash_percent": final_ash_percent,
        "gas_composition_molar": gas_composition_molar,
        "mass_profile": mass_profile,
        "k_devol_eff": k_devol_eff,
        "initial_hhv": initial_hhv_mj_kg,
        "biochar_hhv": biochar_hhv_mj_kg,
        "energy_yield_percent": energy_yield_percent,
        "parameters": {
            "biomass": biomass, "moisture": moisture, "temperature": temp_C, 
            "duration": duration_min, "size": size, "initial_mass": initial_mass_kg,
            "reactor": reactor_type
        }
    }

# --- 4. PDF Report Generation Function (No change needed here for UI/AI) ---
def generate_pdf_report(results):
    # ... (PDF generation logic remains the same) ...
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
    elements.append(Paragraph("CHEMISCO TORREFACTION REPORT", title_style))
    elements.append(Paragraph("Project presented to: د. عمرو الرفاعي", styles["Heading3"])) 
    
    try:
        img_path = LOGO_PATH 
        logo_pdf = ReportImage(img_path, width=1.5*inch, height=1.5*inch)
        logo_pdf.hAlign = 'CENTER'
        elements.append(logo_pdf)
    except FileNotFoundError:
        elements.append(Paragraph("CHEMISCO (Logo Failed to Load)", title_style)) 
        
    elements.append(Paragraph(f"Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # -- 2. Parameters Table --
    elements.append(Paragraph("1. Simulation Parameters & Kinetics", heading_style))
    p = results["parameters"]
    param_data = [
        ["Parameter", "Value"],
        ["Biomass Type", p['biomass']],
        ["Reactor Type", p['reactor']],
        ["Initial Mass", f"{p['initial_mass']} kg"],
        ["Moisture Content", f"{p['moisture']}%"],
        ["Temperature", f"{p['temperature']} °C"],
        ["Duration", f"{p['duration']} min"],
        ["Particle Size", p["size"]],
        ["Eff. Devol Rate", f"{results['k_devol_eff']:.4f} min-1"]
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
    elements.append(Paragraph("2. Product Yields & Energy", heading_style))
    yield_data = [["Component", "Mass (kg)", "Yield (%)"]]
    for idx, row in results["yields_percent"].iterrows():
        mass = results["yields_mass"].loc[idx, "Mass (kg)"]
        yield_data.append([idx, f"{mass:.2f}", f"{row['Yield (%)']:.2f}"])
    
    yield_data.append(["Initial HHV", f"{results['initial_hhv']:.2f} MJ/kg", "-"])
    yield_data.append(["Biochar HHV", f"{results['biochar_hhv']:.2f} MJ/kg", "-"])
    yield_data.append(["Energy Yield", "-", f"{results['energy_yield_percent']:.2f}"])
        
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
    
    # -- 4. Visualizations (Matplotlib for Report) --
    def get_image_bytes(fig):
        img_buf = BytesIO()
        fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=150, transparent=True)
        img_buf.seek(0)
        return img_buf

    elements.append(Paragraph("3. Results Visualization", heading_style))

    # A. Two Pie Charts Side-by-Side 
    fig_pie1, ax_pie1 = plt.subplots(figsize=(4.5, 4.5)) 
    colors_solid_pdf = ['#6A1B9A', '#AB47BC', '#BDBDBD'] 
    ax_pie1.pie(results["solid_composition"]["Mass (kg)"], labels=results["solid_composition"].index, 
                autopct='%1.1f%%', colors=colors_solid_pdf, startangle=140, pctdistance=0.85, 
                textprops={'fontsize': 8})
    ax_pie1.set_title("Solid Product Composition", fontsize=10, weight='bold')
    img1 = ReportImage(get_image_bytes(fig_pie1), width=3.25*inch, height=3.25*inch) 
    
    # Chart 2: Global Balance
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
    
    # B. Dual Axis Line Chart
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

    # C. Bar Chart (Gas)
    fig_bar, ax_bar = plt.subplots(figsize=(8, 3))
    results["gas_composition_molar"].plot(kind='bar', ax=ax_bar, legend=False, color='#1565C0') 
    ax_bar.set_title("Dry Gas Composition (Molar %)", fontsize=12)
    ax_bar.set_ylabel("Molar %")
    plt.xticks(rotation=0)
    ax_bar.grid(axis='y', alpha=0.4, linestyle='--', color='lightgrey')
    
    img_bar = ReportImage(get_image_bytes(fig_bar), width=6.5*inch, height=2.5*inch)
    elements.append(img_bar)
    plt.close(fig_bar)
    
    # Build
    doc.build(elements)
    buffer.seek(0)
    return buffer


# --- 5. AI Chatbot Logic (MOCK FUNCTION) - Fully Updated ---
def mock_ai_response(prompt, results):
    """
    MOCK AI function: Provides simulated responses based on keywords.
    """
    p = results["parameters"]
    
    prompt_lower = prompt.lower()
    
    # 1. Pyrolysis vs Torrefaction Comparison
    if "pyrolysis" in prompt_lower and "torrefaction" in prompt_lower or "pyrolysis" in prompt_lower and "vs" in prompt_lower or "مقارنة" in prompt:
        
        return """Torrefaction and Pyrolysis are both thermal processes but differ significantly in **temperature**, **product focus**, and **biomass change**. Here is a quick comparison:

| Feature | Torrefaction (Mild Pyrolysis) | Pyrolysis (Fast/Intermediate) |
| :--- | :--- | :--- |
| **Temperature** | $200^\circ C$ to $300^\circ C$ (Low) | $400^\circ C$ to $700^\circ C$ (High) |
| **Atmosphere** | Inert or low oxygen (Mild) | Inert (Strictly oxygen-free) |
| **Main Product** | **Solid Biochar** (Torrefied Biomass) | **Bio-Oil/Bio-Fuel** (Liquid) |
| **Yield Focus** | Maximize **Solid** yield (70-90% mass) | Maximize **Liquid** yield (50-75% mass) |
| **Biomass Change** | Removes water, slight devolatilization. Biochar is **hydrophobic** and brittle. | Heavy devolatilization. Structure completely breaks down. |
| **Process Duration** | Minutes to Hours (Slower) | Seconds (Very Fast) |

**Conclusion:** Our simulator models a **Torrefaction** process, aiming for a high-quality solid fuel."""

    # 2. HHV / Energy Yield
    if "hhv" in prompt_lower or "heating value" in prompt_lower or "حرارية" in prompt or "طاقة" in prompt:
        
        energy_gain = results['biochar_hhv'] - results['initial_hhv']
        
        return f"""The **Higher Heating Value (HHV)** represents the potential energy of the fuel.
        * **Initial Biomass HHV (Dry):** **{results['initial_hhv']:.2f} MJ/kg**
        * **Torrefied Biochar HHV:** **{results['biochar_hhv']:.2f} MJ/kg**
        
        The increase is **{energy_gain:.2f} MJ/kg**. This enrichment happens because the low-energy volatile components are removed, increasing the relative concentration of **carbon (Fixed Carbon)** in the final solid product.
        
        The **Energy Yield** is **{results['energy_yield_percent']:.1f}%**, meaning this percentage of the initial energy content is retained in the biochar."""
        
    # 3. Heat Reclaim / Autothermal
    if "reclaim" in prompt_lower or "heat" in prompt_lower and "gas" in prompt_lower or "autothermal" in prompt_lower:
        
        total_gas_mass = results["yields_mass"].loc["Non-Condensable Gases", "Mass (kg)"]
        
        return f"""During torrefaction, the volatile gases produced ({total_gas_mass:.2f} kg in this simulation) have a significant calorific value.
        The concept of **Autothermal Torrefaction** is to combust a portion of these gases to provide the necessary heat for the drying and torrefaction steps, greatly reducing external energy consumption. This is a crucial strategy for making the process economically and environmentally sustainable."""


    # 4. Optimization
    if "optimize" in prompt_lower or "increase yield" in prompt_lower or "best conditions" in prompt_lower:
        
        return f"""To achieve higher **Biochar Yield** and lower **Ash Concentration** for {p['biomass']}, consider the following:
        1. **Lower Temperature:** Try reducing the temperature to around **250 °C** to minimize volatile losses.
        2. **Shorter Duration:** Use a duration of **30-40 minutes**.
        3. **Current Results:** Your current yield is {results["yields_percent"].loc["Biochar (Solid Product)", "Yield (%)"]:.1f}% with {results['final_ash_percent']:.1f}% ash. For optimization, the solid mass yield is the most crucial metric to maximize."""
    
    # 5. Ash Concentration
    if "ash" in prompt_lower or "ash concentration" in prompt_lower or "inerts" in prompt_lower:
        
        initial_ash_percentage = EMPIRICAL_DATA[p['biomass']]["Ash"] * 100
        
        if initial_ash_percentage > 0:
            enrichment_factor = results['final_ash_percent'] / initial_ash_percentage
        else:
            enrichment_factor = 1.0 if results['final_ash_percent'] < 0.01 else 999.0

        return f"""Ash concentration increases because the inert ash remains while other components (moisture and volatiles) are removed. 
        Current Ash Concentration: **{results['final_ash_percent']:.2f}%**. 
        This is an **enrichment factor** of {enrichment_factor:.2f} compared to the initial biomass. Lowering the initial ash content in the feedstock is the only way to reduce this value, as ash is not consumed during torrefaction."""

    # 6. Reactor Type
    if "reactor" in prompt_lower or "type" in prompt_lower:
        return f"""You are currently simulating a **{p['reactor']}**. The choice of reactor affects heat transfer and mixing. **Fluidized Bed Reactors** provide excellent heat transfer but have high operating costs, while **Rotary Drum Reactors** are often preferred for continuous, large-scale production."""

    # 7. Gas Composition
    if "gas" in prompt_lower or "composition" in prompt_lower or "volatiles" in prompt_lower:
        gas_total = results["yields_mass"].loc["Non-Condensable Gases", "Mass (kg)"]
        return f"""The main gaseous components are CO2, CO, CH4, and H2. The total mass of non-condensable gases produced is **{gas_total:.2f} kg**. This gas is rich in energy and can be combusted to provide the heat required for the torrefaction process itself (autothermal operation), thus lowering your operational cost."""

    # 8. General Definition
    if "torrefaction" in prompt_lower or "process" in prompt_lower or "what is" in prompt_lower:
        return "Torrefaction is a mild pyrolysis process conducted typically between $200^\circ C$ and $300^\circ C$ in an inert or low-oxygen atmosphere. It removes moisture and volatile organic compounds, resulting in a hydrophobic, brittle, and energy-dense solid product called **biochar** (or torrefied biomass)."

    # Default/General response
    return "I am the Chemisco AI Assistant. I can help explain the torrefaction process kinetics, analyze the results, or suggest optimized parameters based on your current inputs. Try asking about the **HHV**, **heat reclaim**, or the difference between **pyrolysis vs torrefaction**."


# --- 6. Main Streamlit App ---
def main():
    # Set page configuration with a wide layout
    st.set_page_config(page_title="Chemisco Torrefaction Simulator", layout="wide", initial_sidebar_state="expanded")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Welcome! I am the Chemisco AI Assistant. How can I help you analyze your torrefaction simulation?"}]
    
    # --- Sidebar ---
    with st.sidebar:
        if LOGO_BASE64_STRING:
            st.markdown(f"""
                <div class="sidebar-logo-container">
                    <img src="data:image/png;base64,{LOGO_BASE64_STRING}" style="width: 80%; display: block; margin: 0 auto;">
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("## Chemisco")
            st.warning("⚠️ Logo file not found.")

        # Updated Sidebar Header (Better colors, Arabic Name)
        st.markdown(f"""
            <div class="sidebar-header-box">
                <h1>CHEMISCO</h1>
                <p>Torrefaction Process Simulator</p>
                <hr style='margin: 10px 0; border-color: #388E3C;'>
                <p style='color: #C8E6C9;'>Project presented to:</p>
                <h3>د. عمرو الرفاعي</h3>
            </div>
            """, unsafe_allow_html=True)
        
        st.header("⚙️ Input Parameters")
        
        reactor_type = st.selectbox("Reactor Type", 
            ["Rotary Drum Reactor", "Fluidized Bed Reactor", "Auger/Screw Reactor", "Fixed Bed Reactor"])
        
        with st.expander("🌲 Biomass Properties", expanded=True):
            initial_mass_kg = st.number_input("Initial Biomass Mass (kg)", min_value=1.0, value=100.0, step=10.0)
            biomass_type = st.selectbox("Biomass Type", list(EMPIRICAL_DATA.keys()))
            moisture_content = st.slider("Initial Moisture Content (%)", 0.0, 50.0, 10.0, step=1.0)
            particle_size = st.selectbox("Particle Size", list(SIZE_FACTOR.keys()))
            
        with st.expander("🌡️ Process Conditions", expanded=True):
            temperature = st.slider("Torrefaction Temperature (°C)", 200, 350, 275, step=5)
            duration = st.slider("Process Duration (min)", 10, 120, 45, step=5)
            ash_percent_init = EMPIRICAL_DATA[biomass_type]["Ash"] * 100
            st.info(f"Initial Ash Content: **{ash_percent_init:.1f}%**")
            
        with st.expander("💰 Cost Management", expanded=False):
            st.caption("Economic Feasibility Parameters")
            cost_biomass_per_ton = st.number_input("Biomass Feedstock Cost ($/ton)", min_value=0.0, value=30.0, step=5.0)
            cost_energy_per_hour = st.number_input("Operational/Energy Cost ($/hour)", min_value=0.0, value=5.0, step=0.5, help="Total cost of electricity + labor per hour of operation")
            price_biochar_per_kg = st.number_input("Biochar Selling Price ($/kg)", min_value=0.0, value=1.20, step=0.1)
            
        st.markdown("---")
        st.subheader("🎮 Gamification")
        game_mode = st.checkbox("Activate 'Plant Manager Challenge'", value=False)


    # --- Main Header ---
    st.title("CHEMISCO: Advanced Torrefaction Simulator")
    st.subheader("Optimizing Biochar Production through Advanced Modeling")
    st.markdown("---")
    
    # BFD (Block Flow Diagram)
    bfd_html = f"""
    <div class="bfd-container">
        <div class="bfd-block">
            FEED PREPARATION
            <p style="color: #1565C0;">Mass: {initial_mass_kg:.0f} kg</p>
            <p style="color: #0277BD;">Moist: {moisture_content:.1f}%</p>
        </div>
        <div class="bfd-stream"></div>
        <div class="bfd-block">
            DRYING & PREHEATING
            <p>100 °C - 200 °C</p>
            <div class="side-stream"></div>
            <div class="side-stream-label">Water Vapor</div>
        </div>
        <div class="bfd-stream"></div>
        <div class="bfd-block" style="border-color: #D32F2F; background-color: #FFCDD2; color: #B71C1C;">
            {reactor_type.upper()}
            <p style="color: #B71C1C;">Temp: {temperature} °C</p>
            <p style="color: #B71C1C;">Duration: {duration} min</p>
            <div class="side-stream" style="background-color: #FFD700;"></div>
            <div class="side-stream-label" style="color: #FFD700;">Volatile Gases</div>
        </div>
        <div class="bfd-stream"></div>
        <div class="bfd-block" style="border-color: #1D7948; background-color: #C8E6C9; color: #1D7948;">
            COOLING & PRODUCT
            <p>Torrefied Biochar</p>
        </div>
    </div>
    <div style="height: 40px;"></div>
    """
    st.subheader("Process Flow Block Diagram (BFD)")
    st.markdown(bfd_html, unsafe_allow_html=True)
    
    # Input validation
    if moisture_content / 100 + EMPIRICAL_DATA[biomass_type]["Ash"] > 1:
        st.error("**Input Error:** Initial Moisture and Ash content exceed 100%. Please adjust input parameters.")
        return 
        
    # Run Simulation
    results = simulate_torrefaction(biomass_type, moisture_content, temperature, duration, particle_size, initial_mass_kg, reactor_type)
    
    # --- GAME LOGIC SECTION (Celebration Added) ---
    if game_mode:
        st.markdown("---")
        st.markdown("""
        <div style="background-color: #E8F5E9; padding: 20px; border-radius: 10px; border-left: 6px solid #2EAF6C; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style="color: #1D7948; margin-top:0;">🏭 Plant Manager Challenge</h3>
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
            # Scoring adjusted: 10 points deduction per 1% yield difference, 20 points deduction per 1% ash difference
            score = max(0, 100 - (diff_yield * 10 + diff_ash * 20))
            st.metric("🏆 Your Efficiency Score", f"{score:.1f} / 100")
            
            if score >= 90:
                st.success("🎉 **PERFECT MATCH! Order fulfilled successfully.**")
                if not st.session_state.has_won:
                    st.session_state.has_won = True
                    st.balloons() # CELEBRATION
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
                st.experimental_rerun()
        st.markdown("---")
    # --------------------------

    # --- Display Results ---
    st.header("📊 Simulation Results & Analysis")
    
    # AI Assistant Tab Added Here
    tab1, tab2, tab3, tab4, tab5, tab_ai = st.tabs(["Yields & Ash Enrichment", "Ash & Mass Kinetics", "Energy & HHV", "💰 Cost Analysis", "PDF Report", "🤖 AI Assistant"])
    
    with tab1:
        st.subheader(f"Product Yields & Ash Enrichment")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        biochar_mass = results["yields_mass"].loc["Biochar (Solid Product)", "Mass (kg)"]
        col_m1.metric("⚖️ Total Biochar Mass", f"{biochar_mass:.2f} kg", delta=f"{results['yields_percent'].loc['Biochar (Solid Product)', 'Yield (%)']:.1f}% Yield")
        
        final_ash = results["final_ash_percent"]
        ash_increase = final_ash - ash_percent_init
        col_m2.metric("⚗️ Final Ash Concentration", f"{final_ash:.2f} %", delta=f"+{ash_increase:.2f}% (Enrichment)")
        
        moisture_loss = results["yields_mass"].loc["Moisture Loss (Water Vapor)", "Mass (kg)"]
        col_m3.metric("💧 Moisture Removed", f"{moisture_loss:.2f} kg")
        
        gas_total = results["yields_mass"].loc["Non-Condensable Gases", "Mass (kg)"]
        col_m4.metric("💨 Volatiles Mass", f"{gas_total:.2f} kg")


        st.markdown("---")
        
        col_t1, col_t2 = st.columns(2)
        
        # --- PLOTLY PIE CHARTS ---
        with col_t1:
            st.markdown("##### Final Biochar Composition")
            
            df_solid = results["solid_composition"].reset_index()
            df_solid.columns = ["Component", "Mass (kg)"]
            
            fig1 = px.pie(df_solid, values='Mass (kg)', names='Component', hole=0.5,
                            color='Component',
                            color_discrete_map={
                                "Fixed Carbon": "#6A1B9A", 
                                "Remaining Volatiles": "#AB47BC", 
                                "Ash": "#BDBDBD" 
                            })
            
            fig1.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(t=20, b=50, l=10, r=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig1, use_container_width=True)

        with col_t2:
            st.markdown("##### Global Mass Balance")
            
            filtered_yields = results["yields_percent"].iloc[[0, 1, 2]].reset_index()
            filtered_yields.columns = ["Component", "Yield (%)"]
            
            fig2 = px.pie(filtered_yields, values='Yield (%)', names='Component', hole=0.5,
                            color='Component',
                            color_discrete_map={
                                "Biochar (Solid Product)": "#1D7948", 
                                "Non-Condensable Gases": "#2EAF6C", 
                                "Moisture Loss (Water Vapor)": "#9CCC65" 
                            })
            
            fig2.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(t=20, b=50, l=10, r=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            fig2.update_traces(textposition='inside', textinfo='percent')
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Ash Concentration & Mass Depletion Kinetics")
        
        # --- PLOTLY DUAL-AXIS CHART ---
        fig_dual = go.Figure()

        # Line 1: Total Mass (Left Axis)
        fig_dual.add_trace(go.Scatter(
            x=results["mass_profile"].index,
            y=results["mass_profile"]["Total Mass Yield (%)"],
            name="Total Mass %",
            line=dict(color="#1D7948", width=3), 
            yaxis="y1"
        ))

        # Line 2: Ash Concentration (Right Axis)
        fig_dual.add_trace(go.Scatter(
            x=results["mass_profile"].index,
            y=results["mass_profile"]["Ash Concentration in Solid (%)"],
            name="Ash Concentration %",
            line=dict(color="#D32F2F", width=3, dash='dot'), 
            yaxis="y2"
        ))

        fig_dual.update_layout(
            title="Dynamic Ash Enrichment Logic",
            xaxis=dict(title="Time (min)", showgrid=False),
            
            # Left Axis (Primary)
            yaxis=dict(
                title=dict(text="Total Mass Remaining (%)", font=dict(color="#1D7948")),
                tickfont=dict(color="#1D7948"),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            
            # Right Axis (Secondary)
            yaxis2=dict(
                title=dict(text="Ash Concentration (%)", font=dict(color="#D32F2F")),
                tickfont=dict(color="#D32F2F"),
                overlaying="y",
                side="right",
                showgrid=False
            ),
            
            legend=dict(x=0.1, y=1.1, orientation="h"),
            hovermode="x unified",
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            height=450
        )

        st.plotly_chart(fig_dual, use_container_width=True)
        
        st.info("""
        **Logic Explanation:** The Total Mass Yield (Green line) decreases as moisture and volatiles are removed. The Ash Concentration (Red line) mathematically increases because the inert ash remains while the total mass shrinks.
        """)

    with tab3:
        st.subheader("Energy Yield and Heating Value")
        col_e1, col_e2, col_e3 = st.columns(3)
        
        col_e1.metric("🔥 Initial HHV (Dry)", f"{results['initial_hhv']:.2f} MJ/kg")
        col_e2.metric("✨ Biochar HHV", f"{results['biochar_hhv']:.2f} MJ/kg", delta=f"+{results['biochar_hhv'] - results['initial_hhv']:.2f} MJ/kg")
        col_e3.metric("⚡ Energy Yield", f"{results['energy_yield_percent']:.1f} %", delta="Energy retained in the biochar")

        st.markdown("---")
        st.subheader("Gas Composition (Potential Energy Source)")
        st.bar_chart(results["gas_composition_molar"])
        st.info("The volatile gases (CO, CH4) can be used to provide the process heat, potentially making the reactor 'autothermal'.")

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
            x = ["Feedstock Cost", "Operational Cost", "Revenue", "Net Profit"],
            textposition = "outside",
            y = [-cost_feedstock_total, -cost_operations_total, revenue_total, net_profit],
            connector = {"line": {"color": "rgb(63, 63, 63)"}},
            decreasing = {"marker":{"color": "#D32F2F"}}, 
            increasing = {"marker":{"color": "#1D7948"}}, 
            totals = {"marker":{"color": "#FFD700", "line":{"width":1, "color":"#FFD700"}}}
        ))

        fig_waterfall.update_layout(
            title = "Revenue vs. Cost Breakdown",
            showlegend = False,
            margin=dict(t=50, b=50, l=50, r=50),
            height=450
        )

        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        st.info(f"Total Initial Mass: **{initial_mass_kg:.2f} kg** | Biochar Produced: **{biochar_produced_kg:.2f} kg**")

    with tab5:
        st.subheader("📥 Generate Report")
        st.info("Create a comprehensive PDF document of the simulation results, parameters, and visualizations.")
        
        pdf_buffer = generate_pdf_report(results)
        
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_buffer,
            file_name=f"Chemisco_Torrefaction_Report_{biomass_type}_{temperature}C.pdf",
            mime="application/pdf"
        )
    
    # --- AI ASSISTANT TAB ---
    with tab_ai:
        st.header("🤖 AI Assistant: Torrefaction Expert")
        st.info("Ask me about process optimization, result analysis, the difference between Pyrolysis and Torrefaction, the **HHV**, or **heat reclaim**!")

        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input handling
        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing simulation data..."):
                    time.sleep(1) 
                    ai_response = mock_ai_response(prompt, results)
                
                st.markdown(ai_response)
                
                st.session_state.messages.append({"role": "assistant", "content": ai_response})


# --- Execution Entry Point ---
if __name__ == "__main__":
    main()

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

# --- 1. Chemical and Kinetic Constants (Multi-Component Model) ---
R_GAS = 8.314 # J/(mol.K)
HHV_INITIAL = { "Wood": 18.0, "Agricultural Waste": 16.5, "Municipal Waste": 15.0 }
HHV_ENRICHMENT_FACTOR = 1.3 

# KINETIC PARAMETERS (Parallel First-Order Reactions for Torrefaction)
# A (min^-1), Ea (J/mol)
KINETICS = {
    # Component: [A, Ea, Yield_Solid_Product_Factor]
    "Hemicellulose": [1.5e10, 110000, 0.40], # Devolatilizes first, lower Ea
    "Cellulose":     [1.0e12, 130000, 0.55], # Devolatilizes second, medium Ea
    "Lignin":        [2.0e9, 100000, 0.70]   # Most stable, higher solid yield factor
}
# Initial mass fractions of components (Dry Ash Free Basis)
BIOMASS_COMPOSITION = {
    "Wood": {"Hemicellulose": 0.35, "Cellulose": 0.45, "Lignin": 0.20, "Ash": 0.02},
    "Agricultural Waste": {"Hemicellulose": 0.45, "Cellulose": 0.35, "Lignin": 0.20, "Ash": 0.08},
    "Municipal Waste": {"Hemicellulose": 0.30, "Cellulose": 0.40, "Lignin": 0.30, "Ash": 0.15}
}
# Other Constants
DRYING_RATE_CONST = 0.05 
SIZE_FACTOR = {"Fine (<1mm)": 1.0, "Medium (1-5mm)": 0.85, "Coarse (>5mm)": 0.65}
BASE_FC_FACTOR = 0.20 # Approximate Fixed Carbon Fraction in Total Volatiles (DAF)

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


# --- 2. Global CSS (Enhanced Aesthetic) ---
GLOBAL_CSS = """
<style>
    .stApp { padding-top: 10px; background-color: #F8F9FA; }
    
    /* Metrics Style */
    [data-testid="stMetricValue"] { font-size: 36px; color: #1D7948; font-weight: 900; }
    [data-testid="stMetricLabel"] { font-size: 14px; color: #388E3C; font-weight: 600;}
    [data-testid="stMetricDelta"] { font-size: 16px; font-weight: bold;}
    
    /* Sidebar Styling */
    .sidebar-header-box {
        background-color: #1D7948; /* Dark Green */
        padding: 20px;
        border-radius: 10px;
        margin-top: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .sidebar-header-box h1 { color: #FFFFFF; margin: 0; font-size: 2.8em; letter-spacing: 2px; }
    .sidebar-header-box p { color: #81C784; margin: 0; font-size: 1.0em; font-weight: 500;}
    .sidebar-header-box h3 { color: #FFD700; margin: 5px 0 0; font-size: 1.4em; font-family: 'GE SS Unique', Arial, sans-serif;} /* GOLD for Doctor's Name */
    
    /* Headers and Tabs */
    h1, h2, h3 { color: #1D7948; }
    div[data-testid="stTabs"] button {
        color: #1D7948 !important;
        font-weight: bold !important;
        border-bottom: 3px solid #FFD700 !important;
    }
    
    /* Info Blocks */
    div.stAlert.info {
        background-color: #E6F0E8;
        border-left: 5px solid #2EAF6C;
    }
    
    /* Block Flow Diagram (BFD) - Enhanced Visuals */
    .bfd-container { display: flex; justify-content: center; align-items: center; margin: 30px 0 60px 0; position: relative; }
    .bfd-block { padding: 18px 30px; border: 3px solid #2EAF6C; border-radius: 10px; text-align: center; background-color: #FFFFFF; box-shadow: 0 6px 15px rgba(0, 0, 0, 0.15); font-weight: bold; color: #1D7948; position: relative; min-width: 200px; transition: all 0.4s;}
    .bfd-block:hover { transform: scale(1.02); box-shadow: 0 10px 20px rgba(0, 0, 0, 0.25); }
    .bfd-stream { width: 80px; height: 4px; background-color: #2EAF6C; position: relative; }
    .bfd-stream::before { content: ''; position: absolute; right: -12px; top: -6px; border-top: 8px solid transparent; border-bottom: 8px solid transparent; border-left: 12px solid #2EAF6C; }
    .side-stream { position: absolute; left: 50%; transform: translateX(-50%); width: 4px; height: 50px; background-color: #FF9800; bottom: -50px; }
    .side-stream-label { position: absolute; bottom: -80px; left: 50%; transform: translateX(-50%); font-size: 12px; white-space: nowrap; color: #FF9800; font-weight: bold;}
    
    /* Chatbot Styling */
    .stChatMessage { 
        border-radius: 15px; 
        background-color: #FFFFFF;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    .stChatMessage [data-testid="stMarkdownContainer"] { padding: 10px 15px; }
</style>
"""

# --- 3. Simulation Core Logic (Multi-Component Model) ---
def simulate_torrefaction(biomass, moisture, temp_C, duration_min, size, initial_mass_kg, reactor_type="N/A"): 
    temp_K = temp_C + 273.15
    comp = BIOMASS_COMPOSITION.get(biomass)
    R_GAS_LOCAL = R_GAS 
    
    # 1. Initial Fractions & Masses
    initial_moisture_frac = moisture / 100
    initial_ash_frac = comp["Ash"]
    
    # Dry Ash Free Fraction
    daf_frac = 1.0 - initial_moisture_frac - initial_ash_frac
    
    # Initial DAF component masses (normalized to DAF fraction)
    m_h_init = comp["Hemicellulose"] * daf_frac
    m_c_init = comp["Cellulose"] * daf_frac
    m_l_init = comp["Lignin"] * daf_frac
    
    # Fixed carbon and remaining volatiles assumption based on DAF
    # This factor is simplified but necessary for initial mass balance closure
    initial_mass_fixed_carbon_daf = daf_frac * BASE_FC_FACTOR 
    
    # 2. Rate Constants (Arrhenius for all components)
    k_drying = DRYING_RATE_CONST * SIZE_FACTOR.get(size)
    
    k_h_arr = KINETICS["Hemicellulose"][0] * np.exp(-KINETICS["Hemicellulose"][1] / (R_GAS_LOCAL * temp_K))
    k_c_arr = KINETICS["Cellulose"][0] * np.exp(-KINETICS["Cellulose"][1] / (R_GAS_LOCAL * temp_K))
    k_l_arr = KINETICS["Lignin"][0] * np.exp(-KINETICS["Lignin"][1] / (R_GAS_LOCAL * temp_K))
    
    # Apply particle size factor to reaction kinetics
    size_factor_val = SIZE_FACTOR.get(size)
    k_h_eff = k_h_arr * size_factor_val
    k_c_eff = k_c_arr * size_factor_val
    k_l_eff = k_l_arr * size_factor_val

    # 3. ODE System (Differential Equations for mass fractions)
    # y = [m_moist, m_h, m_c, m_l]
    def model(y, t, k_dry, kh, kc, kl):
        m_moist, m_h, m_c, m_l = y
        
        # Drying (only if moisture is present)
        d_moist = -k_dry * m_moist if m_moist > 0.001 else 0 
        
        # Devolatilization (parallel first-order reactions)
        d_h = -kh * m_h
        d_c = -kc * m_c
        d_l = -kl * m_l
        
        return [d_moist, d_h, d_c, d_l]
    
    t = np.linspace(0, duration_min, 100)
    y0 = [initial_moisture_frac, m_h_init, m_c_init, m_l_init]
    
    sol = odeint(model, y0, t, args=(k_drying, k_h_eff, k_c_eff, k_l_eff))
    sol[sol < 0] = 0
    
    # 4. Final Mass Balance & Products
    
    final_moisture_remaining = sol[:, 0][-1]
    final_h_remaining = sol[:, 1][-1]
    final_c_remaining = sol[:, 2][-1]
    final_l_remaining = sol[:, 3][-1]
    
    # Mass of remaining volatiles (from the initial DAF component masses)
    # This is the mass fraction that remains in the solid product after the process
    final_solid_volatiles_frac = final_h_remaining + final_c_remaining + final_l_remaining
    
    # Mass of components that reacted (Lost volatiles)
    lost_h_frac = m_h_init - final_h_remaining
    lost_c_frac = m_c_init - final_c_remaining
    lost_l_frac = m_l_init - final_l_remaining
    
    total_volatiles_lost_frac = lost_h_frac + lost_c_frac + lost_l_frac

    # Calculate final fixed carbon based on component conversion factors
    # Fixed Carbon remains in the solid, it is NOT lost as volatile gas during torrefaction
    # Total fixed carbon in product = Initial Fixed Carbon + (Conversion_Factor * Lost_Component_Mass)
    
    # Simplification: Assume fixed carbon comes from unconverted material and charring of the lost fraction.
    # We use the initial fixed carbon estimate (BASE_FC_FACTOR) and assume it is NOT lost.
    mass_fixed_carbon_kg = initial_mass_kg * initial_mass_fixed_carbon_daf
    
    # Biochar Mass = Fixed Carbon + Remaining Components (H, C, L) + Ash
    mass_ash_kg = initial_mass_kg * initial_ash_frac
    mass_remaining_components = (final_h_remaining + final_c_remaining + final_l_remaining) * initial_mass_kg

    mass_biochar_total = mass_fixed_carbon_kg + mass_remaining_components + mass_ash_kg
    
    final_solid_yield_percent = (mass_biochar_total / initial_mass_kg) * 100
    
    # Lost Water and Gas
    mass_moisture_loss_kg = (initial_moisture_frac - final_moisture_remaining) * initial_mass_kg
    mass_non_condensable_gas_kg = total_volatiles_lost_frac * initial_mass_kg * BIOMASS_COMPOSITION[biomass]["Gas_Factor"] 
    mass_bio_oil_kg = total_volatiles_lost_frac * initial_mass_kg * (1 - BIOMASS_COMPOSITION[biomass]["Gas_Factor"]) 

    # Final Ash Concentration
    final_ash_percent = (mass_ash_kg / mass_biochar_total) * 100

    # 5. Output Data Structure
    yields_percent = pd.DataFrame({
        "Yield (%)": [final_solid_yield_percent, (mass_bio_oil_kg / initial_mass_kg) * 100, (mass_non_condensable_gas_kg / initial_mass_kg) * 100, (mass_moisture_loss_kg / initial_mass_kg) * 100]},
        index=["Biochar (Solid Product)", "Bio-Oil (Condensable)", "Non-Condensable Gases", "Moisture Loss (Water Vapor)"]
    )
    
    yields_mass = yields_percent.copy()
    yields_mass["Mass (kg)"] = yields_percent["Yield (%)"] * initial_mass_kg / 100
    yields_mass.drop(columns=["Yield (%)"], inplace=True)
    
    solid_composition = pd.DataFrame({
        "Mass (kg)": [mass_fixed_carbon_kg, mass_remaining_components, mass_ash_kg]
    }, index=["Fixed Carbon", "Volatile Matter Remaining", "Ash"])

    # Gas Composition (Simplified - more detailed components for better AI answers)
    gas_comp_mass_fractions = {"CO2": 0.50, "CO": 0.30, "CH4": 0.15, "H2": 0.05}
    gas_composition_molar = pd.DataFrame.from_dict(
        {k: v * 100 for k, v in gas_comp_mass_fractions.items()}, 
        orient="index", columns=["Molar % in Dry Gas"]
    )

    # Mass Profile for the chart
    current_solid_mass_fraction_curve = sol[:, 1] + sol[:, 2] + sol[:, 3] + initial_mass_fixed_carbon_daf + initial_ash_frac
    ash_concentration_percent = (initial_ash_frac / current_solid_mass_fraction_curve) * 100
    
    mass_profile = pd.DataFrame({
        "Time (min)": t,
        "Total Mass Yield (%)": current_solid_mass_fraction_curve * 100,
        "Ash Concentration in Solid (%)": ash_concentration_percent
    }).set_index("Time (min)")
    
    # 6. Energy & Sustainability Metrics
    initial_hhv_mj_kg = HHV_INITIAL.get(biomass, 17.0) 
    biochar_hhv_mj_kg = initial_hhv_mj_kg * HHV_ENRICHMENT_FACTOR
    
    initial_energy_mj = initial_mass_kg * initial_hhv_mj_kg * (1 - initial_moisture_frac)
    final_biochar_energy_mj = mass_biochar_total * biochar_hhv_mj_kg
    energy_yield_percent = (final_biochar_energy_mj / initial_energy_mj) * 100
    
    # Carbon Efficiency (Carbon Retained in Biochar / Carbon in Initial Biomass)
    # Simplified calculation based on mass yield and HHV enrichment factor
    carbon_efficiency = final_solid_yield_percent * (biochar_hhv_mj_kg / initial_hhv_mj_kg) / 100 
    
    # Overall Kinetics Factor (for AI)
    avg_devol_rate = (k_h_eff + k_c_eff + k_l_eff) / 3
    
    return {
        "yields_percent": yields_percent,
        "yields_mass": yields_mass,
        "solid_composition": solid_composition,
        "final_ash_percent": final_ash_percent,
        "gas_composition_molar": gas_composition_molar,
        "mass_profile": mass_profile,
        "initial_hhv": initial_hhv_mj_kg,
        "biochar_hhv": biochar_hhv_mj_kg,
        "energy_yield_percent": energy_yield_percent,
        "carbon_efficiency": carbon_efficiency,
        "avg_devol_rate": avg_devol_rate,
        "mass_bio_oil_kg": mass_bio_oil_kg,
        "mass_non_condensable_gas_kg": mass_non_condensable_gas_kg,
        "parameters": {
            "biomass": biomass, "moisture": moisture, "temperature": temp_C, 
            "duration": duration_min, "size": size, "initial_mass": initial_mass_kg,
            "reactor": reactor_type
        }
    }

# --- 4. PDF Report Generation Function (Retained/Updated) ---
def generate_pdf_report(results):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleStyleSheet()
    elements = []
    
    # ... (PDF generation logic updated for new metrics) ...
    # Styles
    title_style = styles["Title"]
    title_style.textColor = colors.HexColor("#1D7948")
    heading_style = styles["Heading2"]
    heading_style.textColor = colors.HexColor("#2EAF6C")
    normal_style = styles["Normal"]
    
    # Header
    elements.append(Paragraph("CHEMISCO ADVANCED TORREFACTION REPORT", title_style))
    elements.append(Paragraph("Project presented to: د. عمرو الرفاعي", styles["Heading3"])) 
    elements.append(Paragraph(f"Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # 1. Parameters Table
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
        ["Avg. Devol Rate", f"{results['avg_devol_rate']:.4f} min-1"]
    ]
    
    t_param = Table(param_data, colWidths=[3*inch, 3*inch])
    t_param.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E8F5E9")),
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
    ]))
    elements.append(t_param)
    elements.append(Spacer(1, 0.2*inch))
    
    # 2. Yields Table
    elements.append(Paragraph("2. Product Yields & Energy Metrics", heading_style))
    yield_data = [["Component", "Mass (kg)", "Yield (%)"]]
    for idx, row in results["yields_percent"].iterrows():
        mass = results["yields_mass"].loc[idx, "Mass (kg)"]
        yield_data.append([idx, f"{mass:.2f}", f"{row['Yield (%)']:.2f}"])
    
    # Add Energy/Carbon Metrics
    yield_data.append(["Biochar HHV", f"{results['biochar_hhv']:.2f} MJ/kg", "-"])
    yield_data.append(["Energy Yield", "-", f"{results['energy_yield_percent']:.2f} %"])
    yield_data.append(["Carbon Efficiency", "-", f"{results['carbon_efficiency'] * 100:.2f} %"])
        
    t_yield = Table(yield_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
    t_yield.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E8F5E9")),
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ]))
    elements.append(t_yield)
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph(f"<b>Final Ash Concentration:</b> {results['final_ash_percent']:.2f}%", normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # 3. Visualizations (Placeholder for brevity - full charts are in Streamlit app)
    elements.append(Paragraph("3. Results Visualization", heading_style))
    elements.append(Paragraph("Charts included in the interactive web report.", normal_style))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer


# --- 5. AI Chatbot Logic (MOCK FUNCTION) - Maximized Answers ---
def mock_ai_response(prompt, results):
    """
    MOCK AI function: Provides simulated responses based on keywords, 
    with richer detail and better handling of Arabic context.
    """
    p = results["parameters"]
    prompt_lower = prompt.lower()
    
    # --- 1. Pyrolysis vs Torrefaction Comparison (Expanded) ---
    if "pyrolysis" in prompt_lower and "torrefaction" in prompt_lower or "pyrolysis" in prompt_lower and "vs" in prompt_lower or "مقارنة" in prompt_lower or "فرق" in prompt_lower:
        
        return """
        ## ⚖️ مقارنة شاملة بين البيروليسيس و التوريفكشن (Torrefaction vs Pyrolysis)

| الميزة (Feature) | التوريفكشن (Torrefaction) | البيروليسيس (Pyrolysis) |
| :--- | :--- | :--- |
| **نطاق الحرارة** | $200^\circ C$ - $300^\circ C$ (منخفض) | $400^\circ C$ - $700^\circ C$ (مرتفع) |
| **هدف المنتج الرئيسي** | **فحم حيوي صلب** عالي الجودة (Biochar) | **زيت حيوي سائل** (Bio-Oil) |
| **كفاءة الكتلة (Mass Yield)** | عالية (65-90%) | منخفضة/متوسطة (30-50%) |
| **كفاءة الطاقة (Energy Yield)** | عالية جداً (85-95%) | متوسطة (60-80%) |
| **تغير الكتلة الحيوية** | إزالة الرطوبة والمركبات قليلة التطاير (Hemicellulose). المنتج يصبح **كاره للماء** (Hydrophobic). | تكسير حراري كامل (Thermal Decomposition) لجميع المكونات. |
| **الاستخدام الأساسي** | وقود صلب بديل للفحم (Coal Substitute) | إنتاج مواد كيميائية، وقود سائل متجدد. |

**النتيجة:** المحاكي الحالي يركز على **التوريفكشن المعتدل (Mild Pyrolysis)** ضمن النطاق الحراري الذي يهدف لتعظيم إنتاج الوقود الصلب.
"""

    # --- 2. Kinetic Model Explanation (New Advanced Answer) ---
    if "kinetics" in prompt_lower or "حركية" in prompt_lower or "مكونات" in prompt_lower:
        
        return f"""
        ## 🧪 الحركية الكيميائية المتقدمة (Advanced Kinetics)

        هذه المحاكاة تستخدم نموذج **تفاعلات متوازية من الدرجة الأولى (Parallel First-Order Reactions)**، حيث تتم معالجة المكونات الرئيسية الثلاثة للكتلة الحيوية بشكل منفصل:
        * **الهيميسليلوز (Hemicellulose):** يتفاعل بسرعة أكبر (أقل طاقة تنشيط $E_a$) ويفقد جزءاً كبيراً من كتلته أولاً.
        * **السليلوز (Cellulose):** يتفاعل بمعدل متوسط.
        * **الليجنين (Lignin):** أبطأ التفاعلات (أكثر استقراراً حرارياً) و يساهم بأكبر قدر في **الفحم الثابت (Fixed Carbon)** النهائي.

        **معدل التفكك المتوسط الفعال (Avg. Devol Rate) عند هذه الشروط هو:** {results['avg_devol_rate']:.4f} $\\text{min}^{-1}$
        """

    # --- 3. HHV / Energy Yield / Carbon Efficiency (Expanded) ---
    if "hhv" in prompt_lower or "heating value" in prompt_lower or "حرارية" in prompt_lower or "طاقة" in prompt_lower or "كربون" in prompt_lower:
        
        energy_gain = results['biochar_hhv'] - results['initial_hhv']
        
        return f"""
        ## ⚡ مقاييس الأداء الحراري والكربوني

        1.  **قيمة التسخين العليا (HHV):**
            * **HHV للكتلة الأولية (جافة):** **{results['initial_hhv']:.2f} $\\text{MJ/kg}$**
            * **HHV للفحم الحيوي النهائي:** **{results['biochar_hhv']:.2f} $\\text{MJ/kg}$**
            * **مكسب الطاقة:** **{energy_gain:.2f} $\\text{MJ/kg}$**. هذا المكسب ناتج عن زيادة تركيز الكربون الثابت بعد إزالة الأجزاء ذات القيمة الحرارية المنخفضة (الأكسجين).

        2.  **كفاءة الطاقة (Energy Yield):**
            * **{results['energy_yield_percent']:.1f}\\%** من الطاقة الكلية الأولية تم الاحتفاظ بها في المنتج الصلب (Biochar).

        3.  **كفاءة الكربون (Carbon Efficiency):**
            * **{results['carbon_efficiency'] * 100:.1f}\\%**. هذا يعني أن هذه النسبة من الكربون الموجود في الكتلة الحيوية الأولية قد تم تثبيتها في الفحم الحيوي النهائي، مما يمثل عامل حاسم في مشاريع **إزالة الكربون وتثبيته**.
        """
        
    # --- 4. Optimization / Cost ---
    if "optimize" in prompt_lower or "increase yield" in prompt_lower or "تكلفة" in prompt_lower or "افضل ظروف" in prompt_lower:
        
        cost_feedstock_total = (p['initial_mass'] / 1000) * st.session_state.cost_biomass_per_ton
        revenue_total = results["yields_mass"].loc["Biochar (Solid Product)", "Mass (kg)"] * st.session_state.price_biochar_per_kg
        net_profit = revenue_total - cost_feedstock_total # Simplified profit for recommendation

        recommendation = ""
        if p['temperature'] > 280 and results["yields_percent"].loc["Biochar (Solid Product)", "Yield (%)"] < 70:
            recommendation = "لزيادة المردود الكتلي، يجب **خفض الحرارة** إلى 240-260 درجة مئوية وتقصير المدة لتقليل فقدان السليلوز والليجنين."
        elif results['final_ash_percent'] > 10:
             recommendation = "لتحسين جودة المنتج، يجب البحث عن مادة خام ذات **نسبة رماد أولية أقل**، حيث أن الرماد يتركز أثناء العملية."
        else:
             recommendation = "الظروف الحالية تبدو جيدة. لزيادة الربحية، ركز على خفض التكاليف التشغيلية (الغازات/الطاقة)."

        return f"""
        ## 📈 تحليل الربحية والتحسين

        * **الإيرادات المتوقعة (من Biochar فقط):** ${revenue_total:.2f}
        * **تكلفة المواد الخام:** ${cost_feedstock_total:.2f}
        * **الربح الأولي (Pre-Op Profit):** ${net_profit:.2f}

        **توصية التحسين الفوري:** {recommendation}

        **ملحوظة:** يجب النظر في قيمة الطاقة الكامنة في الزيت الحيوي ({results['mass_bio_oil_kg']:.2f} kg) والغازات القابلة للاحتراق ({results['mass_non_condensable_gas_kg']:.2f} kg) لتعظيم العائد الاقتصادي.
        """
        
    # --- 5. Ash / Quality ---
    if "ash" in prompt_lower or "رماد" in prompt_lower or "جودة" in prompt_lower:
        
        initial_ash_percentage = BIOMASS_COMPOSITION[p['biomass']]["Ash"] * 100
        enrichment_factor = results['final_ash_percent'] / initial_ash_percentage if initial_ash_percentage > 0 else 0

        return f"""
        ## 💎 تحليل الرماد وجودة المنتج

        تركيز الرماد يزيد لأن الرماد **عنصر خامل (Inert)** لا يتفاعل ولا يتبخر. عندما تفقد الكتلة الحيوية الماء والمركبات العضوية، يتركز الرماد المتبقي في الكتلة النهائية.

        * **الرماد الأولي (Initial):** {initial_ash_percentage:.2f}\\%
        * **الرماد النهائي (Final):** **{results['final_ash_percent']:.2f}\\%**
        * **عامل التركيز (Enrichment Factor):** {enrichment_factor:.2f} مرات.

        لتحسين الجودة، يجب فحص ومعالجة المادة الخام الأولية لتقليل محتوى الرماد، خاصةً في النفايات الزراعية والبلدية.
        """

    # --- 6. Reactor / Operation ---
    if "reactor" in prompt_lower or "مفاعل" in prompt_lower or "تشغيل" in prompt_lower:
        
        reactor_details = {
            "Rotary Drum Reactor": "مناسب للإنتاج المستمر على نطاق واسع، ويوفر خلطاً جيداً وتجانساً حرارياً. يتطلب صيانة للختم (Seals).",
            "Fluidized Bed Reactor": "يوفر أفضل نقل حرارة وأسرع تفاعلات، مثالي للأحجام الدقيقة. تكلفته التشغيلية والطاقية عالية.",
            "Auger/Screw Reactor": "بسيط في التصميم، جيد لنقل المواد اللزجة. نقل الحرارة فيه متوسط وقد يعاني من تدرجات حرارية.",
            "Fixed Bed Reactor": "الأبسط والأرخص، ولكنه يعاني من أسوأ تجانس حراري وتدرجات حرارة كبيرة داخل الفرن، مما يقلل من جودة المنتج النهائي."
        }
        
        return f"""
        ## ⚙️ نوع المفاعل وظروف التشغيل

        أنت تحاكي باستخدام **{p['reactor']}**. {reactor_details.get(p['reactor'])}

        **نصيحة التشغيل:** التحكم الدقيق في درجة الحرارة ($T$) ومدة التفاعل ($t$) هو مفتاح جودة المنتج. أي زيادة في $T$ أو $t$ فوق نقطة التحول تزيد من فقدان الكتلة بشكل كبير وتؤدي إلى منتج أكثر تفحماً وأقل مردوداً.
        """

    # --- Default/General response ---
    return "أنا مساعد Chemisco للذكاء الاصطناعي. يمكنني تحليل البيانات، شرح نموذج الحركية الكيميائية المتقدم، أو مقارنة العمليات. جرب أن تسأل عن **كفاءة الكربون (Carbon Efficiency)** أو **الحركية (Kinetics)**."

# --- 6. Main Streamlit App ---
def main():
    st.set_page_config(page_title="Chemisco Torrefaction Simulator", layout="wide", initial_sidebar_state="expanded")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    
    # Initialize chat history and cost variables
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Welcome! I am the Chemisco AI Assistant. How can I help you analyze your torrefaction simulation?"}]
    if 'target_yield' not in st.session_state:
        st.session_state['target_yield'] = 75
        st.session_state['target_ash'] = 8.0
        st.session_state['has_won'] = False
        # Initialize costs in session state
        st.session_state['cost_biomass_per_ton'] = 30.0
        st.session_state['cost_energy_per_hour'] = 5.0
        st.session_state['price_biochar_per_kg'] = 1.20

    # --- Sidebar ---
    with st.sidebar:
        # Header and Logo
        if LOGO_BASE64_STRING:
            st.markdown(f"""
                <div class="sidebar-logo-container">
                    <img src="data:image/png;base64,{LOGO_BASE64_STRING}" style="width: 80%; display: block; margin: 0 auto;">
                </div>
            """, unsafe_allow_html=True)

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
            biomass_type = st.selectbox("Biomass Type", list(BIOMASS_COMPOSITION.keys()))
            moisture_content = st.slider("Initial Moisture Content (%)", 0.0, 50.0, 10.0, step=1.0)
            particle_size = st.selectbox("Particle Size", list(SIZE_FACTOR.keys()))
            
        with st.expander("🌡️ Process Conditions", expanded=True):
            temperature = st.slider("Torrefaction Temperature (°C)", 200, 350, 275, step=5)
            duration = st.slider("Process Duration (min)", 10, 120, 45, step=5)
            ash_percent_init = BIOMASS_COMPOSITION[biomass_type]["Ash"] * 100
            st.info(f"Initial Ash Content: **{ash_percent_init:.1f}%**")
            
        with st.expander("💰 Cost Management", expanded=False):
            st.caption("Economic Feasibility Parameters")
            # Store costs in session state
            st.session_state.cost_biomass_per_ton = st.number_input("Biomass Feedstock Cost ($/ton)", min_value=0.0, value=st.session_state.cost_biomass_per_ton, step=5.0)
            st.session_state.cost_energy_per_hour = st.number_input("Operational/Energy Cost ($/hour)", min_value=0.0, value=st.session_state.cost_energy_per_hour, step=0.5)
            st.session_state.price_biochar_per_kg = st.number_input("Biochar Selling Price ($/kg)", min_value=0.0, value=st.session_state.price_biochar_per_kg, step=0.1)
            
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
        <div class="bfd-block" style="border-color: #D32F2F; background-color: #FFF3E0; color: #B71C1C;">
            {reactor_type.upper()}
            <p style="color: #B71C1C;">Temp: {temperature} °C</p>
            <p style="color: #B71C1C;">Duration: {duration} min</p>
            <div class="side-stream" style="background-color: #FFD700;"></div>
            <div class="side-stream-label" style="color: #FFD700;">Volatiles (Gas & Oil)</div>
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
    if moisture_content / 100 + BIOMASS_COMPOSITION[biomass_type]["Ash"] > 1:
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
        
        # Game targets initialization (randomized on first run or New Order)
        if st.button("🔄 New Client Order", key="new_order_btn"):
            st.session_state.target_yield = random.randint(60, 85)
            st.session_state.target_ash = round(random.uniform(ash_percent_init + 1.0, ash_percent_init + 5.0), 1)
            st.session_state.has_won = False
            st.experimental_rerun()

        col_g1, col_g2, col_g3 = st.columns([1.5, 2, 1])
        
        with col_g1:
            st.info(f"📋 **CLIENT ORDER:**\n\n🎯 Target Yield: **{st.session_state.target_yield}%**\n\n🎯 Max Ash: **{st.session_state.target_ash}%**")
            
        with col_g2:
            curr_yield = results["yields_percent"].loc["Biochar (Solid Product)", "Yield (%)"]
            curr_ash = results["final_ash_percent"]
            diff_yield = abs(curr_yield - st.session_state.target_yield)
            diff_ash = abs(curr_ash - st.session_state.target_ash)
            
            # Scoring: 10 points deduction per 1% yield difference, 20 points deduction per 1% ash difference
            score = max(0, 100 - (diff_yield * 10 + diff_ash * 20))
            st.metric("🏆 Your Efficiency Score", f"{score:.1f} / 100")
            
            if score >= 90:
                st.success("🎉 **PERFECT MATCH! Order fulfilled successfully.**")
                if not st.session_state.has_won:
                    st.session_state.has_won = True
                    st.balloons() # CELEBRATION!
            elif score >= 70:
                st.warning("⚠️ Acceptable, but try to optimize further.")
            else:
                st.error("❌ Specification mismatch. Quality too low.")
        
        # Empty column for spacing
        with col_g3:
             st.markdown("###")

        st.markdown("---")
    # --------------------------

    # --- Display Results ---
    st.header("📊 Simulation Results & Analysis")
    
    tab1, tab2, tab3, tab4, tab5, tab_ai = st.tabs(["Yields & Products", "Kinetics & Ash", "Energy & Sustainability", "💰 Cost Analysis", "PDF Report", "🤖 AI Assistant"])
    
    with tab1:
        st.subheader(f"Product Yields & Mass Balance")
        
        # Enhanced Metrics Display
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        biochar_mass = results["yields_mass"].loc["Biochar (Solid Product)", "Mass (kg)"]
        col_m1.metric("⚖️ Final Biochar Mass", f"{biochar_mass:.2f} kg", delta=f"{results['yields_percent'].loc['Biochar (Solid Product)', 'Yield (%)']:.1f}% Mass Yield")
        
        bio_oil_mass = results["yields_mass"].loc["Bio-Oil (Condensable)", "Mass (kg)"]
        col_m2.metric("💧 Bio-Oil Produced", f"{bio_oil_mass:.2f} kg", delta=f"{results['yields_percent'].loc['Bio-Oil (Condensable)', 'Yield (%)']:.1f}% Oil Yield")
        
        gas_total = results["yields_mass"].loc["Non-Condensable Gases", "Mass (kg)"]
        col_m3.metric("💨 Non-Condensable Gas", f"{gas_total:.2f} kg", delta="Potential Heat Source")
        
        moisture_loss = results["yields_mass"].loc["Moisture Loss (Water Vapor)", "Mass (kg)"]
        col_m4.metric("💦 Moisture Removed", f"{moisture_loss:.2f} kg", delta=f"{results['yields_percent'].loc['Moisture Loss (Water Vapor)', 'Yield (%)']:.1f}% Loss")


        st.markdown("---")
        
        col_t1, col_t2 = st.columns(2)
        
        # --- PLOTLY PIE CHARTS ---
        with col_t1:
            st.markdown("##### Final Biochar Composition")
            st.caption("Mass Breakdown of the Solid Product (Fixed Carbon + Volatiles + Ash)")
            
            df_solid = results["solid_composition"].reset_index()
            df_solid.columns = ["Component", "Mass (kg)"]
            
            fig1 = px.pie(df_solid, values='Mass (kg)', names='Component', hole=0.5,
                            color='Component',
                            color_discrete_map={
                                "Fixed Carbon": "#6A1B9A", 
                                "Volatile Matter Remaining": "#AB47BC", 
                                "Ash": "#BDBDBD" 
                            })
            
            fig1.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#000000', width=1)))
            st.plotly_chart(fig1, use_container_width=True)

        with col_t2:
            st.markdown("##### Overall Product Yields")
            st.caption("Distribution of Initial Mass into Final Products")
            
            filtered_yields = results["yields_percent"].iloc[[0, 1, 2, 3]].reset_index()
            filtered_yields.columns = ["Component", "Yield (%)"]
            
            fig2 = px.pie(filtered_yields, values='Yield (%)', names='Component', hole=0.5,
                            color='Component',
                            color_discrete_map={
                                "Biochar (Solid Product)": "#1D7948", 
                                "Bio-Oil (Condensable)": "#2EAF6C",
                                "Non-Condensable Gases": "#FFD700",
                                "Moisture Loss (Water Vapor)": "#9CCC65" 
                            })
            
            fig2.update_traces(textposition='inside', textinfo='percent', marker=dict(line=dict(color='#000000', width=1)))
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Dynamic Ash Enrichment and Multi-Component Kinetics")
        
        col_k1, col_k2 = st.columns(2)
        
        with col_k1:
            # Dual Axis Chart for Mass/Ash
            fig_dual = go.Figure()
            fig_dual.add_trace(go.Scatter(
                x=results["mass_profile"].index, y=results["mass_profile"]["Total Mass Yield (%)"],
                name="Total Mass %", line=dict(color="#1D7948", width=3), yaxis="y1"
            ))
            final_ash = results["final_ash_percent"]
            ash_increase = final_ash - ash_percent_init
            st.metric("⚗️ Final Ash Concentration", f"{final_ash:.2f} %", delta=f"+{ash_increase:.2f}% (Enrichment)")
            
            fig_dual.add_trace(go.Scatter(
                x=results["mass_profile"].index, y=results["mass_profile"]["Ash Concentration in Solid (%)"],
                name="Ash Concentration %", line=dict(color="#D32F2F", width=3, dash='dot'), yaxis="y2"
            ))

            fig_dual.update_layout(title="Ash Enrichment vs. Mass Depletion", height=400,
                yaxis=dict(title=dict(text="Total Mass Remaining (%)", font=dict(color="#1D7948")), tickfont=dict(color="#1D7948")),
                yaxis2=dict(title=dict(text="Ash Concentration (%)", font=dict(color="#D32F2F")), tickfont=dict(color="#D32F2F"), overlaying="y", side="right"),
                legend=dict(x=0.1, y=1.1, orientation="h"), hovermode="x unified",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_dual, use_container_width=True)

        with col_k2:
            st.markdown("##### Multi-Component Kinetic Rates")
            st.caption("How fast each main component (DAF) is breaking down at the selected conditions.")

            kinetics_data = {
                "Hemicellulose": KINETICS["Hemicellulose"][0] * np.exp(-KINETICS["Hemicellulose"][1] / (results['parameters']['temperature'] + 273.15) * R_GAS) * SIZE_FACTOR.get(particle_size) * 1000, # A * exp(-Ea/RT) * size_factor * 1000 (for better visual scale)
                "Cellulose": KINETICS["Cellulose"][0] * np.exp(-KINETICS["Cellulose"][1] / (results['parameters']['temperature'] + 273.15) * R_GAS) * SIZE_FACTOR.get(particle_size) * 1000,
                "Lignin": KINETICS["Lignin"][0] * np.exp(-KINETICS["Lignin"][1] / (results['parameters']['temperature'] + 273.15) * R_GAS) * SIZE_FACTOR.get(particle_size) * 1000,
            }
            
            df_kinetics = pd.DataFrame(kinetics_data, index=["Rate Factor (a.u.)"]).T
            
            fig_rates = px.bar(df_kinetics, y='Rate Factor (a.u.)', 
                               color=df_kinetics.index, 
                               color_discrete_map={"Hemicellulose": "#1565C0", "Cellulose": "#42A5F5", "Lignin": "#64B5F6"})
            
            fig_rates.update_layout(height=400, title="Devolatilization Rate Factors (Scaled)")
            fig_rates.update_traces(showlegend=False)
            st.plotly_chart(fig_rates, use_container_width=True)
            
            st.info(f"Rate Factor for Hemicellulose is often the highest, confirming its rapid breakdown in the $200^\circ C$ - $300^\circ C$ range.")


    with tab3:
        st.subheader("Energy and Environmental Sustainability Metrics")
        
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        
        col_e1.metric("🔥 Biochar HHV", f"{results['biochar_hhv']:.2f} MJ/kg", delta=f"+{results['biochar_hhv'] - results['initial_hhv']:.2f} MJ/kg")
        col_e2.metric("⚡ Energy Yield", f"{results['energy_yield_percent']:.1f} %", delta="Energy Retained in Biochar")
        col_e3.metric("♻️ Carbon Efficiency", f"{results['carbon_efficiency'] * 100:.1f} %", delta="Carbon Retained in Solid")
        
        # New Metric: Fuel Ratio (Fixed Carbon / Volatile Matter)
        fixed_carbon_kg = results["solid_composition"].loc["Fixed Carbon", "Mass (kg)"]
        volatile_matter_remaining_kg = results["solid_composition"].loc["Volatile Matter Remaining", "Mass (kg)"]
        fuel_ratio = fixed_carbon_kg / volatile_matter_remaining_kg if volatile_matter_remaining_kg > 0 else 999
        col_e4.metric("📈 Fuel Ratio (FC/VM)", f"{fuel_ratio:.2f}", delta="Higher is better for combustion stability")
        
        st.markdown("---")
        st.subheader("Gas Composition (Potential Energy Source)")
        st.bar_chart(results["gas_composition_molar"])

    with tab4:
        st.subheader("💰 Economic Feasibility Analysis")
        
        # Get costs from session state
        cost_biomass_per_ton = st.session_state.cost_biomass_per_ton
        cost_energy_per_hour = st.session_state.cost_energy_per_hour
        price_biochar_per_kg = st.session_state.price_biochar_per_kg
        
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
            x = ["Feedstock Cost", "Operational Cost", "Revenue (Biochar)", "Net Profit"],
            textposition = "outside",
            y = [-cost_feedstock_total, -cost_operations_total, revenue_total, net_profit],
            connector = {"line": {"color": "rgb(63, 63, 63)"}},
            decreasing = {"marker":{"color": "#D32F2F"}}, 
            increasing = {"marker":{"color": "#1D7948"}}, 
            totals = {"marker":{"color": "#FFD700", "line":{"width":1, "color":"#FFD700"}}}
        ))

        fig_waterfall.update_layout(title = "Revenue vs. Cost Breakdown", showlegend = False, height=450)
        st.plotly_chart(fig_waterfall, use_container_width=True)

    with tab5:
        st.subheader("📥 Generate Report")
        st.info("Create a comprehensive PDF document of the simulation results, parameters, and visualizations.")
        
        pdf_buffer = generate_pdf_report(results)
        
        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_buffer,
            file_name=f"Chemisco_Adv_Report_{biomass_type}_{temperature}C.pdf",
            mime="application/pdf"
        )
    
    # --- AI ASSISTANT TAB ---
    with tab_ai:
        st.header("🤖 AI Assistant: Torrefaction Expert")
        st.info("اسألني عن التحسين، كفاءة الكربون، أو نموذج الحركية الكيميائية (Kinetics).")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing simulation data..."):
                    time.sleep(1.5) # Longer thinking time for the advanced AI
                    ai_response = mock_ai_response(prompt, results)
                
                st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})


# --- Execution Entry Point ---
if __name__ == "__main__":
    main()

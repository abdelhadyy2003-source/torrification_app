import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
from reportlab.lib import colors
from datetime import datetime
import math
import streamlit.components.v1 as components 

# --- 1. Constants & Defaults ---
R_GAS = 8.314
CP_BIOMASS = 1500.0
CP_WATER = 4180.0
H_VAPOR = 2260000.0
HHV_DRY_INITIAL_DEFAULT = 18.0

# --- 2. Styles (Premium Enterprise Theme) ---
GLOBAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    /* 1. Global Reset & Main Body */
    .stApp { 
        background-color: #F4F7F6; /* Very light grey-green tint for eye comfort */
        font-family: 'Inter', sans-serif; 
    }
    
    /* 2. Professional Sidebar Styling */
    section[data-testid="stSidebar"] { 
        /* Deep Gradient for a premium look */
        background: linear-gradient(180deg, #051e18 0%, #0d332d 100%);
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Sidebar Text & Inputs */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 { 
        color: #FFFFFF !important; 
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    /* Input Labels in Sidebar (Micro-typography) */
    section[data-testid="stSidebar"] label {
        color: #A7C4BC !important; /* Soft Teal Grey */
        font-size: 12px !important;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 1px;
    }
    
    /* Styling Streamlit Inputs within Sidebar to blend in */
    section[data-testid="stSidebar"] .stNumberInput, 
    section[data-testid="stSidebar"] .stSelectbox, 
    section[data-testid="stSidebar"] .stSlider {
        color: white;
    }
    
    /* 3. Main Content Typography */
    h1, h2, h3 { 
        color: #0d332d !important; 
        font-family: 'Inter', sans-serif; 
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    p, div, span, li { color: #2C3E50; }

    /* 4. Metric Cards (KPIs) - Minimalist & Clean */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #EAEAEA;
        border-radius: 8px; 
        padding: 15px 20px; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        position: relative;
        overflow: hidden;
    }
    /* Add a colored accent strip to the top of metrics */
    div[data-testid="stMetric"]::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: #26A69A;
    }
    div[data-testid="stMetricValue"] { 
        color: #0d332d !important; 
        font-weight: 700; 
        font-size: 28px !important;
    }
    div[data-testid="stMetricLabel"] { 
        color: #546E7A !important; 
        font-size: 13px; 
        font-weight: 500;
    }

    /* 5. Buttons - High End Action */
    .stButton > button { 
        background: linear-gradient(135deg, #26A69A 0%, #00897B 100%) !important;
        color: white !important; 
        border: none; 
        font-weight: 600; 
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 4px 6px rgba(38, 166, 154, 0.2);
        transition: transform 0.1s, box-shadow 0.1s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 8px rgba(38, 166, 154, 0.3);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* 6. Process Flow Blocks */
    .bfd-block {
        padding: 18px; 
        border-radius: 8px; 
        text-align: center; 
        background: #FFFFFF;
        border: 1px solid #E0E0E0; 
        color: #0d332d; 
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }
    .bfd-stream { 
        color: #26A69A; 
        font-size: 24px; 
        padding-top: 10px; 
        font-weight: bold; 
    }

    /* 7. Sidebar Header Box (Logo Area) */
    .header-box {
        background: rgba(255, 255, 255, 0.05); 
        padding: 20px; 
        border-radius: 8px; 
        text-align: center; 
        margin-bottom: 30px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .header-box h1 { 
        color: #FFFFFF !important; 
        margin: 0; 
        font-size: 24px; 
        font-weight: 900; 
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .header-box p { 
        color: #4DB6AC !important; 
        margin: 0; 
        font-size: 10px; 
        margin-top: 6px; 
        letter-spacing: 3px; 
        text-transform: uppercase; 
        opacity: 0.9;
    }

    /* 8. Tabs Customization */
    div[data-testid="stTabs"] button { 
        color: #78909C; 
        font-weight: 500; 
        font-size: 14px;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] { 
        color: #0d332d !important; 
        border-bottom: 2px solid #26A69A !important; 
        font-weight: 700;
    }
    
    /* 9. Expander styling in Sidebar */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        background-color: transparent;
        color: #ECEFF1 !important; /* Off-white for headers */
        font-size: 14px;
    }
    section[data-testid="stSidebar"] .streamlit-expanderContent {
        background-color: rgba(0,0,0,0.2); /* Slightly darker bg for inputs */
        border-radius: 4px;
        padding: 10px;
        margin-bottom: 10px;
    }

    #MainMenu, footer, .stDeployButton {visibility: hidden;}
</style>
"""

# --- 3. Mathematical Models (UNCHANGED) ---
def moisture_evap_linear(initial_moisture_kg, T_C, t_min, k_f=0.02):
    if T_C <= 100: return 0.0
    evap_kg = k_f * (T_C - 100) * t_min * initial_moisture_kg
    return min(initial_moisture_kg, max(0.0, evap_kg))

def Y_solid_empirical(T_C, t_min, a=0.35, b=0.004):
    severity = max(0.0, T_C - 200) * t_min
    return 1.0 - a * (1.0 - math.exp(-b * severity))

def m_oil(dry_mass_kg, T_C, t_min, C_oil=0.25):
    k_oil = 0.0008 * max(0.0, T_C - 200)
    return dry_mass_kg * C_oil * (1.0 - math.exp(-k_oil * t_min))

def m_gas(dry_mass_kg, T_C, t_min, C_gas=0.20):
    k_gas = 0.0015 * max(0.0, T_C - 180)
    return dry_mass_kg * C_gas * (1.0 - math.exp(-k_gas * t_min))

def hhv_improved_model(Y_solid, temp_c, enhancement_factor=0.85):
    mass_loss_fraction = 1.0 - Y_solid
    base_increase = mass_loss_fraction * enhancement_factor
    temp_bonus = 0.0
    if temp_c > 280: temp_bonus = 0.02 * ((temp_c - 280) / 50.0)
    return base_increase + temp_bonus

def run_simulation(mass_in, moisture_pct, ash_pct_dry, temp_c, time_min, params):
    moisture_frac = moisture_pct / 100.0
    M0_water = mass_in * moisture_frac
    M0_dry = mass_in * (1.0 - moisture_frac)
    M_ash = M0_dry * (ash_pct_dry / 100.0)
    
    w_evap = moisture_evap_linear(M0_water, temp_c, time_min, k_f=params['k_f'])
    oil_kg = m_oil(M0_dry, temp_c, time_min, C_oil=params['C_oil'])
    gas_kg = m_gas(M0_dry, temp_c, time_min, C_gas=params['C_gas'])
    char_dry = max(0, M0_dry - oil_kg - gas_kg) 
    char_total_mass = char_dry + (M0_water - w_evap)
    
    y_solid_val = Y_solid_empirical(temp_c, time_min, a=params['a_solid'], b=params['b_solid'])
    hhv_inc_frac = hhv_improved_model(y_solid_val, temp_c, enhancement_factor=params.get('energy_factor', 0.85))
    hhv_final = HHV_DRY_INITIAL_DEFAULT * (1.0 + hhv_inc_frac)
    
    energy_in = M0_dry * HHV_DRY_INITIAL_DEFAULT
    energy_out = char_dry * hhv_final
    
    T_K = temp_c + 273.15
    Q_total_kJ = ((M0_dry * CP_BIOMASS * (T_K - 298.15)) + (M0_water * CP_WATER * (373.15 - 298.15)) + (w_evap * H_VAPOR)) / 1000
    
    return {
        "mass_in": mass_in, "char_kg": char_total_mass, "water_evap_kg": w_evap,
        "oil_kg": oil_kg, "gas_kg": gas_kg, "ash_kg": M_ash,
        "hhv_final": hhv_final, "mass_yield_pct": (char_total_mass / mass_in) * 100,
        "energy_yield_pct": (energy_out / energy_in) * 100 if energy_in > 0 else 0,
        "hhv_increase_pct": hhv_inc_frac * 100, "Q_total_kJ": Q_total_kJ
    }

def get_time_series(mass_in, moisture_pct, ash_pct_dry, temp_c, time_min, params):
    times = np.linspace(0, time_min, 50)
    data = []
    for t in times:
        res = run_simulation(mass_in, moisture_pct, ash_pct_dry, temp_c, t, params)
        data.append({
            "Time (min)": t, "Char (kg)": res['char_kg'], "Bio-Oil (kg)": res['oil_kg'],
            "Gases (kg)": res['gas_kg'], "Water Vapor (kg)": res['water_evap_kg'],
            "HHV Increase (%)": res['hhv_increase_pct']
        })
    return pd.DataFrame(data)

# --- 4. Professional PDF Generator ---
def create_pdf(res, profit, fig1, fig2):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Matching PDF colors to the dark green/teal theme
    CHEMISCO_PRIMARY = colors.HexColor('#0d332d') # Dark Green
    CHEMISCO_ACCENT = colors.HexColor('#26A69A')  # Teal
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Header with Logo
    logo_style = ParagraphStyle(name='LogoText', fontName='Helvetica-Bold', fontSize=18, textColor=colors.white, alignment=1)
    sub_logo_style = ParagraphStyle(name='SubLogo', fontName='Helvetica', fontSize=8, textColor=colors.whitesmoke, alignment=1)
    
    logo_content = [[Paragraph("CHEMISCO", logo_style)], [Paragraph("Torrefaction Simulator", sub_logo_style)]]
    t_logo = Table(logo_content, colWidths=[2.5*inch])
    t_logo.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), CHEMISCO_PRIMARY),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,-1), 1, colors.white), ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))

    info_text = Paragraph(f"<b>Date:</b> {current_time}<br/><b>Status:</b> Success", styles['Normal'])
    header_layout = [[t_logo, info_text]]
    t_header_main = Table(header_layout, colWidths=[3*inch, 3*inch])
    t_header_main.setStyle(TableStyle([('ALIGN', (0,0), (0,0), 'LEFT'), ('ALIGN', (1,0), (1,0), 'RIGHT'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    
    story.append(t_header_main)
    story.append(Spacer(1, 25))
    
    # Title & Metrics
    story.append(Paragraph("Technical Engineering Report", ParagraphStyle(name='Title', parent=styles['Heading2'], textColor=CHEMISCO_PRIMARY, fontSize=16)))
    story.append(Spacer(1, 10))
    story.append(Paragraph("This document summarizes the simulation results for the biomass torrefaction process.", styles['Normal']))
    story.append(Spacer(1, 20))

    data = [
        ["Metric", "Value"], 
        ["Mass Yield", f"{res['mass_yield_pct']:.1f} %"], 
        ["Energy Density (HHV)", f"{res['hhv_final']:.2f} MJ/kg"],
        ["Energy Yield", f"{res['energy_yield_pct']:.1f} %"],
        ["Bio-Oil Produced", f"{res['oil_kg']:.2f} kg"],
        ["Profit Estimate", f"${profit:.2f}"]
    ]
    t = Table(data, colWidths=[3.5*inch, 2.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), CHEMISCO_PRIMARY), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12), ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 1, colors.grey)
    ]))
    story.append(t); story.append(Spacer(1, 30))

    # Charts
    def add_plot(fig, title):
        try:
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", font=dict(color="black"))
            img_bytes = fig.to_image(format="png", width=800, height=450, scale=2)
            story.append(Paragraph(f"<b>{title}</b>", styles['Heading3']))
            story.append(Image(BytesIO(img_bytes), width=6*inch, height=3.5*inch))
            story.append(Spacer(1, 20))
        except Exception:
            story.append(Paragraph(f"<font color=red>Error rendering chart: {title}. Ensure 'kaleido==0.2.1' is installed.</font>", styles['Normal']))

    add_plot(fig1, "Figure 1: Mass Balance Distribution")
    add_plot(fig2, "Figure 2: Solid Composition")
    
    story.append(Spacer(1, 30))
    story.append(Paragraph("<font color=grey size=8>Chemisco Simulator v3.4 | Confidential & Proprietary</font>", styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- 5. Main Streamlit App ---
def main():
    st.set_page_config(page_title="Chemisco Pro", layout="wide", initial_sidebar_state="expanded")
    
    # *** 🚀 INJECT BOTPRESS ***
    js_code = """
    <script>
        if (!window.parent.document.getElementById('botpress-inject')) {
            var script1 = window.parent.document.createElement('script');
            script1.id = 'botpress-inject';
            script1.src = 'https://cdn.botpress.cloud/webchat/v3.4/inject.js';
            window.parent.document.head.appendChild(script1);
            
            script1.onload = function() {
                var script2 = window.parent.document.createElement('script');
                script2.src = 'https://files.bpcontent.cloud/2025/11/28/23/20251128230307-F5JAD1ML.js';
                script2.defer = true;
                window.parent.document.body.appendChild(script2);
            };
        }
    </script>
    """
    components.html(js_code, height=0, width=0)
    # **************************

    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    
    if 'cost_biomass' not in st.session_state: 
        st.session_state.update({'cost_biomass': 30.0, 'cost_energy': 0.15, 'price_char': 1.20})

    # Sidebar with Professional Dark Theme
    with st.sidebar:
        st.markdown("""
            <div class="header-box">
                <h1>CHEMISCO</h1>
                <p>Torrefaction Systems</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.header("Control Panel")
        reactor = st.selectbox("Reactor Configuration", ["Rotary Drum", "Fluidized Bed", "Screw Reactor"])
        
        # Using Expanders for organization, styled via CSS
        with st.expander("Parameters: Feedstock", expanded=True):
            mass = st.number_input("Input Mass (kg)", 1.0, 10000.0, 100.0, 10.0)
            moisture = st.slider("Moisture Content (%)", 0.0, 60.0, 15.0)
            ash = st.slider("Ash Content (Dry %)", 0.0, 30.0, 5.0)
            
        with st.expander("Parameters: Reactor Process", expanded=True):
            temp = st.slider("Temperature (°C)", 150, 350, 275)
            time_min = st.slider("Residence Time (min)", 10, 120, 30)

        with st.expander("Advanced: Kinetic Model", expanded=False):
            p_kf = st.number_input("Drying rate (k_f)", 0.0, 0.1, 0.02, format="%.4f")
            p_Coil = st.number_input("Max Oil Fraction", 0.0, 0.5, 0.25)
            p_Cgas = st.number_input("Max Gas Fraction", 0.0, 0.5, 0.20)
            p_a = st.number_input("Solid Yield Coeff (a)", 0.1, 0.5, 0.35)
            p_b = st.number_input("Degradation Coeff (b)", 0.001, 0.01, 0.004, format="%.4f")
            st.markdown("---")
            p_enh = st.slider("HHV Enhancement Factor", 0.2, 1.5, 0.85)
            
        params = {"k_f": p_kf, "C_oil": p_Coil, "C_gas": p_Cgas, "a_solid": p_a, "b_solid": p_b, "energy_factor": p_enh}

        with st.expander("Settings: Economics", expanded=False):
            st.session_state.cost_biomass = st.number_input("Feedstock Cost ($/ton)", value=st.session_state.cost_biomass)
            st.session_state.cost_energy = st.number_input("Energy Cost ($/kWh)", value=st.session_state.cost_energy)
            st.session_state.price_char = st.number_input("Biochar Price ($/kg)", value=st.session_state.price_char)
        
        st.markdown("<br>", unsafe_allow_html=True)
        game_mode = st.checkbox("Enable Optimization Challenge")

    # Calculations
    res = run_simulation(mass, moisture, ash, temp, time_min, params)
    cost_feed = (mass / 1000) * st.session_state.cost_biomass
    energy_kwh = res['Q_total_kJ'] / 3600.0
    cost_ops = energy_kwh * st.session_state.cost_energy
    revenue = res['char_kg'] * st.session_state.price_char
    profit = revenue - (cost_feed + cost_ops)

    # Visualization Theme Settings
    APP_TXT_COLOR = "#0d332d"
    APP_BG_COLOR = "#F4F7F6"
    # Professional Palette: Deep Green, Teal, Slate Blue, Muted Red
    colors_seq = ["#0d332d", "#26A69A", "#5c6bc0", "#ef5350"]

    # 1. Pie Chart
    df_pie = pd.DataFrame({
        "Component": ["Biochar", "Water Vapor", "Bio-Oil", "Gases"],
        "Mass (kg)": [res['char_kg'], res['water_evap_kg'], res['oil_kg'], res['gas_kg']]
    })
    fig1 = px.pie(df_pie, values='Mass (kg)', names='Component', hole=0.7, color_discrete_sequence=colors_seq)
    fig1.update_layout(
        title=dict(text="Mass Balance Distribution", font=dict(size=18, color=APP_TXT_COLOR)),
        paper_bgcolor=APP_BG_COLOR, plot_bgcolor=APP_BG_COLOR, 
        font=dict(color=APP_TXT_COLOR, family="Inter"),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    # Add text in the center
    fig1.add_annotation(text=f"{res['char_kg']:.1f} kg<br>Solid", x=0.5, y=0.5, font_size=16, showarrow=False, font_color=APP_TXT_COLOR)

    # 2. Bar Chart
    organic_char = res['char_kg'] - res['ash_kg']
    df_bar = pd.DataFrame({
        "Type": ["Organic Carbon", "Ash Content"],
        "Mass (kg)": [organic_char, res['ash_kg']]
    })
    fig2 = px.bar(df_bar, x='Type', y='Mass (kg)', color='Type', color_discrete_sequence=['#0d332d', '#90A4AE'])
    fig2.update_layout(
        title=dict(text="Solid Product Composition", font=dict(size=18, color=APP_TXT_COLOR)),
        paper_bgcolor=APP_BG_COLOR, plot_bgcolor=APP_BG_COLOR, font=dict(color=APP_TXT_COLOR, family="Inter"),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#E0E0E0'),
        showlegend=False
    )

    # Dashboard Layout
    st.title("Process Overview")
    st.markdown("Real-time simulation results based on current reactor parameters.")
    st.markdown("---")
    
    # Process Flow Visualization
    c1, c2, c3, c4, c5 = st.columns([1.5, 0.5, 1.5, 0.5, 1.5])
    with c1: st.markdown(f'<div class="bfd-block">FEEDSTOCK<br><span style="color:#78909C; font-weight:normal; font-size:0.9em">{mass} kg<br>{moisture}% H2O</span></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="bfd-stream" style="text-align:center;">➜</div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="bfd-block">REACTOR<br><span style="color:#78909C; font-weight:normal; font-size:0.9em">{temp}°C<br>{time_min} min</span></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="bfd-stream" style="text-align:center;">➜</div>', unsafe_allow_html=True)
    with c5: st.markdown(f'<div class="bfd-block" style="border-color:#26A69A;">PRODUCT<br><span style="color:#26A69A; font-size:1.1em">{res["char_kg"]:.1f} kg</span></div>', unsafe_allow_html=True)
    
    st.markdown("### Key Performance Indicators")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mass Yield", f"{res['mass_yield_pct']:.1f}%", f"{res['char_kg']:.1f} kg")
    k2.metric("HHV (Energy Density)", f"{res['hhv_final']:.2f} MJ/kg", f"+{res['hhv_increase_pct']:.1f}%")
    k3.metric("Bio-Oil Generated", f"{res['oil_kg']:.1f} kg", "By-product")
    k4.metric("Net Profit Estimate", f"${profit:.2f}", f"Cost: ${cost_ops:.2f}")
    st.markdown("---")

    t1, t2, t3, t4 = st.tabs(["Analytics & Charts", "Reaction Kinetics", "Export Report", "Engineer Challenge"])
    
    with t1:
        cc1, cc2 = st.columns(2)
        with cc1: st.plotly_chart(fig1, use_container_width=True)
        with cc2: st.plotly_chart(fig2, use_container_width=True)

    with t2:
        df_time = get_time_series(mass, moisture, ash, temp, time_min, params)
        fig_area = go.Figure()
        # Clean area chart with opacity
        fig_area.add_trace(go.Scatter(x=df_time['Time (min)'], y=df_time['Char (kg)'], stackgroup='one', name='Solid Char', line=dict(width=0, color='#0d332d')))
        fig_area.add_trace(go.Scatter(x=df_time['Time (min)'], y=df_time['Bio-Oil (kg)'], stackgroup='one', name='Bio-Oil', line=dict(width=0, color='#26A69A')))
        fig_area.add_trace(go.Scatter(x=df_time['Time (min)'], y=df_time['Gases (kg)'], stackgroup='one', name='Gases', line=dict(width=0, color='#90A4AE')))
        fig_area.add_trace(go.Scatter(x=df_time['Time (min)'], y=df_time['Water Vapor (kg)'], stackgroup='one', name='Water Vapor', line=dict(width=0, color='#CFD8DC')))
        fig_area.update_layout(
            title="Mass Evolution over Time",
            paper_bgcolor=APP_BG_COLOR, plot_bgcolor=APP_BG_COLOR,
            font=dict(color=APP_TXT_COLOR, family="Inter"),
            xaxis=dict(title="Residence Time (min)", showgrid=False),
            yaxis=dict(title="Mass Component (kg)", showgrid=True, gridcolor='#E0E0E0'),
            hovermode="x unified"
        )
        st.plotly_chart(fig_area, use_container_width=True)

    with t3:
        st.markdown("### 📄 Generate Technical Report")
        st.markdown("Download a comprehensive PDF report including all current simulation parameters and results.")
        try:
            import kaleido
            pdf = create_pdf(res, profit, fig1, fig2)
            st.download_button("Download PDF Report", pdf, f"Chemisco_Report.pdf", "application/pdf")
        except ImportError:
            st.error("⚠️ Library Missing: Please ensure 'kaleido==0.2.1' is in requirements.txt")

    with t4:
        if game_mode:
            TARGET_HHV, MIN_YIELD, TARGET_PROFIT = 22.0, 55.0, 0.0
            st.markdown("### 🎯 Optimization Targets"); 
            
            c_target1, c_target2, c_target3 = st.columns(3)
            with c_target1:
                st.info(f"Target HHV: > {TARGET_HHV} MJ/kg")
            with c_target2:
                st.info(f"Target Yield: > {MIN_YIELD}%")
            with c_target3:
                st.info(f"Target Profit: > $0")
                
            st.markdown("---")
            col_g1, col_g2, col_g3 = st.columns(3)
            delta_hhv = res['hhv_final'] - TARGET_HHV
            col_g1.metric("Current HHV", f"{res['hhv_final']:.2f}", f"{delta_hhv:.2f}", delta_color="normal" if res['hhv_final'] >= TARGET_HHV else "inverse")
            delta_yield = res['mass_yield_pct'] - MIN_YIELD
            col_g2.metric("Current Yield", f"{res['mass_yield_pct']:.1f}%", f"{delta_yield:.1f}%", delta_color="normal" if res['mass_yield_pct'] >= MIN_YIELD else "inverse")
            col_g3.metric("Current Profit", f"${profit:.2f}", "Net", delta_color="normal" if profit > 0 else "inverse")
            
            if res['hhv_final'] >= TARGET_HHV and res['mass_yield_pct'] >= MIN_YIELD and profit > TARGET_PROFIT:
                st.balloons(); st.success("🏆 Excellent Work! Optimization Goals Met.")
            else:
                st.warning("Goals not met. Try adjusting Temperature and Time.")
        else:
            st.info("Activate 'Optimization Challenge' in the sidebar to start.")

if __name__ == "__main__":
    main()

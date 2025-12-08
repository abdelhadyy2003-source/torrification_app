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
import random 
import streamlit.components.v1 as components 

# --- 1. Constants & Defaults ---
R_GAS = 8.314
CP_BIOMASS = 1500.0
CP_WATER = 4180.0
H_VAPOR = 2260000.0
HHV_DRY_INITIAL_DEFAULT = 18.0

# --- 2. Styles ---
GLOBAL_CSS = """
<style>
    /* 1. Main Background - Clean Professional Grey-White */
    .stApp { 
        background-color: #F9FAFA; 
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
    }
    
    /* --- SIDEBAR STYLING (UNCHANGED as requested) --- */
    section[data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #00743c 0%, #005029 100%);
        border-right: 1px solid rgba(255,255,255,0.2);
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { 
        color: #FFFFFF !important; font-weight: 800; text-shadow: 0px 1px 2px rgba(0,0,0,0.1);
    }
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div[data-testid="stTickBar"] > div,
    section[data-testid="stSidebar"] div[data-testid="stThumbValue"],
    section[data-testid="stSidebar"] .stSlider div { 
        color: #FFFFFF !important; 
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        color: #000000 !important; background-color: #ffffff !important; border-radius: 8px;
    }
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.15) !important; color: #FFFFFF !important;
        border-radius: 6px; font-weight: 600;
    }

    /* --- MAIN CONTENT HARMONIZATION --- */
    
    /* Headings - Matching the Sidebar Green */
    h1, h2, h3 { 
        color: #005029 !important; /* Slightly darker than sidebar for contrast on white */
        font-weight: 800; 
        letter-spacing: -0.5px;
    }
    
    /* Metrics Cards - Clean & Professional */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E0E0E0; 
        border-left: 5px solid #00743c; /* Accent Color */
        border-radius: 8px; 
        padding: 15px; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    div[data-testid="stMetricValue"] { color: #00743c !important; font-weight: 800; font-size: 26px !important; }
    div[data-testid="stMetricLabel"] { color: #546E7A !important; font-weight: 600; font-size: 14px; }

    /* Tabs - Professional Look */
    div[data-testid="stTabs"] button { 
        color: #607D8B !important; 
        font-weight: 600; 
        font-size: 15px; 
    }
    div[data-testid="stTabs"] button[aria-selected="true"] { 
        color: #00743c !important; 
        border-bottom: 3px solid #00743c !important; 
        font-weight: 800;
    }

    /* Flow Chart Blocks - Harmonized */
    .bfd-block {
        padding: 15px; 
        border-radius: 8px; 
        text-align: center; 
        background: #FFFFFF; 
        border: 1px solid #CFD8DC; 
        border-top: 5px solid #00743c;
        color: #005029; 
        font-weight: 800;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        font-size: 15px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .bfd-sub {
        font-weight: 500;
        font-size: 13px;
        color: #546E7A; /* Blue Grey for subtitle */
        margin-top: 5px;
    }
    .arrow-container {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        font-size: 24px;
        color: #00743c; /* Green Arrows */
        font-weight: bold;
    }

    /* Buttons */
    .stButton > button {
        background-color: #00743c !important;
        color: white !important;
        border-radius: 6px;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 5px rgba(0, 116, 60, 0.2);
    }
    .stButton > button:hover {
        background-color: #005029 !important;
        box-shadow: 0 4px 8px rgba(0, 116, 60, 0.3);
    }
    
    /* Header Box in Sidebar */
    .header-box {
        background: rgba(255, 255, 255, 0.2); padding: 15px; border-radius: 8px; 
        text-align: center; margin-bottom: 25px; border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    #MainMenu, footer, .stDeployButton {visibility: hidden;}
</style>
"""

# --- 3. Mathematical Models ---
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

# --- 4. PDF Generator ---
def create_pdf(res, profit, fig1, fig2):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    CHEMISCO_GREEN = colors.HexColor('#00743c')
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Header with Logo
    logo_style = ParagraphStyle(name='LogoText', fontName='Helvetica-Bold', fontSize=18, textColor=colors.white, alignment=1)
    sub_logo_style = ParagraphStyle(name='SubLogo', fontName='Helvetica', fontSize=8, textColor=colors.whitesmoke, alignment=1)
    
    logo_content = [[Paragraph("CHEMISCO", logo_style)], [Paragraph("Torrefaction Simulator", sub_logo_style)]]
    t_logo = Table(logo_content, colWidths=[2.5*inch])
    t_logo.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), CHEMISCO_GREEN),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,-1), 1, colors.white), ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))

    info_text = Paragraph(f"<b>Date:</b> {current_time}<br/><b>Status:</b> Success", styles['Normal'])
    header_layout = [[t_logo, info_text]]
    t_header_main = Table(header_layout, colWidths=[3*inch, 3*inch])
    t_header_main.setStyle(TableStyle([('ALIGN', (0,0), (0,0), 'LEFT'), ('ALIGN', (1,0), (1,0), 'RIGHT'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')]))
    
    story.append(t_header_main)
    story.append(Spacer(1, 25))
    
    story.append(Paragraph("Technical Engineering Report", ParagraphStyle(name='Title', parent=styles['Heading2'], textColor=CHEMISCO_GREEN, fontSize=16)))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Simulation results based on current reactor parameters.", styles['Normal']))
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
        ('BACKGROUND', (0,0), (-1,0), CHEMISCO_GREEN), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12), ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 1, colors.grey)
    ]))
    story.append(t); story.append(Spacer(1, 30))

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
    story.append(Paragraph("<font color=grey size=8>Chemisco Simulator v3.6</font>", styles['Normal']))
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

    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    
    if 'cost_biomass' not in st.session_state: 
        st.session_state.update({'cost_biomass': 30.0, 'cost_energy': 0.15, 'price_char': 1.20})

    # Sidebar
    with st.sidebar:
        st.markdown("""
            <div class="header-box">
                <h1 style="color:white !important; text-shadow:none;">CHEMISCO</h1>
                <p style="color:#E8F5E9 !important; font-weight:bold;">TORREFACTION SIMULATOR</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.header("⚙️ Inputs")
        reactor = st.selectbox("Reactor Type", ["Rotary Drum", "Fluidized Bed", "Screw Reactor"])
        
        with st.expander("🌲 Feedstock", expanded=True):
            mass = st.slider("Mass (kg)", 1.0, 10000.0, 100.0, 10.0, key='mass_input')
            moisture = st.slider("Moisture (%)", 0.0, 60.0, 15.0)
            ash = st.slider("Ash (Dry %)", 0.0, 30.0, 5.0)
            
        with st.expander("🔥 Process", expanded=True):
            temp = st.slider("Temp (°C)", 150, 350, 275, key='temp_input')
            time_min = st.slider("Time (min)", 10, 120, 30, key='time_input')

        with st.expander("🔧 Model Params", expanded=False):
            p_kf = st.slider("Drying rate", 0.0, 0.1, 0.02, step=0.001, format="%.3f")
            p_Coil = st.slider("Max Oil frac", 0.0, 0.5, 0.25, step=0.01)
            p_Cgas = st.slider("Max Gas frac", 0.0, 0.5, 0.20, step=0.01)
            p_a = st.slider("Solid Yield Factor", 0.1, 0.5, 0.35, step=0.01)
            p_b = st.slider("Degradation", 0.001, 0.01, 0.004, step=0.0005, format="%.4f")
            st.markdown("---")
            p_enh = st.slider("Energy Factor", 0.2, 1.5, 0.85, step=0.05)
            
        params = {"k_f": p_kf, "C_oil": p_Coil, "C_gas": p_Cgas, "a_solid": p_a, "b_solid": p_b, "energy_factor": p_enh}

        with st.expander("💰 Economics", expanded=False):
            st.session_state.cost_biomass = st.slider("Feed ($/ton)", 0.0, 200.0, st.session_state.cost_biomass, step=1.0)
            st.session_state.cost_energy = st.slider("Energy ($/kWh)", 0.0, 1.0, st.session_state.cost_energy, step=0.01)
            st.session_state.price_char = st.slider("Char Price ($/kg)", 0.0, 5.0, st.session_state.price_char, step=0.1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        game_mode = st.checkbox("🎮 Optimization Challenge")

    # Calculations
    res = run_simulation(mass, moisture, ash, temp, time_min, params)
    cost_feed = (mass / 1000) * st.session_state.cost_biomass
    energy_kwh = res['Q_total_kJ'] / 3600.0
    cost_ops = energy_kwh * st.session_state.cost_energy
    revenue = res['char_kg'] * st.session_state.price_char
    profit = revenue - (cost_feed + cost_ops)

    # --- UPDATED COLORS & FONTS ---
    APP_TXT_COLOR = "#005029"  # Darker shade of #00743c for better readability
    APP_BG_COLOR = "#F9FAFA"   # Very clean, professional white-grey
    
    # Harmonized Palette: #00743c (Primary), #009688 (Teal), #66BB6A (Light Green), #BDBDBD (Grey)
    colors_seq = ["#00743c", "#26A69A", "#66BB6A", "#B0BEC5"] 

    # 1. Pie Chart
    df_pie = pd.DataFrame({
        "Component": ["Biochar", "Water Vapor", "Bio-Oil", "Gases"],
        "Mass (kg)": [res['char_kg'], res['water_evap_kg'], res['oil_kg'], res['gas_kg']]
    })
    fig1 = px.pie(df_pie, values='Mass (kg)', names='Component', hole=0.6, color_discrete_sequence=colors_seq, title="Mass Balance")
    fig1.update_traces(
        textposition='inside', 
        textinfo='percent+label', 
        textfont=dict(color='white', size=14, family="Arial", weight="bold")
    )
    fig1.update_layout(
        paper_bgcolor=APP_BG_COLOR, 
        plot_bgcolor=APP_BG_COLOR, 
        font=dict(color=APP_TXT_COLOR, size=15, family="Segoe UI"),
        title_font=dict(size=18, color=APP_TXT_COLOR, family="Segoe UI", weight="bold")
    )

    # 2. Bar Chart
    organic_char = res['char_kg'] - res['ash_kg']
    df_bar = pd.DataFrame({
        "Type": ["Organic Carbon", "Ash"],
        "Mass (kg)": [organic_char, res['ash_kg']]
    })
    fig2 = px.bar(df_bar, x='Type', y='Mass (kg)', color='Type', color_discrete_sequence=['#00743c', '#90A4AE'], title="Solid Composition")
    fig2.update_layout(
        paper_bgcolor=APP_BG_COLOR, 
        plot_bgcolor=APP_BG_COLOR, 
        font=dict(color=APP_TXT_COLOR, size=14, family="Segoe UI"),
        title_font=dict(size=18, color=APP_TXT_COLOR, family="Segoe UI", weight="bold"),
        xaxis=dict(title_font=dict(color=APP_TXT_COLOR), tickfont=dict(color=APP_TXT_COLOR, size=12)),
        yaxis=dict(title_font=dict(color=APP_TXT_COLOR), tickfont=dict(color=APP_TXT_COLOR, size=12)),
        showlegend=False
    )

    # Dashboard
    st.title("CHEMISCO: Process Dashboard")
    st.markdown("---")
    
    # --- FLOW CHART ---
    c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 0.2, 1, 0.2, 1, 0.2, 1])
    
    with c1: 
        st.markdown(f'''
            <div class="bfd-block">
                <div>FEEDSTOCK</div>
                <div class="bfd-sub">{mass} kg<br>{moisture}% H2O</div>
            </div>
        ''', unsafe_allow_html=True)
        
    with c2: st.markdown('<div class="arrow-container">➜</div>', unsafe_allow_html=True)
    
    with c3: 
        st.markdown(f'''
            <div class="bfd-block">
                <div>DRYING</div>
                <div class="bfd-sub">Removing H2O<br>Evap: {res['water_evap_kg']:.1f} kg</div>
            </div>
        ''', unsafe_allow_html=True)

    with c4: st.markdown('<div class="arrow-container">➜</div>', unsafe_allow_html=True)

    with c5: 
        st.markdown(f'''
            <div class="bfd-block">
                <div>{reactor.upper().split()[0]}</div>
                <div class="bfd-sub">{temp}°C | {time_min} min<br>Pyrolysis</div>
            </div>
        ''', unsafe_allow_html=True)

    with c6: st.markdown('<div class="arrow-container">➜</div>', unsafe_allow_html=True)

    with c7: 
        st.markdown(f'''
            <div class="bfd-block" style="border-top-color: #00743c;">
                <div style="color:#00743c;">BIOCHAR</div>
                <div class="bfd-sub" style="font-size:14px; font-weight:bold;">{res["char_kg"]:.1f} kg</div>
            </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mass Yield", f"{res['mass_yield_pct']:.1f}%", f"{res['char_kg']:.1f} kg")
    k2.metric("Energy Density (HHV)", f"{res['hhv_final']:.2f} MJ/kg", f"+{res['hhv_increase_pct']:.1f}% Increase")
    k3.metric("Bio-Oil Output", f"{res['oil_kg']:.1f} kg", "Condensable Volatiles")
    k4.metric("Est. Profit", f"${profit:.2f}", f"Energy Cost: ${cost_ops:.2f}")
    st.markdown("---")

    t1, t2, t3, t4 = st.tabs(["📊 Analytics", "📈 Kinetics", "📄 Export", "🎯 Challenge"])
    
    with t1:
        cc1, cc2 = st.columns(2)
        with cc1: st.plotly_chart(fig1, use_container_width=True)
        with cc2: st.plotly_chart(fig2, use_container_width=True)

    with t2:
        df_time = get_time_series(mass, moisture, ash, temp, time_min, params)
        fig_area = go.Figure()
        # Area chart matching the new green palette
        fig_area.add_trace(go.Scatter(x=df_time['Time (min)'], y=df_time['Char (kg)'], stackgroup='one', name='Char', line=dict(width=0, color='#00743c')))
        fig_area.add_trace(go.Scatter(x=df_time['Time (min)'], y=df_time['Bio-Oil (kg)'], stackgroup='one', name='Bio-Oil', line=dict(width=0, color='#26A69A')))
        fig_area.add_trace(go.Scatter(x=df_time['Time (min)'], y=df_time['Gases (kg)'], stackgroup='one', name='Gases', line=dict(width=0, color='#81C784')))
        fig_area.add_trace(go.Scatter(x=df_time['Time (min)'], y=df_time['Water Vapor (kg)'], stackgroup='one', name='Water Vapor', line=dict(width=0, color='#E8F5E9')))
        
        fig_area.update_layout(
            paper_bgcolor=APP_BG_COLOR, 
            plot_bgcolor=APP_BG_COLOR, 
            title="Product Evolution", 
            title_font=dict(size=18, color=APP_TXT_COLOR, family="Segoe UI", weight="bold"),
            font=dict(color=APP_TXT_COLOR, family="Segoe UI"),
            xaxis=dict(title="Time (min)", tickfont=dict(color=APP_TXT_COLOR), title_font=dict(color=APP_TXT_COLOR, weight="bold")),
            yaxis=dict(title="Mass (kg)", tickfont=dict(color=APP_TXT_COLOR), title_font=dict(color=APP_TXT_COLOR, weight="bold"))
        )
        st.plotly_chart(fig_area, use_container_width=True)

    with t3:
        st.markdown("### 📄 Professional Report Generation")
        try:
            import kaleido
            pdf = create_pdf(res, profit, fig1, fig2)
            st.download_button("Download PDF Report", pdf, f"Chemisco_Report.pdf", "application/pdf")
        except ImportError:
            st.error("⚠️ Library Missing: Please ensure 'kaleido==0.2.1' is in requirements.txt")

    # --- Game Mode Updates ---
    with t4:
        if game_mode:
            # --- Initialize Game State if not present ---
            if 'game_target_hhv' not in st.session_state:
                st.session_state['game_target_hhv'] = 22.0
                st.session_state['game_min_yield'] = 55.0
                st.session_state['game_target_profit'] = 0.0
                st.session_state['client_name'] = "Default Corp"

            # --- Callback: Generate New Targets (Randomize Challenge) ---
            def generate_new_client():
                # Randomize targets within realistic bounds to create a "New Client"
                # HHV: 20 to 24 MJ/kg
                st.session_state['game_target_hhv'] = round(random.uniform(20.0, 24.0), 1)
                # Yield: 45% to 65%
                st.session_state['game_min_yield'] = round(random.uniform(45.0, 65.0), 1)
                # Profit: $10 to $100
                st.session_state['game_target_profit'] = round(random.uniform(10.0, 100.0), 1)
                
                # Random Client Name
                companies = ["EcoChar Solutions", "GreenEnergy Inc", "CarbonFix Ltd", "AgriFuel Systems", "Sustainable Tech"]
                st.session_state['client_name'] = f"{random.choice(companies)} #{random.randint(100, 999)}"

            # --- NEW CLIENT BUTTON (Does NOT reset sliders, only targets) ---
            st.markdown(f"### 🎯 Engineering Challenge: {st.session_state['client_name']}")
            
            col_reset, col_info = st.columns([1, 4])
            with col_reset:
                st.button("🔄 New Client", help="Get a new client contract (New Targets)", on_click=generate_new_client)
            with col_info:
                st.caption("Pressing this will generate a new set of requirements (Targets) without resetting your process parameters.")

            st.markdown("---")
            
            # Use Session State Targets
            TARGET_HHV = st.session_state['game_target_hhv']
            MIN_YIELD = st.session_state['game_min_yield']
            TARGET_PROFIT = st.session_state['game_target_profit']

            score = 0
            if res['hhv_final'] >= TARGET_HHV: score += 33
            if res['mass_yield_pct'] >= MIN_YIELD: score += 33
            if profit > TARGET_PROFIT: score += 34
            
            st.write(f"**Mission Progress: {score}%**")
            st.progress(score)
            
            col_g1, col_g2, col_g3 = st.columns(3)
            
            # 1. HHV Target
            with col_g1:
                is_hhv_ok = res['hhv_final'] >= TARGET_HHV
                st.metric(f"Target HHV (>{TARGET_HHV})", f"{res['hhv_final']:.2f}", 
                         delta=f"{res['hhv_final'] - TARGET_HHV:.2f}", 
                         delta_color="normal" if is_hhv_ok else "inverse")
                if not is_hhv_ok:
                    st.info("💡 **Hint:** Increase Temperature or Residence Time.")

            # 2. Yield Target
            with col_g2:
                is_yield_ok = res['mass_yield_pct'] >= MIN_YIELD
                st.metric(f"Target Yield (>{MIN_YIELD}%)", f"{res['mass_yield_pct']:.1f}%", 
                         delta=f"{res['mass_yield_pct'] - MIN_YIELD:.1f}%", 
                         delta_color="normal" if is_yield_ok else "inverse")
                if not is_yield_ok:
                    st.info("💡 **Hint:** Temperature is too high! Lower it.")

            # 3. Profit Target
            with col_g3:
                is_profit_ok = profit > TARGET_PROFIT
                st.metric(f"Target Profit (>${TARGET_PROFIT})", f"${profit:.2f}", 
                         delta="Net Profit", 
                         delta_color="normal" if is_profit_ok else "inverse")
                if not is_profit_ok:
                    st.info("💡 **Hint:** Reduce Time or Increase Mass.")

            st.markdown("---")
            
            if score >= 100:
                st.balloons()
                st.success(f"🏆 **CONTRACT FULFILLED!** {st.session_state['client_name']} is satisfied.")
            else:
                st.warning("⚠️ Optimization Incomplete. Adjust the sliders in the sidebar.")
                
        else:
            st.info("👋 Activate **'Optimization Challenge'** in the sidebar to start the game.")

if __name__ == "__main__":
    main()

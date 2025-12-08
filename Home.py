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

# --- 2. Styles (High Contrast & Modern Fixed) ---
GLOBAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&family=Inter:wght@400;600;800&display=swap');

    /* 1. Main Background & Font */
    .stApp { 
        background-color: #F8F9FA; 
        font-family: 'Inter', sans-serif; 
    }
    
    /* 2. SIDEBAR STYLING (FIXED CONTRAST) */
    section[data-testid="stSidebar"] { 
        background-color: #1A3C34; /* Dark Pine */
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Force ALL text in sidebar to be light */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] p {
        color: #FFFFFF !important;
    }

    /* Input Labels */
    section[data-testid="stSidebar"] label {
        color: #B2DFDB !important; /* Light Teal */
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    
    /* Input Widgets Background & Text */
    section[data-testid="stSidebar"] .stNumberInput input,
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] .stSlider {
        color: #333333 !important; /* Text inside boxes usually needs to be dark if box is white */
    }
    
    /* Fix Expander Headers in Sidebar */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        color: #FFFFFF !important;
        font-weight: bold;
        background-color: rgba(255,255,255,0.05);
        border-radius: 5px;
    }
    section[data-testid="stSidebar"] .streamlit-expanderHeader p {
        font-size: 15px;
    }

    /* 3. METRIC CARDS */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border-radius: 10px; 
        padding: 15px; 
        border-left: 5px solid #26A69A;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricValue"] { color: #1A3C34 !important; font-weight: 800; font-size: 26px !important; }
    div[data-testid="stMetricLabel"] { color: #546E7A !important; font-size: 14px; font-weight: 600; }

    /* 4. BUTTONS */
    .stButton > button { 
        background-color: #26A69A !important; 
        color: white !important; 
        border: none; 
        border-radius: 6px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #1A3C34 !important;
        box-shadow: 0 5px 15px rgba(26, 60, 52, 0.2);
    }

    /* 5. FLOW CHART BLOCKS */
    .bfd-block {
        background: #FFFFFF;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #E0E0E0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        color: #1A3C34;
        font-weight: bold;
    }
    .bfd-subtitle { color: #78909C; font-size: 0.85em; font-weight: normal; margin-top: 5px; }
    
    /* 6. HEADERS */
    h1, h2, h3 { color: #1A3C34 !important; }
    
    /* 7. TABS */
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #26A69A !important;
        border-bottom-color: #26A69A !important;
        font-weight: bold;
    }

    /* Hide Footer */
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

# --- 4. PDF Generator (Logo & Colors) ---
def create_pdf(res, profit, fig1, fig2):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Colors for PDF
    CHEMISCO_PRIMARY = colors.HexColor('#1A3C34')
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Header
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
    
    story.append(Paragraph("Technical Engineering Report", ParagraphStyle(name='Title', parent=styles['Heading2'], textColor=CHEMISCO_PRIMARY, fontSize=16)))
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
    story.append(Paragraph("<font color=grey size=8>Chemisco Simulator v3.5</font>", styles['Normal']))
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- 5. Main Streamlit App ---
def main():
    st.set_page_config(page_title="Chemisco Pro", layout="wide", initial_sidebar_state="expanded")
    
    # Inject Botpress
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

    # --- SIDEBAR (High Contrast) ---
    with st.sidebar:
        # Header Box
        st.markdown("""
            <div style="background:rgba(255,255,255,0.1); border-radius:10px; padding:20px; text-align:center; margin-bottom:20px; border:1px solid rgba(255,255,255,0.2);">
                <h1 style="margin:0; font-size:28px; letter-spacing:2px; color:white !important;">CHEMISCO</h1>
                <p style="margin:5px 0 0 0; font-size:10px; color:#B2DFDB !important; letter-spacing:1px;">TORREFACTION LAB</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.header("⚙️ Control Panel")
        reactor = st.selectbox("Reactor Type", ["Rotary Drum", "Fluidized Bed", "Screw Reactor"])
        
        with st.expander("📝 Feedstock Properties", expanded=True):
            mass = st.number_input("Input Mass (kg)", 1.0, 10000.0, 100.0, 10.0)
            moisture = st.slider("Moisture (%)", 0.0, 60.0, 15.0)
            ash = st.slider("Ash Content (%)", 0.0, 30.0, 5.0)
            
        with st.expander("🔥 Process Conditions", expanded=True):
            temp = st.slider("Temperature (°C)", 150, 350, 275)
            time_min = st.slider("Time (min)", 10, 120, 30)

        with st.expander("🛠️ Advanced Parameters", expanded=False):
            p_kf = st.number_input("Drying Rate", 0.0, 0.1, 0.02)
            p_Coil = st.number_input("Oil Coeff", 0.0, 0.5, 0.25)
            p_Cgas = st.number_input("Gas Coeff", 0.0, 0.5, 0.20)
            p_a = st.number_input("Yield Coeff (a)", 0.1, 0.5, 0.35)
            p_b = st.number_input("Degradation (b)", 0.001, 0.01, 0.004, format="%.4f")
            st.markdown("---")
            p_enh = st.slider("Energy Factor", 0.2, 1.5, 0.85)
            
        params = {"k_f": p_kf, "C_oil": p_Coil, "C_gas": p_Cgas, "a_solid": p_a, "b_solid": p_b, "energy_factor": p_enh}

        with st.expander("💰 Market Values", expanded=False):
            st.session_state.cost_biomass = st.number_input("Biomass ($/ton)", value=st.session_state.cost_biomass)
            st.session_state.cost_energy = st.number_input("Elec ($/kWh)", value=st.session_state.cost_energy)
            st.session_state.price_char = st.number_input("Char ($/kg)", value=st.session_state.price_char)
        
        st.markdown("<br>", unsafe_allow_html=True)
        game_mode = st.checkbox("🎮 Enable Challenge Mode")

    # Calculations
    res = run_simulation(mass, moisture, ash, temp, time_min, params)
    cost_feed = (mass / 1000) * st.session_state.cost_biomass
    energy_kwh = res['Q_total_kJ'] / 3600.0
    cost_ops = energy_kwh * st.session_state.cost_energy
    revenue = res['char_kg'] * st.session_state.price_char
    profit = revenue - (cost_feed + cost_ops)

    # Theme Colors
    COLOR_PRIMARY = "#1A3C34"
    COLOR_ACCENT = "#26A69A"
    COLOR_BG_LIGHT = "#F8F9FA"

    # --- PLOTLY CONFIG (FIXING LABELS) ---
    
    # 1. PIE CHART - Fixed Labels
    df_pie = pd.DataFrame({
        "Component": ["Biochar", "Water Vapor", "Bio-Oil", "Gases"],
        "Mass (kg)": [res['char_kg'], res['water_evap_kg'], res['oil_kg'], res['gas_kg']]
    })
    
    fig1 = px.pie(df_pie, values='Mass (kg)', names='Component', hole=0.6, 
                  color_discrete_sequence=["#1A3C34", "#26A69A", "#80CBC4", "#B2DFDB"])
    
    fig1.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        textfont=dict(color='white', size=14, family="Arial Black") # Force White Bold Text
    )
    
    fig1.update_layout(
        title=dict(text="Mass Balance", font=dict(color=COLOR_PRIMARY, size=20)),
        paper_bgcolor=COLOR_BG_LIGHT, 
        plot_bgcolor=COLOR_BG_LIGHT,
        showlegend=True,
        legend=dict(font=dict(color=COLOR_PRIMARY))
    )

    # 2. BAR CHART - Fixed Labels
    organic_char = res['char_kg'] - res['ash_kg']
    df_bar = pd.DataFrame({
        "Type": ["Organic Carbon", "Ash"],
        "Mass (kg)": [organic_char, res['ash_kg']]
    })
    fig2 = px.bar(df_bar, x='Type', y='Mass (kg)', color='Type', 
                  color_discrete_sequence=['#1A3C34', '#90A4AE'],
                  text='Mass (kg)') # Show value on bar
    
    fig2.update_traces(texttemplate='%{text:.1f}', textposition='auto')
    
    fig2.update_layout(
        title=dict(text="Solid Composition", font=dict(color=COLOR_PRIMARY, size=20)),
        paper_bgcolor=COLOR_BG_LIGHT, 
        plot_bgcolor=COLOR_BG_LIGHT,
        font=dict(color=COLOR_PRIMARY), # General font color dark
        xaxis=dict(title=dict(font=dict(color=COLOR_PRIMARY)), tickfont=dict(color=COLOR_PRIMARY)),
        yaxis=dict(title=dict(font=dict(color=COLOR_PRIMARY)), tickfont=dict(color=COLOR_PRIMARY)),
        showlegend=False
    )

    # Main Dashboard
    st.title("CHEMISCO: Process Simulation")
    st.markdown("---")
    
    # Flow Diagram
    c1, c2, c3, c4, c5 = st.columns([1.5, 0.5, 1.5, 0.5, 1.5])
    with c1: st.markdown(f'<div class="bfd-block">FEED<div class="bfd-subtitle">{mass} kg | {moisture}% H2O</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div style="text-align:center; font-size:30px; color:#26A69A; padding-top:10px;">➜</div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="bfd-block">PYROLYSIS<div class="bfd-subtitle">{temp}°C | {time_min} min</div></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div style="text-align:center; font-size:30px; color:#26A69A; padding-top:10px;">➜</div>', unsafe_allow_html=True)
    with c5: st.markdown(f'<div class="bfd-block" style="border-color:#26A69A; border-width:2px;">PRODUCT<div class="bfd-subtitle" style="color:#26A69A; font-weight:bold;">{res["char_kg"]:.1f} kg</div></div>', unsafe_allow_html=True)
    
    st.markdown("### Performance Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mass Yield", f"{res['mass_yield_pct']:.1f}%", f"{res['char_kg']:.1f} kg Output")
    k2.metric("HHV Energy", f"{res['hhv_final']:.2f} MJ/kg", f"+{res['hhv_increase_pct']:.1f}% Gain")
    k3.metric("Liquid Oil", f"{res['oil_kg']:.1f} kg", "Condensable")
    k4.metric("Net Profit", f"${profit:.2f}", f"Op. Cost ${cost_ops:.2f}")
    st.markdown("---")

    t1, t2, t3, t4 = st.tabs(["📊 Data Viz", "📈 Kinetics", "📄 Report", "🎯 Challenge"])
    
    with t1:
        cc1, cc2 = st.columns(2)
        with cc1: st.plotly_chart(fig1, use_container_width=True)
        with cc2: st.plotly_chart(fig2, use_container_width=True)

    with t2:
        df_time = get_time_series(mass, moisture, ash, temp, time_min, params)
        fig_area = go.Figure()
        fig_area.add_trace(go.Scatter(x=df_time['Time (min)'], y=df_time['Char (kg)'], stackgroup='one', name='Char', line=dict(width=0, color='#1A3C34')))
        fig_area.add_trace(go.Scatter(x=df_time['Time (min)'], y=df_time['Bio-Oil (kg)'], stackgroup='one', name='Oil', line=dict(width=0, color='#26A69A')))
        fig_area.add_trace(go.Scatter(x=df_time['Time (min)'], y=df_time['Gases (kg)'], stackgroup='one', name='Gas', line=dict(width=0, color='#80CBC4')))
        fig_area.add_trace(go.Scatter(x=df_time['Time (min)'], y=df_time['Water Vapor (kg)'], stackgroup='one', name='Water', line=dict(width=0, color='#E0F2F1')))
        
        # Area Chart Fixes
        fig_area.update_layout(
            title="Mass Evolution",
            paper_bgcolor=COLOR_BG_LIGHT, plot_bgcolor=COLOR_BG_LIGHT,
            font=dict(color=COLOR_PRIMARY),
            xaxis=dict(title="Time (min)", tickfont=dict(color=COLOR_PRIMARY)),
            yaxis=dict(title="Mass (kg)", tickfont=dict(color=COLOR_PRIMARY)),
            hovermode="x unified"
        )
        st.plotly_chart(fig_area, use_container_width=True)

    with t3:
        st.markdown("### 📄 Export Results")
        try:
            import kaleido
            pdf = create_pdf(res, profit, fig1, fig2)
            st.download_button("Download Report (PDF)", pdf, f"Chemisco_Report.pdf", "application/pdf")
        except ImportError:
            st.warning("Install 'kaleido' to generate PDFs.")

    with t4:
        if game_mode:
            TARGET_HHV, MIN_YIELD, TARGET_PROFIT = 22.0, 55.0, 0.0
            st.markdown("### 🎯 Challenge Targets")
            col_g1, col_g2, col_g3 = st.columns(3)
            col_g1.metric("Target HHV", f"> {TARGET_HHV}", f"Current: {res['hhv_final']:.2f}", delta_color="normal" if res['hhv_final'] >= TARGET_HHV else "inverse")
            col_g2.metric("Target Yield", f"> {MIN_YIELD}%", f"Current: {res['mass_yield_pct']:.1f}%", delta_color="normal" if res['mass_yield_pct'] >= MIN_YIELD else "inverse")
            col_g3.metric("Target Profit", f"> $0", f"Current: ${profit:.2f}", delta_color="normal" if profit > 0 else "inverse")
            
            if res['hhv_final'] >= TARGET_HHV and res['mass_yield_pct'] >= MIN_YIELD and profit > TARGET_PROFIT:
                st.balloons(); st.success("🎉 Target Achieved!")
            else:
                st.info("Adjust Temp/Time to meet targets.")
        else:
            st.info("Enable Checkbox in Sidebar.")

if __name__ == "__main__":
    main()

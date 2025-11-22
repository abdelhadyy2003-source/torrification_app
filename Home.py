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

# --- 1. Chemical and Empirical Constants (Unchanged) ---
R_GAS = 8.314  # Universal Gas Constant (J/mol·K)

EMPIRICAL_DATA = {
    "Wood": {
        "A": 2.5e10, "Ea": 135000, "k_drying_base": 0.05, 
        "Ash": 0.02, "Gas_Factor": 0.35, "Feedstock_Cost": 0.05
    },
    "Agricultural Waste": {
        "A": 5.0e11, "Ea": 150000, "k_drying_base": 0.07, 
        "Ash": 0.08, "Gas_Factor": 0.45, "Feedstock_Cost": 0.03
    },
    "Municipal Waste": {
        "A": 1.0e12, "Ea": 165000, "k_drying_base": 0.10, 
        "Ash": 0.15, "Gas_Factor": 0.55, "Feedstock_Cost": 0.02
    }
}

SIZE_FACTOR = {
    "Fine (<1mm)": 1.0,
    "Medium (1-5mm)": 0.85,
    "Coarse (>5mm)": 0.65
}

# --- 2. Static UI Components (CSS & Helper) ---
GLOBAL_CSS = """
<style>
    /* Global Styling for Professional Look */
    .stApp { padding-top: 20px; background-color: #f4f7f6; } 
    
    /* Custom Banner Style */
    .main-banner {
        background: linear-gradient(135deg, #1e704b, #388E3C); /* Gradient Green */
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
    }
    .main-banner h1 { color: #FFFFFF; margin: 0; font-size: 2.8em; }
    .main-banner p { color: #E8F5E9; margin-top: 5px; font-size: 1.2em; }
    
    /* Sidebar Customization */
    .st-emotion-cache-1na6f8g, .st-emotion-cache-1d391kg { 
        background-color: #e3f2fd; /* Light Blue for contrast */
    }
    
    /* Input Expander Style */
    .st-emotion-cache-p5m8m8 { 
        border-radius: 10px;
        border-left: 5px solid #00BCD4; /* Cyan accent */
        padding: 10px;
        margin-bottom: 15px;
        background-color: #FFFFFF;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* Scoreboard Metrics */
    .scorecard-container {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        background-color: #FFFFFF;
    }
    .scorecard-value {
        font-size: 38px !important;
        font-weight: bold;
        color: #1e704b !important; 
        margin: 5px 0 0;
    }
</style>
"""
# --- Helper function for Gauge Chart (Visual KPI) ---
def plot_gauge(value, title, min_val, max_val, color_map, unit=""):
    """Creates a visually appealing gauge chart using Matplotlib."""
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={'aspect': 'equal'})
    
    # Draw background arc (Scale)
    ax.add_patch(plt.Circle((0, 0), 1.0, color='lightgray', fill=False, linewidth=10))
    
    # Calculate angular position for the value
    norm_val = (value - min_val) / (max_val - min_val)
    angle = 180 * (1 - norm_val) # 180 (start) to 0 (end) degrees
    
    # Draw colored arcs (Color Map)
    for limit, color in color_map.items():
        if value > limit:
            continue
        
        # Calculate angle for color limit
        limit_angle = 180 * (1 - (limit - min_val) / (max_val - min_val))
        
        # Draw the colored arc
        ax.add_patch(plt.Wedge((0, 0), 1.0, limit_angle, 180, color=color, linewidth=0, alpha=0.6))

    # Draw the pointer (Needle)
    x = 0.9 * np.cos(np.deg2rad(angle))
    y = 0.9 * np.sin(np.deg2rad(angle))
    ax.plot([0, x], [0, y], color='black', linewidth=3)
    
    # Center circle
    ax.add_patch(plt.Circle((0, 0), 0.1, color='black', zorder=10))

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontsize=10, pad=10)
    ax.text(0, -0.15, f"{value:.2f}{unit}", ha='center', va='center', fontsize=16, weight='bold')
    
    # Hide axes
    ax.axis('off')
    
    return fig

# --- 3. Simulation Core Logic (Unchanged) ---
def simulate_torrefaction(biomass, moisture, temp_C, duration_min, size, initial_mass_kg):
    """Core torrefaction simulation logic using Arrhenius and particle size correction."""
    temp_K = temp_C + 273.15
    data = EMPIRICAL_DATA.get(biomass)
    
    k_devol_arrhenius = data["A"] * np.exp(-data["Ea"] / (R_GAS * temp_K))
    k_devol_eff = k_devol_arrhenius * SIZE_FACTOR.get(size)
    k_drying = data["k_drying_base"]
    ash_content = data["Ash"]

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

    final_moisture = sol[-1, 0]
    final_volatiles_remaining = sol[-1, 1]
    
    final_biochar_fraction = (1 - final_moisture - final_volatiles_remaining - ash_content)
    
    # Calculate mass fractions lost/remaining
    final_volatiles_lost_fraction = initial_volatiles_fraction - final_volatiles_remaining
    moisture_lost_fraction = initial_moisture_fraction - final_moisture
    
    yields_percent = pd.DataFrame({
        "Yield (%)": [
            (final_biochar_fraction + ash_content) * 100,
            final_volatiles_lost_fraction * 100,
            moisture_lost_fraction * 100,
            ash_content * 100
        ]},
        index=["Biochar (Solid) & Ash", "Non-Condensable Gases", "Moisture Loss (Water Vapor)", "Initial Ash Content"]
    )
    
    yields_mass = yields_percent.copy()
    yields_mass["Mass (kg)"] = yields_percent["Yield (%)"] * initial_mass_kg / 100
    yields_mass.drop(columns=["Yield (%)"], inplace=True)

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

# --- 4. Tycoon Game Logic (Updated) ---

def calculate_tycoon_profit(results):
    """Calculates the economic and energy outcomes for the Scorecard."""
    params = results["parameters"]
    data = EMPIRICAL_DATA.get(params["biomass"])
    
    initial_mass_kg = params["initial_mass"]
    final_biochar_mass = results["yields_mass"].loc["Biochar (Solid) & Ash", "Mass (kg)"]
    
    # 1. Energy & Quality Calculations (New Sustainability KPI)
    M_total = results["yields_percent"].loc["Biochar (Solid) & Ash", "Yield (%)"] / 100
    initial_moisture = params["moisture"] / 100
    initial_ash = data["Ash"]
    
    M_daf = (M_total - initial_ash) / (1 - initial_moisture - initial_ash) if (1 - initial_moisture - initial_ash) > 0.001 else 1.0
    
    # EDR (Energy Density Ratio) ~ 1/M_daf (simplified)
    EDR = 1 / M_daf if M_daf > 0.001 else 1.0 

    # Thermal Efficiency (TE): Energy in Biochar / Energy in Feedstock
    # TE = M_daf * EDR (simplified)
    Thermal_Efficiency = M_daf * EDR * (0.8 + 0.2 * (EDR - 1)) # Add a slight correction factor
    Thermal_Efficiency = min(Thermal_Efficiency, 0.99) # Max 99%
    
    # 2. Economic Calculations
    cost_feedstock = initial_mass_kg * data["Feedstock_Cost"]
    temp_factor = params["temperature"] / 200
    time_factor = params["duration"] / 60
    cost_operating = initial_mass_kg * 0.01 * temp_factor * time_factor 
    total_costs = cost_feedstock + cost_operating
    
    BASE_SELLING_PRICE = 0.15 # $/kg
    price_multiplier = 1 + (EDR - 1.0) * 2.5
    selling_price = BASE_SELLING_PRICE * price_multiplier
    revenues = final_biochar_mass * selling_price
    
    net_profit = revenues - total_costs
    
    return {
        "final_biochar_mass": final_biochar_mass,
        "total_costs": total_costs,
        "revenues": revenues,
        "net_profit": net_profit,
        "EDR": EDR,
        "Thermal_Efficiency": Thermal_Efficiency * 100, # Return as percentage
        "selling_price": selling_price
    }

# --- 5. Main Streamlit App ---
def main():
    # Session state initialization
    if 'capital' not in st.session_state:
        st.session_state.capital = 5000.0
    if 'batch_count' not in st.session_state:
        st.session_state.batch_count = 0
        
    st.set_page_config(page_title="Chemisco Pro Torrefaction Simulator", layout="wide", initial_sidebar_state="expanded")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # 5.1. Sidebar (Inputs) - (Simplified and moved to sidebar)
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 15px; border-radius: 8px; background-color: #1B5E20;'>
                <h1 style='color: white; margin: 0; font-size: 1.8em;'>CHEMISCO PRO</h1>
                <p style='color: #A5D6A7; margin: 0; font-size: 0.9em;'>Torrefaction Process Simulator</p>
            </div>
            """, unsafe_allow_html=True)
        st.header("⚙️ Input Parameters")
        
        with st.expander("🌲 Biomass & Batch Size", expanded=True):
            initial_mass_kg = st.number_input("Initial Batch Mass (kg)", min_value=1.0, value=100.0, step=50.0)
            biomass_type = st.selectbox("Biomass Type", list(EMPIRICAL_DATA.keys()))
            moisture_content = st.slider("Initial Moisture Content (%)", 0.0, 50.0, 10.0, step=1.0)
            particle_size = st.selectbox("Particle Size", list(SIZE_FACTOR.keys()))
        
        with st.expander("🌡️ Process Conditions", expanded=True):
            temperature = st.slider("Torrefaction Temperature (°C)", 200, 350, 275, step=5)
            duration = st.slider("Process Duration (min)", 10, 120, 45, step=5)
            
        st.markdown("---")
        
        # Tycoon Game Controls in Sidebar
        st.subheader("💰 Tycoon Controls")
        
        # Run Button
        if st.button("▶️ تشغيل دورة الإنتاج الحالية", help="يشغل دورة إنتاج واحدة بالإعدادات المحددة"):
            if moisture_content / 100 + EMPIRICAL_DATA[biomass_type]["Ash"] > 1:
                st.error("Input Error: Moisture and Ash > 100%.")
            else:
                st.session_state.run_batch = True
        else:
            st.session_state.run_batch = False
        
        if st.session_state.batch_count > 0:
            if st.button("🔄 إعادة تعيين اللعبة", help="إعادة رأس المال والعدادات إلى القيمة الافتراضية"):
                st.session_state.capital = 5000.0
                st.session_state.batch_count = 0
                st.rerun()

    # 5.2. Main Content
    st.markdown("""
        <div class="main-banner">
            <h1>🔥 Advanced Torrefaction Simulator</h1>
            <p>Strategic Dashboard for Process Optimization</p>
        </div>
        """, unsafe_allow_html=True)
    
    # --- Scoreboard Bar ---
    st.subheader("لوحة التحكم الاستراتيجية (Scoreboard)")
    col_sc1, col_sc2, col_sc3 = st.columns(3)
    
    col_sc1.markdown('<div class="scorecard-container"><h4>🏦 رأس المال الكلي</h4><p class="scorecard-value">$%s</p></div>' % f"{st.session_state.capital:,.2f}", unsafe_allow_html=True)
    col_sc2.markdown('<div class="scorecard-container"><h4>🏭 دورات الإنتاج المنفذة</h4><p class="scorecard-value">%s</p></div>' % f"{st.session_state.batch_count}", unsafe_allow_html=True)
    
    if st.session_state.capital >= 15000:
        col_sc3.markdown('<div class="scorecard-container" style="background-color: #E8F5E9;"><h4>🏆 حالة التحدي</h4><p class="scorecard-value" style="color:#2E7D32;">هدف تم تحقيقه!</p></div>', unsafe_allow_html=True)
    elif st.session_state.capital >= 5000:
        col_sc3.markdown('<div class="scorecard-container" style="background-color: #FFFDE7;"><h4>🎯 هدف الربح</h4><p class="scorecard-value" style="color:#FBC02D;">$15,000</p></div>', unsafe_allow_html=True)
    else:
        col_sc3.markdown('<div class="scorecard-container" style="background-color: #FFEBEE;"><h4>🚨 هدف الربح</h4><p class="scorecard-value" style="color:#C62828;">$15,000</p></div>', unsafe_allow_html=True)


    st.markdown("---")
    
    # --- Run Logic and Display Tabs ---
    
    results = simulate_torrefaction(biomass_type, moisture_content, temperature, duration, particle_size, initial_mass_kg)
    
    tab1, tab2, tab3 = st.tabs(["🔥 بطاقة الأداء (KPIs)", "📈 تفاصيل المحاكاة", "📄 تقرير PDF"])

    with tab1:
        st.subheader("بطاقة الأداء الاستراتيجية ودورة الإنتاج")
        
        # 1. KPIs Visuals
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        
        tycoon_results = calculate_tycoon_profit(results)
        
        # KPI 1: Profitability
        profit_map = {-50: 'red', 0: 'orange', 100: 'green'}
        profit_fig = plot_gauge(tycoon_results['net_profit'], "1. الأداء الاقتصادي (الربح الصافي)", min_val=-100, max_val=200, color_map=profit_map, unit="$")
        col_kpi1.pyplot(profit_fig)

        # KPI 2: Biochar Quality (EDR)
        edr_map = {1.1: 'red', 1.25: 'yellow', 1.4: 'green'}
        edr_fig = plot_gauge(tycoon_results['EDR'], "2. جودة المنتج (نسبة كثافة الطاقة)", min_val=1.0, max_val=1.5, color_map=edr_map)
        col_kpi2.pyplot(edr_fig)

        # KPI 3: Thermal Efficiency
        te_map = {60: 'red', 75: 'yellow', 90: 'green'}
        te_fig = plot_gauge(tycoon_results['Thermal_Efficiency'], "3. الكفاءة الحرارية (%)", min_val=50, max_val=100, color_map=te_map, unit="%")
        col_kpi3.pyplot(te_fig)

        st.markdown("---")
        
        # 2. Batch Execution Summary
        st.markdown("### 📝 نتائج دورة الإنتاج الأخيرة")
        if st.session_state.run_batch:
            
            # Update capital and count here after successful run check in sidebar
            if st.session_state.capital >= initial_mass_kg * EMPIRICAL_DATA[biomass_type]["Feedstock_Cost"]:
                st.session_state.capital += tycoon_results["net_profit"]
                st.session_state.batch_count += 1
                
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                col_sum1.metric("الربح الصافي ($)", f"${tycoon_results['net_profit']:,.2f}", delta=f"{tycoon_results['net_profit']/tycoon_results['total_costs'] * 100:.1f} % هامش ربح")
                col_sum2.metric("سعر البيع المعدل ($/kg)", f"${tycoon_results['selling_price']:.4f}")
                col_sum3.metric("عائد الكتلة (%)", f"{results['yields_percent'].loc['Biochar (Solid) & Ash', 'Yield (%)']:.2f} %")

                st.markdown("##### تحليل التكاليف والإيرادات:")
                profit_df = pd.DataFrame({
                    "البند": ["تكلفة المواد الخام", "تكلفة التشغيل", "إجمالي التكاليف", "الإيرادات الكلية"],
                    "القيمة ($)": [
                        tycoon_results["cost_feedstock"], 
                        tycoon_results["cost_operating"], 
                        tycoon_results["total_costs"], 
                        tycoon_results["revenues"]
                    ]
                }).set_index("البند")
                st.dataframe(profit_df.style.format("${:,.2f}"), use_container_width=True)
                
            else:
                st.error("⚠️ فشلت الدورة! رأس مالك الحالي غير كافٍ لشراء المواد الخام. حاول تقليل حجم الدفعة أو اختيار مادة خام أرخص.")
                
            st.session_state.run_batch = False # Reset flag

    # --- Tab 2: Simulation Details ---
    with tab2:
        st.subheader("تفاصيل ميزان الكتلة والمحاكاة الحركية")
        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.markdown("##### 1. Mass Conversion Kinetics")
            st.line_chart(results["mass_profile"], use_container_width=True)
            st.markdown("##### 2. Gas Composition")
            st.bar_chart(results["gas_composition_molar"], use_container_width=True)
            
        with col_d2:
            st.markdown("##### 3. Yield Distribution (kg)")
            st.dataframe(results["yields_mass"].style.format("{:.2f}"), use_container_width=True)
            st.markdown("##### 4. Mass Balance Pie Chart")
            fig1, ax1 = plt.subplots(figsize=(5, 5))
            filtered_yields = results["yields_percent"].iloc[[0, 1, 2]] 
            ax1.pie(filtered_yields["Yield (%)"].values, labels=filtered_yields.index, autopct='%1.1f%%', startangle=90, colors=['#8B4513', '#A9A9A9', '#ADD8E6'])
            ax1.axis('equal')
            st.pyplot(fig1)

    # --- Tab 3: PDF Report ---
    with tab3:
        st.subheader("Generate Comprehensive PDF Report")
        st.markdown("Click the button below to generate and download a detailed report of the simulation.")
        
        if st.button("⬇️ Download PDF Report", key="pdf_download"):
            pdf_buffer = generate_pdf_report(results)
            st.download_button(
                label="Download Report",
                data=pdf_buffer,
                file_name=f"Torrefaction_Report_{biomass_type}_{temperature}C.pdf",
                mime="application/pdf"
            )

# --- 6. PDF Report Generation Function (Unchanged) ---
# ... (The rest of the generate_pdf_report function remains the same as in V8)
def generate_pdf_report(results):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter, title="Torrefaction Report",
        leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch
    )
    styles = getSampleStyleSheet()
    elements = []
    
    # Header & Banner
    elements.append(Paragraph("<font size=16 color='#4CAF50'>CHEMISCO PRO TORREFACTION REPORT</font>", styles["Title"]))
    elements.append(Paragraph(f"Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", styles["Italic"]))
    elements.append(Spacer(1, 0.25*inch))
    
    # 1. Parameters Table
    elements.append(Paragraph("1. Simulation Parameters & Kinetics", styles["h2"]))
    p = results["parameters"]
    param_data = [
        ["Parameter", "Value"],
        ["Initial Biomass Mass", f"{p['initial_mass']:.0f} kg"],
        ["Moisture Content", f"{p['moisture']}%"],
        ["Temperature", f"{p['temperature']} °C"],
        ["Duration", f"{p['duration']} min"],
        ["Particle Size", p["size"]],
        [f"Effective Devol. Rate ($k_{{devol,eff}}$)", f"{results['k_devol_eff']:.3f} min⁻¹"],
    ]
    param_table = Table(param_data, colWidths=[2.5*inch, 3*inch], 
                        style=[('GRID', (0,0), (-1,-1), 1, colors.black)])
    elements.append(param_table)
    elements.append(Spacer(1, 0.25*inch))
    
    # 2. Yields Tables
    elements.append(Paragraph("2. Product Yields", styles["h2"]))
    
    # Mass Yields Table
    elements.append(Paragraph("2.1. Mass Yields (kg)", styles["h3"]))
    mass_data = [["Component", "Mass (kg)"]] + \
                 [[idx, f"{val[0]:.2f}"] for idx, val in results["yields_mass"].iterrows()]
    mass_table = Table(mass_data, colWidths=[3.5*inch, 2*inch], style=[('GRID', (0,0), (-1,-1), 1, colors.black)])
    elements.append(mass_table)
    elements.append(Spacer(1, 0.1*inch))
    
    # Percentage Yields Table
    elements.append(Paragraph("2.2. Percentage Yields (%)", styles["h3"]))
    percent_data = [["Component", "Yield (%)"]] + \
                 [[idx, f"{val[0]:.2f}"] for idx, val in results["yields_percent"].iterrows()]
    percent_table = Table(percent_data, colWidths=[3.5*inch, 2*inch], style=[('GRID', (0,0), (-1,-1), 1, colors.black)])
    elements.append(percent_table)
    elements.append(Spacer(1, 0.5*inch))
    
    # 3. Charts
    elements.append(Paragraph("3. Results Visualization", styles["h2"]))
    
    # Chart 1: Mass Conversion Plot 
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    results["mass_profile"].plot(ax=ax3)
    plt.title("Mass Component Conversion Over Time")
    plt.xlabel("Time (min)")
    plt.ylabel("Mass Fraction")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    imgdata3 = BytesIO()
    fig3.savefig(imgdata3, format='png', dpi=300, bbox_inches='tight')
    imgdata3.seek(0)
    elements.append(ReportImage(imgdata3, width=5.5*inch, height=3.7*inch))
    elements.append(Spacer(1, 0.25*inch))
    
    # Chart 2: Mass balance pie chart
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    filtered_yields = results["yields_percent"].iloc[[0, 1, 2]]
    ax1.pie(filtered_yields["Yield (%)"].values, labels=filtered_yields.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    plt.title("Mass Balance Distribution (%)")
    imgdata1 = BytesIO()
    fig1.savefig(imgdata1, format='png', dpi=300)
    imgdata1.seek(0)
    elements.append(ReportImage(imgdata1, width=3*inch, height=3*inch))
    elements.append(Spacer(1, 0.25*inch))
    
    # Chart 3: Gas composition bar chart
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    results["gas_composition_molar"].plot(kind='bar', ax=ax2, legend=False)
    plt.title("Dry Gas Composition (Molar %)")
    plt.ylabel("Molar %")
    plt.xticks(rotation=0)
    imgdata2 = BytesIO()
    fig2.savefig(imgdata2, format='png', dpi=300)
    imgdata2.seek(0)
    elements.append(ReportImage(imgdata2, width=4*inch, height=3.2*inch))
    
    plt.close('all')
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

if __name__ == "__main__":
    main()

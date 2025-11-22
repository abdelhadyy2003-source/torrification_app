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

# --- 2. Static UI Components (CSS) ---
GLOBAL_CSS = """
<style>
    /* Main Content Styling */
    .stApp { padding-top: 20px; }
    
    /* Custom Banner Style */
    .main-banner {
        background-color: #388E3C; 
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .main-banner h1 { color: #FFFFFF; margin: 0; font-size: 2.5em; }
    .main-banner p { color: #C8E6C9; margin-top: 5px; font-size: 1.1em; }
    
    /* Sidebar Customization */
    .st-emotion-cache-1na6f8g, .st-emotion-cache-1d391kg { 
        background-color: #F0F8FF;
    }
    /* Expander (Input) styling */
    .st-emotion-cache-p5m8m8 { 
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        padding: 10px;
        margin-bottom: 15px;
        background-color: #FFFFFF;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Metric Styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #388E3C;
    }
    /* Tycoon Scoreboard */
    .tycoon-metric-value {
        font-size: 32px !important;
        font-weight: bold;
        color: #FF4500 !important; /* Orange Red */
    }
</style>
"""

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

# --- 4. Tycoon Game Logic ---

def calculate_tycoon_profit(results):
    """Calculates the economic outcomes for the Tycoon Game."""
    params = results["parameters"]
    data = EMPIRICAL_DATA.get(params["biomass"])
    
    initial_mass_kg = params["initial_mass"]
    final_biochar_mass = results["yields_mass"].loc["Biochar (Solid) & Ash", "Mass (kg)"]
    
    # 1. Cost Calculations
    
    # Cost of Raw Material (Feedstock)
    cost_feedstock = initial_mass_kg * data["Feedstock_Cost"] # $/kg * kg
    
    # Operating Cost (Proportional to Mass, Temperature, and Duration)
    # Base cost: 0.01 $/kg. Increased by temperature and duration impact.
    temp_factor = params["temperature"] / 200 # Normalized to a base of 200°C
    time_factor = params["duration"] / 60   # Normalized to a base of 60 min
    cost_operating = initial_mass_kg * 0.01 * temp_factor * time_factor 
    
    total_costs = cost_feedstock + cost_operating
    
    # 2. Revenue Calculations
    
    # Estimate Energy Density Ratio (EDR) for quality assessment
    M_total = results["yields_percent"].loc["Biochar (Solid) & Ash", "Yield (%)"] / 100
    initial_moisture = params["moisture"] / 100
    initial_ash = data["Ash"]
    
    # Mass Yield on Dry-Ash-Free basis (M_daf)
    M_daf = (M_total - initial_ash) / (1 - initial_moisture - initial_ash) if (1 - initial_moisture - initial_ash) > 0.001 else 1.0
    
    # EDR (HHV_Product / HHV_Feed) ~ 1/M_daf (simplified)
    EDR = 1 / M_daf if M_daf > 0.001 else 1.0 
    
    # Selling Price is adjusted based on EDR (Quality)
    BASE_SELLING_PRICE = 0.15 # $/kg
    # Price multiplier: high EDR (e.g., 1.3) boosts price; low EDR (e.g., 1.1) reduces it slightly.
    price_multiplier = 1 + (EDR - 1.0) * 2.5 # Aggressive pricing curve
    
    selling_price = BASE_SELLING_PRICE * price_multiplier
    
    revenues = final_biochar_mass * selling_price
    
    # 3. Final Profit
    net_profit = revenues - total_costs
    
    return {
        "final_biochar_mass": final_biochar_mass,
        "cost_feedstock": cost_feedstock,
        "cost_operating": cost_operating,
        "total_costs": total_costs,
        "revenues": revenues,
        "net_profit": net_profit,
        "EDR": EDR,
        "selling_price": selling_price
    }

# --- 5. Main Streamlit App ---
def main():
    # Session state for Tycoon Game
    if 'capital' not in st.session_state:
        st.session_state.capital = 5000.0  # Initial capital
    if 'batch_count' not in st.session_state:
        st.session_state.batch_count = 0
        
    st.set_page_config(page_title="Chemisco Pro Torrefaction Simulator", layout="wide", initial_sidebar_state="expanded")
    
    # Inject Global CSS
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    # 5.1. Sidebar (Inputs)
    with st.sidebar:
        # Logo and Title
        st.markdown("""
            <div style='text-align: center; padding: 15px; border-radius: 8px; background-color: #1B5E20;'>
                <h1 style='color: white; margin: 0; font-size: 1.8em;'>CHEMISCO PRO</h1>
                <p style='color: #A5D6A7; margin: 0; font-size: 0.9em;'>Torrefaction Process Simulator</p>
            </div>
            """, unsafe_allow_html=True)
        st.header("⚙️ Input Parameters")
        
        # Input Sections
        with st.expander("🌲 Biomass Properties", expanded=True):
            initial_mass_kg = st.number_input("Initial Biomass Mass (kg)", min_value=1.0, value=100.0, step=10.0, help="Initial mass of the feedstock entering the process.")
            biomass_type = st.selectbox("Biomass Type", list(EMPIRICAL_DATA.keys()))
            moisture_content = st.slider("Initial Moisture Content (%)", 0.0, 50.0, 10.0, step=1.0, help="Moisture percentage on a wet basis.")
            particle_size = st.selectbox("Particle Size", list(SIZE_FACTOR.keys()))
        
        with st.expander("🌡️ Process Conditions", expanded=True):
            temperature = st.slider("Torrefaction Temperature (°C)", 200, 350, 275, step=5, help="Target operating temperature in the reactor.")
            duration = st.slider("Process Duration (min)", 10, 120, 45, step=5, help="Time spent in the torrefaction zone.")
            
            ash_percent = EMPIRICAL_DATA[biomass_type]["Ash"] * 100
            st.info(f"Assumed Initial Ash Content: **{ash_percent:.1f}%**")
            
    # 5.2. Main Content (Banner and Run Simulation)
    
    # Main Banner
    st.markdown("""
        <div class="main-banner">
            <h1>🔥 Advanced Torrefaction Simulator</h1>
            <p>Enhanced Kinetic Model for Process Optimization</p>
        </div>
        """, unsafe_allow_html=True)
    
    # --- Run Simulation ---
    if moisture_content / 100 + EMPIRICAL_DATA[biomass_type]["Ash"] > 1:
        st.error("**Input Error:** Initial Moisture and Ash content exceed 100%. Please adjust the parameters.")
        return 
        
    results = simulate_torrefaction(biomass_type, moisture_content, temperature, duration, particle_size, initial_mass_kg)
    
    # --- Display Results ---
    st.header("📊 Simulation Results & Analysis")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Yields & Mass Balance", "Mass Conversion Kinetics", "Gas Composition", "PDF Report", "💰 Torrefaction Tycoon"])
    
    # --- Tab 1: Yields & Mass Balance ---
    with tab1:
        st.subheader(f"Product Yields (Based on {initial_mass_kg:.0f} kg Input)")
        col_m1, col_m2, col_m3 = st.columns(3)
        
        biochar_mass_metric = results["yields_mass"].loc["Biochar (Solid) & Ash", "Mass (kg)"]
        col_m1.metric("⚖️ Total Solid Product (kg)", f"{biochar_mass_metric:.2f} kg", delta=f"{results['k_devol_eff']:.3f} min⁻¹ (Rate)")
        
        gas_mass_metric = results["yields_mass"].loc["Non-Condensable Gases", "Mass (kg)"]
        col_m2.metric("💨 Non-Condensable Gas Mass (kg)", f"{gas_mass_metric:.2f} kg")
        
        moisture_mass_metric = results["yields_mass"].loc["Moisture Loss (Water Vapor)", "Mass (kg)"]
        col_m3.metric("💧 Water Vapor Loss (kg)", f"{moisture_mass_metric:.2f} kg")

        st.markdown("---")
        
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

    # --- Tab 2 & 3 ---
    with tab2:
        st.subheader("Mass Component Conversion Over Time")
        st.line_chart(results["mass_profile"])
        st.caption("The curves show how Moisture and Volatiles fractions decrease as the Biochar fraction forms over time.")

    with tab3:
        st.subheader("Non-Condensable Dry Gas Composition")
        st.bar_chart(results["gas_composition_molar"])
        st.caption("Molar percentages of gaseous products from devolatilization (dry basis).")
        
    # --- Tab 4 (PDF Report) ---
    with tab4:
        st.subheader("Generate Comprehensive PDF Report")
        st.markdown("Click the button below to generate and download a detailed report of the simulation.")
        
        if st.button("⬇️ Download PDF Report"):
            pdf_buffer = generate_pdf_report(results)
            st.download_button(
                label="Download Report",
                data=pdf_buffer,
                file_name=f"Torrefaction_Report_{biomass_type}_{temperature}C.pdf",
                mime="application/pdf"
            )

    # --- Tab 5: Tycoon Game ---
    with tab5:
        st.title("💰 Torrefaction Tycoon: تحدي الربح")
        st.write("هدفك هو الوصول إلى أعلى ربح صافي ممكن عن طريق ضبط ظروف التفحيم (درجة الحرارة والمدة).")
        st.markdown("---")
        
        # Display Current Capital
        col_cap1, col_cap2 = st.columns([1, 2])
        col_cap1.markdown(f"## 🏦 رأس مالك الحالي:")
        col_cap1.markdown(f"<p class='tycoon-metric-value'>${st.session_state.capital:,.2f}</p>", unsafe_allow_html=True)
        col_cap2.info(f"**تنبيه:** يتم استخدام إعدادات المحاكاة الحالية ({initial_mass_kg:.0f} كجم من {biomass_type}) في دورة الإنتاج.")

        st.markdown("---")

        if st.button("▶️ تشغيل دورة الإنتاج الحالية", help="يشغل دورة إنتاج واحدة بالإعدادات المحددة في الشريط الجانبي"):
            
            # Check if capital is enough to cover feedstock cost
            feedstock_cost_check = initial_mass_kg * EMPIRICAL_DATA[biomass_type]["Feedstock_Cost"]
            if st.session_state.capital < feedstock_cost_check:
                st.error("⚠️ فشلت الدورة! رأس مالك الحالي غير كافٍ لشراء المواد الخام.")
            else:
                # Calculate Profit
                tycoon_results = calculate_tycoon_profit(results)
                
                # Update Capital and Batch Count
                st.session_state.capital += tycoon_results["net_profit"]
                st.session_state.batch_count += 1
                
                st.success(f"✅ تم تشغيل الدورة رقم {st.session_state.batch_count} بنجاح!")
                
                # Display Results Table
                st.markdown("### 📈 ملخص الدورة الإنتاجية")
                
                # Metrics
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("الإيرادات الكلية ($)", f"${tycoon_results['revenues']:,.2f}", delta_color="off")
                col_r2.metric("التكاليف الكلية ($)", f"${tycoon_results['total_costs']:,.2f}", delta_color="off")
                col_r3.metric("الربح الصافي ($)", f"${tycoon_results['net_profit']:,.2f}", delta=f"{tycoon_results['net_profit']/tycoon_results['total_costs'] * 100:.1f} % هامش ربح", delta_color="normal")
                
                st.markdown("##### تفاصيل الحسابات:")
                profit_df = pd.DataFrame({
                    "البند": ["تكلفة المواد الخام", "تكلفة التشغيل", "إجمالي التكاليف", "الإيرادات الكلية", "الربح الصافي"],
                    "القيمة ($)": [
                        tycoon_results["cost_feedstock"], 
                        tycoon_results["cost_operating"], 
                        tycoon_results["total_costs"], 
                        tycoon_results["revenues"], 
                        tycoon_results["net_profit"]
                    ]
                }).set_index("البند")
                st.dataframe(profit_df.style.format("${:,.2f}"), use_container_width=True)

                st.info(f"""
                    * **جودة المنتج (EDR):** {tycoon_results['EDR']:.3f} (كلما زادت القيمة، زادت كثافة الطاقة).
                    * **سعر البيع المعدل:** ${tycoon_results['selling_price']:.4f} / كجم.
                    * **الكتلة الحيوية النهائية:** {tycoon_results['final_biochar_mass']:.2f} كجم.
                """)

        if st.session_state.batch_count > 0:
            if st.button("🔄 إعادة تعيين اللعبة"):
                st.session_state.capital = 5000.0
                st.session_state.batch_count = 0
                st.rerun()

# --- 6. PDF Report Generation Function ---
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

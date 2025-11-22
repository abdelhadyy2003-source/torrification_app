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

# --- 1. الثوابت الكيميائية الحرارية ---
R_GAS = 8.314  # ثابت الغازات العام (J/mol·K)

# المعاملات التجريبية (A: عامل التردد [1/min], Ea: طاقة التنشيط [J/mol])
# k_drying_base: ثابت معدل التجفيف الأساسي عند 250°C (1/min)
# Gas_Factor: عامل قياس للغازات المتكونة
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

# عامل تصحيح لحجم الجسيمات (يؤثر على انتقال الحرارة)
SIZE_FACTOR = {
    "Fine (<1mm)": 1.0,
    "Medium (1-5mm)": 0.85,
    "Coarse (>5mm)": 0.65
}

# --- 2. دالة المحاكاة (simulate_torrefaction) ---
def simulate_torrefaction(biomass, moisture, temp_C, duration_min, size):
    """منطق محاكاة التفحيم الاحترافي باستخدام Arrhenius وتأثير حجم الجسيمات."""
    temp_K = temp_C + 273.15
    data = EMPIRICAL_DATA.get(biomass)
    
    # 1. حساب ثابت معدل التطاير (k_devol) باستخدام Arrhenius
    k_devol_arrhenius = data["A"] * np.exp(-data["Ea"] / (R_GAS * temp_K))
    
    # 2. تطبيق عامل تصحيح حجم الجسيمات
    size_correction = SIZE_FACTOR.get(size)
    k_devol_eff = k_devol_arrhenius * size_correction
    k_drying = data["k_drying_base"]
    ash_content = data["Ash"]

    # 3. حل المعادلات التفاضلية العادية (ODEs)
    def model(y, t, k1, k2):
        moisture, volatiles = y
        # يتم فقدان الرطوبة حتى تجف تماماً
        d_moisture = -k1 * moisture if moisture > 0.001 else 0
        # يتم فقدان المواد المتطايرة
        d_volatiles = -k2 * volatiles
        return [d_moisture, d_volatiles]
    
    t = np.linspace(0, duration_min, 100)
    # y0: [الرطوبة الابتدائية (كسر الكتلة), المتطايرات العضوية الابتدائية (كسر الكتلة)]
    initial_moisture_fraction = moisture / 100
    initial_volatiles_fraction = 1 - initial_moisture_fraction - ash_content
    y0 = [initial_moisture_fraction, initial_volatiles_fraction]
    
    if y0[1] < 0:
        y0[1] = 0
        # لا نعرض الخطأ هنا لتجنب التكرار في Streamlit، بل نعتمد على st.error في الواجهة
        
    sol = odeint(model, y0, t, args=(k_drying, k_devol_eff))
    
    # ضمان عدم وجود كسور سالبة بسبب أخطاء التكامل
    sol[sol < 0] = 0

    # 4. حساب تحول الكتلة بمرور الزمن
    final_moisture = sol[-1, 0]
    final_volatiles_remaining = sol[-1, 1]
    final_biochar_yield = (1 - final_moisture - final_volatiles_remaining - ash_content)
    final_volatiles_lost = initial_volatiles_fraction - final_volatiles_remaining
    moisture_lost = initial_moisture_fraction - final_moisture

    mass_profile = pd.DataFrame({
        "Time (min)": t,
        "Moisture Fraction": sol[:, 0],
        "Volatiles Fraction": sol[:, 1],
        "Biochar Fraction": 1 - sol[:, 0] - sol[:, 1] - ash_content,
    }).set_index("Time (min)")
    
    # 5. توزيع نواتج التحلل
    gas_fraction = final_volatiles_lost * data["Gas_Factor"]
    
    gas_comp = {
        "CO2": 0.45 * gas_fraction,
        "CO": 0.35 * gas_fraction,
        "CH4": 0.15 * gas_fraction,
        "H2": 0.05 * gas_fraction
    }
    
    # 6. تجهيز مخرجات النتائج
    yields = pd.DataFrame({
        "Yield (%)": [
            (final_biochar_yield + ash_content) * 100,
            final_volatiles_lost * 100,
            moisture_lost * 100,
            ash_content * 100
        ]},
        index=["Biochar (Solid) & Ash", "Non-Condensable Gases", "Moisture Loss (Water Vapor)", "Initial Ash Content"]
    )
    
    gas_composition = pd.DataFrame.from_dict(
        # تحويل ناتج الغاز إلى نسب مولارية مئوية (على أساس جاف)
        {k: v * 100 / final_volatiles_lost for k, v in gas_comp.items() if final_volatiles_lost > 0.001}, 
        orient="index", columns=["Molar % in Dry Gas"]
    ).fillna(0)

    return {
        "yields": yields,
        "temp_profile": pd.DataFrame({"Temperature (°C)": temp_C * np.ones_like(t)}, index=t),
        "gas_composition": gas_composition,
        "mass_profile": mass_profile,
        "k_devol_eff": k_devol_eff,
        "parameters": {
            "biomass": biomass, "moisture": moisture, "temperature": temp_C, 
            "duration": duration_min, "size": size
        }
    }

# --- 3. دالة واجهة المستخدم الرئيسية (main) ---
def main():
    st.set_page_config(page_title="Chemisco Pro Torrefaction Simulator", layout="wide")
    
    # 3.1. الشعار والبانر (استخدام HTML/Markdown للعرض الاحترافي)
    with st.sidebar:
        # الشعار في الشريط الجانبي (Placeholder - استبدله بـ st.image)
        st.markdown(
            """
            <div style='text-align: center; padding: 10px; background-color: #0E1117; border-radius: 5px;'>
                <h1 style='color: #4CAF50;'>CHEMISCO PRO</h1>
                <p style='color: #F0F2F6;'>Torrefaction Analytics</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.header("⚙️ معلمات التشغيل")
        
        # تجميع المدخلات في Sidebar
        with st.expander("مدخلات الكتلة الحيوية", expanded=True):
            biomass_type = st.selectbox("نوع الكتلة الحيوية", list(EMPIRICAL_DATA.keys()))
            moisture_content = st.slider("محتوى الرطوبة الأولي (%)", 0.0, 50.0, 10.0, step=1.0)
            particle_size = st.selectbox("حجم الجسيمات", list(SIZE_FACTOR.keys()))
        
        with st.expander("معلمات العملية", expanded=True):
            temperature = st.slider("درجة حرارة التفحيم (°C)", 200, 350, 275, step=5)
            duration = st.slider("مدة العملية (دقيقة)", 10, 120, 45, step=5)
            
            ash_percent = EMPIRICAL_DATA[biomass_type]["Ash"] * 100
            st.info(f"محتوى الرماد الأولي المُفترض: **{ash_percent:.1f}%**")
            
    # البانر (العنوان الرئيسي)
    st.markdown(
        """
        <div style='background-color: #4CAF50; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;'>
            <h1 style='color: white; margin: 0;'>🔥 محاكي التفحيم المتقدم</h1>
            <p style='color: white; margin: 0;'>نموذج حركي مُعزز لأفضل نتائج تحليلية</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # --- تشغيل المحاكاة ---
    results = simulate_torrefaction(biomass_type, moisture_content, temperature, duration, particle_size)
    
    # التحقق من المدخلات (تجنب الرماد والرطوبة > 100%)
    if results["parameters"]["moisture"] / 100 + EMPIRICAL_DATA[biomass_type]["Ash"] > 1:
        st.error("**خطأ في المدخلات:** مجموع الرطوبة الأولية ومحتوى الرماد يتجاوز 100%. يرجى خفض الرطوبة.")
        return # إيقاف العرض إذا كانت المدخلات غير صالحة

    # --- عرض النتائج الاحترافية في Tabs ---
    st.header("📊 مخرجات المحاكاة والتحليل")
    tab1, tab2, tab3, tab4 = st.tabs(["نتائج العائد (Yields)", "تحول الكتلة (Mass Conversion)", "تركيبة الغاز", "تقرير PDF"])
    
    with tab1:
        st.subheader("إجمالي نواتج العملية")
        col_m1, col_m2, col_m3 = st.columns(3)
        
        # عرض المقاييس (Metrics)
        biochar_yield_metric = results["yields"].loc["Biochar (Solid) & Ash", "Yield (%)"]
        col_m1.metric("⚖️ كسر المنتج الصلب (Biochar + Ash)", f"{biochar_yield_metric:.2f} %", delta=f"{results['k_devol_eff']:.3f} min⁻¹ (Rate)")
        
        gas_yield_metric = results["yields"].loc["Non-Condensable Gases", "Yield (%)"]
        col_m2.metric("💨 كسر الغازات", f"{gas_yield_metric:.2f} %")
        
        moisture_loss_metric = results["yields"].loc["Moisture Loss (Water Vapor)", "Yield (%)"]
        col_m3.metric("💧 بخار الماء المفقود", f"{moisture_loss_metric:.2f} %")

        st.markdown("---")
        
        # جدول ورسم بياني للعائد
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("جدول توزيع الكتلة")
            st.dataframe(results["yields"].style.format("{:.2f}"), use_container_width=True)
        
        with col_t2:
            st.subheader("مخطط الميزان الكتلي")
            fig1, ax1 = plt.subplots(figsize=(6, 6))
            # نأخذ العوائد الرئيسية الثلاثة للعرض
            filtered_yields = results["yields"].iloc[[0, 1, 2]] 
            ax1.pie(filtered_yields["Yield (%)"].values, labels=filtered_yields.index, autopct='%1.1f%%', startangle=90, colors=['#8B4513', '#A9A9A9', '#ADD8E6'])
            ax1.axis('equal')
            st.pyplot(fig1)
            

[Image of mass balance pie chart for torrefaction products]
 # صورة تمثيلية للمخطط الدائري

    with tab2:
        st.subheader("تتبع تحول مكونات الكتلة الحيوية عبر الزمن")
        st.line_chart(results["mass_profile"])
        st.caption("المنحنى يوضح كيف تتناقص كسور الرطوبة والمواد المتطايرة وتتشكل حصة الفحم الحيوي مع تقدم العملية.")

    with tab3:
        st.subheader("تركيبة الغازات غير القابلة للتكثف (على أساس جاف)")
        st.bar_chart(results["gas_composition"])
        st.caption("النسب المئوية المولارية للغازات الناتجة عن التحلل الحراري للمواد المتطايرة.")

    with tab4:
        st.subheader("إنشاء تقرير PDF شامل")
        st.markdown("يحتوي التقرير على جميع المدخلات والجداول والرسوم البيانية للمحاكاة.")
        
        if st.button("⬇️ تنزيل تقرير PDF"):
            pdf_buffer = generate_pdf_report(results)
            st.download_button(
                label="تنزيل التقرير",
                data=pdf_buffer,
                file_name=f"Torrefaction_Report_{biomass_type}_{temperature}C.pdf",
                mime="application/pdf"
            )

# --- 4. دالة إنشاء تقرير PDF مُحسَّنة (generate_pdf_report) ---
def generate_pdf_report(results):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        title="Torrefaction Report",
        leftMargin=inch, rightMargin=inch, topMargin=inch, bottomMargin=inch
    )
    styles = getSampleStyleSheet()
    elements = []
    
    # Header & Banner
    elements.append(Paragraph("<font size=16 color='#4CAF50'>CHEMISCO PRO TORREFACTION REPORT</font>", styles["Title"]))
    elements.append(Paragraph(f"تاريخ التقرير: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", styles["Italic"]))
    elements.append(Spacer(1, 0.25*inch))
    
    # 1. Parameters Table
    elements.append(Paragraph("1. Simulation Parameters & Kinetics", styles["h2"]))
    p = results["parameters"]
    param_data = [
        ["Parameter", "Value"],
        ["Biomass Type", p["biomass"]],
        ["Moisture Content", f"{p['moisture']}%"],
        ["Temperature", f"{p['temperature']} °C"],
        ["Duration", f"{p['duration']} min"],
        ["Particle Size", p["size"]],
        ["Effective Devol. Rate ($k_{devol,eff}$)", f"{results['k_devol_eff']:.3f} min⁻¹"],
    ]
    param_table = Table(param_data, colWidths=[2.5*inch, 3*inch], 
                        style=[('GRID', (0,0), (-1,-1), 1, colors.black)])
    elements.append(param_table)
    elements.append(Spacer(1, 0.25*inch))
    
    # 2. Yields Table
    elements.append(Paragraph("2. Product Yields (Mass Balance)", styles["h2"]))
    yield_data = [["Component", "Yield (%)"]] + \
                 [[idx, f"{val[0]:.2f}"] for idx, val in results["yields"].iterrows()]
    yield_table = Table(yield_data, colWidths=[3.5*inch, 2*inch],
                        style=[('GRID', (0,0), (-1,-1), 1, colors.black)])
    elements.append(yield_table)
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
    filtered_yields = results["yields"].iloc[[0, 1, 2]]
    ax1.pie(filtered_yields["Yield (%)"].values, labels=filtered_yields.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    plt.title("Mass Balance Distribution")
    imgdata1 = BytesIO()
    fig1.savefig(imgdata1, format='png', dpi=300)
    imgdata1.seek(0)
    elements.append(ReportImage(imgdata1, width=3*inch, height=3*inch))
    elements.append(Spacer(1, 0.25*inch))
    
    # Chart 3: Gas composition bar chart
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    results["gas_composition"].plot(kind='bar', ax=ax2, legend=False)
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

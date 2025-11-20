import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from reportlab.pdfgen import canvas
import graphviz

# --------- واجهة التطبيق ---------
st.set_page_config(page_title="SRF Production Simulator", layout="wide")
st.title("♻️ SRF Production Simulator")
st.write("تطبيق لحساب إنتاج SRF مع تحليل المكونات، الرطوبة، الحرارة، التصنيف، وتوليد تقارير PDF")

# --------- Sidebar Inputs ---------
st.sidebar.header("Input Waste Data")

waste_type = st.sidebar.selectbox("Type of Waste", ["Municipal", "Industrial", "Biomass"])
total_mass = st.sidebar.number_input("Total Waste Mass (kg)", 0.0, 10000.0, 0.0)

st.sidebar.subheader("🧪 Compositional Data (%)")
plastic = st.sidebar.number_input("Plastic", 0.0, 100.0, 0.0)
paper = st.sidebar.number_input("Paper & Cardboard", 0.0, 100.0, 0.0)
metals = st.sidebar.number_input("Metals", 0.0, 100.0, 0.0)
textiles = st.sidebar.number_input("Textiles", 0.0, 100.0, 0.0)
organic = st.sidebar.number_input("Organic Waste", 0.0, 100.0, 0.0)
inert = st.sidebar.number_input("Inert Materials", 0.0, 100.0, 0.0)
other = st.sidebar.number_input("Other Materials", 0.0, 100.0, 0.0)

total_composition = plastic + paper + metals + textiles + organic + inert + other
st.sidebar.write(f"Total: {total_composition:.2f}%")

st.sidebar.subheader("Initial Moisture Content (%)")
moisture = st.sidebar.slider("", 0.0, 100.0, 85.0)

st.sidebar.subheader("⚠ Contaminants (Optional)")
chlorine = st.sidebar.number_input("Chlorine (%)", 0.0, 10.0, 0.07)
mercury_median = st.sidebar.number_input("Mercury (Median) mg/MJ", 0.0, 1.0, 0.07)
mercury_80th = st.sidebar.number_input("Mercury (80th Percentile) mg/MJ", 0.0, 1.0, 0.06)

# --------- Tabs ---------
tab1, tab2, tab3, tab4 = st.tabs(["Waste Input", "Process Flow", "Production Results", "Reports"])

# --------- Tab 1: Waste Input ---------
with tab1:
    st.header("Input Waste Data Summary")
    st.write(f"**Waste Type:** {waste_type}")
    st.write(f"**Total Mass:** {total_mass} kg")
    comp_data = {
        "Plastic": plastic,
        "Paper & Cardboard": paper,
        "Metals": metals,
        "Textiles": textiles,
        "Organic": organic,
        "Inert": inert,
        "Other": other
    }
    comp_df = pd.DataFrame(list(comp_data.items()), columns=["Component", "Percentage"])
    st.dataframe(comp_df)

# --------- Tab 2: Process Flow ---------
with tab2:
    st.header("Process Flow Diagram")
    dot = graphviz.Digraph()
    dot.node('A', 'Presorting')
    dot.node('B', 'Primary Shredding')
    dot.node('C', 'Mechanical Separation')
    dot.node('D', 'Torrefaction / SRF Production')
    dot.edges(['AB', 'BC', 'CD'])
    st.graphviz_chart(dot)

# --------- Tab 3: Production Results ---------
with tab3:
    st.header("Production Results")

    # Simple simulation calculations
    final_mass = total_mass * (1 - moisture/100)  # crude estimate
    effective_moisture = max(0, moisture - 20)  # crude formula
    HHV = final_mass * 0.1  # dummy heating value in MJ/kg

    st.write(f"**Final SRF Mass:** {final_mass:.2f} kg")
    st.write(f"**Effective Moisture:** {effective_moisture:.2f} %")
    st.write(f"**Heating Value (HHV):** {HHV:.2f} MJ/kg")

    # Classification
    st.write("📈 SRF Quality Classification")
    st.write(f"NCV: {HHV:.2f} MJ/kg (Class 5)")
    st.write(f"Chlorine: {chlorine:.2f}% (Class 1)")
    st.write(f"Mercury: Median={mercury_median:.3f}, 80th={mercury_80th:.3f} (Class 3)")

# --------- Tab 4: Reports ---------
with tab4:
    st.header("Download PDF Report")

    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(50, 800, "SRF Production Report")
    c.drawString(50, 780, f"Waste Type: {waste_type}")
    c.drawString(50, 760, f"Total Mass: {total_mass} kg")
    c.drawString(50, 740, f"Components: {comp_data}")
    c.drawString(50, 720, f"Initial Moisture: {moisture}%")
    c.drawString(50, 700, f"Contaminants: Chlorine={chlorine}%, Mercury Median={mercury_median}, 80th={mercury_80th}")
    c.drawString(50, 680, f"Final SRF Mass: {final_mass:.2f} kg")
    c.drawString(50, 660, f"Effective Moisture: {effective_moisture:.2f}%")
    c.drawString(50, 640, f"Heating Value: {HHV:.2f} MJ/kg")
    c.showPage()
    c.save()
    st.download_button("Download PDF", data=buffer, file_name="SRF_report.pdf", mime="application/pdf")

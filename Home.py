import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from reportlab.pdfgen import canvas
import graphviz

# --------- إعداد الصفحة ---------
st.set_page_config(page_title="Torrification Simulator", layout="wide")
st.title("🌿 Torrification Simulator")
st.write("محاكي كامل لتوريفيكاشن: إدخال البيانات، محاكاة، Flow Sheet أفقي، رسومات، تقارير PDF/CSV")

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
st.sidebar.write(f"**Total:** {total_composition:.2f}%")

st.sidebar.subheader("Initial Moisture Content (%)")
moisture = st.sidebar.slider("", 0.0, 100.0, 85.0)

st.sidebar.subheader("⚠ Contaminants (Optional)")
chlorine = st.sidebar.number_input("Chlorine (%)", 0.0, 10.0, 0.07)
mercury_median = st.sidebar.number_input("Mercury Median (mg/MJ)", 0.0, 1.0, 0.07)
mercury_80th = st.sidebar.number_input("Mercury 80th Percentile (mg/MJ)", 0.0, 1.0, 0.06)

# --------- محاكاة الحسابات ---------
# Final Mass و Moisture تقريبي
final_mass = total_mass * (1 - moisture/100)
effective_moisture = max(0, moisture - 20)
HHV = final_mass * 0.1  # مثال dummy heating value

# --------- الرسومات ---------
st.subheader("📊 Waste Composition")
comp_data = {"Plastic": plastic, "Paper": paper, "Metals": metals,
             "Textiles": textiles, "Organic": organic, "Inert": inert, "Other": other}
comp_df = pd.DataFrame(list(comp_data.items()), columns=["Component", "Percentage"])
fig_pie = px.pie(comp_df, names='Component', values='Percentage', title="Waste Composition")
st.plotly_chart(fig_pie, use_container_width=True)

st.subheader("📈 Mass vs Moisture Simulation")
time = np.linspace(0, 5, 100)
moisture_profile = moisture * np.exp(-0.05*time)
fig_line = px.line(x=time, y=moisture_profile, labels={"x":"Time (hours)", "y":"Moisture (%)"}, title="Moisture over Time")
st.plotly_chart(fig_line, use_container_width=True)

# --------- Flow Sheet أفقي ---------
st.subheader("⚙ Process Flow (Horizontal)")
dot = graphviz.Digraph()
dot.attr(rankdir='LR', size='10')  # Horizontal
dot.node('A', f'Raw Material\nMass={total_mass}kg')
dot.node('B', f'Presorting\nMass={total_mass*0.9:.1f}kg')
dot.node('C', f'Shredding\nMass={total_mass*0.8:.1f}kg')
dot.node('D', f'Torrefaction\nMass={final_mass:.1f}kg\nMoisture={effective_moisture:.1f}%')
dot.node('E', 'Final Product')
dot.edges(['AB', 'BC', 'CD', 'DE'])
st.graphviz_chart(dot)

# --------- Production Results ---------
st.subheader("📦 Production Results")
st.write(f"**Final SRF Mass:** {final_mass:.2f} kg")
st.write(f"**Effective Moisture:** {effective_moisture:.2f}%")
st.write(f"**Heating Value (HHV):** {HHV:.2f} MJ/kg")
st.write("📈 SRF Quality Classification")
st.write(f"NCV: {HHV:.2f} MJ/kg (Class 5)")
st.write(f"Chlorine: {chlorine:.2f}% (Class 1)")
st.write(f"Mercury: Median={mercury_median:.3f}, 80th={mercury_80th:.3f} (Class 3)")

# --------- PDF و CSV ---------
st.subheader("📄 Download Reports")
csv = pd.DataFrame({"Component": comp_df['Component'], "Percentage": comp_df['Percentage']}).to_csv(index=False).encode()
st.download_button("Download CSV", data=csv, file_name="SRF_results.csv", mime="text/csv")

# PDF
buffer = BytesIO()
c = canvas.Canvas(buffer)
c.drawString(50, 800, "SRF Production Report")
c.drawString(50, 780, f"Waste Type: {waste_type}")
c.drawString(50, 760, f"Total Mass: {total_mass} kg")
c.drawString(50, 740, f"Components: {comp_data}")
c.drawString(50, 720, f"Initial Moisture: {moisture}%")
c.drawString(50, 700, f"Contaminants: Cl={chlorine}%, Mercury Median={mercury_median}, 80th={mercury_80th}")
c.drawString(50, 680, f"Final SRF Mass: {final_mass:.2f} kg")
c.drawString(50, 660, f"Effective Moisture: {effective_moisture:.2f}%")
c.drawString(50, 640, f"Heating Value: {HHV:.2f} MJ/kg")
c.showPage()
c.save()
st.download_button("Download PDF", data=buffer, file_name="SRF_report.pdf", mime="application/pdf")

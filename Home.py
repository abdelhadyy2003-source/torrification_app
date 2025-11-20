# ===== Imports =====
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF
import io

# ===== Simulation Function =====
def simulate_torrefaction(waste_type, mass, moisture, temp, residence_time):
    water_loss = mass * moisture / 100 * (1 - np.exp(-0.5 * residence_time))
    volatile_fraction = 0.3 + 0.1 * (temp - 200) / 100
    volatile_loss = (mass - water_loss) * volatile_fraction
    ash_fraction = 0.05
    ash_mass = mass * ash_fraction
    biochar_mass = mass - water_loss - volatile_loss - ash_mass
    fixed_carbon = biochar_mass * 0.8
    return {
        'Biochar (kg)': biochar_mass,
        'Gas & Volatiles (kg)': volatile_loss,
        'Ash (kg)': ash_mass,
        'Fixed Carbon (kg)': fixed_carbon,
        'Water Loss (kg)': water_loss
    }

# ===== Cost Analysis =====
def calculate_costs(mass, processing_cost_per_kg):
    return mass * processing_cost_per_kg

# ===== PDF Generation =====
def create_pdf_report(simulation_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "🔥 Torrefaction Simulation Report 🔥", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    for key, value in simulation_data.items():
        pdf.cell(0, 10, f"{key}: {value:.2f}", ln=True)
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer

# ===== Streamlit App Setup =====
st.set_page_config(page_title="Torrefaction Simulator", layout="wide")
st.title("🔥 Torrefaction Simulator - Single Page")

# ===== Session State =====
if "simulations" not in st.session_state:
    st.session_state.simulations = []

# ===== Input Section =====
st.subheader("Input Parameters")
col1, col2 = st.columns([1,1])
with col1:
    waste_type = st.selectbox("Waste Type", ['Municipal', 'Wood', 'Agricultural', 'Plastic'])
    mass = st.slider("Mass (kg)", 1.0, 100.0, 10.0)
    moisture = st.slider("Moisture (%)", 0.0, 100.0, 20.0)
with col2:
    temp = st.slider("Temperature (°C)", 200, 300, 250)
    residence_time = st.slider("Residence Time (hr)", 0.1, 5.0, 1.0)
    processing_cost_per_kg = st.number_input("Processing Cost per kg ($)", 0.1, 10.0, 1.0)

if st.button("Run Simulation"):
    results = simulate_torrefaction(waste_type, mass, moisture, temp, residence_time)
    total_cost = calculate_costs(mass, processing_cost_per_kg)
    results['Total Cost ($)'] = total_cost
    sim_entry = {
        "Waste Type": waste_type,
        "Mass": mass,
        "Moisture": moisture,
        "Temperature": temp,
        "Residence Time": residence_time,
        **results
    }
    st.session_state.simulations.append(sim_entry)
    st.success("Simulation added successfully!")

# ===== Simulation Results =====
if st.session_state.simulations:
    st.subheader("Latest Simulation Results")
    latest = st.session_state.simulations[-1]
    cols = st.columns(5)
    metric_keys = ['Biochar (kg)', 'Gas & Volatiles (kg)', 'Ash (kg)', 'Fixed Carbon (kg)', 'Total Cost ($)']
    for col, key in zip(cols, metric_keys):
        col.metric(key, f"{latest[key]:.2f}")

# ===== Charts Section =====
if st.session_state.simulations:
    st.subheader("Charts")
    df = pd.DataFrame(st.session_state.simulations)
    keys = ['Biochar (kg)', 'Gas & Volatiles (kg)', 'Ash (kg)', 'Fixed Carbon (kg)', 'Water Loss (kg)']
    fig_pie = go.Figure(data=[go.Pie(labels=keys, values=[df.iloc[-1][k] for k in keys])])
    fig_pie.update_layout(title="Product Distribution (Last Simulation)")
    st.plotly_chart(fig_pie, use_container_width=True)
    
    st.bar_chart(df[['Biochar (kg)', 'Gas & Volatiles (kg)', 'Total Cost ($)']])

# ===== Flow Sheet Section =====
if st.session_state.simulations:
    st.subheader("Torrefaction Process Flow Sheet")
    labels = ["Input Waste", "Water Loss", "Gas & Volatiles", "Ash", "Biochar"]
    node_colors = ['#8B4513','#1E90FF','#FFA500','#808080','#2E8B57']
    sources, targets, values, link_colors = [], [], [], []
    for sim_index, sim in enumerate(st.session_state.simulations):
        sources.extend([0,0,0,0])
        targets.extend([1,2,3,4])
        values.extend([sim['Water Loss (kg)'], sim['Gas & Volatiles (kg)'], sim['Ash (kg)'], sim['Biochar (kg)']])
        link_colors.extend(node_colors)
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=15, thickness=20, color=node_colors),
        link=dict(source=sources, target=targets, value=values, color=link_colors)
    )])
    fig_sankey.update_layout(title_text="Flow Sheet (All Simulations)", font_size=12)
    st.plotly_chart(fig_sankey, use_container_width=True)

# ===== PDF Reports Section =====
if st.session_state.simulations:
    st.subheader("Download PDF Reports")
    for i, sim in enumerate(st.session_state.simulations):
        st.markdown(f"**Simulation #{i + 1}: {sim['Waste Type']}**")
        pdf_key = f"pdf_{i}"
        pdf_file = create_pdf_report(sim)
        st.download_button("Download PDF", data=pdf_file, file_name=f"Torrefaction_Report_{i+1}.pdf", mime="application/pdf", key=pdf_key)

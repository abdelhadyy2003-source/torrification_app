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
    total_cost = mass * processing_cost_per_kg
    return total_cost

# ===== PDF Generation =====
def create_pdf_report(simulation_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "🔥 Torrefaction Simulation Report 🔥", ln=True, align="C")
    
    pdf.set_font("Arial", "", 12)
    for key, value in simulation_data.items():
        pdf.cell(0, 10, f"{key}: {value:.2f}", ln=True)
    
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer

# ===== Streamlit App =====
st.set_page_config(page_title="Torrefaction Simulator", layout="wide")
st.title("🔥 Torrefaction Simulator with Comparison 🔥")

# ===== Session State =====
if "simulations" not in st.session_state:
    st.session_state.simulations = []

# Sidebar Inputs
st.sidebar.header("Input Parameters")
waste_type = st.sidebar.selectbox("Waste Type", ['Municipal', 'Wood', 'Agricultural', 'Plastic'])
mass = st.sidebar.slider("Mass (kg)", 1.0, 100.0, 10.0)
moisture = st.sidebar.slider("Moisture (%)", 0.0, 100.0, 20.0)
temp = st.sidebar.slider("Temperature (°C)", 200, 300, 250)
residence_time = st.sidebar.slider("Residence Time (hr)", 0.1, 5.0, 1.0)
processing_cost_per_kg = st.sidebar.number_input("Processing Cost per kg ($)", 0.1, 10.0, 1.0)

if st.sidebar.button("Run Simulation"):
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

# ===== Comparisons Tab =====
if st.sidebar.button("Show Comparisons"):
    if st.session_state.simulations:
        df = pd.DataFrame(st.session_state.simulations)
        st.dataframe(df)
        st.bar_chart(df[['Biochar (kg)', 'Gas & Volatiles (kg)', 'Total Cost ($)']])
    else:
        st.info("No simulations run yet.")

# ===== PDF Reports Tab =====
if st.sidebar.button("Download Reports"):
    if st.session_state.simulations:
        for i, sim in enumerate(st.session_state.simulations):
            st.markdown(f"**Simulation #{i + 1}: {sim['Waste Type']}**")
            pdf_file = create_pdf_report(sim)
            st.download_button("Download PDF", data=pdf_file, file_name=f"Report_{i + 1}.pdf", mime="application/pdf")
    else:
        st.info("Run some simulations to download reports.")

# ===== Imports =====
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF
import io

# ===== CSS for Background and Company Name =====
st.markdown(
    """
    <style>
    /* Background Image */
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1605902711622-cfb43c4430d6?auto=format&fit=crop&w=1950&q=80');
        background-size: cover;
        background-attachment: fixed;
        color: white;
    }
    /* Centered Company Name */
    .company-name {
        text-align: center;
        font-size: 50px;
        font-weight: bold;
        color: #FFD700;
        margin-bottom: 20px;
        text-shadow: 2px 2px #000000;
    }
    /* Headers */
    h1, h2, h3, h4 {
        color: #FFFFFF;
        text-shadow: 1px 1px #000000;
    }
    </style>
    """, unsafe_allow_html=True
)

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
st.set_page_config(page_title="Chemisco Torrefaction Simulator", layout="wide")

# ===== Company Name =====
st.markdown('<div class="company-name">Chemisco</div>', unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center;'>🔥 Torrefaction Simulator 🔥</h2>", unsafe_allow_html=True)

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
    colors = ['#2E8B57', '#1E90FF', '#FFA500', '#808080', '#8B4513']
    for col, key, color in zip(cols, metric_keys, colors):
        col.metric(label=key, value=f"{latest[key]:.2f}", delta_color="normal")

# ===== Charts, Block Diagram, Flow Sheet, PDF Reports =====
# (Add the same code for Charts, Block Flow Diagram, Flow Sheet, and PDF Reports as قبل)
# مع الاحتفاظ بنفس التحسينات للألوان والخطوط

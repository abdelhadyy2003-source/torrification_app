# ===== Imports =====
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF
import io
import plotly.io as pio

# ===== Simulation Function =====
def simulate_torrefaction(waste_type, mass, moisture, temp, residence_time):
    water_loss = mass * moisture/100 * (1 - np.exp(-0.5*residence_time))
    volatile_fraction = 0.3 + 0.1*(temp-200)/100
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

# ===== PDF Function without write_image =====
def create_pdf_with_charts(results, waste_type, mass, moisture, temp, residence_time):
    keys = ["Biochar (kg)","Gas & Volatiles (kg)","Ash (kg)","Fixed Carbon (kg)","Water Loss (kg)"]
    
    # Pie Chart in memory
    fig_pie = go.Figure(data=[go.Pie(labels=keys, 
                                     values=[results[k] for k in keys],
                                     marker=dict(colors=['#2E8B57','#FFA500','#808080','#1E90FF','#654321']))])
    fig_pie.update_layout(title="Torrefaction Product Distribution")
    pie_bytes = pio.to_image(fig_pie, format='png')
    
    # Line Chart in memory
    total_mass = sum([results[k] for k in keys])
    time = np.linspace(0,1,100)
    mass_curve = total_mass * (1 - time*(1-0.7))
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=time, y=mass_curve, mode='lines', line=dict(color='red', width=2)))
    fig_line.update_layout(title="Mass Loss Over Time", xaxis_title="Normalized Time", yaxis_title="Mass (kg)")
    line_bytes = pio.to_image(fig_line, format='png')
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0,10,"🔥 Torrefaction Simulation Report 🔥", ln=True, align="C")
    
    pdf.set_font("Arial", "", 12)
    pdf.ln(5)
    pdf.cell(0,10,f"Waste Type: {waste_type}", ln=True)
    pdf.cell(0,10,f"Mass (kg): {mass}", ln=True)
    pdf.cell(0,10,f"Moisture (%): {moisture}", ln=True)
    pdf.cell(0,10,f"Temperature (°C): {temp}", ln=True)
    pdf.cell(0,10,f"Residence Time (hr): {residence_time}", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0,10,"Simulation Results:", ln=True)
    pdf.set_font("Arial", "", 12)
    for k in keys:
        pdf.cell(0,10,f"{k}: {results[k]:.2f} kg", ln=True)
    
    pdf.ln(5)
    pdf.image(io.BytesIO(pie_bytes), x=30, w=150)
    pdf.ln(80)
    pdf.image(io.BytesIO(line_bytes), x=30, w=150)
    
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer

# ===== Streamlit App =====
st.set_page_config(page_title="Torrefaction Simulator", layout="wide")
st.markdown("<style>body {font-family: 'Roboto', sans-serif;}</style>", unsafe_allow_html=True)
st.title("🔥 Torrefaction Simulator with Comparison 🔥")

# ===== Session State for Comparisons =====
if "simulations" not in st.session_state:
    st.session_state.simulations = []

# Sidebar Inputs
st.sidebar.header("Input Parameters")
waste_type = st.sidebar.selectbox("Waste Type", ['Municipal', 'Wood', 'Agricultural', 'Plastic'])
mass = st.sidebar.slider("Mass (kg)", 1.0, 100.0, 10.0)
moisture = st.sidebar.slider("Moisture (%)", 0.0, 100.0, 20.0)
temp = st.sidebar.slider("Temperature (°C)", 200, 300, 250)
residence_time = st.sidebar.slider("Residence Time (hr)", 0.1, 5.0, 1.0)

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Run Simulation"):
        results = simulate_torrefaction(waste_type, mass, moisture, temp, residence_time)
        sim_entry = {
            "Waste Type": waste_type,
            "Mass": mass,
            "Moisture": moisture,
            "Temperature": temp,
            "Residence Time": residence_time,
        }
        sim_entry.update(results)
        st.session_state.simulations.append(sim_entry)
with col2:
    if st.button("Reset All"):
        st.session_state.simulations = []
        st.experimental_rerun()

# ===== Tabs =====
tabs = st.tabs(["Simulations Summary", "Charts", "Flow Sheet", "Download PDFs"])

# ===== Simulations Summary Tab =====
with tabs[0]:
    st.subheader("All Simulation Results")
    if st.session_state.simulations:
        df = pd.DataFrame(st.session_state.simulations)
        st.dataframe(df)
    else:
        st.info("Run a simulation to see results.")

# ===== Charts Tab =====
with tabs[1]:
    st.subheader("Comparison Charts")
    if st.session_state.simulations:
        last_sim = st.session_state.simulations[-1]
        keys = ["Biochar (kg)","Gas & Volatiles (kg)","Ash (kg)","Fixed Carbon (kg)","Water Loss (kg)"]

        # Pie Chart of last simulation
        fig_pie = go.Figure(data=[go.Pie(labels=keys,
                                         values=[last_sim[k] for k in keys],
                                         marker=dict(colors=['#2E8B57','#FFA500','#808080','#1E90FF','#654321']))])
        fig_pie.update_layout(title="Last Simulation Product Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

        # Line Chart comparing Biochar across simulations
        fig_line = go.Figure()
        for i, sim in enumerate(st.session_state.simulations):
            fig_line.add_trace(go.Scatter(x=[sim["Residence Time"]], y=[sim["Biochar (kg)"]],
                                          mode='markers+lines', name=f"{sim['Waste Type']} #{i+1}"))
        fig_line.update_layout(title="Biochar Production vs Residence Time", xaxis_title="Residence Time (hr)", yaxis_title="Biochar (kg)")
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Run a simulation to see charts.")

# ===== Flow Sheet Tab =====
with tabs[2]:
    st.subheader("Torrefaction Process Flow Sheet (All Simulations)")
    if st.session_state.simulations:
        labels = ["Input Waste", "Water Loss", "Gas & Volatiles", "Ash", "Biochar"]
        node_colors = ['#8B4513','#1E90FF','#FFA500','#808080','#2E8B57']

        sources = []
        targets = []
        values = []
        link_colors = []

        for sim_index, sim in enumerate(st.session_state.simulations):
            sources.extend([0,0,0,0])
            targets.extend([1,2,3,4])
            values.extend([sim['Water Loss (kg)'], sim['Gas & Volatiles (kg)'], sim['Ash (kg)'], sim['Biochar (kg)']])
            link_colors.extend(node_colors)

        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(label=labels, pad=15, thickness=20, color=node_colors),
            link=dict(source=sources, target=targets, value=values, color=link_colors)
        )])
        fig_sankey.update_layout(title_text="Torrefaction Process Flow Sheet (All Simulations)", font_size=12)
        st.plotly_chart(fig_sankey, use_container_width=True)
    else:
        st.info("Run a simulation to see the process flow sheet.")

# ===== Download PDFs Tab =====
with tabs[3]:
    st.subheader("Download PDF Reports")
    if st.session_state.simulations:
        keys = ["Biochar (kg)","Gas & Volatiles (kg)","Ash (kg)","Fixed Carbon (kg)","Water Loss (kg)"]
        for i, sim in enumerate(st.session_state.simulations):
            st.markdown(f"**Simulation #{i+1}: {sim['Waste Type']}**")
            if st.button(f"📄 Download PDF #{i+1}", key=f"pdf_{i}"):
                pdf_file = create_pdf_with_charts(
                    {k: sim[k] for k in keys},
                    sim["Waste Type"], sim["Mass"], sim["Moisture"], sim["Temperature"], sim["Residence Time"]
                )
                st.download_button("Download PDF", data=pdf_file, file_name=f"Torrefaction_Report_{i+1}.pdf", mime="application/pdf")
    else:
        st.info("Run a simulation to enable PDF downloads.")

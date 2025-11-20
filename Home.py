# ==============================
# Chemisco Ultra-Professional Torrefaction Simulator
# ==============================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF
import io

# ==============================
# CSS Styling (Ultra-Professional)
# ==============================
st.markdown("""
<style>
.stApp {
    background-image: url('https://images.unsplash.com/photo-1599058917216-52c6cd19f2d1?auto=format&fit=crop&w=1950&q=80');
    background-size: cover;
    background-attachment: fixed;
    color: white;
    font-family: 'Helvetica', sans-serif;
}
.company-name {
    text-align: center;
    font-size: 72px;
    font-weight: bold;
    color: #FFD700;
    margin-bottom: 15px;
    text-shadow: 3px 3px #000000;
}
h1, h2, h3, h4 {
    color: #FFFFFF;
    text-shadow: 2px 2px #000000;
}
.stButton>button {
    background-color: #2E8B57;
    color: white;
    font-size: 16px;
    border-radius: 12px;
    padding: 10px 25px;
    margin-top: 10px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #3CB371;
    transform: scale(1.08);
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Simulation Functions
# ==============================
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

def calculate_costs(mass, processing_cost_per_kg):
    return mass * processing_cost_per_kg

# ==============================
# PDF Report Generation
# ==============================
def create_pdf_report(sim_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, "Chemisco Torrefaction Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 13)
    for key, value in sim_data.items():
        if isinstance(value, (int,float)):
            pdf.cell(0,10,f"{key}: {value:.2f}", ln=True)
        else:
            pdf.cell(0,10,f"{key}: {value}", ln=True)
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer

# ==============================
# Streamlit Page Setup
# ==============================
st.set_page_config(page_title="Chemisco Torrefaction Simulator", layout="wide")
st.markdown('<div class="company-name">Chemisco</div>', unsafe_allow_html=True)
st.markdown("<h2 style='text-align:center;'>🔥 Ultra-Professional Torrefaction Simulator 🔥</h2>", unsafe_allow_html=True)

if "simulations" not in st.session_state:
    st.session_state.simulations = []

# ==============================
# Centered Input Section
# ==============================
st.subheader("Input Parameters")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    waste_type = st.selectbox("Waste Type", ['Municipal', 'Wood', 'Agricultural', 'Plastic'])
    mass = st.slider("Mass (kg)", 1.0, 100.0, 10.0)
    moisture = st.slider("Moisture (%)", 0.0, 100.0, 20.0)
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
        st.success("✅ Simulation added successfully!")

# ==============================
# Latest Metrics
# ==============================
if st.session_state.simulations:
    st.subheader("Latest Simulation Metrics")
    latest = st.session_state.simulations[-1]
    metric_keys = ['Biochar (kg)','Gas & Volatiles (kg)','Ash (kg)','Fixed Carbon (kg)','Total Cost ($)']
    colors = ['#2E8B57','#1E90FF','#FFA500','#808080','#8B4513']
    cols = st.columns(5)
    for col, key, color in zip(cols, metric_keys, colors):
        col.metric(label=key, value=f"{latest[key]:.2f}", delta_color="normal")

# ==============================
# Charts Section
# ==============================
if st.session_state.simulations:
    st.subheader("Interactive Charts")
    df = pd.DataFrame(st.session_state.simulations)
    keys = ['Biochar (kg)','Gas & Volatiles (kg)','Ash (kg)','Fixed Carbon (kg)','Water Loss (kg)']

    # Pie
    fig_pie = go.Figure(data=[go.Pie(
        labels=keys,
        values=[df.iloc[-1][k] for k in keys],
        marker=dict(colors=colors),
        hoverinfo='label+percent+value',
        hole=0.3
    )])
    fig_pie.update_layout(title="Last Simulation Product Distribution", title_font_size=20)
    st.plotly_chart(fig_pie, use_container_width=True)

    # Bar
    fig_bar = go.Figure(data=[go.Bar(
        x=keys,
        y=[df.iloc[-1][k] for k in keys],
        marker_color=colors,
        text=[f"{v:.2f}" for v in [df.iloc[-1][k] for k in keys]],
        textposition='auto'
    )])
    fig_bar.update_layout(title="Bar Chart of Last Simulation", title_font_size=20)
    st.plotly_chart(fig_bar, use_container_width=True)

# ==============================
# Block Flow Diagram
# ==============================
st.subheader("Block Flow Diagram - Torrefaction Process")
fig_block = go.Figure()
blocks = [
    {"name":"Input Waste","x0":0,"x1":2,"y0":2,"y1":3,"color":"#8B4513"},
    {"name":"Drying","x0":3,"x1":5,"y0":2,"y1":3,"color":"#1E90FF"},
    {"name":"Torrefaction","x0":6,"x1":8,"y0":2,"y1":3,"color":"#FFA500"},
    {"name":"Products","x0":9,"x1":11,"y0":2,"y1":3,"color":"#2E8B57"}
]
for block in blocks:
    fig_block.add_shape(type="rect", x0=block["x0"], x1=block["x1"], y0=block["y0"], y1=block["y1"],
                        line=dict(color="black", width=2), fillcolor=block["color"], layer="below")
    fig_block.add_annotation(
        x=(block["x0"]+block["x1"])/2,
        y=(block["y0"]+block["y1"])/2,
        text=f"<b>{block['name']}</b>",
        showarrow=False,
        font=dict(color="white", size=16)
    )
arrows = [(2,2.5,3,2.5),(5,2.5,6,2.5),(8,2.5,9,2.5)]
for x0,y0,x1,y1 in arrows:
    fig_block.add_annotation(x=x1, y=y1, ax=x0, ay=y0,
                             xref="x", yref="y", axref="x", ayref="y",
                             showarrow=True, arrowhead=3, arrowsize=2, arrowwidth=3, arrowcolor="#333333")
fig_block.update_xaxes(range=[-1,12], showticklabels=False, showgrid=False, zeroline=False)
fig_block.update_yaxes(range=[1,4], showticklabels=False, showgrid=False, zeroline=False)
fig_block.update_layout(height=350, margin=dict(l=20,r=20,t=20,b=20), paper_bgcolor="#F5F5F5")
st.plotly_chart(fig_block, use_container_width=True)

# ==============================
# Flow Sheet Sankey
# ==============================
if st.session_state.simulations:
    st.subheader("Torrefaction Process Flow Sheet")
    labels = ["Input Waste","Water Loss","Gas & Volatiles","Ash","Biochar"]
    node_colors = ['#8B4513','#1E90FF','#FFA500','#808080','#2E8B57']
    sources, targets, values, link_colors = [], [], [], []
    for sim in st.session_state.simulations:
        sources.extend([0,0,0,0])
        targets.extend([1,2,3,4])
        values.extend([sim['Water Loss (kg)'],sim['Gas & Volatiles (kg)'],sim['Ash (kg)'],sim['Biochar (kg)']])
        link_colors.extend(node_colors)
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(label=labels,pad=15,thickness=20,color=node_colors),
        link=dict(source=sources,target=targets,value=values,color=link_colors)
    )])
    fig_sankey.update_layout(title_text="Flow Sheet (All Simulations)", font_size=12)
    st.plotly_chart(fig_sankey, use_container_width=True)

# ==============================
# PDF Download
# ==============================
if st.session_state.simulations:
    st.subheader("Download PDF Reports")
    for i, sim in enumerate(st.session_state.simulations):
        pdf_file = create_pdf_report(sim)
        st.download_button("Download PDF", data=pdf_file,
                           file_name=f"Torrefaction_Report_{i+1}.pdf",
                           mime="application/pdf",
                           key=f"pdf_{i}")

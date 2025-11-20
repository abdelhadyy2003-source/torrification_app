import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import tempfile, io, os, glob
import base64

# ----- CONFIG -----
st.set_page_config(page_title="Chemisco - Torrefaction", layout="wide", initial_sidebar_state="collapsed")

# ----- SESSION STATE INIT -----
if 'simulations' not in st.session_state:
    st.session_state.simulations = []

# --- Utility: find uploaded image ---
def find_first_file(containing):
    candidates = glob.glob("/mnt/data/*")
    for c in candidates:
        if containing.lower() in os.path.basename(c).lower():
            return c
    return ""

# --- Image Handling ---
HERO_COVER = find_first_file("cover") or ""
BANNER_COVER = find_first_file("banner") or HERO_COVER
LOGO_PATH = find_first_file("logo") or ""

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

HERO_B64 = img_to_base64(HERO_COVER) if HERO_COVER else ""
BANNER_B64 = img_to_base64(BANNER_COVER) if BANNER_COVER else ""

# CSS for Hero & Banner
hero_css = f"""
<style>
.hero {{
  {"background-image: url('data:image/png;base64," + HERO_B64 + "');" if HERO_B64 else ""}
  background-size: cover;
  background-position: center;
  height: 40vh;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  text-shadow: 2px 2px #000;
}}
.hero h1 {{ font-size: 48px; margin:0; color:#FFD700; }}

.banner {{
  {"background-image: url('data:image/png;base64," + BANNER_B64 + "');" if BANNER_B64 else ""}
  background-size: cover;
  background-position: center;
  height: 12vh;
  display:flex;
  align-items:center;
  padding-left:2rem;
  color:#fff;
  text-shadow:1px 1px #000;
  border-radius:8px;
  margin-top:1rem;
  margin-bottom:1rem;
}}

.glass {{
    background: rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 18px;
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.12);
}}
</style>
"""
st.markdown(hero_css, unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Chemisco Pro — Advanced Torrefaction</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="banner"><h3>Torrefaction Simulator — Realistic process & analytics</h3></div>', unsafe_allow_html=True)

# ---------- Torrefaction Simulation ----------
def simulate_torrefaction(waste_type, mass, moisture, temp, residence_time):
    water_loss = mass * (moisture / 100.0) * (1.0 - np.exp(-0.6 * residence_time))
    volatile_fraction = np.clip(0.30 + 0.12 * ((temp - 200.0) / 100.0), 0.0, 0.9)
    volatile_loss = max(0.0, (mass - water_loss) * volatile_fraction)
    ash_mass = mass * 0.05
    biochar_mass = max(0.0, mass - water_loss - volatile_loss - ash_mass)
    fixed_carbon = biochar_mass * 0.78
    return {
        'Biochar (kg)': biochar_mass,
        'Gas & Volatiles (kg)': volatile_loss,
        'Ash (kg)': ash_mass,
        'Fixed Carbon (kg)': fixed_carbon,
        'Water Loss (kg)': water_loss
    }

# ---------- Matplotlib Charts ----------
def _make_matplotlib_charts(sim):
    keys = ['Biochar (kg)', 'Gas & Volatiles (kg)', 'Ash (kg)', 'Fixed Carbon (kg)', 'Water Loss (kg)']
    values = [sim.get(k, 0.0) for k in keys]
    colors_list = ['#2E8B57', '#1E90FF', '#FFA500', '#808080', '#8B4513']

    pie_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    if sum(values) == 0: values = [1e-6] * len(values)
    ax1.pie(values, labels=keys, colors=colors_list, autopct=lambda pct: f"{pct:.1f}%", startangle=140, textprops={'fontsize': 8})
    ax1.axis('equal')
    fig1.savefig(pie_tmp.name, dpi=150, bbox_inches='tight', transparent=True)
    plt.close(fig1)

    bar_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.bar(keys, values, color=colors_list)
    ax2.set_xticklabels(keys, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('kg')
    fig2.savefig(bar_tmp.name, dpi=150, bbox_inches='tight', transparent=True)
    plt.close(fig2)

    return pie_tmp.name, bar_tmp.name

# ---------- PDF Report ----------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.setFont("Helvetica", 9)
        footer_text = f"Chemisco • Torrefaction Report • Page {self._pageNumber} of {page_count}"
        self.setFillColor(colors.HexColor("#666666"))
        self.drawRightString(19 * cm, 1 * cm, footer_text)

# ---------- App UI ----------
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("Input Parameters")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    waste_type = st.selectbox("Waste Type", ['Municipal', 'Wood', 'Agricultural', 'Plastic'])
    if waste_type == 'Plastic':
        plastic_type = st.selectbox("Plastic Type", ['Mixed LDPE', 'PET', 'PP'])
with col2:
    mass = st.number_input("Mass (kg)", min_value=1.0, max_value=10000.0, value=50.0, step=1.0, format="%.2f")
    moisture = st.slider("Moisture (%)", 0.0, 100.0, 15.0)
with col3:
    temp = st.slider("Temperature (°C)", 200, 300, 250)
    residence_time = st.slider("Residence Time (hr)", 0.1, 5.0, 1.0)

if st.checkbox("Show advanced settings"):
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        processing_cost_per_kg = st.number_input("Processing Cost per kg ($)", 0.01, 50.0, 1.0, format="%.2f")
        heating_rate = st.slider("Heating Rate (°C/min)", 1, 50, 10)
    with adv_col2:
        reactor_type = st.selectbox("Reactor Type", ['Fixed Bed', 'Rotary', 'Fluidized'])
        atmosphere = st.selectbox("Atmosphere", ['Inert (N2)', 'Air', 'Steam'])

if st.button("Run Simulation"):
    try:
        processing_cost_per_kg
    except NameError:
        processing_cost_per_kg = 1.0
    sim = {
        "Waste Type": waste_type,
        "Mass": mass,
        "Moisture": moisture,
        "Temperature": temp,
        "Residence Time": residence_time
    }
    sim_res = simulate_torrefaction(waste_type, mass, moisture, temp, residence_time)
    sim.update(sim_res)
    sim["Total Cost ($)"] = mass * processing_cost_per_kg
    st.session_state.simulations.append(sim)
    st.success("Simulation run added to dashboard.")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Dashboard ----------
if st.session_state.simulations:
    st.markdown("---")
    st.subheader("Dashboard — Simulations Overview")
    df = pd.DataFrame(st.session_state.simulations)
    st.dataframe(df.style.format("{:.2f}", subset=[c for c in df.columns if df[c].dtype == float]))

    # Latest Simulation
    latest = st.session_state.simulations[-1]

    # KPIs
    kcols = st.columns(5)
    keys = ['Biochar (kg)', 'Gas & Volatiles (kg)', 'Ash (kg)', 'Fixed Carbon (kg)', 'Total Cost ($)']
    kcolors = ['#2E8B57', '#1E90FF', '#FFA500', '#808080', '#8B4513']
    for c, k, col_color in zip(kcols, keys, kcolors):
        c.metric(k, f"{latest.get(k, 0):.2f}")

    # Charts
    st.subheader("Charts — Latest Simulation")
    try:
        pie_path, bar_path = _make_matplotlib_charts(latest)
        st.image(pie_path, caption="Mass Distribution Pie Chart", use_column_width=True)
        st.image(bar_path, caption="Mass Distribution Bar Chart", use_column_width=True)
        for path in (pie_path, bar_path):
            if os.path.exists(path):
                os.unlink(path)
    except Exception as e:
        st.warning(f"Charts could not be generated: {e}")
# ---------- Visualizations Charts ----------
def plot_charts(sim):
    keys = ['Biochar (kg)', 'Gas & Volatiles (kg)', 'Ash (kg)', 'Fixed Carbon (kg)', 'Water Loss (kg)']
    values = [sim.get(k, 0.0) for k in keys]
    colors_list = ['#2E8B57', '#1E90FF', '#FFA500', '#808080', '#8B4513']

    # Pie chart
    fig_pie = go.Figure(data=[go.Pie(labels=keys, values=values, marker=dict(colors=colors_list), hole=0.3)])
    fig_pie.update_layout(title_text="Mass Distribution (kg)", height=400)

    # Bar chart
    fig_bar = go.Figure(data=[go.Bar(x=keys, y=values, marker_color=colors_list)])
    fig_bar.update_layout(title_text="Mass Components (kg)", height=400, yaxis_title="kg")

    return fig_pie, fig_bar

if st.session_state.simulations:
    latest = st.session_state.simulations[-1]
    st.subheader("Charts")
    fig_pie, fig_bar = plot_charts(latest)
    st.plotly_chart(fig_pie, use_container_width=True)
    st.plotly_chart(fig_bar, use_container_width=True)
# ---------- Charts ----------
def plot_charts(sim):
    keys = ['Biochar (kg)', 'Gas & Volatiles (kg)', 'Ash (kg)', 'Fixed Carbon (kg)', 'Water Loss (kg)']
    values = [sim.get(k, 0.0) for k in keys]
    colors_list = ['#2E8B57', '#1E90FF', '#FFA500', '#808080', '#8B4513']

    # Pie chart
    fig_pie = go.Figure(data=[go.Pie(labels=keys, values=values, marker=dict(colors=colors_list), hole=0.3)])
    fig_pie.update_layout(title_text="Mass Distribution (kg)", height=400)

    # Bar chart
    fig_bar = go.Figure(data=[go.Bar(x=keys, y=values, marker_color=colors_list)])
    fig_bar.update_layout(title_text="Mass Components (kg)", height=400, yaxis_title="kg")

    return fig_pie, fig_bar

# ---------- Dashboard and Charts ----------
if st.session_state.simulations:
    st.markdown("---")
    st.subheader("Dashboard — Simulations Overview")
    df = pd.DataFrame(st.session_state.simulations)
    st.dataframe(df.style.format("{:.2f}", subset=[c for c in df.columns if df[c].dtype == float]))

    latest = st.session_state.simulations[-1]

    # KPIs
    kcols = st.columns(5)
    keys = ['Biochar (kg)', 'Gas & Volatiles (kg)', 'Ash (kg)', 'Fixed Carbon (kg)', 'Total Cost ($)']
    kcolors = ['#2E8B57', '#1E90FF', '#FFA500', '#808080', '#8B4513']
    for c, k, col_color in zip(kcols, keys, kcolors):
        c.metric(k, f"{latest.get(k, 0):.2f}")

    # Charts
    st.subheader("Visualizations")
    fig_pie, fig_bar = plot_charts(latest)
    st.plotly_chart(fig_pie, use_container_width=True)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Printing the report
    if st.button("Download Report (PDF)"):
        pdf_buffer = create_pdf_report(latest)
        st.download_button(
            label="Download Report (PDF)",
            data=pdf_buffer,
            file_name="torrefaction_report.pdf",
            mime="application/pdf"
        )

    # Process Flow Diagram
    st.subheader("Process Flow Diagram")
    fig_block = go.Figure()
    blocks = [
        {"name": "Input Waste", "x0": 0, "x1": 2, "y0": 2, "y1": 3, "color": "#8B4513"},
        {"name": "Drying", "x0": 3, "x1": 5, "y0": 2, "y1": 3, "color": "#1E90FF"},
        {"name": "Torrefaction", "x0": 6, "x1": 8, "y0": 2, "y1": 3, "color": "#FFA500"},
        {"name": "Products", "x0": 9, "x1": 11, "y0": 2, "y1": 3, "color": "#2E8B57"}
    ]

    for block in blocks:
        fig_block.add_shape(type="rect", x0=block["x0"], x1=block["x1"], y0=block["y0"], y1=block["y1"],
                             line=dict(color="black", width=2), fillcolor=block["color"], layer="below")
        fig_block.add_annotation(x=(block["x0"] + block["x1"]) / 2, y=(block["y0"] + block["y1"]) / 2,
                                 text=f"<b>{block['name']}</b>", showarrow=False, font=dict(color="white", size=14))

    arrows = [(2, 2.5, 3, 2.5), (5, 2.5, 6, 2.5), (8, 2.5, 9, 2.5)]
    for x0, y0, x1, y1 in arrows:
        fig_block.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                                 showarrow=True, arrowhead=3, arrowsize=2, arrowwidth=3, arrowcolor="#333333")

    fig_block.update_xaxes(range=[-1, 12], showticklabels=False, showgrid=False, zeroline=False)
    fig_block.update_yaxes(range=[1, 4], showticklabels=False, showgrid=False, zeroline=False)
    fig_block.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_block, use_container_width=True)

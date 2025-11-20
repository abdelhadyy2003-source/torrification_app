# chemisco_pro_clean.py
"""
Chemisco Pro — Torrefaction Simulator (clean, organized, and numerically robust)

How this version improves on the original:
- Clear separation: constants, utilities, simulation model, report/chart builders, and Streamlit UI.
- Units consistent and explicit (mass in kg, time in minutes unless specified).
- Physically plausible, tunable empirical models for drying, devolatilization, ash and char yields.
- Defensive input validation and clear docstrings for all functions.
- Clean PDF export with numbered pages and temporary-file cleanup.
- Comments and type hints for maintainability.
"""

import os
import io
import glob
import tempfile
import base64
import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="Chemisco Pro — Torrefaction", layout="wide", initial_sidebar_state="collapsed")

APP_TITLE = "Chemisco Pro — Torrefaction Simulator"
APP_SUBTITLE = "منظّمة، قابلة للتعديل، ودقيقة لحساب نتائج التوريفكشن"

# ----------------------------
# HELPERS: file finds & images
# ----------------------------
def find_first_file(containing: str, search_dir: str = "/mnt/data") -> str:
    """Find first file in search_dir with 'containing' substring (case-insensitive)."""
    try:
        for p in glob.glob(os.path.join(search_dir, "*")):
            if containing.lower() in os.path.basename(p).lower():
                return p
    except Exception:
        pass
    return ""

def img_to_b64(path: str) -> str:
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

# optionally load uploaded images in /mnt/data
HERO_PATH = find_first_file("cover")
BANNER_PATH = find_first_file("banner") or HERO_PATH
LOGO_PATH = find_first_file("logo")

HERO_B64 = img_to_b64(HERO_PATH)
BANNER_B64 = img_to_b64(BANNER_PATH)

# simple hero/banner CSS
hero_css = f"""
<style>
.hero {{
  {"background-image: url('data:image/png;base64," + HERO_B64 + "');" if HERO_B64 else ""}
  background-size: cover;
  background-position: center;
  height: 36vh;
  display:flex; align-items:center; justify-content:center;
  color: white; text-shadow: 1px 1px #000;
}}
.banner {{
  {"background-image: url('data:image/png;base64," + BANNER_B64 + "');" if BANNER_B64 else ""}
  background-size: cover; background-position:center;
  height: 8vh; display:flex; align-items:center; padding-left:1rem;
  color:#fff; text-shadow:1px 1px #000; border-radius:8px; margin-top:0.8rem;
}}
.card {{ background: rgba(255,255,255,0.03); border-radius:10px; padding:14px; }}
</style>
"""
st.markdown(hero_css, unsafe_allow_html=True)
st.markdown(f'<div class="hero"><h1 style="font-size:34px;">{APP_TITLE}</h1></div>', unsafe_allow_html=True)
st.markdown(f'<div class="banner"><h3 style="margin:0;">{APP_SUBTITLE}</h3></div>', unsafe_allow_html=True)

# ----------------------------
# PHYSICS / EMPIRICAL MODELS
# ----------------------------
# Tunable constants (can be exposed in "Advanced" UI)
DEFAULTS = {
    "drying_rate_constant_per_min": 0.20,  # sec^-1-ish constant for moisture removal (higher -> faster drying)
    "devol_k0": 1.0e6,                     # Arrhenius pre-exponential for devolatilization
    "activation_energy_j_per_mol": 1.2e5,  # activation energy (J/mol) — controls temperature sensitivity
    "gas_fraction_base": 0.45,             # baseline fraction of volatile mass that becomes gas/volatiles
    "ash_fraction_by_feed": {              # approximate ash content per feed type (mass fraction)
        "Municipal": 0.08,
        "Wood": 0.02,
        "Agricultural": 0.04,
        "Plastic": 0.005
    },
    "fixed_carbon_fraction_of_char": 0.75,  # fraction of produced char that is fixed carbon
}

R_GAS = 8.31446261815324  # J/(mol*K)


def _drying_mass_loss(initial_mass: float, moisture_pct: float, residence_min: float,
                      k_dry: float) -> float:
    """
    Estimate water mass evaporated during processing using first-order drying kinetics:
      mass_water_initial = initial_mass * moisture_pct
      water_evap_frac = 1 - exp(-k_dry * residence_time)
    Returns water mass (kg).
    """
    moisture_frac = max(0.0, min(1.0, moisture_pct / 100.0))
    water_initial = initial_mass * moisture_frac
    evaporated = water_initial * (1.0 - math.exp(-k_dry * residence_min))
    # clip numeric artifacts
    evaporated = max(0.0, min(evaporated, water_initial))
    return evaporated


def _devolatilization_fraction(temp_c: float, residence_min: float,
                               k0: float, Ea: float) -> float:
    """
    Empirical Arrhenius-like devolatilization severity factor.
    Returns fraction of dry organic mass that is converted into volatiles (0..0.95).
    This is not a mechanistic pyrolysis model but a tunable approximation:
      rate ∝ k0 * exp(-Ea/(R*T))
    Residence time multiplies the rate to give an effective conversion.
    """
    T = temp_c + 273.15
    # reaction rate (per minute) approximate:
    try:
        rate = k0 * math.exp(-Ea / (R_GAS * T))  # units: 1/min (scaled)
    except OverflowError:
        rate = 0.0
    # convert to a fraction via 1 - exp(-rate * t)
    frac = 1.0 - math.exp(-rate * max(0.0, residence_min))
    # physically plausible bounds
    return float(np.clip(frac, 0.0, 0.95))


def simulate_torrefaction(
    feed_type: str,
    mass_kg: float,
    moisture_pct: float,
    temp_c: float,
    residence_time_min: float,
    params: dict = None
) -> Dict[str, float]:
    """
    Run a single torrefaction simulation.
    Inputs:
      - feed_type: 'Municipal'|'Wood'|'Agricultural'|'Plastic'
      - mass_kg: input wet mass (kg)
      - moisture_pct: percentage (0-100) of water in the feed (wet basis)
      - temp_c: torrefaction temperature in °C (recommended 200-300)
      - residence_time_min: residence time in minutes
      - params: tuning constants (defaults in DEFAULTS)
    Returns dict with mass balance (biochar, gas, ash, water_evap, fixed_carbon) and metadata.
    """
    if params is None:
        params = DEFAULTS

    # validation & sanitization
    mass = float(max(0.0, mass_kg))
    moisture_pct = float(max(0.0, min(100.0, moisture_pct)))
    temp_c = float(temp_c)
    t_min = float(max(0.0, residence_time_min))

    # 1) Drying (water loss)
    k_dry = float(params.get("drying_rate_constant_per_min", DEFAULTS["drying_rate_constant_per_min"]))
    water_loss = _drying_mass_loss(mass, moisture_pct, t_min, k_dry)
    mass_after_drying = mass - water_loss
    if mass_after_drying < 0:
        mass_after_drying = 0.0

    # 2) Ash (inorganic) — assumed non-volatile and remains in char/ash
    ash_fraction = float(params.get("ash_fraction_by_feed", DEFAULTS["ash_fraction_by_feed"]).get(feed_type, 0.05))
    ash_mass = mass * ash_fraction  # ash referenced to original mass (wet-basis approximation)

    # 3) Devolatilization of organic fraction (excluding ash and water)
    dry_organic_mass = max(0.0, mass_after_drying - ash_mass)
    # Arrhenius-like fraction of dry organic mass that becomes volatiles
    k0 = float(params.get("devol_k0", DEFAULTS["devol_k0"]))
    Ea = float(params.get("activation_energy_j_per_mol", DEFAULTS["activation_energy_j_per_mol"]))
    vol_frac = _devolatilization_fraction(temp_c, t_min, k0, Ea)

    volatiles_mass = dry_organic_mass * vol_frac
    # remaining solid from organics -> biochar precursor
    char_precursor = dry_organic_mass - volatiles_mass
    # 4) Char and fixed carbon
    # Not all char_precursor becomes stable biochar (some secondary decompositions): apply empirical char_yield factor which depends on severity
    # char_yield decreases with higher temp & longer time.
    severity = vol_frac  # proxy: more vol_frac -> more severe -> less char yield
    char_yield = float(np.clip(0.85 - 0.5 * severity, 0.05, 0.85))  # between ~5% and 85% of char_precursor
    biochar_mass = char_precursor * char_yield

    fixed_carbon_fraction = float(params.get("fixed_carbon_fraction_of_char", DEFAULTS["fixed_carbon_fraction_of_char"]))
    fixed_carbon = biochar_mass * fixed_carbon_fraction

    # 5) Gas vs condensable volatiles split (simple partition)
    gas_fraction_of_volatiles = float(params.get("gas_fraction_base", DEFAULTS["gas_fraction_base"]))
    gas_mass = volatiles_mass * gas_fraction_of_volatiles
    condensable_mass = volatiles_mass - gas_mass

    # 6) Mass balance check and final corrections
    # Compute residual error and adjust small negative rounding issues
    total_out = biochar_mass + gas_mass + ash_mass + water_loss + condensable_mass
    residual = mass - total_out
    # If there's small residual numeric error, assign to gas (safe)
    if abs(residual) > 1e-6:
        gas_mass += residual
        total_out = biochar_mass + gas_mass + ash_mass + water_loss + condensable_mass

    # Ensure no negative values
    for v in ("biochar_mass", "gas_mass", "ash_mass", "water_loss", "condensable_mass", "fixed_carbon"):
        val = locals().get(v)
        if val is None:
            continue
        if val < 0:
            locals()[v] = 0.0

    return {
        "Feed Type": feed_type,
        "Input Mass (kg)": mass,
        "Moisture (%)": moisture_pct,
        "Temperature (°C)": temp_c,
        "Residence Time (min)": t_min,
        "Water Loss (kg)": float(water_loss),
        "Ash (kg)": float(ash_mass),
        "Volatiles (kg)": float(volatiles_mass),
        "Condensables (kg)": float(condensable_mass),
        "Gas & Non-condensables (kg)": float(gas_mass),
        "Biochar (kg)": float(biochar_mass),
        "Fixed Carbon (kg)": float(fixed_carbon)
    }

# ----------------------------
# Reporting / plotting utilities
# ----------------------------
def make_matplotlib_pie_and_bar(sim_result: dict) -> Tuple[str, str]:
    """Create small pie and bar charts and return their file paths (temporary files)."""
    labels = ["Biochar", "Gas", "Ash", "Condensables", "Water"]
    values = [
        sim_result.get("Biochar (kg)", 0.0),
        sim_result.get("Gas & Non-condensables (kg)", 0.0),
        sim_result.get("Ash (kg)", 0.0),
        sim_result.get("Condensables (kg)", 0.0),
        sim_result.get("Water Loss (kg)", 0.0)
    ]
    # avoid all-zero
    if sum(values) <= 0:
        values = [1e-6] * len(values)

    # Pie
    pie_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig1, ax1 = plt.subplots(figsize=(3.2, 3.2))
    ax1.pie(values, labels=labels, autopct=lambda p: f"{p:.1f}%", startangle=140, textprops={'fontsize': 6})
    ax1.axis('equal')
    fig1.savefig(pie_tmp.name, bbox_inches='tight', dpi=150, transparent=True)
    plt.close(fig1)

    # Bar
    bar_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig2, ax2 = plt.subplots(figsize=(4.5, 2.2))
    ax2.bar(labels, values)
    ax2.set_ylabel("kg", fontsize=8)
    ax2.tick_params(axis='x', labelrotation=30, labelsize=7)
    fig2.tight_layout()
    fig2.savefig(bar_tmp.name, bbox_inches='tight', dpi=150, transparent=True)
    plt.close(fig2)

    return pie_tmp.name, bar_tmp.name


class NumberedCanvas(canvas.Canvas):
    """ReportLab canvas with page numbers in footer."""
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
        self.setFont("Helvetica", 8)
        footer_text = f"Chemisco • Torrefaction Report • Page {self._pageNumber} of {page_count}"
        self.setFillColor(colors.HexColor("#666666"))
        self.drawRightString(19 * cm, 1 * cm, footer_text)


def create_pdf_report(sim_result: dict, logo_path: str = LOGO_PATH) -> io.BytesIO:
    """Create a PDF report for one simulation result and return a BytesIO buffer."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2.5*cm)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('title', parent=styles['Title'], alignment=1, fontSize=20,
                                 textColor=colors.HexColor("#1E90FF"))
    body_style = styles["BodyText"]

    # logo if exists
    if logo_path and os.path.exists(logo_path):
        try:
            logo = RLImage(logo_path, width=5*cm, height=5*cm)
            logo.hAlign = 'CENTER'
            story.append(logo)
            story.append(Spacer(1, 0.2*cm))
        except Exception:
            pass

    story.append(Paragraph("Chemisco Pro — Torrefaction Report", title_style))
    story.append(Spacer(1, 0.4*cm))
    meta = [["Generated by", "Chemisco Pro Simulator"], ["Report date", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")]]
    meta_table = Table(meta, colWidths=[5*cm, 9*cm])
    meta_table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Helvetica'),
                                    ('FONTSIZE',(0,0),(-1,-1),9),
                                    ('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.append(meta_table)
    story.append(PageBreak())

    # Simulation table
    story.append(Paragraph("Simulation Summary", styles['Heading2']))
    story.append(Spacer(1, 0.2*cm))
    data = [["Parameter", "Value"]]
    for k, v in sim_result.items():
        data.append([k, f"{v:.4f}" if isinstance(v, (int, float)) else str(v)])
    table = Table(data, colWidths=[9*cm, 6*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1E90FF")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.6*cm))

    try:
        pie_path, bar_path = make_matplotlib_pie_and_bar(sim_result)
        story.append(PageBreak())
        story.append(Paragraph("Visual Summary", styles['Heading2']))
        story.append(Spacer(1, 0.2*cm))
        story.append(RLImage(pie_path, width=8*cm, height=8*cm))
        story.append(Spacer(1, 0.4*cm))
        story.append(RLImage(bar_path, width=12*cm, height=4.5*cm))
    except Exception:
        story.append(Paragraph("Charts could not be generated.", body_style))

    story.append(PageBreak())
    story.append(Paragraph("Notes", styles['Heading2']))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "This report contains empirical simulation outputs. Values are approximate and intended for "
        "process design and estimation. For design-critical decisions, use a validated pyrolysis model "
        "or lab-scale experiments.", body_style))
    doc.build(story, canvasmaker=NumberedCanvas)

    # cleanup temporary image files if any
    for p in (locals().get('pie_path'), locals().get('bar_path')):
        if p and os.path.exists(p):
            try:
                os.unlink(p)
            except Exception:
                pass

    buffer.seek(0)
    return buffer

# ----------------------------
# Streamlit UI
# ----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("أدخل معطيات المحاكاة")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    feed_type = st.selectbox("نوع المادة (Feed Type)", ["Municipal", "Wood", "Agricultural", "Plastic"])
    if feed_type == "Plastic":
        plastic_subtype = st.selectbox("نوع البلاستيك (إذا اخترت Plastic)", ["Mixed LDPE", "PET", "PP"])
with col2:
    mass_input = st.number_input("الكتلة (kg) — Wet mass", min_value=0.1, max_value=100000.0, value=50.0, format="%.3f")
    moisture_input = st.slider("نسبة الرطوبة (%)", 0.0, 100.0, 15.0)
with col3:
    temperature_input = st.slider("درجة الحرارة (°C)", 150, 350, 260)
    residence_time_input = st.number_input("زمن الإقامة (دقائق)", min_value=0.1, max_value=1440.0, value=60.0, format="%.2f")

if st.checkbox("إظهار الإعدادات المتقدمة (Advanced settings)"):
    adv1, adv2 = st.columns(2)
    with adv1:
        k_dry_ui = st.number_input("Drying rate constant (1/min)", value=DEFAULTS["drying_rate_constant_per_min"], format="%.5f")
        devol_k0_ui = st.number_input("Devolatilization k0 (scale)", value=DEFAULTS["devol_k0"], format="%.1e")
    with adv2:
        Ea_ui = st.number_input("Activation energy (J/mol)", value=DEFAULTS["activation_energy_j_per_mol"], format="%.0f")
        fixed_c_frac_ui = st.slider("Fixed carbon fraction of char", 0.4, 0.95, DEFAULTS["fixed_carbon_fraction_of_char"])

    custom_params = {
        "drying_rate_constant_per_min": float(k_dry_ui),
        "devol_k0": float(devol_k0_ui),
        "activation_energy_j_per_mol": float(Ea_ui),
        "fixed_carbon_fraction_of_char": float(fixed_c_frac_ui)
    }
else:
    custom_params = None

if st.button("Run Simulation"):
    try:
        sim = simulate_torrefaction(
            feed_type=feed_type,
            mass_kg=mass_input,
            moisture_pct=moisture_input,
            temp_c=temperature_input,
            residence_time_min=residence_time_input,
            params=custom_params
        )
        # persist in session for dashboard/listing
        if "simulations" not in st.session_state:
            st.session_state.simulations = []
        st.session_state.simulations.append(sim)
        st.success("Simulation completed and added to session.")
    except Exception as e:
        st.error(f"حدث خطأ أثناء المحاكاة: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# Dashboard
if "simulations" in st.session_state and st.session_state.simulations:
    st.markdown("---")
    st.subheader("Dashboard — نتائج المحاكاة")
    df = pd.DataFrame(st.session_state.simulations)
    # present numeric columns nicely
    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    st.dataframe(df.style.format({c: "{:.3f}" for c in numeric_cols}))

    latest = st.session_state.simulations[-1]
    # Key metrics
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Biochar (kg)", f"{latest.get('Biochar (kg)', 0.0):.3f}")
    k2.metric("Gas (kg)", f"{latest.get('Gas & Non-condensables (kg)', 0.0):.3f}")
    k3.metric("Ash (kg)", f"{latest.get('Ash (kg)', 0.0):.3f}")
    k4.metric("Fixed Carbon (kg)", f"{latest.get('Fixed Carbon (kg)', 0.0):.3f}")
    k5.metric("Water Loss (kg)", f"{latest.get('Water Loss (kg)', 0.0):.3f}")

    # Simple process flow diagram (horizontal blocks)
    st.subheader("Process Flow Diagram")
    fig_block = go.Figure()
    blocks = [
        {"name": "Input", "x0": 0, "x1": 1.6, "y0": 1.2, "y1": 2.2, "color": "#8B4513"},
        {"name": "Drying", "x0": 2, "x1": 3.6, "y0": 1.2, "y1": 2.2, "color": "#1E90FF"},
        {"name": "Torrefaction", "x0": 4, "x1": 5.6, "y0": 1.2, "y1": 2.2, "color": "#FFA500"},
        {"name": "Products", "x0": 6, "x1": 7.6, "y0": 1.2, "y1": 2.2, "color": "#2E8B57"}
    ]
    for b in blocks:
        fig_block.add_shape(type="rect", x0=b["x0"], x1=b["x1"], y0=b["y0"], y1=b["y1"],
                            line=dict(color="black", width=1.5), fillcolor=b["color"], layer="below")
        fig_block.add_annotation(x=(b["x0"] + b["x1"]) / 2, y=(b["y0"] + b["y1"]) / 2,
                                 text=f"<b>{b['name']}</b>", showarrow=False, font=dict(color="white", size=11))
    arrows = [(1.6, 1.7, 2.0, 1.7), (3.6, 1.7, 4.0, 1.7), (5.6, 1.7, 6.0, 1.7)]
    for x0, y0, x1, y1 in arrows:
        fig_block.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                                 showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=2)
    fig_block.update_xaxes(range=[-0.5, 8.5], showgrid=False, visible=False)
    fig_block.update_yaxes(range=[1.0, 2.6], showgrid=False, visible=False)
    fig_block.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_block, use_container_width=True)

    # Charts and PDF export for the latest simulation
    st.subheader("Visual Summary & Report")
    try:
        pie_path, bar_path = make_matplotlib_pie_and_bar(latest)
        cols = st.columns([1, 2])
        with cols[0]:
            st.image(pie_path, caption="Mass distribution (pie)", use_column_width=True)
        with cols[1]:
            st.image(bar_path, caption="Mass breakdown (bar)", use_column_width=True)
        # download PDF
        pdf_buffer = create_pdf_report(latest)
        st.download_button("Download PDF Report", data=pdf_buffer, file_name="torrefaction_report.pdf", mime="application/pdf")
        # cleanup temps
        for p in (pie_path, bar_path):
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass
    except Exception as e:
        st.warning(f"تعذر رسم المخططات أو إنشاء التقرير: {e}")

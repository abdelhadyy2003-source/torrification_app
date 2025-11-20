# chemisco_final_fixed.py
# Streamlit Torrefaction Simulator (Single Page) — corrected

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import tempfile, io, os, glob

# ----- CONFIG -----
st.set_page_config(page_title="Chemisco - Ultimate Torrefaction", layout="wide", initial_sidebar_state="collapsed")

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

HERO_COVER = find_first_file("cover") or find_first_file("tor") or ""
BANNER_COVER = find_first_file("refaction") or find_first_file("torrefaction") or HERO_COVER
LOGO_PATH = find_first_file("enzyme") or find_first_file("logo") or ""

# ---------- Torrefaction simulation ----------
def simulate_torrefaction(waste_type, mass, moisture, temp, residence_time):
    water_loss = mass * (moisture / 100.0) * (1.0 - np.exp(-0.6 * residence_time))
    volatile_fraction = 0.30 + 0.12 * ((temp - 200.0) / 100.0)
    volatile_fraction = max(0.0, min(0.9, volatile_fraction))
    volatile_loss = max(0.0, (mass - water_loss) * volatile_fraction)
    ash_mass = mass * 0.05
    biochar_mass = max(0.0, mass - water_loss - volatile_loss - ash_mass)
    fixed_carbon = biochar_mass * 0.78
    return {
        'Biochar (kg)': float(biochar_mass),
        'Gas & Volatiles (kg)': float(volatile_loss),
        'Ash (kg)': float(ash_mass),
        'Fixed Carbon (kg)': float(fixed_carbon),
        'Water Loss (kg)': float(water_loss)
    }

# ---------- ReportLab PDF ----------
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
        self.drawRightString(19*cm, 1*cm, footer_text)

def _make_matplotlib_charts(sim):
    keys = ['Biochar (kg)', 'Gas & Volatiles (kg)', 'Ash (kg)', 'Fixed Carbon (kg)', 'Water Loss (kg)']
    values = [float(sim.get(k, 0.0)) for k in keys]
    colors_list = ['#2E8B57','#1E90FF','#FFA500','#808080','#8B4513']

    pie_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig1, ax1 = plt.subplots(figsize=(4,4))
    if sum(values) == 0: values = [1e-6]*len(values)
    ax1.pie(values, labels=keys, colors=colors_list, autopct=lambda pct: f"{pct:.1f}%", startangle=140, textprops={'fontsize':8})
    ax1.axis('equal')
    fig1.savefig(pie_tmp.name, dpi=150, bbox_inches='tight', transparent=True)
    plt.close(fig1)

    bar_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig2, ax2 = plt.subplots(figsize=(6,3))
    ax2.bar(keys, values, color=colors_list)
    ax2.set_xticklabels(keys, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('kg')
    fig2.savefig(bar_tmp.name, dpi=150, bbox_inches='tight', transparent=True)
    plt.close(fig2)

    return pie_tmp.name, bar_tmp.name

def create_pdf_report(sim, logo_path=LOGO_PATH):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2.5*cm)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('title', parent=styles['Title'], alignment=1, fontSize=28, textColor=colors.HexColor("#1E90FF"))
    subtitle_style = ParagraphStyle('subtitle', parent=styles['Heading2'], alignment=1, fontSize=14, textColor=colors.HexColor("#444444"))
    body_style = styles['BodyText']

    if logo_path and os.path.exists(logo_path):
        try:
            img = RLImage(logo_path, width=6*cm, height=6*cm)
            img.hAlign = 'CENTER'
            story.append(Spacer(1, 0.5*cm))
            story.append(img)
        except: pass

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Chemisco", title_style))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("Torrefaction Simulation Report", subtitle_style))
    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph("Simulation results from Chemisco Torrefaction Simulator.", body_style))
    story.append(Spacer(1, 1.8*cm))

    meta = [["Generated by", "Chemisco Torrefaction Simulator"],
            ["Report generated", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")]]
    meta_table = Table(meta, colWidths=[5*cm, 8*cm])
    meta_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor("#333333")),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(meta_table)
    story.append(PageBreak())

    # Summary Table
    story.append(Paragraph("Simulation Summary", styles['Heading2']))
    story.append(Spacer(1, 0.2*cm))
    data = [["Parameter","Value"]]
    for k,v in sim.items():
        if isinstance(v,(int,float)):
            data.append([k,f"{v:.2f}"])
        else:
            data.append([k,str(v)])
    table = Table(data, colWidths=[9*cm,6*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E90FF")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 11),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F7F7F7")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#DDDDDD")),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.8*cm))

    try:
        pie_path, bar_path = _make_matplotlib_charts(sim)
        story.append(PageBreak())
        story.append(Paragraph("Visual Summary", styles['Heading2']))
        story.append(Spacer(1, 0.3*cm))
        story.append(RLImage(pie_path, width=10*cm, height=10*cm))
        story.append(Spacer(1, 0.5*cm))
        story.append(RLImage(bar_path, width=14*cm, height=6*cm))
    except: story.append(Paragraph("Charts could not be generated.", body_style))

    story.append(PageBreak())
    story.append(Paragraph("Technical details & notes", styles['Heading2']))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("Values are approximate simulation outputs for demonstration.", body_style))

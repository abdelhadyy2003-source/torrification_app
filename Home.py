# chemisco_final_fixed.py
# Final fixed Streamlit app (Single page) with Hero cover A, Banner cover B, branching menus, ReportLab PDF
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

# --- Utility: find uploaded image by hint (robust against odd filenames) ---
def find_first_file(containing):
    """Search /mnt/data for a filename containing the given substring. Return full path or ''."""
    candidates = glob.glob("/mnt/data/*")
    for c in candidates:
        if containing.lower() in os.path.basename(c).lower():
            return c
    return ""

# If user uploaded with odd chars, try to find sensible files
HERO_COVER = find_first_file("cover") or find_first_file("tor") or ""
BANNER_COVER = find_first_file("refaction") or find_first_file("torrefaction") or HERO_COVER
LOGO_PATH = find_first_file("enzyme") or find_first_file("logo") or ""

# ---------- Torrefaction simulation function (added to fix NameError) ----------
def simulate_torrefaction(waste_type, mass, moisture, temp, residence_time):
    """
    Simple physics-inspired torrefaction model (deterministic approx).
    Returns dict of product masses / losses.
    """
    # water removed (kg) - first-order style dependent on residence time
    water_loss = mass * (moisture / 100.0) * (1.0 - np.exp(-0.6 * residence_time))

    # volatile fraction increases with temperature (simple linear approx)
    volatile_fraction = 0.30 + 0.12 * ((temp - 200.0) / 100.0)  # 0.30..0.42 across 200..300
    volatile_fraction = max(0.0, min(0.9, volatile_fraction))

    volatile_loss = max(0.0, (mass - water_loss) * volatile_fraction)

    ash_fraction = 0.05  # fixed ash fraction (approx)
    ash_mass = mass * ash_fraction

    biochar_mass = mass - water_loss - volatile_loss - ash_mass
    if biochar_mass < 0:
        biochar_mass = 0.0

    fixed_carbon = biochar_mass * 0.78  # assume 78% of biochar is fixed carbon

    return {
        'Biochar (kg)': float(biochar_mass),
        'Gas & Volatiles (kg)': float(volatile_loss),
        'Ash (kg)': float(ash_mass),
        'Fixed Carbon (kg)': float(fixed_carbon),
        'Water Loss (kg)': float(water_loss)
    }

# ---------- Helper: ReportLab PDF (professional, multi-page, charts embedded) ----------
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
    colors = ['#2E8B57','#1E90FF','#FFA500','#808080','#8B4513']

    pie_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig1, ax1 = plt.subplots(figsize=(4,4))
    # handle case values all zero -> pie would error
    if sum(values) == 0:
        values = [1e-6 for _ in values]
    ax1.pie(values, labels=keys, colors=colors, autopct=lambda pct: f"{pct:.1f}%", startangle=140, textprops={'fontsize':8})
    ax1.axis('equal')
    plt.tight_layout()
    fig1.savefig(pie_tmp.name, dpi=150, bbox_inches='tight', transparent=True)
    plt.close(fig1)

    bar_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig2, ax2 = plt.subplots(figsize=(6,3))
    ax2.bar(keys, values, color=colors)
    ax2.set_xticklabels(keys, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('kg')
    plt.tight_layout()
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

    # Cover
    title_style = ParagraphStyle('title', parent=styles['Title'], alignment=1, fontSize=28, textColor=colors.HexColor("#1E90FF"))
    subtitle_style = ParagraphStyle('subtitle', parent=styles['Heading2'], alignment=1, fontSize=14, textColor=colors.HexColor("#444444"))
    body_style = styles['BodyText']

    if logo_path and os.path.exists(logo_path):
        try:
            img = RLImage(logo_path, width=6*cm, height=6*cm)
            img.hAlign = 'CENTER'
            story.append(Spacer(1, 0.5*cm))
            story.append(img)
        except Exception:
            pass

    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("Chemisco", title_style))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("Torrefaction Simulation Report", subtitle_style))
    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph("This report contains the results and summary of the torrefaction simulation performed using the Chemisco Torrefaction Simulator.", body_style))
    story.append(Spacer(1, 1.8*cm))

    meta = [["Generated by", "Chemisco Torrefaction Simulator"],
            ["Report generated", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")]]
    meta_table = Table(meta, colWidths=[5*cm, 8*cm])
    meta_table.setStyle(TableStyle([('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
                                    ('FONTSIZE', (0,0), (-1,-1), 9),
                                    ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor("#333333")),
                                    ('BOTTOMPADDING', (0,0), (-1,-1), 6),]))
    story.append(meta_table)
    story.append(PageBreak())

    # Summary Table
    story.append(Paragraph("Simulation Summary", styles['Heading2']))
    story.append(Spacer(1, 0.2*cm))
    data = [["Parameter", "Value"]]
    input_keys = ["Waste Type", "Mass", "Moisture", "Temperature", "Residence Time"]
    for k in input_keys:
        if k in sim:
            data.append([k, f"{sim[k]}"])
    result_keys = [k for k in sim.keys() if k not in input_keys]
    for k in result_keys:
        v = sim[k]
        if isinstance(v, (int, float)):
            data.append([k, f"{v:.2f}"])
        else:
            data.append([k, str(v)])

    table = Table(data, colWidths=[9*cm, 6*cm])
    table_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1E90FF")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 11),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#F7F7F7")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#DDDDDD")),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
    ])
    table.setStyle(table_style)
    story.append(table)
    story.append(Spacer(1, 0.8*cm))

    # Charts page
    try:
        pie_path, bar_path = _make_matplotlib_charts(sim)
        story.append(PageBreak())
        story.append(Paragraph("Visual Summary", styles['Heading2']))
        story.append(Spacer(1, 0.3*cm))
        story.append(RLImage(pie_path, width=10*cm, height=10*cm))
        story.append(Spacer(1, 0.5*cm))
        story.append(RLImage(bar_path, width=14*cm, height=6*cm))
    except Exception:
        story.append(Paragraph("Charts could not be generated.", body_style))

    # Technical notes & footer
    story.append(PageBreak())
    story.append(Paragraph("Technical details & notes", styles['Heading2']))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("This report is generated by the Chemisco Torrefaction Simulator. Values are simulation outputs and approximate estimates for demonstration purposes.", body_style))
    story.append(Spacer(1, 0.6*cm))
    story.append(Paragraph("<i>Generated by Chemisco Torrefaction Simulator © 2025</i>", styles['Normal']))

    doc.build(story, canvasmaker=NumberedCanvas)

    # cleanup temp files
    try:
        if 'pie_path' in locals() and os.path.exists(pie_path):
            os.unlink(pie_path)
        if 'bar_path' in locals() and os.path.exists(bar_path):
            os.unlink(bar_path)
    except Exception:
        pass

    buffer.seek(0)
    return buffer

# ---------- App UI (Single page) ----------
# CSS for hero and banner, using uploaded cover images
# If HERO_COVER/BANNER_COVER are empty, CSS will not break; the hero/banner simply show solid background.
hero_css = f"""
<style>
.hero {{
  background-image: url("file://{HERO_COVER}") ;
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
  background-image: url("file://{BANNER_COVER}") ;
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

# Hero section (cover A)
st.markdown('<div class="hero"><h1>Chemisco — Advanced Torrefaction</h1></div>', unsafe_allow_html=True)

# Banner (cover B)
st.markdown('<div class="banner"><h3>Torrefaction Simulator — Realistic process & analytics</h3></div>', unsafe_allow_html=True)

# Centered inputs in glass card with branching menus
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("Input Parameters")
col1, col2, col3 = st.columns([1,1,1])
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

# extra branching: advanced settings collapsible
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
        "Mass": float(mass),
        "Moisture": float(moisture),
        "Temperature": float(temp),
        "Residence Time": float(residence_time)
    }
    sim_res = simulate_torrefaction(waste_type, float(mass), float(moisture), float(temp), float(residence_time))
    sim.update(sim_res)
    sim["Total Cost ($)"] = float(mass) * float(processing_cost_per_kg)
    st.session_state.simulations.append(sim)
    st.success("Simulation run added to dashboard.")

st.markdown('</div>', unsafe_allow_html=True)

# Results & Dashboard area
if st.session_state.simulations:
    st.markdown("---")
    st.subheader("Dashboard — Simulations Overview")
    df = pd.DataFrame(st.session_state.simulations)
    st.dataframe(df.style.format("{:.2f}", subset=[c for c in df.columns if df[c].dtype==float]))

    # KPIs (latest)
    latest = st.session_state.simulations[-1]
    kcols = st.columns(5)
    keys = ['Biochar (kg)','Gas & Volatiles (kg)','Ash (kg)','Fixed Carbon (kg)','Total Cost ($)']
    kcolors = ['#2E8B57','#1E90FF','#FFA500','#808080','#8B4513']
    for c,k,col_color in zip(kcols, keys, kcolors):
        c.metric(k, f"{latest.get(k,0):.2f}")

    # Charts
    st.subheader("Visualizations")
    chart_cols = st.columns([1,1])
    with chart_cols[0]:
        keys_chart = ['Biochar (kg)','Gas & Volatiles (kg)','Ash (kg)','Fixed Carbon (kg)','Water Loss (kg)']
        fig_pie = go.Figure(data=[go.Pie(labels=keys_chart, values=[df.iloc[-1][k] for k in keys_chart], marker=dict(colors=kcolors))])
        fig_pie.update_layout(title="Last Simulation Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
    with chart_cols[1]:
        st.line_chart(df[['Biochar (kg)','Gas & Volatiles (kg)','Total Cost ($)']])

    # Block Flow
    st.subheader("Process Flow Diagram")
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
        fig_block.add_annotation(x=(block["x0"]+block["x1"])/2, y=(block["y0"]+block["y1"])/2,
                                 text=f"<b>{block['name']}</b>", showarrow=False, font=dict(color="white", size=14))
    arrows = [(2,2.5,3,2.5),(5,2.5,6,2.5),(8,2.5,9,2.5)]
    for x0,y0,x1,y1 in arrows:
        fig_block.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                                 showarrow=True, arrowhead=3, arrowsize=2, arrowwidth=3, arrowcolor="#333333")
    fig_block.update_xaxes(range=[-1,12], showticklabels=False, showgrid=False, zeroline=False)
    fig_block.update_yaxes(range=[1,4], showticklabels=False, showgrid=False, zeroline=False)
    fig_block.update_layout(height=300, margin=dict(l=20,r=20,t=20,b=20), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_block, use_container_width=True)

    # Sankey overall
    st.subheader("Sankey Flow Sheet (All Simulations)")
    labels = ["Input Waste","Water Loss","Gas & Volatiles","Ash","Biochar"]
    node_colors = ['#8B4513','#1E90FF','#FFA500','#808080','#2E8B57']
    sources, targets, values, link_colors = [], [], [], []
    for sim in st.session_state.simulations:
        sources.extend([0,0,0,0])
        targets.extend([1,2,3,4])
        values.extend([sim['Water Loss (kg)'], sim['Gas & Volatiles (kg)'], sim['Ash (kg)'], sim['Biochar (kg)']])
        link_colors.extend(node_colors)
    fig_sankey = go.Figure(data=[go.Sankey(node=dict(label=labels, pad=15, thickness=20, color=node_colors),
                                           link=dict(source=sources, target=targets, value=values, color=link_colors))])
    fig_sankey.update_traces(hovertemplate='From %{source.label} to %{target.label}: %{value} kg<extra></extra>')
    fig_sankey.update_layout(title_text="Flow Sheet (All Simulations)", font_size=12)
    st.plotly_chart(fig_sankey, use_container_width=True)

    # Export & Reports
    st.subheader("Reports & Export")
    last_sim = st.session_state.simulations[-1]
    pdf_buf = create_pdf_report(last_sim)
    st.download_button("Download Premium PDF Report (with cover & charts)", data=pdf_buf,
                       file_name="Chemisco_Torrefaction_Report.pdf", mime="application/pdf")

# End of app

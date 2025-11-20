import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from reportlab.pdfgen import canvas
import graphviz

# --------- واجهة التطبيق ---------
st.set_page_config(page_title="Torrification Simulator", layout="wide")
st.title("🌿 Torrification Simulator")
st.write("تطبيق محاكاة التوريفيكاشن: احسب الرطوبة، أعمل سيميوليشن، اعمل Flow Sheet وحمل تقارير PDF/Excel")

# Sidebar لإدخال البيانات
st.sidebar.header("Input Parameters")
initial_moisture = st.sidebar.number_input("Initial Moisture (%)", 0, 100, 40)
temperature = st.sidebar.slider("Temperature (°C)", 50, 300, 200)
time_hours = st.sidebar.number_input("Time (hours)", 0.0, 10.0, 2.0)
samples = st.sidebar.number_input("Number of Samples", 1, 10, 1)

# --------- Tabs ---------
tab1, tab2, tab3 = st.tabs(["Simulation", "Flow Sheet", "Reports"])

# --------- Tab 1: Simulation ---------
with tab1:
    st.header("Simulation Results")
    times = np.linspace(0, time_hours, 100)
    fig = px.line()
    
    all_data = pd.DataFrame()
    
    for s in range(1, samples+1):
        decay_rate = 0.05 + 0.0005*(temperature-100)
        moistures = initial_moisture * np.exp(-decay_rate * times)
        fig = px.line(x=times, y=moistures, labels={"x":"Time (hours)", "y":"Moisture (%)"}, title=f"Sample {s}")
        st.plotly_chart(fig, use_container_width=True)
        
        df = pd.DataFrame({"Time": times, f"Sample {s}": moistures})
        all_data = pd.concat([all_data, df.set_index("Time")], axis=1)

# --------- Tab 2: Flow Sheet ---------
with tab2:
    st.header("Flow Sheet Diagram")
    dot = graphviz.Digraph()
    dot.node('A', 'Raw Material')
    dot.node('B', 'Drying')
    dot.node('C', 'Torrefaction')
    dot.node('D', 'Final Product')
    dot.edges(['AB', 'BC', 'CD'])
    st.graphviz_chart(dot)

# --------- Tab 3: Reports ---------
with tab3:
    st.header("Download Reports")
    
    # CSV
    csv = all_data.reset_index().to_csv(index=False).encode()
    st.download_button("Download CSV", data=csv, file_name="torrification_results.csv")
    
    # PDF
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.drawString(50, 800, "Torrification Report")
    c.drawString(50, 780, f"Initial Moisture: {initial_moisture}%")
    c.drawString(50, 760, f"Temperature: {temperature}°C")
    c.drawString(50, 740, f"Time: {time_hours} hours")
    c.drawString(50, 720, f"Number of Samples: {samples}")
    c.showPage()
    c.save()
    st.download_button("Download PDF", data=buffer, file_name="torrification_report.pdf", mime="application/pdf")

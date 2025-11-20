


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

# Constants
R = 8.314  # Universal gas constant (J/mol·K)

def main():
    st.set_page_config(page_title="Chemisco Pro Torrefaction Simulator", layout="wide")
    
    st.title("Chemisco Pro Torrefaction Simulator")
    st.markdown("Professional simulation of biomass torrefaction processes")
    
    with st.sidebar:
        st.header("Input Parameters")
        biomass_type = st.selectbox("Biomass Type", ["Wood", "Agricultural Waste", "Municipal Waste"])
        moisture_content = st.slider("Moisture Content (%)", 0.0, 50.0, 10.0)
        temperature = st.slider("Torrefaction Temperature (°C)", 200, 350, 250)
        duration = st.slider("Process Duration (min)", 10, 120, 30)
        particle_size = st.selectbox("Particle Size", ["Fine (<1mm)", "Medium (1-5mm)", "Coarse (>5mm)"])
    
    # Model calculations
    results = simulate_torrefaction(biomass_type, moisture_content, temperature, duration, particle_size)
    
    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Product Yields")
        st.dataframe(results["yields"].style.format("{:.2f}"), use_container_width=True)
        
        st.subheader("Mass Balance")
        fig1, ax1 = plt.subplots()
        ax1.pie(results["yields"].values.flatten(), labels=results["yields"].index, autopct='%1.1f%%')
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Temperature Profile")
        st.line_chart(results["temp_profile"])
        
        st.subheader("Gas Composition")
        st.bar_chart(results["gas_composition"])
    
    # Report generation
    if st.button("Generate PDF Report"):
        pdf_buffer = generate_pdf_report(results, biomass_type, moisture_content, temperature, duration, particle_size)
        st.download_button(
            label="Download Report",
            data=pdf_buffer,
            file_name="torrefaction_report.pdf",
            mime="application/pdf"
        )

def simulate_torrefaction(biomass, moisture, temp_C, duration_min, size):
    """Core torrefaction simulation logic"""
    temp_K = temp_C + 273.15
    
    # Empirical coefficients based on biomass type
    if biomass == "Wood":
        k_drying = 0.05
        k_devol = 0.03
        ash_content = 0.02
    elif biomass == "Agricultural Waste":
        k_drying = 0.07
        k_devol = 0.04
        ash_content = 0.08
    else:  # Municipal Waste
        k_drying = 0.10
        k_devol = 0.06
        ash_content = 0.15
    
    # Time points for simulation
    t = np.linspace(0, duration_min, 100)
    
    # Solve drying and devolatilization ODEs
    def model(y, t, k1, k2):
        moisture, volatiles = y
        dydt = [-k1 * moisture, -k2 * volatiles]
        return dydt
    
    y0 = [moisture/100, 1 - moisture/100 - ash_content]
    sol = odeint(model, y0, t, args=(k_drying, k_devol))
    
    # Calculate yields
    final_moisture = sol[-1, 0]
    final_volatiles = sol[-1, 1]
    biochar_yield = 1 - final_moisture - final_volatiles - ash_content
    
    # Estimate gas composition (empirical relationships)
    gas_comp = {
        "CO": 0.4 * final_volatiles,
        "CO2": 0.3 * final_volatiles,
        "CH4": 0.2 * final_volatiles,
        "H2": 0.1 * final_volatiles
    }
    
    # Prepare results
    yields = pd.DataFrame({
        "Yield (%)": [
            (1 - final_moisture) * 100,  # Moisture loss
            final_volatiles * 100,       # Volatiles
            biochar_yield * 100,         # Biochar
            ash_content * 100            # Ash
        ]},
        index=["Moisture Loss", "Volatiles", "Biochar", "Ash"]
    )
    
    temp_profile = pd.DataFrame({
        "Time (min)": t,
        "Temperature (°C)": temp_C * np.ones_like(t)
    }).set_index("Time (min)")
    
    gas_composition = pd.DataFrame.from_dict(gas_comp, orient="index", columns=["Composition (%)"])
    
    return {
        "yields": yields,
        "temp_profile": temp_profile,
        "gas_composition": gas_composition,
        "parameters": {
            "biomass": biomass,
            "moisture": moisture,
            "temperature": temp_C,
            "duration": duration_min,
            "size": size
        }
    }

def generate_pdf_report(results, biomass, moisture, temp, duration, size):
    """Generate PDF report of simulation results"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Header
    elements.append(Paragraph("Chemisco Pro Torrefaction Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    
    # Parameters
    elements.append(Paragraph("Simulation Parameters:", styles["Heading2"]))
    param_text = f"""
    Biomass Type: {biomass}<br/>
    Moisture Content: {moisture}%<br/>
    Temperature: {temp}°C<br/>
    Duration: {duration} minutes<br/>
    Particle Size: {size}
    """
    elements.append(Paragraph(param_text, styles["Normal"]))
    elements.append(Spacer(1, 12))
    
    # Yields table
    elements.append(Paragraph("Product Yields:", styles["Heading2"]))
    yield_data = [["Component", "Yield (%)"]] + [[idx, f"{val[0]:.2f}"] 
                 for idx, val in results["yields"].iterrows()]
    yield_table = Table(yield_data)
    elements.append(yield_table)
    elements.append(Spacer(1, 12))
    
    # Charts
    elements.append(Paragraph("Results Visualization:", styles["Heading2"]))
    
    # Mass balance pie chart
    fig1, ax1 = plt.subplots()
    ax1.pie(results["yields"].values.flatten(), labels=results["yields"].index, autopct='%1.1f%%')
    plt.title("Mass Balance")
    imgdata = BytesIO()
    plt.savefig(imgdata, format='png')
    imgdata.seek(0)
    elements.append(Image(imgdata, width=400, height=300))
    
    # Gas composition bar chart
    fig2, ax2 = plt.subplots()
    results["gas_composition"].plot(kind='bar', ax=ax2)
    plt.title("Gas Composition")
    plt.ylabel("Percentage")
    imgdata2 = BytesIO()
    plt.savefig(imgdata2, format='png')
    imgdata2.seek(0)
    elements.append(Image(imgdata2, width=400, height=300))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

if __name__ == "__main__":
    main()



# ===== Import Libraries =====
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== Functions =====
def simulate_torrefaction(waste_type, mass, moisture, temp, residence_time):
    """
    Simulate torrefaction process.
    Returns dict of outputs.
    """
    # Moisture loss
    water_loss = mass * moisture/100 * (1 - np.exp(-0.5*residence_time))
    
    # Volatile loss (simplified)
    volatile_fraction = 0.3 + 0.1*(temp-200)/100
    volatile_loss = (mass - water_loss) * volatile_fraction
    
    # Ash fraction fixed
    ash_fraction = 0.05
    ash_mass = mass * ash_fraction
    
    # Biochar & fixed carbon
    biochar_mass = mass - water_loss - volatile_loss - ash_mass
    fixed_carbon = biochar_mass * 0.8
    
    return {
        'Biochar (kg)': biochar_mass,
        'Gas & Volatiles (kg)': volatile_loss,
        'Ash (kg)': ash_mass,
        'Fixed Carbon (kg)': fixed_carbon,
        'Water Loss (kg)': water_loss
    }

def plot_results(results):
    labels = list(results.keys())
    values = list(results.values())
    
    # Pie chart
    fig, axs = plt.subplots(1,2, figsize=(12,5))
    axs[0].pie(values, labels=labels, autopct='%1.1f%%', startangle=140,
               colors=['#654321','#FFA500','#808080','#2E8B57','#1E90FF'])
    axs[0].set_title("Torrefaction Product Distribution")
    
    # Mass loss line plot (simplified)
    total_mass = sum(values)
    time = np.linspace(0,1,100)
    mass_curve = total_mass * (1 - time*(1-0.7))  # simple approximation
    axs[1].plot(time, mass_curve, color='red', lw=2)
    axs[1].set_xlabel('Normalized Time')
    axs[1].set_ylabel('Mass (kg)')
    axs[1].set_title('Mass Loss Over Time')
    
    plt.tight_layout()
    st.pyplot(fig)

# ===== Streamlit App =====
st.set_page_config(page_title="Torrefaction Simulator", layout="wide")
st.title("🔥 Torrefaction Simulator 🔥")
st.markdown("Simulate the torrefaction process of different waste types with interactive sliders.")

# --- Sidebar Inputs ---
st.sidebar.header("Input Parameters")
waste_type = st.sidebar.selectbox("Waste Type", ['Municipal', 'Wood', 'Agricultural', 'Plastic'])
mass = st.sidebar.slider("Initial Mass (kg)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
moisture = st.sidebar.slider("Moisture (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
temp = st.sidebar.slider("Torrefaction Temperature (°C)", min_value=200, max_value=300, value=250, step=5)
residence_time = st.sidebar.slider("Residence Time (hr)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

# --- Simulation ---
results = simulate_torrefaction(waste_type, mass, moisture, temp, residence_time)

# --- Display Results ---
st.subheader("Simulation Results")
for k, v in results.items():
    st.metric(label=k, value=f"{v:.2f} kg")

# --- Plots ---
st.subheader("Visualizations")
plot_results(results)

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Torrefaction Simulator")

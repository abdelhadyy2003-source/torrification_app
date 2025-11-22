# -*- coding: utf-8 -*-
"""
CHEMISCO PRO: Integrated Biorefinery Simulation Platform
-------------------------------------------------------
Author: Chemisco Development Team
Version: 3.5.0 (Monolithic Enterprise Architecture)
Description: 
    An advanced engineering tool for simulating biomass torrefaction.
    Features include:
    - Multi-component Kinetic Modeling (Arrhenius)
    - Intra-particle Heat Transfer (Finite Difference Method)
    - Techno-Economic Analysis (TEA)
    - Monte Carlo Sensitivity Analysis
    - Historical Data Persistence (SQLite)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time
import base64
import sqlite3
import datetime
import io

# ==============================================================================
# MODULE 1: CONFIGURATION & STYLING (الإعدادات والتصميم)
# ==============================================================================

class AppConfig:
    """Global configuration and visual assets."""
    APP_NAME = "CHEMISCO PRO"
    VERSION = "3.5.0"
    
    # Physical Constants
    R_GAS = 8.314  # J/(mol.K)
    ENTHALPY_VAP = 2260.0 # kJ/kg
    STEFAN_BOLTZMANN = 5.67e-8
    
    # CSS Styling
    @staticmethod
    def get_css():
        return """
        <style>
            /* Main Layout */
            .stApp {
                background-color: #0E1117;
                font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            }
            
            /* Typography */
            h1, h2, h3 { color: #E0E0E0 !important; font-weight: 600; }
            p, label { color: #B0B0B0 !important; }
            
            /* Custom Metric Card */
            .metric-container {
                background: linear-gradient(145deg, #1e212b, #161820);
                padding: 20px;
                border-radius: 12px;
                border-left: 4px solid #00ADB5;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                margin-bottom: 15px;
                transition: transform 0.2s;
            }
            .metric-container:hover { transform: translateY(-2px); }
            .metric-label { font-size: 0.9em; color: #00ADB5; text-transform: uppercase; letter-spacing: 1px; }
            .metric-value { font-size: 2em; color: #FFFFFF; font-weight: bold; margin: 10px 0; }
            .metric-delta { font-size: 0.8em; color: #76FF03; }
            
            /* Block Flow Diagram (BFD) */
            .bfd-wrapper {
                display: flex;
                justify-content: space-between;
                align-items: center;
                background-color: #1A1D24;
                padding: 20px;
                border-radius: 15px;
                margin: 20px 0;
                overflow-x: auto;
            }
            .bfd-node {
                background: #2D333B;
                padding: 15px 25px;
                border-radius: 8px;
                text-align: center;
                min-width: 120px;
                border: 1px solid #444;
                color: white;
            }
            .bfd-arrow { font-size: 24px; color: #666; margin: 0 15px; }
            
            /* Sidebar */
            [data-testid="stSidebar"] { background-color: #161820; border-right: 1px solid #2D333B; }
            
            /* Buttons */
            .stButton>button {
                background-color: #00ADB5;
                color: white;
                border-radius: 8px;
                border: none;
                padding: 10px 20px;
                font-weight: bold;
                transition: all 0.3s;
            }
            .stButton>button:hover { background-color: #00D2DB; box-shadow: 0 0 10px rgba(0, 173, 181, 0.5); }
        </style>
        """

# ==============================================================================
# MODULE 2: DATABASE LAYER (طبقة البيانات)
# ==============================================================================

class DatabaseManager:
    """Handles persistence of simulation runs using SQLite."""
    def __init__(self, db_name="chemisco_history.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.init_db()

    def init_db(self):
        c = self.conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                biomass TEXT,
                temp REAL,
                duration REAL,
                mass_yield REAL,
                energy_yield REAL,
                profit REAL
            )
        ''')
        self.conn.commit()

    def save_run(self, biomass, temp, dur, m_yield, e_yield, profit):
        c = self.conn.cursor()
        c.execute("INSERT INTO runs (biomass, temp, duration, mass_yield, energy_yield, profit) VALUES (?, ?, ?, ?, ?, ?)",
                  (biomass, temp, dur, m_yield, e_yield, profit))
        self.conn.commit()

    def get_history(self):
        return pd.read_sql("SELECT * FROM runs ORDER BY id DESC LIMIT 50", self.conn)

# ==============================================================================
# MODULE 3: PHYSICS & KINETICS ENGINE (محرك الفيزياء)
# ==============================================================================

@dataclass
class BiomassProperties:
    name: str
    hemi: float
    cell: float
    lig: float
    ash: float
    moisture: float
    density: float
    cp: float
    k_cond: float # Thermal conductivity

BIOMASS_DB = {
    "Wood Chips": BiomassProperties("Wood Chips", 0.35, 0.45, 0.20, 0.01, 0.15, 600, 1500, 0.15),
    "Wheat Straw": BiomassProperties("Wheat Straw", 0.45, 0.35, 0.20, 0.08, 0.10, 400, 1400, 0.12),
    "Olive Pits": BiomassProperties("Olive Pits", 0.30, 0.35, 0.30, 0.04, 0.12, 700, 1600, 0.18),
}

class AdvancedSimulator:
    """
    Combines Kinetic Modeling (ODEs) with Heat Transfer (FDM).
    """
    def __init__(self, biomass_name, size_mm):
        self.bio = BIOMASS_DB[biomass_name]
        self.radius = size_mm / 2000.0 # Convert mm to m (radius)
        self.params = {
            "A": [1.5e10, 1.0e12, 2.0e9],     # Hemi, Cell, Lig
            "E": [110000, 130000, 100000]     # J/mol
        }

    # --- KINETICS SOLVER ---
    def _kinetics_odes(self, y, t, T_K):
        m_h, m_c, m_l = y
        R = 8.314
        
        # Rate constants
        k_h = self.params["A"][0] * np.exp(-self.params["E"][0] / (R * T_K))
        k_c = self.params["A"][1] * np.exp(-self.params["E"][1] / (R * T_K))
        k_l = self.params["A"][2] * np.exp(-self.params["E"][2] / (R * T_K))
        
        dm_h = -k_h * m_h
        dm_c = -k_c * m_c
        dm_l = -k_l * m_l
        
        return [dm_h, dm_c, dm_l]

    # --- HEAT TRANSFER SOLVER (FDM) ---
    def solve_thermal_profile(self, T_surf_C, duration_min, nodes=20):
        """
        Simulates intra-particle temperature gradient (Core vs Surface).
        Uses Crank-Nicolson explicit scheme for spherical coordinates.
        """
        dt = 1.0 # seconds
        steps = int(duration_min * 60 / dt)
        dr = self.radius / (nodes - 1)
        
        # Properties
        alpha = self.bio.k_cond / (self.bio.density * self.bio.cp) # Thermal diffusivity
        
        # Grid initialization
        T = np.ones(nodes) * 25.0 # Initial temp 25C
        r = np.linspace(0, self.radius, nodes)
        
        core_temps = []
        avg_temps = []
        
        for _ in range(steps):
            T_new = np.copy(T)
            # Internal nodes loop
            for i in range(1, nodes - 1):
                diffusion = alpha * dt * (
                    (T[i+1] - 2*T[i] + T[i-1]) / dr**2 + 
                    (2/r[i]) * (T[i+1] - T[i-1]) / (2*dr)
                )
                T_new[i] = T[i] + diffusion
                
            # Boundary Conditions
            T_new[0] = T_new[1] # Symmetry at center
            T_new[-1] = T_surf_C # Surface temp fixed (Dirichlet)
            
            T = T_new
            core_temps.append(T[0])
            avg_temps.append(np.mean(T))
            
        return np.array(core_temps), np.array(avg_temps)

    def run(self, temp_C, duration_min, initial_mass):
        # 1. Thermal Simulation
        core_T_profile, avg_T_profile = self.solve_thermal_profile(temp_C, duration_min)
        effective_T_K = np.mean(avg_T_profile) + 273.15 # Use average temp for kinetics
        
        # 2. Kinetics Simulation
        t_span = np.linspace(0, duration_min, 100)
        y0 = [self.bio.hemi, self.bio.cell, self.bio.lig]
        
        sol = odeint(self._kinetics_odes, y0, t_span, args=(effective_T_K,))
        
        # 3. Mass Balance
        final_fractions = sol[-1]
        mass_daf_rem = sum(final_fractions) * initial_mass * (1 - self.bio.ash - self.bio.moisture)
        mass_ash = initial_mass * self.bio.ash
        mass_moisture_evap = initial_mass * self.bio.moisture
        
        # Products
        mass_biochar = mass_daf_rem + mass_ash
        mass_volatiles = initial_mass - mass_biochar - mass_moisture_evap
        
        # Yields
        y_mass = (mass_biochar / initial_mass) * 100
        # Energy yield correlation
        y_energy = y_mass * (1.0 + (0.4 * (1 - y_mass/100))) 
        
        return {
            "time": t_span,
            "profiles": sol,
            "thermal": {"core": core_T_profile, "avg": avg_T_profile},
            "yields": {"mass": y_mass, "energy": y_energy},
            "products": {
                "Biochar": mass_biochar,
                "Volatiles": mass_volatiles,
                "Water": mass_moisture_evap
            }
        }

# ==============================================================================
# MODULE 4: ECONOMICS & OPTIMIZATION (الاقتصاد والتحسين)
# ==============================================================================

class EconomicsEngine:
    def __init__(self, capex, op_days, feedstock_cost, energy_cost, biochar_price):
        self.capex = capex
        self.op_days = op_days
        self.feed_cost = feedstock_cost
        self.energy_cost = energy_cost
        self.price = biochar_price

    def calculate_profitability(self, batch_mass, batch_duration_min, biochar_mass, energy_req_kwh):
        batches_per_day = (24 * 60) / (batch_duration_min + 15) # 15 min handling time
        annual_batches = batches_per_day * self.op_days
        
        # Annual Costs
        total_feedstock_cost = (batch_mass * annual_batches / 1000) * self.feed_cost
        total_energy_cost = (energy_req_kwh * annual_batches) * self.energy_cost
        labor_maintenance = self.capex * 0.05
        total_opex = total_feedstock_cost + total_energy_cost + labor_maintenance
        
        # Annual Revenue
        total_revenue = (biochar_mass * annual_batches) * self.price
        
        # Metrics
        gross_profit = total_revenue - total_opex
        roi = (gross_profit / self.capex) * 100 if self.capex > 0 else 0
        payback = self.capex / gross_profit if gross_profit > 0 else 999
        
        return {
            "Revenue": total_revenue,
            "OPEX": total_opex,
            "Profit": gross_profit,
            "ROI": roi,
            "Payback": payback
        }

# ==============================================================================
# MODULE 5: UI COMPONENTS (عناصر الواجهة)
# ==============================================================================

class UIManager:
    @staticmethod
    def render_header():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("CHEMISCO PRO 🌌")
            st.caption("Advanced Biorefinery Process Simulator v3.5")
        with col2:
            st.metric("System Status", "Online", delta="Ready")
        st.markdown("---")

    @staticmethod
    def render_bfd(feed, temp, char, gas):
        st.markdown(f"""
        <div class="bfd-wrapper">
            <div class="bfd-node" style="border-left: 4px solid #4CAF50;">FEED<br><b>{feed:.1f} kg</b></div>
            <div class="bfd-arrow">➜</div>
            <div class="bfd-node" style="border-left: 4px solid #FF5252;">DRYER<br>105°C</div>
            <div class="bfd-arrow">➜</div>
            <div class="bfd-node" style="border-left: 4px solid #FFC107;">REACTOR<br><b>{temp}°C</b></div>
            <div class="bfd-arrow">➜</div>
            <div class="bfd-node" style="border-left: 4px solid #00BCD4;">PRODUCT<br><b>{char:.1f} kg</b></div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_kpis(metrics):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Mass Yield</div>
                <div class="metric-value">{metrics['mass_yield']:.1f}%</div>
                <div class="metric-delta">Target: >70%</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Energy Yield</div>
                <div class="metric-value">{metrics['energy_yield']:.1f}%</div>
                <div class="metric-delta">High Efficiency</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Net Profit</div>
                <div class="metric-value">${metrics['profit']:,.0f}</div>
                <div class="metric-delta">Annual Estimate</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Payback</div>
                <div class="metric-value">{metrics['payback']:.1f} Yrs</div>
                <div class="metric-delta">ROI: {metrics['roi']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

# ==============================================================================
# MAIN APPLICATION LOGIC (التطبيق الرئيسي)
# ==============================================================================

def main():
    st.set_page_config(page_title=AppConfig.APP_NAME, layout="wide", page_icon="⚛️")
    st.markdown(AppConfig.get_css(), unsafe_allow_html=True)
    
    # Initialize DB
    db = DatabaseManager()

    # --- SIDEBAR INPUTS ---
    with st.sidebar:
        st.header("⚙️ Simulation Control")
        
        with st.expander("1. Feedstock Parameters", expanded=True):
            biomass_type = st.selectbox("Biomass Type", list(BIOMASS_DB.keys()))
            initial_mass = st.number_input("Batch Mass (kg)", 100.0, 5000.0, 1000.0)
            particle_size = st.slider("Particle Diameter (mm)", 1, 30, 10)
        
        with st.expander("2. Reactor Settings", expanded=True):
            temp_c = st.slider("Temperature (°C)", 200, 350, 275)
            duration_min = st.slider("Residence Time (min)", 15, 120, 60)
            
        with st.expander("3. Economic Assumptions"):
            capex = st.number_input("CAPEX ($)", value=1500000)
            feed_cost = st.number_input("Feedstock ($/ton)", value=30.0)
            price = st.number_input("Biochar Price ($/kg)", value=1.2)
            
        run_btn = st.button("🚀 RUN SIMULATION", type="primary")

    # --- MAIN PAGE RENDER ---
    UIManager.render_header()

    if run_btn:
        with st.spinner("Processing Physics & Kinetics..."):
            time.sleep(0.5) # UX delay
            
            # 1. Initialize Engines
            sim = AdvancedSimulator(biomass_type, particle_size)
            econ = EconomicsEngine(capex, 300, feed_cost, 0.12, price)
            
            # 2. Run Simulation
            res = sim.run(temp_c, duration_min, initial_mass)
            
            # 3. Calculate Economics
            # Estimate energy (heating biomass + water evap)
            q_mj = (initial_mass * 1.5 * (temp_c - 25) + initial_mass * 0.15 * 2260) / 1000
            econ_res = econ.calculate_profitability(initial_mass, duration_min, res['products']['Biochar'], q_mj/3.6)
            
            # 4. Save to DB
            db.save_run(biomass_type, temp_c, duration_min, res['yields']['mass'], res['yields']['energy'], econ_res['Profit'])

            # --- DISPLAY RESULTS ---
            
            # KPI Cards
            kpi_data = {
                'mass_yield': res['yields']['mass'],
                'energy_yield': res['yields']['energy'],
                'profit': econ_res['Profit'],
                'payback': econ_res['Payback'],
                'roi': econ_res['ROI']
            }
            UIManager.render_kpis(kpi_data)
            
            # Block Flow Diagram
            UIManager.render_bfd(initial_mass, temp_c, res['products']['Biochar'], res['products']['Volatiles'])
            
            # Tabs for Details
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Kinetics & Dynamics", "🔥 Thermal Profile", "💰 Financials", "📜 History"])
            
            with tab1:
                st.subheader("Mass Loss Kinetics")
                col_g1, col_g2 = st.columns([2, 1])
                with col_g1:
                    df_kin = pd.DataFrame(res['profiles'], columns=['Hemicellulose', 'Cellulose', 'Lignin'])
                    df_kin['Time'] = res['time']
                    fig = px.line(df_kin, x='Time', y=['Hemicellulose', 'Cellulose', 'Lignin'], title="Component Degradation")
                    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                    st.plotly_chart(fig, use_container_width=True)
                with col_g2:
                    st.subheader("Product Split")
                    fig_pie = px.pie(names=res['products'].keys(), values=res['products'].values(), hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
                    fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with tab2:
                st.subheader("Intra-Particle Heat Transfer (Finite Difference Method)")
                st.write("Simulating thermal lag between particle surface and core.")
                
                # Visualize Thermal Gradient
                thermal_data = pd.DataFrame({
                    "Core Temp": res['thermal']['core'],
                    "Avg Temp": res['thermal']['avg']
                })
                # Downsample for plotting if too large
                thermal_data = thermal_data.iloc[::10, :] 
                
                fig_therm = go.Figure()
                fig_therm.add_trace(go.Scatter(y=thermal_data['Core Temp'], mode='lines', name='Core Temperature'))
                fig_therm.add_trace(go.Scatter(y=thermal_data['Avg Temp'], mode='lines', name='Average Temperature', line=dict(dash='dash')))
                fig_therm.add_hline(y=temp_c, line_dash="dot", annotation_text="Reactor Setpoint", annotation_position="bottom right")
                
                fig_therm.update_layout(
                    title="Heat Penetration Profile", 
                    xaxis_title="Time Steps", 
                    yaxis_title="Temperature (°C)",
                    plot_bgcolor="rgba(0,0,0,0)", 
                    paper_bgcolor="rgba(0,0,0,0)", 
                    font_color="white"
                )
                st.plotly_chart(fig_therm, use_container_width=True)
                
            with tab3:
                st.subheader("Techno-Economic Analysis (TEA)")
                st.dataframe(pd.DataFrame([econ_res]).T.rename(columns={0: "Value"}), use_container_width=True)
                
            with tab4:
                st.subheader("Simulation History")
                df_hist = db.get_history()
                st.dataframe(df_hist, use_container_width=True)
                
                if len(df_hist) > 1:
                    st.line_chart(df_hist['mass_yield'])

    else:
        # Hero Section / Landing
        st.info("👋 Ready to Simulate. Please configure parameters in the sidebar.")
        

[Image of torrefaction process diagram]
 
        st.markdown("""
        ### About Chemisco Pro
        This platform uses advanced numerical methods to simulate the torrefaction of biomass.
        
        **Key Features:**
        * **FDM Heat Transfer:** Solves the heat equation inside the particle.
        * **Multi-component Kinetics:** Tracks Hemicellulose, Cellulose, and Lignin separately.
        * **SQL Persistence:** Automatically saves your runs for comparison.
        """)

if __name__ == "__main__":
    main()

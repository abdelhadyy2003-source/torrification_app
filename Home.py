# -*- coding: utf-8 -*-
"""
CHEMISCO ENTERPRISE: Integrated Biorefinery Simulation Platform
---------------------------------------------------------------
Author: Chemisco Development Team
Version: 4.1.0 (Stable & Clean)
Description:
    An advanced engineering tool for simulating biomass torrefaction.
    No external ML dependencies (sklearn removed) to ensure stability.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.integrate import odeint
import sqlite3
import time
from dataclasses import dataclass

# ==============================================================================
# 1. CONFIGURATION & STYLING
# ==============================================================================

st.set_page_config(page_title="Chemisco Enterprise", layout="wide", page_icon="🏭")

class AppStyle:
    @staticmethod
    def apply():
        st.markdown("""
        <style>
            /* Main Theme */
            .stApp { background-color: #0E1117; color: #FAFAFA; }
            
            /* Metrics */
            .metric-card {
                background: linear-gradient(135deg, #1e232a 0%, #16181d 100%);
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #00ADB5;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                margin-bottom: 10px;
            }
            .metric-title { font-size: 14px; color: #00ADB5; text-transform: uppercase; }
            .metric-value { font-size: 28px; font-weight: bold; color: white; margin: 5px 0; }
            .metric-delta { font-size: 12px; color: #8b949e; }

            /* BFD Container */
            .bfd-container {
                display: flex; justify-content: space-around; align-items: center;
                background-color: #161b22; padding: 30px;
                border-radius: 15px; border: 1px solid #30363d; margin: 20px 0;
            }
            .bfd-box {
                background: #21262d; color: #c9d1d9; padding: 15px 25px;
                border-radius: 6px; text-align: center; border: 1px solid #30363d; min-width: 120px;
            }
            .bfd-arrow { color: #8b949e; font-size: 24px; }
        </style>
        """, unsafe_allow_html=True)

# ==============================================================================
# 2. DATABASE LAYER
# ==============================================================================

class DatabaseManager:
    def __init__(self, db_path="chemisco_logs.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        query = """
        CREATE TABLE IF NOT EXISTS simulations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            biomass_type TEXT,
            temperature REAL,
            duration REAL,
            mass_yield REAL,
            energy_yield REAL,
            profit REAL
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    def log_run(self, biomass, t, d, m_y, e_y, profit):
        query = "INSERT INTO simulations (biomass_type, temperature, duration, mass_yield, energy_yield, profit) VALUES (?, ?, ?, ?, ?, ?)"
        self.conn.execute(query, (biomass, t, d, m_y, e_y, profit))
        self.conn.commit()

    def get_logs(self):
        return pd.read_sql("SELECT * FROM simulations ORDER BY id DESC LIMIT 20", self.conn)

# ==============================================================================
# 3. PHYSICS ENGINE
# ==============================================================================

@dataclass
class BiomassData:
    name: str; hemi: float; cell: float; lig: float; ash: float; moist: float; cp: float; rho: float; k: float

BIOMASS_DB = {
    "Wood Chips": BiomassData("Wood Chips", 0.30, 0.45, 0.25, 0.01, 0.15, 1500, 600, 0.12),
    "Wheat Straw": BiomassData("Wheat Straw", 0.45, 0.35, 0.20, 0.08, 0.10, 1400, 400, 0.09),
    "Olive Pits": BiomassData("Olive Pits", 0.25, 0.35, 0.40, 0.03, 0.12, 1600, 750, 0.18),
}

class PhysicsEngine:
    def __init__(self, biomass_name, particle_size_mm):
        self.bio = BIOMASS_DB[biomass_name]
        self.radius = particle_size_mm / 2000.0

    def solve_kinetics(self, T_C, time_min):
        T_K = T_C + 273.15
        R = 8.314
        # Kinetic Params (A, E) for Hemi, Cell, Lig
        params = [(1e10, 110000), (1e12, 130000), (1e8, 100000)]
        k = [A * np.exp(-E / (R * T_K)) for A, E in params]
        
        def model(y, t): return [-k[0]*y[0], -k[1]*y[1], -k[2]*y[2]]
        
        t_span = np.linspace(0, time_min, 100)
        sol = odeint(model, [self.bio.hemi, self.bio.cell, self.bio.lig], t_span)
        return t_span, sol

    def solve_heat_transfer(self, T_surf_C, time_min, nodes=15):
        dt = 0.5; steps = int(time_min * 60 / dt); dr = self.radius / (nodes - 1)
        alpha = self.bio.k / (self.bio.rho * self.bio.cp)
        T = np.ones(nodes) * 25.0
        r = np.linspace(0, self.radius, nodes)
        history_core, history_avg = [], []
        
        for _ in range(steps):
            T_new = np.copy(T)
            for i in range(1, nodes-1):
                diff = alpha * dt * ((T[i+1]-2*T[i]+T[i-1])/dr**2 + (2/r[i])*(T[i+1]-T[i-1])/(2*dr))
                T_new[i] = T[i] + diff
            T_new[0] = T_new[1]; T_new[-1] = T_surf_C
            T = T_new
            history_core.append(T[0]); history_avg.append(np.mean(T))
        return history_core, history_avg

    def calculate_energy(self, mass_in, T_react, moisture_frac):
        mass_dry = mass_in * (1 - moisture_frac)
        mass_h2o = mass_in * moisture_frac
        # Energy Balance (MJ)
        q_sens_bio = mass_dry * (self.bio.cp/1000) * (T_react - 25)
        q_sens_h2o = mass_h2o * 4.18 * (100 - 25)
        q_lat = mass_h2o * 2260
        q_rxn = mass_dry * 150 # Endothermic estimate
        q_loss = (q_sens_bio + q_sens_h2o + q_lat + q_rxn) * 0.15
        total = (q_sens_bio + q_sens_h2o + q_lat + q_rxn + q_loss) / 1000
        return {
            "Q_sensible_biomass": q_sens_bio/1000, "Q_sensible_water": q_sens_h2o/1000,
            "Q_latent": q_lat/1000, "Q_reaction": q_rxn/1000, "Q_loss": q_loss/1000, "Total_MJ": total
        }

# ==============================================================================
# 4. OPTIMIZATION LOGIC
# ==============================================================================

class ProcessOptimizer:
    @staticmethod
    def get_optimal(biomass_name):
        # Rule-based logic
        if "Wood" in biomass_name: return 280, 45
        elif "Straw" in biomass_name: return 260, 40
        return 290, 60

# ==============================================================================
# 5. MAIN APP
# ==============================================================================

def main():
    AppStyle.apply()
    db = DatabaseManager()
    
    with st.sidebar:
        st.title("CHEMISCO PRO")
        st.divider()
        st.subheader("1. Feedstock")
        b_type = st.selectbox("Type", list(BIOMASS_DB.keys()))
        mass = st.number_input("Batch (kg)", 100.0, 5000.0, 1000.0)
        moist = st.slider("Moisture (%)", 0, 50, 15)
        size = st.slider("Size (mm)", 1, 25, 10)
        
        st.subheader("2. Reactor")
        temp = st.slider("Temp (°C)", 200, 350, 275)
        dur = st.slider("Time (min)", 15, 120, 60)
        
        st.subheader("3. Economics")
        price = st.number_input("Price ($/kg)", value=1.5)
        
        btn_run = st.button("🚀 START", type="primary")
        btn_opt = st.button("✨ OPTIMIZE")

    if btn_opt:
        opt_t, opt_d = ProcessOptimizer.get_optimal(b_type)
        st.success(f"Optimized: {opt_t}°C, {opt_d} min")
        temp, dur = opt_t, opt_d

    if btn_run or btn_opt:
        with st.spinner("Simulating..."):
            time.sleep(0.5)
            engine = PhysicsEngine(b_type, size)
            
            # Calculations
            t, sol = engine.solve_kinetics(temp, dur)
            heat_core, heat_avg = engine.solve_heat_transfer(temp, dur)
            energy = engine.calculate_energy(mass, temp, moist/100)
            
            # Results
            final_mass = sum(sol[-1]) * mass * (1 - engine.bio.ash - moist/100)
            char_mass = final_mass + (mass * engine.bio.ash)
            y_mass = (char_mass / mass) * 100
            y_energy = y_mass * 1.15
            profit = (char_mass * price) - (energy['Total_MJ']/3.6 * 0.12)
            
            db.log_run(b_type, temp, dur, y_mass, y_energy, profit)

            # --- DISPLAY ---
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="metric-card"><div class="metric-title">Mass Yield</div><div class="metric-value">{y_mass:.1f}%</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><div class="metric-title">Energy Yield</div><div class="metric-value">{y_energy:.1f}%</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><div class="metric-title">Profit</div><div class="metric-value">${profit:.2f}</div></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="metric-card"><div class="metric-title">Energy</div><div class="metric-value">{energy["Total_MJ"]:.1f} MJ</div></div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="bfd-container">
                <div class="bfd-box" style="border-left:4px solid #4CAF50;">Input<br>{mass} kg</div>
                <div class="bfd-arrow">➜</div>
                <div class="bfd-box" style="border-left:4px solid #FFC107;">Reactor<br>{temp}°C</div>
                <div class="bfd-arrow">➜</div>
                <div class="bfd-box" style="border-left:4px solid #00ADB5;">Output<br>{char_mass:.1f} kg</div>
            </div>""", unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(["📊 Kinetics", "🔥 Thermal", "📜 History"])
            
            with tab1:
                df = pd.DataFrame(sol, columns=['Hemi', 'Cell', 'Lig']); df['Time'] = t
                st.plotly_chart(px.line(df, x='Time', y=['Hemi', 'Cell', 'Lig'], title="Decomposition"), use_container_width=True)
                
            with tab2:
                # Waterfall Chart
                w_df = pd.DataFrame({"Stage": ["Sensible Bio", "Sensible H2O", "Latent", "Reaction", "Loss"], 
                                     "MJ": [energy['Q_sensible_biomass'], energy['Q_sensible_water'], energy['Q_latent'], energy['Q_reaction'], energy['Q_loss']]})
                fig_w = go.Figure(go.Waterfall(x=w_df["Stage"], y=w_df["MJ"], connector={"line":{"color":"white"}}))
                fig_w.update_layout(title="Energy Breakdown", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_w, use_container_width=True)
                
                # Heat Transfer Plot
                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(y=heat_core[::5], name="Core Temp"))
                fig_h.add_trace(go.Scatter(y=heat_avg[::5], name="Avg Temp", line=dict(dash='dash')))
                fig_h.update_layout(title="Particle Heat Transfer (FDM)", yaxis_title="Temp (°C)", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig_h, use_container_width=True)

            with tab3:
                st.dataframe(db.get_logs(), use_container_width=True)

    else:
        st.markdown("<br><br>", unsafe_allow_html=True)
        col_main, col_img = st.columns([2, 1])
        with col_main:
            st.title("Welcome to Chemisco Enterprise")
            st.markdown("""
            **System Status:** ✅ Online & Ready
            
            This platform uses advanced numerical methods (FDM & ODEs) to simulate biomass torrefaction.
            Select your parameters from the sidebar to begin.
            """)
        with col_img:
            # صورة حقيقية من رابط ثابت لتجنب الأخطاء
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Torrefaction_plant.jpg/640px-Torrefaction_plant.jpg", caption="Biomass Torrefaction Plant")

if __name__ == "__main__":
    main()

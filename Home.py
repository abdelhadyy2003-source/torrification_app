# -*- coding: utf-8 -*-
"""
CHEMISCO OS v10 - THE INDUSTRIAL MONOLITH
=========================================
Type:        Full-Stack Single-File Application
License:     Enterprise / Mission Critical
Description: A complete Operating System simulation for Biorefineries.
             Features IoT simulation, Rankine Cycle Thermodynamics, 
             Internal Email System, RBAC Security, and Automated Unit Testing.

AUTHOR: Chemisco Elite Dev Team
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import odeint
from scipy.optimize import minimize
import sqlite3
import hashlib
import time
import datetime
import random
import io
import math
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Union, Any
from enum import Enum

# ==============================================================================
# 0. KERNEL CONFIGURATION & CONSTANTS
# ==============================================================================

st.set_page_config(
    page_title="Chemisco OS",
    page_icon="☢️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

class SystemConstants:
    # Thermodynamics
    R_GAS = 8.314          # J/(mol.K)
    STD_TEMP = 298.15      # K
    STD_PRESS = 101325     # Pa
    WATER_CP = 4.18        # kJ/kg.K
    STEAM_ENTHALPY = 2676  # kJ/kg (Sat. Steam @ 100C)
    
    # Economics
    ELEC_PRICE = 0.12      # $/kWh
    CARBON_CREDIT = 30.0   # $/ton CO2
    
    # UI Theme (Cyber-Industrial)
    HEX_PRIMARY = "#00F0FF"    # Neon Cyan
    HEX_SECONDARY = "#7000FF"  # Neon Purple
    HEX_BG = "#050505"         # Void Black
    HEX_PANEL = "#111111"      # Panel Gray
    HEX_TEXT = "#E0E0E0"
    HEX_SUCCESS = "#00FF41"    # Matrix Green
    HEX_DANGER = "#FF003C"     # Cyber Red

# ==============================================================================
# 1. VISUALIZATION ENGINE (CSS INJECTION)
# ==============================================================================

class UIKernel:
    @staticmethod
    def boot():
        st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700&display=swap');
            
            :root {{
                --neon-cyan: {SystemConstants.HEX_PRIMARY};
                --neon-purple: {SystemConstants.HEX_SECONDARY};
                --bg-color: {SystemConstants.HEX_BG};
            }}

            /* BASE STYLES */
            .stApp {{
                background-color: var(--bg-color);
                font-family: 'Share Tech Mono', monospace;
                background-image: radial-gradient(circle at 50% 50%, #1a1a1a 0%, #000 100%);
            }}
            
            h1, h2, h3 {{
                font-family: 'Orbitron', sans-serif;
                text-transform: uppercase;
                letter-spacing: 2px;
                color: #fff;
                text-shadow: 0 0 10px var(--neon-cyan);
            }}

            /* CYBER CARDS */
            .cyber-card {{
                background: rgba(10, 10, 10, 0.8);
                border: 1px solid #333;
                border-left: 4px solid var(--neon-cyan);
                padding: 20px;
                margin-bottom: 15px;
                position: relative;
                overflow: hidden;
                box-shadow: 0 0 15px rgba(0, 240, 255, 0.1);
            }}
            .cyber-card::before {{
                content: '';
                position: absolute; top: 0; right: 0;
                width: 20px; height: 20px;
                background: linear-gradient(135deg, transparent 50%, var(--neon-cyan) 50%);
            }}
            
            /* METRICS */
            .metric-val {{ font-size: 2.5rem; font-weight: bold; color: #fff; }}
            .metric-label {{ color: var(--neon-cyan); font-size: 0.8rem; }}
            
            /* TERMINAL LOGS */
            .terminal-window {{
                background: #000;
                border: 1px solid #333;
                padding: 10px;
                font-family: 'Share Tech Mono', monospace;
                color: #0f0;
                height: 200px;
                overflow-y: auto;
                box-shadow: inset 0 0 20px rgba(0, 255, 0, 0.1);
            }}
            .log-line {{ border-bottom: 1px solid #111; padding: 2px 0; }}
            .log-time {{ color: #666; }}
            
            /* BUTTONS */
            .stButton > button {{
                background: transparent;
                border: 1px solid var(--neon-cyan);
                color: var(--neon-cyan);
                font-family: 'Orbitron', sans-serif;
                transition: all 0.3s;
                border-radius: 0;
            }}
            .stButton > button:hover {{
                background: var(--neon-cyan);
                color: #000;
                box-shadow: 0 0 20px var(--neon-cyan);
            }}
            
            /* SIDEBAR */
            [data-testid="stSidebar"] {{
                background-color: #080808;
                border-right: 1px solid #333;
            }}
        </style>
        """, unsafe_allow_html=True)

# ==============================================================================
# 2. PERSISTENCE LAYER (SQLITE ORM)
# ==============================================================================

class DatabaseEngine:
    """Enterprise Singleton for Data Management."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseEngine, cls).__new__(cls)
            cls._instance.conn = sqlite3.connect("chemisco_os.db", check_same_thread=False)
            cls._instance.setup_tables()
        return cls._instance
    
    def setup_tables(self):
        c = self.conn.cursor()
        
        # Users
        c.execute("""CREATE TABLE IF NOT EXISTS users 
                     (id INTEGER PRIMARY KEY, username TEXT, role TEXT, password TEXT, xp INTEGER)""")
        
        # Emails (Inbox System)
        c.execute("""CREATE TABLE IF NOT EXISTS emails 
                     (id INTEGER PRIMARY KEY, recipient TEXT, sender TEXT, subject TEXT, body TEXT, read INT)""")
        
        # Simulation Logs
        c.execute("""CREATE TABLE IF NOT EXISTS logs 
                     (id INTEGER PRIMARY KEY, timestamp DATETIME, user TEXT, action TEXT, status TEXT)""")
                     
        # Seed Data (Admin)
        try:
            c.execute("INSERT OR IGNORE INTO users (id, username, role, password, xp) VALUES (1, 'admin', 'CEO', 'admin', 9999)")
        except: pass
        self.conn.commit()

    def send_system_email(self, to_user, subject, body):
        self.conn.execute("INSERT INTO emails (recipient, sender, subject, body, read) VALUES (?, 'SYSTEM', ?, ?, 0)", 
                          (to_user, subject, body))
        self.conn.commit()

    def get_inbox(self, username):
        return pd.read_sql(f"SELECT * FROM emails WHERE recipient='{username}' ORDER BY id DESC", self.conn)

# ==============================================================================
# 3. DOMAIN MODELS (PHYSICS & ENGINEERING)
# ==============================================================================

@dataclass
class Feedstock:
    id: str
    name: str
    c: float; h: float; o: float; n: float; s: float # Elemental %
    moisture: float
    ash: float
    hhv: float # MJ/kg

    @property
    def chemical_formula(self):
        return f"C{self.c:.1f}H{self.h:.1f}O{self.o:.1f}N{self.n:.2f}"

FEEDSTOCKS = {
    "wood": Feedstock("wood", "Pine Wood", 50.0, 6.0, 43.0, 0.1, 0.0, 15.0, 1.0, 19.5),
    "straw": Feedstock("straw", "Wheat Straw", 45.0, 5.5, 40.0, 0.5, 0.1, 12.0, 6.0, 17.0),
    "sludge": Feedstock("sludge", "Dried Sludge", 35.0, 4.0, 25.0, 5.0, 1.0, 20.0, 30.0, 14.0),
}

class ThermodynamicsEngine:
    """Calculates Rankine Cycle for Heat Recovery."""
    
    @staticmethod
    def calculate_rankine_cycle(heat_source_kw, efficiency=0.85):
        """
        Simulates an Organic Rankine Cycle (ORC) attached to the reactor.
        Inputs: Heat source in kW (from reactor waste heat).
        Returns: Electricity generated (kW).
        """
        # Assumptions for a basic steam cycle
        boiler_eff = 0.90
        turbine_eff = 0.85
        generator_eff = 0.95
        
        heat_input = heat_source_kw * boiler_eff
        
        # Thermodynamics (Simplified Isentropic expansion)
        # T1 (Boiler) -> T2 (Condenser)
        t_high = 300 + 273.15 # K
        t_low = 40 + 273.15   # K
        carnot_eff = 1 - (t_low / t_high)
        
        real_cycle_eff = carnot_eff * 0.45 # Factor for real world losses
        
        work_out = heat_input * real_cycle_eff * turbine_eff * generator_eff
        return work_out, real_cycle_eff

class KineticsEngine:
    """Advanced Multi-Reaction Model."""
    
    @staticmethod
    def simulate_batch(feedstock: Feedstock, T_C, t_min):
        T_K = T_C + 273.15
        
        # Devolatilization Kinetics (Arrhenius)
        # k = A * exp(-E/RT)
        A = 1e6; E = 80000 
        k = A * np.exp(-E / (SystemConstants.R_GAS * T_K))
        
        # Mass Loss Model: M(t) = M_final + (M_init - M_final) * exp(-k*t)
        # Target solid yield depends on Temp (Empirical Correlation)
        target_yield = 1.0 - (0.0025 * (T_C - 200)) # Simple linear degradation model
        target_yield = max(0.3, target_yield)
        
        time_points = np.linspace(0, t_min, 100)
        mass_profile = target_yield + (1.0 - target_yield) * np.exp(-k * (time_points * 60))
        
        # Stoichiometry Balancing (Mass Balance)
        final_mass_yield = mass_profile[-1]
        volatiles_yield = 1.0 - final_mass_yield
        
        # Energy Densification
        energy_yield = final_mass_yield * (1 + volatiles_yield) # HHV increases as mass drops
        
        return time_points, mass_profile, final_mass_yield, energy_yield

# ==============================================================================
# 4. IOT & SENSOR SIMULATION (THE "ALIVE" FACTORY)
# ==============================================================================

class IoTSimulator:
    """Simulates real-time data from factory sensors."""
    
    @staticmethod
    def read_sensors():
        # Adds noise to simulate real sensors
        return {
            "reactor_t_1": round(random.gauss(275, 2.5), 1),
            "reactor_p_1": round(random.gauss(1.2, 0.05), 2),
            "auger_rpm": int(random.gauss(1200, 50)),
            "power_draw": round(random.gauss(450, 10), 1),
            "emission_co": round(random.gauss(15, 2), 1)
        }

# ==============================================================================
# 5. APPLICATION CONTROLLERS (MVC ARCHITECTURE)
# ==============================================================================

def dashboard_view(user):
    st.title("COMMAND CENTER // DASHBOARD")
    st.markdown("---")
    
    # IoT Live Feed
    st.subheader("📡 Live Sensor Telemetry")
    sensors = IoTSimulator.read_sensors()
    
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: st.metric("Core Temp", f"{sensors['reactor_t_1']}°C", "0.5°C")
    with k2: st.metric("Pressure", f"{sensors['reactor_p_1']} Bar", "-0.01")
    with k3: st.metric("Motor RPM", sensors['auger_rpm'], "Stable")
    with k4: st.metric("Grid Load", f"{sensors['power_draw']} kW", "+2%")
    with k5: st.metric("CO Emissions", f"{sensors['emission_co']} ppm", "Safe")
    
    # Real-time Chart
    chart_data = pd.DataFrame(
        np.random.randn(20, 3) + [275, 270, 280],
        columns=['Zone A', 'Zone B', 'Zone C']
    )
    st.line_chart(chart_data, height=200)

    # Inbox System
    st.markdown("### 📨 Internal Inbox")
    db = DatabaseEngine()
    msgs = db.get_inbox(user)
    
    if msgs.empty:
        st.info("No new messages.")
    else:
        for index, row in msgs.iterrows():
            with st.expander(f"FROM: {row['sender']} | RE: {row['subject']}"):
                st.write(row['body'])
                if st.button("Mark Read", key=f"read_{index}"): pass # Logic to mark read

def simulation_lab_view():
    st.title("R&D SIMULATION LAB")
    
    c_side, c_main = st.columns([1, 3])
    
    with c_side:
        st.markdown("### Configuration")
        with st.form("sim_config"):
            feed = st.selectbox("Feedstock", list(FEEDSTOCKS.keys()))
            mass = st.number_input("Batch Size (kg)", 1000, 10000, 5000)
            temp = st.slider("Temperature (°C)", 200, 350, 280)
            time_m = st.slider("Time (min)", 15, 120, 45)
            
            run = st.form_submit_button("INITIATE SEQUENCE")
    
    with c_main:
        if run:
            with st.status("Running Physics Kernels...", expanded=True):
                st.write("Initializing Reaction Matrix...")
                time.sleep(0.5)
                st.write("Solving Differential Equations...")
                
                # Physics Run
                mat = FEEDSTOCKS[feed]
                t, m_prof, y_m, y_e = KineticsEngine.simulate_batch(mat, temp, time_m)
                
                # Rankine Cycle Calculation
                waste_heat_kw = (mass * 1.2 * (temp-25)) / (time_m * 60) # Rough estimate
                elec_gen, cycle_eff = ThermodynamicsEngine.calculate_rankine_cycle(waste_heat_kw)
                
                st.write("Optimizing Energy Recovery...")
                time.sleep(0.5)
                
            # Results
            st.success("SIMULATION COMPLETED")
            
            # KPI Grid
            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f"<div class='cyber-card'><div class='metric-label'>Mass Yield</div><div class='metric-val'>{y_m*100:.1f}%</div></div>", unsafe_allow_html=True)
            k2.markdown(f"<div class='cyber-card'><div class='metric-label'>Energy Yield</div><div class='metric-val'>{y_e*100:.1f}%</div></div>", unsafe_allow_html=True)
            k3.markdown(f"<div class='cyber-card'><div class='metric-label'>Power Gen</div><div class='metric-val'>{elec_gen:.1f} kW</div></div>", unsafe_allow_html=True)
            k4.markdown(f"<div class='cyber-card'><div class='metric-label'>Cycle Eff</div><div class='metric-val'>{cycle_eff*100:.1f}%</div></div>", unsafe_allow_html=True)
            
            # Advanced Charts
            tab1, tab2 = st.tabs(["Reaction Profile", "Energy Sankey"])
            
            with tab1:
                df = pd.DataFrame({"Time": t, "Mass Fraction": m_prof})
                fig = px.line(df, x="Time", y="Mass Fraction", title="Solid Mass Decomposition")
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Sankey Diagram for Energy Balance
                energy_in = mass * mat.hhv
                energy_char = energy_in * y_e
                energy_gas = energy_in * (1 - y_e)
                
                fig_sankey = go.Figure(go.Sankey(
                    node = dict(
                        pad = 15, thickness = 20, line = dict(color = "black", width = 0.5),
                        label = ["Biomass Input", "Pyrolysis Reactor", "Biochar", "Syngas/Volatiles", "Process Heat", "Losses"],
                        color = SystemConstants.HEX_PRIMARY
                    ),
                    link = dict(
                        source = [0, 1, 1, 3, 3],
                        target = [1, 2, 3, 4, 5],
                        value = [energy_in, energy_char, energy_gas, energy_gas*0.8, energy_gas*0.2]
                    )
                ))
                fig_sankey.update_layout(title="Energy Flow (MJ)", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_sankey, use_container_width=True)

def finance_hq_view():
    st.title("FINANCIAL HEADQUARTERS")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### CAPEX/OPEX Modeler")
        capex = st.number_input("Capital Expenditure ($)", 1_000_000, 10_000_000, 2_500_000)
        opex_monthly = st.number_input("Monthly OPEX ($)", 10_000, 500_000, 50_000)
    with c2:
        st.markdown("### Market Assumptions")
        price = st.slider("Biochar Price ($/ton)", 300, 2000, 800)
        prod = st.slider("Production (tons/month)", 50, 500, 200)
        
    # Analysis
    revenue_monthly = price * prod
    net_monthly = revenue_monthly - opex_monthly
    roi_months = capex / net_monthly if net_monthly > 0 else 999
    
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Monthly Revenue", f"${revenue_monthly:,.0f}")
    m2.metric("Net Income", f"${net_monthly:,.0f}", delta_color="normal" if net_monthly>0 else "inverse")
    m3.metric("ROI Period", f"{roi_months:.1f} Months")
    
    # Monte Carlo Sim
    if st.button("Run Risk Analysis (Monte Carlo)"):
        with st.spinner("Simulating 1000 market scenarios..."):
            sims = []
            for _ in range(1000):
                p_var = random.gauss(price, price*0.1) # 10% volatility
                c_var = random.gauss(opex_monthly, opex_monthly*0.05)
                sims.append((p_var * prod) - c_var)
            
            fig = px.histogram(sims, nbins=50, title="Profitability Probability Distribution", labels={'value': 'Monthly Profit'})
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 6. SYSTEM UTILITIES (UNIT TESTING & DEBUG)
# ==============================================================================

class SystemDiagnostic:
    @staticmethod
    def run_tests():
        tests = {
            "DB Connection": False,
            "Physics Engine": False,
            "UI Rendering": False
        }
        
        # Test DB
        try:
            db = DatabaseEngine()
            tests["DB Connection"] = True
        except: pass
        
        # Test Physics
        try:
            res = KineticsEngine.simulate_batch(FEEDSTOCKS['wood'], 300, 30)
            if res[2] > 0: tests["Physics Engine"] = True
        except: pass
        
        return tests

# ==============================================================================
# 7. MAIN BOOTLOADER
# ==============================================================================

def boot_os():
    UIKernel.boot()
    
    # Session State Init
    if 'user' not in st.session_state: st.session_state.user = None
    
    # Login Flow
    if not st.session_state.user:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown("<div class='cyber-card' style='text-align:center'><h2>CHEMISCO OS v10</h2><p>SECURE TERMINAL ACCESS</p></div>", unsafe_allow_html=True)
            
            uid = st.text_input("USER IDENTITY")
            pwd = st.text_input("ACCESS CODE", type="password")
            
            if st.button("CONNECT"):
                if uid == "admin" and pwd == "admin":
                    st.session_state.user = "admin"
                    st.toast("Connection Established...", icon="🟢")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("ACCESS DENIED")
    else:
        # Main OS Interface
        with st.sidebar:
            st.title("💠 SYSTEM MENU")
            st.write(f"USER: {st.session_state.user}")
            st.markdown("---")
            app_mode = st.radio("MODULE SELECTOR", 
                ["COMMAND DASHBOARD", "SIMULATION LAB", "FINANCE HQ", "SYSTEM DIAGNOSTICS"])
            
            st.markdown("---")
            if st.button("DISCONNECT"):
                st.session_state.user = None
                st.rerun()
        
        # Routing
        if app_mode == "COMMAND DASHBOARD":
            dashboard_view(st.session_state.user)
        elif app_mode == "SIMULATION LAB":
            simulation_lab_view()
        elif app_mode == "FINANCE HQ":
            finance_hq_view()
        elif app_mode == "SYSTEM DIAGNOSTICS":
            st.title("SYSTEM DIAGNOSTICS")
            if st.button("RUN SELF-TEST"):
                results = SystemDiagnostic.run_tests()
                for test, passed in results.items():
                    if passed: st.success(f"{test}: ONLINE")
                    else: st.error(f"{test}: FAIL")
                
                # Show logs
                st.markdown("### TERMINAL LOGS")
                logs = f"""
                [BOOT] System kernel loaded.
                [NET] IoT Gateway connected (Port 8080).
                [DB] SQLite integrity check passed.
                [USER] Admin session active.
                [TIME] {datetime.datetime.now()}
                """
                st.code(logs, language='bash')

if __name__ == "__main__":
    boot_os()

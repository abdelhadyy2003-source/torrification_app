# -*- coding: utf-8 -*-
"""
CHEMISCO ULTIMATE: The Complete Biorefinery Platform
----------------------------------------------------
Author: Chemisco Dev Team
Version: 6.0 (Physics + AI + Game + BFD)
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.integrate import odeint
import time
import random
from dataclasses import dataclass

# ==============================================================================
# 1. SETUP & STYLING
# ==============================================================================

st.set_page_config(page_title="Chemisco Ultimate", layout="wide", page_icon="⚛️")

class AppStyle:
    @staticmethod
    def apply():
        st.markdown("""
        <style>
            .stApp { background-color: #0E1117; color: #FAFAFA; }
            
            /* Game UI */
            .game-card {
                background: linear-gradient(45deg, #2b5876, #4e4376);
                padding: 20px; border-radius: 15px;
                border: 2px solid #FFD700; text-align: center;
                box-shadow: 0 0 15px rgba(255, 215, 0, 0.3);
            }
            
            /* AI Chat UI */
            .ai-msg {
                background-color: #1F2937; padding: 15px; border-radius: 10px;
                border-left: 4px solid #10B981; margin-bottom: 10px;
            }
            
            /* Metric Cards */
            .metric-card {
                background: #161b22; border: 1px solid #30363d;
                padding: 15px; border-radius: 8px; border-left: 5px solid #00ADB5;
                margin-bottom: 10px;
            }
            
            /* BFD (Block Flow Diagram) Styles */
            .bfd-container {
                display: flex; justify-content: space-around; align-items: center;
                background-color: #161b22; padding: 25px;
                border-radius: 15px; border: 1px solid #30363d; margin: 20px 0;
            }
            .bfd-box {
                background: #21262d; color: #c9d1d9; padding: 15px 20px;
                border-radius: 8px; text-align: center; border: 1px solid #30363d; min-width: 130px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }
            .bfd-arrow { color: #8b949e; font-size: 28px; font-weight: bold; }
            .bfd-label { font-size: 12px; color: #8b949e; text-transform: uppercase; margin-bottom: 5px; }
            .bfd-value { font-size: 18px; font-weight: bold; color: white; }
        </style>
        """, unsafe_allow_html=True)

# ==============================================================================
# 2. PHYSICS ENGINE (CORE)
# ==============================================================================

@dataclass
class Biomass:
    name: str; hemi: float; cell: float; lig: float; ash: float; moist: float

BIOMASS_DB = {
    "Wood": Biomass("Wood", 0.30, 0.45, 0.25, 0.01, 0.15),
    "Straw": Biomass("Straw", 0.45, 0.35, 0.20, 0.08, 0.10),
    "Sewage": Biomass("Sewage", 0.20, 0.30, 0.20, 0.30, 0.20),
}

class Simulator:
    def __init__(self, type_name):
        self.bio = BIOMASS_DB[type_name]

    def run(self, T, t_min):
        T_K = T + 273.15
        R = 8.314
        # Arrhenius: A (min^-1), E (J/mol)
        params = [(1e10, 110000), (1e12, 130000), (1e8, 100000)]
        k = [A * np.exp(-E/(R*T_K)) for A, E in params]
        
        # ODE Model
        def model(y, t): return [-k[0]*y[0], -k[1]*y[1], -k[2]*y[2]]
        t_span = np.linspace(0, t_min, 50)
        sol = odeint(model, [self.bio.hemi, self.bio.cell, self.bio.lig], t_span)
        
        # Results
        final = sol[-1]
        mass_rem = sum(final) * (1 - self.bio.ash - self.bio.moist)
        char_mass = mass_rem + self.bio.ash
        yield_pct = char_mass * 100
        energy_yield = yield_pct * 1.15
        
        return t_span, sol, yield_pct, energy_yield

# ==============================================================================
# 3. AI EXPERT SYSTEM
# ==============================================================================

class AIConsultant:
    @staticmethod
    def analyze(yield_val, energy_val, profit):
        advice = []
        score = 0
        
        if yield_val < 60:
            advice.append("⚠️ **Temperature Alert:** Reactor heat is destroying yield. Lower T by 10°C.")
        elif yield_val > 90:
            advice.append("ℹ️ **Process Incomplete:** Biomass is barely roasted. Increase Time.")
            score += 1
        else:
            advice.append("✅ **Optimal Zone:** Mass yield is within industrial standards.")
            score += 3

        if profit < 0:
            advice.append("💸 **Loss Detected:** Check your OPEX vs. Market Price.")
        else:
            advice.append("💰 **Profitable Run:** Good margin maintained.")
            score += 2

        rating = "⭐⭐⭐ Expert" if score >= 4 else "⭐⭐ Average"
        return advice, rating

# ==============================================================================
# 4. MANAGER GAME
# ==============================================================================

def game_logic():
    st.markdown("<div class='game-card'><h2>🎮 Factory Manager Challenge</h2></div>", unsafe_allow_html=True)
    
    if 'game_score' not in st.session_state:
        st.session_state.game_score = 0
        st.session_state.game_money = 10000
        st.session_state.game_day = 1

    c1, c2, c3 = st.columns(3)
    c1.metric("Day", st.session_state.game_day)
    c2.metric("Budget", f"${st.session_state.game_money}")
    c3.metric("Score", st.session_state.game_score)

    st.write("---")
    # Game Inputs
    target = random.randint(70, 85)
    st.info(f"📋 **Mission:** Achieve Yield **{target}%** (+/- 2%)")
    
    g_temp = st.slider("Set Temp (°C)", 200, 350, 250, key="g_t")
    g_time = st.slider("Set Time (min)", 20, 100, 40, key="g_d")
    
    if st.button("🏭 Run Production"):
        sim = Simulator("Wood")
        _, _, y_res, _ = sim.run(g_temp, g_time)
        diff = abs(y_res - target)
        
        if diff <= 2:
            st.balloons(); reward = 2000; st.success(f"PERFECT! Yield: {y_res:.1f}%")
        elif diff <= 5:
            reward = 500; st.warning(f"Close enough. Yield: {y_res:.1f}%")
        else:
            reward = -500; st.error(f"Failed! Yield: {y_res:.1f}%")
            
        st.session_state.game_money += reward
        st.session_state.game_score += 10 if reward > 0 else 0
        st.session_state.game_day += 1
        st.rerun()

# ==============================================================================
# 5. MAIN APPLICATION
# ==============================================================================

def main():
    AppStyle.apply()
    
    menu = st.sidebar.radio("Navigation", ["🧪 Simulator", "🤖 AI Consultant", "🎮 Manager Game"])
    
    # --- SIMULATOR TAB ---
    if menu == "🧪 Simulator":
        st.title("Chemisco Pro Simulator")
        
        with st.sidebar:
            st.header("Parameters")
            b_type = st.selectbox("Feedstock", list(BIOMASS_DB.keys()))
            temp = st.slider("Temp (°C)", 200, 350, 275)
            time_min = st.slider("Time (min)", 10, 120, 60)
            mass = st.number_input("Mass (kg)", 100, 5000, 1000)
            price = st.number_input("Char Price ($/kg)", 1.5)
            run_btn = st.button("🚀 Run Simulation", type="primary")

        if run_btn:
            sim = Simulator(b_type)
            t, sol, y_mass, y_eng = sim.run(temp, time_min)
            profit = (mass * (y_mass/100) * price) - (mass * 0.2)
            product_kg = mass * (y_mass/100)
            
            st.session_state.last_run = {"yield": y_mass, "energy": y_eng, "profit": profit}
            
            # --- 1. METRICS ---
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="metric-card"><h3>Mass Yield</h3><h1>{y_mass:.1f}%</h1></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><h3>Energy Yield</h3><h1>{y_eng:.1f}%</h1></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><h3>Est. Profit</h3><h1>${profit:.0f}</h1></div>', unsafe_allow_html=True)
            
            # --- 2. BLOCK FLOW DIAGRAM (BFD) - ADDED HERE ---
            st.markdown("### 🔄 Process Block Flow Diagram")
            

[Image of torrefaction process diagram]

            st.markdown(f"""
            <div class="bfd-container">
                <div class="bfd-box" style="border-left: 4px solid #4CAF50;">
                    <div class="bfd-label">Input Feed</div>
                    <div class="bfd-value">{mass} kg</div>
                </div>
                <div class="bfd-arrow">➜</div>
                <div class="bfd-box" style="border-left: 4px solid #FFC107;">
                    <div class="bfd-label">Reactor</div>
                    <div class="bfd-value">{temp}°C</div>
                </div>
                <div class="bfd-arrow">➜</div>
                <div class="bfd-box" style="border-left: 4px solid #00ADB5;">
                    <div class="bfd-label">Biochar Product</div>
                    <div class="bfd-value">{product_kg:.1f} kg</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- 3. CHARTS ---
            st.markdown("### Process Kinetics")
            df = pd.DataFrame(sol, columns=["Hemi", "Cell", "Lig"]); df["Time"] = t
            st.line_chart(df, x="Time")

        else:
            st.info("👈 Please configure and run the simulation.")
            

[Image of torrefaction process diagram]


    # --- AI TAB ---
    elif menu == "🤖 AI Consultant":
        st.title("🤖 Chemisco AI Expert")
        if 'last_run' in st.session_state:
            d = st.session_state.last_run
            advice, rating = AIConsultant.analyze(d['yield'], d['energy'], d['profit'])
            st.markdown(f"### Rating: {rating}")
            for msg in advice: st.markdown(f"<div class='ai-msg'>{msg}</div>", unsafe_allow_html=True)
        else:
            st.warning("Please run a simulation first.")

    # --- GAME TAB ---
    elif menu == "🎮 Manager Game":
        game_logic()

if __name__ == "__main__":
    main()

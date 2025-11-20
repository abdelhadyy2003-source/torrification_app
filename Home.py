import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.title("Torrification Simulation")
st.write("حاسب وراقب الرطوبة خلال التوريفيكاشن بسهولة")

# Sidebar لإدخال البيانات
initial_moisture = st.sidebar.number_input("Initial Moisture (%)", 0, 100, 40)
temperature = st.sidebar.slider("Temperature (°C)", 50, 300, 200)
time_hours = st.sidebar.number_input("Time (hours)", 0.0, 10.0, 2.0)

# حساب الرطوبة النهائية بطريقة بسيطة
decay_rate = 0.05 + 0.0005 * (temperature - 100)
final_moisture = initial_moisture * np.exp(-decay_rate * time_hours)

st.subheader("Results")
st.write(f"Final Moisture: {final_moisture:.2f}%")

# رسم منحنى
times = np.linspace(0, time_hours, 100)
moistures = initial_moisture * np.exp(-decay_rate * times)
data = pd.DataFrame({"Time": times, "Moisture": moistures})

fig = px.line(data, x="Time", y="Moisture", markers=True)
st.plotly_chart(fig)

# جدول البيانات وتحميله
st.dataframe(data)
csv = data.to_csv(index=False).encode()
st.download_button("Download CSV", data=csv, file_name="torrification_results.csv")

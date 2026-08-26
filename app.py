import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Traffic Pattern Forecasting", page_icon="🚦", layout="wide")

DATA_PATH = "dataset/traffic_data.csv"
MODEL_PATH = "model/traffic_model.pkl"

st.title("🚦 Traffic Pattern Forecasting")
st.caption("Machine Learning based traffic volume prediction system")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    return df

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

df = load_data()

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. First run: python train_model.py")
    st.stop()

model = load_model()

tab1, tab2, tab3 = st.tabs(["🔮 Forecast", "📊 Analytics", "📋 Dataset"])

with tab1:
    st.subheader("Predict Traffic Volume")
    c1, c2, c3 = st.columns(3)

    with c1:
        date = st.date_input("Date", value=pd.Timestamp("2026-01-15").date())
        hour = st.slider("Hour", 0, 23, 18)

    with c2:
        weather = st.selectbox("Weather", ["Clear", "Cloudy", "Rain"])
        holiday = st.selectbox("Holiday", [0, 1], format_func=lambda x: "Yes" if x else "No")

    with c3:
        st.write("")
        st.write("")
        predict = st.button("🚦 Predict Traffic", use_container_width=True)

    if predict:
        dt = pd.Timestamp(date)
        row = pd.DataFrame([{
            "Hour": hour,
            "Day": dt.day,
            "Month": dt.month,
            "DayOfWeekNum": dt.dayofweek,
            "Weather": weather,
            "Holiday": holiday
        }])
        prediction = max(0, float(model.predict(row)[0]))
        st.success(f"Predicted Traffic Volume: **{prediction:,.0f} vehicles/hour**")

        if prediction < 2500:
            level = "Low"
        elif prediction < 5000:
            level = "Moderate"
        elif prediction < 7000:
            level = "High"
        else:
            level = "Very High"
        st.info(f"Traffic Level: **{level}**")

with tab2:
    st.subheader("Traffic Analytics")
    a, b, c = st.columns(3)
    a.metric("Average Traffic", f"{df.TrafficVolume.mean():,.0f}")
    b.metric("Maximum Traffic", f"{df.TrafficVolume.max():,.0f}")
    c.metric("Minimum Traffic", f"{df.TrafficVolume.min():,.0f}")

    hourly = df.groupby("DateTime")["TrafficVolume"].mean()
    st.line_chart(hourly.resample("D").mean().tail(60))

    by_hour = df.groupby(df["DateTime"].dt.hour)["TrafficVolume"].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(by_hour.index, by_hour.values, marker="o")
    ax.set_title("Average Traffic by Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Vehicles per Hour")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    by_weather = df.groupby("Weather")["TrafficVolume"].mean().sort_values(ascending=False)
    st.bar_chart(by_weather)

with tab3:
    st.subheader("Traffic Dataset")
    st.dataframe(df.drop(columns=["DateTime"]).tail(500), use_container_width=True)

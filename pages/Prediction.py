import streamlit as st
import pickle
import pandas as pd

st.title("🔮 Weather Prediction")

# Load model
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))
le = pickle.load(open("model/label_encoder.pkl", "rb"))

st.subheader("Enter Input Features")

precipitation = st.number_input("Precipitation", min_value=0.0)
temp_max = st.number_input("Max Temperature")
temp_min = st.number_input("Min Temperature")
wind = st.number_input("Wind Speed")
year = st.number_input("Year", min_value=2000, max_value=2100, value=2012)
month = st.number_input("Month", min_value=1, max_value=12)
day = st.number_input("Day", min_value=1, max_value=31)

if st.button("Predict"):

    input_data = [[precipitation, temp_max, temp_min, wind, year, month, day]]
    input_df = pd.DataFrame(input_data, columns=columns)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    result = le.inverse_transform(prediction)

    # Probability
    prob = model.predict_proba(input_scaled)

    st.success(f"🌤️ Predicted Weather: {result[0]}")
    st.write("Prediction Probabilities:")
    st.write(prob)
import streamlit as st
import pandas as pd

st.title("📈 Model Comparison")

data = {
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy": [0.79, 0.85, 0.83],
    "F1 Score": [0.74, 0.82, 0.82]
}

df = pd.DataFrame(data)

st.dataframe(df)

st.subheader("Performance Comparison")
st.bar_chart(df.set_index("Model"))

st.success("✅ Random Forest selected as best model")
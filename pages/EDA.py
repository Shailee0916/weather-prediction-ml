import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Exploratory Data Analysis")

df = pd.read_csv("data/seattle-weather.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Data Info")
st.write(df.describe())

# Weather distribution
st.subheader("Weather Distribution")
fig, ax = plt.subplots()
sns.countplot(x="weather", data=df, ax=ax)
st.pyplot(fig)

# Temperature distribution
st.subheader("Temperature Distribution")
fig, ax = plt.subplots()
sns.histplot(df["temp_max"], kde=True, ax=ax)
st.pyplot(fig)

# Correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
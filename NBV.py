import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load MultinomialNB model and vectorizer
with open("nb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("count_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit App
st.title("🧠 Fake News Detector — Naive Bayes")
st.write("Enter news content below to predict if it's **Fake or Real** using a MultinomialNB model.")

news_input = st.text_area("Paste your news content here:")

if st.button("Detect"):
    if not news_input.strip():
        st.warning("⚠️ Please enter some news content.")
    else:
        # Vectorize input
        input_vector = vectorizer.transform([news_input])
        prediction = model.predict_proba(input_vector)[0]

        real_prob = prediction[0]
        fake_prob = prediction[1]

        if fake_prob > real_prob:
            st.error(f"🚨 This news is predicted to be **FAKE** with {fake_prob:.2%} confidence.")
        else:
            st.success(f"✅ This news is predicted to be **REAL** with {real_prob:.2%} confidence.")

        # Pie Chart
        fig, ax = plt.subplots()
        ax.pie([real_prob, fake_prob],
               labels=["Real", "Fake"],
               autopct="%1.1f%%",
               startangle=90,
               colors=["green", "red"])
        ax.axis("equal")
        st.pyplot(fig)

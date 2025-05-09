import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import pipeline

# === TOGGLE THIS ===
USE_CUSTOM_LSTM_MODEL = True  # 🔁 Set to False to use HuggingFace BERT model

# Constants (for LSTM)
MAX_SEQUENCE_LENGTH = 200
LSTM_MODEL_PATH = "models/fake_news_lstm_model.h5"
TOKENIZER_PATH = "tokenizer.pickle"

# Load models
if USE_CUSTOM_LSTM_MODEL:
    @st.cache_resource
    def load_lstm_model():
        model = load_model(LSTM_MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer

else:
    @st.cache_resource
    def load_bert_pipeline():
        return pipeline("text-classification", model="jy46604790/Fake-News-Bert-Detect", tokenizer="jy46604790/Fake-News-Bert-Detect")

# App UI
st.title("📰 Fake News Detector")
st.write("Enter a news article below to check if it's **Fake or Real** with confidence.")

news_input = st.text_area("Paste your news content here:")

if st.button("Detect"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:
        if USE_CUSTOM_LSTM_MODEL:
            model, tokenizer = load_lstm_model()
            seq = tokenizer.texts_to_sequences([news_input])
            padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
            pred = model.predict(padded)[0]

            confidence_real = float(pred[0])
            confidence_fake = float(pred[1])
            label = np.argmax(pred)

            if label == 1:
                st.error(f"🚨 This news is predicted to be **FAKE** with {confidence_fake:.2%} confidence.")
            else:
                st.success(f"✅ This news is predicted to be **REAL** with {confidence_real:.2%} confidence.")
        
        else:
            clf = load_bert_pipeline()
            result = clf(news_input)[0]
            label = result['label']
            score = result['score']

            if label == 'LABEL_1':
                st.success(f"✅ This news is predicted to be **REAL** with {score:.2%} confidence.")
                confidence_real, confidence_fake = score, 1 - score
            else:
                st.error(f"🚨 This news is predicted to be **FAKE** with {score:.2%} confidence.")
                confidence_real, confidence_fake = 1 - score, score

        # Pie chart
        fig, ax = plt.subplots()
        ax.pie([confidence_real, confidence_fake],
               labels=['Real', 'Fake'],
               autopct='%1.1f%%',
               startangle=90,
               colors=['green', 'red'])
        ax.axis('equal')
        st.pyplot(fig)

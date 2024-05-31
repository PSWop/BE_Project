import shutil
import os
import streamlit as st
from main import train_ui


# SVM MODEL
def predict_notes():
    predicted_notes = train_ui()
    return predicted_notes


# Streamlit UI
st.title("Guitar to Chords Predictor")

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("Predict"):
        with st.spinner("Processing..."):

            if os.path.exists("chunks/temp"):
                shutil.rmtree("chunks/temp")
            if os.path.exists("songs/temp.wav"):
                os.remove("songs/temp.wav")

            with open("songs/temp.wav", "wb") as f:
                f.write(uploaded_file.getvalue())

            predicted_notes = predict_notes()

            shutil.rmtree("chunks/temp")

        print(predicted_notes)
        st.success(f"Predicted notes: {predicted_notes}")

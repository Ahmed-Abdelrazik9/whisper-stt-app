
import streamlit as st
import whisper
import tempfile
import os

@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

st.title("🎙️ Whisper Speech-to-Text")
st.write("Upload an audio file (MP3/WAV/M4A) and we'll transcribe it using OpenAI's Whisper.")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info("Transcribing...")
    result = model.transcribe(tmp_path)
    st.success("Transcription Complete!")

    st.text_area("Transcribed Text:", result["text"], height=300)
    
    os.remove(tmp_path)

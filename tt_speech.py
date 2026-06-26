from groq import Groq
import os
from dotenv import load_dotenv

import base64
import streamlit as st

load_dotenv()

client = Groq(
    api_key = os.getenv("GROQ_API_KEY")
)
def tts(text:str):
    response = client.audio.speech.create(
        model= "canopylabs/orpheus-v1-english",
        voice="autumn",
        input=text[:200],
        response_format="wav"
    )

    return response.read()


def autoplay_audio(audio_bytes):
    b64 = base64.b64encode(audio_bytes).decode()

    md = f"""
    <audio autoplay style="display:none;">
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """

    st.markdown(md, unsafe_allow_html=True)
import streamlit as st
import whisper
import tempfile
import requests
from deep_translator import GoogleTranslator
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder
import base64
import os

# -------------------------------------------------------------
# Load Whisper Model ONLY once
# -------------------------------------------------------------
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("medium")

model = load_whisper_model()

# -------------------------------------------------------------
# Healthcare AI Model (HuggingFace Router API)
# -------------------------------------------------------------
def healthcare_ai_response(query, hf_token, conversation_history):
    API_URL = "https://router.huggingface.co/v1/chat/completions"

    if not hf_token:
        return "❌ Missing HuggingFace Token", conversation_history

    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}

    messages = conversation_history + [{"role": "user", "content": query}]
    payload = {
        "model": "m42-health/Llama3-Med42-70B:featherless-ai",
        "messages": messages
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        reply = data["choices"][0]["message"]["content"]
        conversation_history.append({"role": "assistant", "content": reply})
        return reply, conversation_history

    except Exception as e:
        return f"❌ AI Request Error: {e}", conversation_history

# -------------------------------------------------------------
# Speech → Text
# -------------------------------------------------------------
def transcribe_audio(file_path, language_code):
    try:
        result = model.transcribe(
            file_path,
            task="transcribe",
            fp16=False,
            language=language_code
        )
        return result.get("text", "").strip()
    except Exception as e:
        return f"❌ Whisper Error: {e}"

# -------------------------------------------------------------
# Text → Speech
# -------------------------------------------------------------
def speak_text(text, lang_code="en"):
    try:
        tts = gTTS(text=text, lang=lang_code)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)

        audio_bytes = open(fp.name, "rb").read()
        b64 = base64.b64encode(audio_bytes).decode()

        return f"""
        <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """

    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

# -------------------------------------------------------------
# UI
# -------------------------------------------------------------
st.set_page_config(page_title="🩺 Multilingual Healthcare Voice Assistant", layout="centered")
st.title("🩺 Multilingual Healthcare Voice Assistant")

# Sidebar
st.sidebar.header("⚙️ Configuration")
hf_token = st.sidebar.text_input("🔑 Enter your Hugging Face Token", type="password")

languages = {"English": "en", "Tamil": "ta", "Hindi": "hi", "Telugu": "te", "Malayalam": "ml"}
language_name = st.sidebar.selectbox("🌐 Select your language", list(languages.keys()))
language_code = languages[language_name]
st.sidebar.markdown("m42-health/Llama3-Med42-70B used")
# Conversation History State
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [
        {"role": "system", "content": "You are a helpful healthcare assistant."}
    ]

# Voice Input
st.markdown("🎙 Speak your healthcare question:")
audio_data = mic_recorder(start_prompt="🎤 Start", stop_prompt="⏹ Stop", key="rec", just_once=False)

if audio_data:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
        fp.write(audio_data["bytes"])
        audio_path = fp.name

    st.audio(audio_path, format="audio/wav")
    st.info("🔍 Transcribing...")

    # Speech → Text
    user_text = transcribe_audio(audio_path, language_code)

    if not user_text:
        st.warning("⚠️ No speech detected.")
    else:
        st.success(f"**You said ({language_code}):** {user_text}")

        # Exit condition
        if user_text.lower().strip() in ["stop", "exit", "bye"]:
            st.warning("👋 Conversation ended.")
        else:
            # Local language → English
            english_query = GoogleTranslator(
                source=language_code, target="en"
            ).translate(user_text) if language_code != "en" else user_text

            # AI Response
            ai_reply, st.session_state.conversation_history = healthcare_ai_response(
                english_query, hf_token, st.session_state.conversation_history
            )

            st.write("🤖 **AI (English):**", ai_reply)

            # English → Local language
            translated = GoogleTranslator(
                source="en", target=language_code
            ).translate(ai_reply) if language_code != "en" else ai_reply

            st.write(f"🗣️ **Assistant ({language_code}):**", translated)

            # Speak
            audio_html = speak_text(translated, lang_code=language_code)
            if audio_html:
                st.markdown(audio_html, unsafe_allow_html=True)

# Conversation history
st.markdown("---")
st.markdown("💬 **Conversation History**")

for msg in st.session_state.conversation_history[1:]:
    role = "🧑‍⚕️ Assistant" if msg["role"] == "assistant" else "👤 You"
    st.markdown(f"**{role}:** {msg['content']}")

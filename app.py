import streamlit as st
from dotenv import load_dotenv
import os

# ==========================
# ENV + SESSION INITIALIZATION
# ==========================
load_dotenv()

st.set_page_config(page_title="AI Toolbox", layout="wide")
st.sidebar.title("🧠 AI Toolbox")

# Initialize session state for API keys
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = os.getenv("GROQ_API_KEY", "")

# ==========================
# SIDEBAR – API KEY MANAGEMENT
# ==========================
st.sidebar.subheader("🔑 API Key Configuration")

with st.sidebar.expander("Manage API Keys", expanded=False):
    st.caption("Keys are stored only in session memory and never displayed or logged.")

    openai_key_input = st.text_input(
        "OpenAI API Key:",
        type="password",
        placeholder="sk-...",
        help="Enter your OpenAI key (not shown again after saving)."
    )

    groq_key_input = st.text_input(
        "Groq API Key:",
        type="password",
        placeholder="gsk-...",
        help="Enter your Groq key (not shown again after saving)."
    )

    if st.button("💾 Save Keys"):
        if openai_key_input.strip():
            st.session_state.openai_api_key = openai_key_input.strip()
        if groq_key_input.strip():
            st.session_state.groq_api_key = groq_key_input.strip()
        st.success("✅ API keys securely saved for this session.")

# Status indicators
def key_status(name, key):
    color = "🟢" if key else "🔴"
    st.sidebar.write(f"{color} {name} Key")

key_status("OpenAI", st.session_state.openai_api_key)
key_status("Groq", st.session_state.groq_api_key)

st.sidebar.markdown("---")

# ==========================
# APP SELECTION
# ==========================
app_choice = st.sidebar.radio(
    "🧩 Select Application:",
    ["📰 Article Summarizer", "🎥 YouTube Summarizer", "🎙️ Voice Agent"],
)

# ==========================
# LOAD SELECTED APP
# ==========================
if app_choice == "📰 Article Summarizer":
    from apps.article_summarizer import run_article_summarizer
    run_article_summarizer()

elif app_choice == "🎥 YouTube Summarizer":
    from apps.youtube_summarizer import run_youtube_summarizer
    run_youtube_summarizer()

elif app_choice == "🎙️ Voice Agent":
    from apps.voice_assistant import run_voice_assistant
    run_voice_assistant()

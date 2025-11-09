import streamlit as st
from youtube_summarizer_backend import YoutubeVideoSummarizer
import time

def run_youtube_summarizer():
    st.title("🎥 YouTube Video Summarizer & Q/A")
    st.markdown(
        """
        Paste a YouTube video link below.  
        This tool will:
        1. Download and transcribe the audio  
        2. Summarize the transcript  
        3. Let you **chat** with the content using vector-based Q/A.
        """
    )

    # ===============================
    # Initialize session state
    # ===============================
    for key in [
        "yt_qa_chain", "yt_chat_history", "yt_user_query",
        "yt_summary_done", "yt_result", "yt_last_url",
        "yt_llm_provider", "yt_llm_model", "yt_embedding_type"
    ]:
        if key not in st.session_state:
            st.session_state[key] = None if "chain" in key or "result" in key else ""

    # ===============================
    # API Keys
    # ===============================
    openai_key = st.session_state.get("openai_api_key", "")
    groq_key = st.session_state.get("groq_api_key", "")
    if not openai_key and not groq_key:
        st.warning("Please set at least one API key from the sidebar before proceeding.")
        st.stop()

    # ===============================
    # Model selections
    # ===============================
    st.sidebar.subheader("⚙️ Model Configuration")
    llm_provider = st.sidebar.selectbox(
        "Select LLM Provider:", ["openai", "groq"], index=0
    )
    llm_model = st.sidebar.selectbox(
        "LLM Model:",
        ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        if llm_provider == "openai"
        else ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
    )
    embedding_type = st.sidebar.selectbox(
        "Embeddings Provider:", ["openai", "huggingface"], index=0
    )

    # ===============================
    # Detect LLM or embedding changes
    # ===============================
    if (
        st.session_state.yt_llm_provider != llm_provider
        or st.session_state.yt_llm_model != llm_model
        or st.session_state.yt_embedding_type != embedding_type
    ):
        # Reset summarizer state on any change
        st.session_state.yt_summary_done = False
        st.session_state.yt_result = {}
        st.session_state.yt_qa_chain = None
        st.session_state.yt_chat_history = []
        st.session_state.yt_user_query = ""

        st.session_state.yt_llm_provider = llm_provider
        st.session_state.yt_llm_model = llm_model
        st.session_state.yt_embedding_type = embedding_type

    # ===============================
    # YouTube URL input
    # ===============================
    youtube_url = st.text_input(
        "📺 Enter YouTube Video URL:", value=st.session_state.get("yt_last_url", "")
    )

    # ===============================
    # Summarize Video Button
    # ===============================
    if st.button("🚀 Summarize Video") and youtube_url:
        with st.spinner("Processing video... this may take a few minutes ⏳"):
            start_time = time.time()
            try:
                summarizer = YoutubeVideoSummarizer(
                    llm_type=llm_provider,
                    llm_model_name=llm_model,
                    embedding_type=embedding_type,
                )

                result = summarizer.process_video(youtube_url)
                if "error" in result:
                    st.error(f"❌ Error: {result['error']}")
                    st.stop()

                # Save results to session state
                st.session_state.yt_result = result
                st.session_state.yt_qa_chain = result["qa_chain"]
                st.session_state.yt_summary_done = True
                st.session_state.yt_last_url = youtube_url
                st.session_state.yt_chat_history = []
                st.session_state.yt_user_query = ""

                st.success(f"✅ Processed in {time.time() - start_time:.1f} seconds!")

            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")
                st.stop()

    # ===============================
    # Display Summary and Q/A
    # ===============================
    if st.session_state.yt_summary_done and st.session_state.yt_result:
        result = st.session_state.yt_result

        st.subheader("🎬 Video Title")
        st.write(result.get("title", "Untitled Video"))

        st.subheader("📝 Summary")
        st.write(result.get("summary", "No summary available."))
        st.divider()

        # -------------------------------
        # Q/A Section
        # -------------------------------
        st.subheader("💬 Ask Questions about the Video")
        st.session_state.yt_user_query = st.text_input(
            "Ask a question:", value=st.session_state.yt_user_query
        )

        if st.session_state.yt_user_query:
            if st.session_state.yt_qa_chain:
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.yt_qa_chain.invoke(
                            {"question": st.session_state.yt_user_query}
                        )
                        answer = response.get("answer", "No response.")
                        st.session_state.yt_chat_history.append(
                            {"user": st.session_state.yt_user_query, "bot": answer}
                        )
                        st.session_state.yt_user_query = ""
                    except Exception as e:
                        st.error(f"Error during Q/A: {str(e)}")
            else:
                st.warning("QA chain not initialized. Please summarize the video first.")

        # Display chat history
        for msg in reversed(st.session_state.yt_chat_history):
            st.markdown(f"**🧑 You:** {msg['user']}")
            st.markdown(f"**🤖 Assistant:** {msg['bot']}")
            st.divider()

        # Full transcript
        with st.expander("🧾 Full Transcript"):
            st.text(result.get("transcript", "No transcript available."))

    elif not youtube_url:
        st.info("👆 Enter a YouTube link to get started.")
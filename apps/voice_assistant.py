import streamlit as st
import os
import tempfile
import numpy as np
from dotenv import load_dotenv

from voice_assistant_backend import VoiceAssistantRAG, DocumentProcessor

load_dotenv()


def run_voice_assistant():
    st.title("🎙️ Voice RAG Assistant")
    st.markdown(
        """
        Upload documents to create a knowledge base, then ask questions using **voice input**.
        """
    )

    # ===============================
    # Session State Initialization
    # ===============================
    for key in [
        "va_vector_store", "va_qa_chain", "va_chat_history",
        "va_user_query", "va_summary_done", "va_audio_data",
        "va_embedding_type", "va_llm_model"
    ]:
        if key not in st.session_state:
            st.session_state[key] = None if "chain" in key or "store" in key else ""

    # ===============================
    # Sidebar Configuration
    # ===============================
    st.sidebar.subheader("⚙️ Model Configuration")
    llm_model = st.sidebar.selectbox(
        "LLM Model:",
        ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
    )
    embedding_type = st.sidebar.selectbox(
        "Embeddings:",
        ["openai", "huggingface"]
    )

    # Reset session state if configs change
    if (
        st.session_state.va_embedding_type != embedding_type
        or st.session_state.va_llm_model != llm_model
    ):
        st.session_state.va_vector_store = None
        st.session_state.va_qa_chain = None
        st.session_state.va_chat_history = []
        st.session_state.va_user_query = ""
        st.session_state.va_summary_done = False
        st.session_state.va_audio_data = None
        st.session_state.va_embedding_type = embedding_type
        st.session_state.va_llm_model = llm_model

    # ===============================
    # Document Upload and Processing
    # ===============================
    uploaded_files = st.file_uploader(
        "📄 Upload documents", 
        accept_multiple_files=True, 
        type=["pdf", "txt", "md"]
    )

    if uploaded_files and st.button("🛠 Process Documents"):
        with st.spinner("Processing documents..."):
            temp_dir = tempfile.mkdtemp()

            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

            try:
                processor = DocumentProcessor(embedding_type)
                documents = processor.load_documents(temp_dir)
                processed_docs = processor.process_documents(documents)
                vector_store = processor.create_vector_store(
                    processed_docs,
                    persist_directory="voice_knowledge_base"
                )

                st.session_state.va_vector_store = vector_store
                st.session_state.va_summary_done = True
                st.session_state.va_chat_history = []

                st.success(f"✅ Processed {len(processed_docs)} document chunks!")

            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")

            finally:
                # Cleanup temp directory
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)

    # ===============================
    # Voice Assistant Q/A
    # ===============================
    if st.session_state.va_summary_done and st.session_state.va_vector_store:
        elevenlabs_key = os.getenv("ELEVEN_LABS_API_KEY")

        assistant = VoiceAssistantRAG(
            elevenlabs_api_key=elevenlabs_key,
            embedding_type=embedding_type,
            llm_model_name=llm_model
        )

        if st.session_state.va_qa_chain is None:
            assistant.setup_vector_store(st.session_state.va_vector_store)
            st.session_state.va_qa_chain = assistant.qa_chain

        col1, col2 = st.columns(2)
        duration = st.sidebar.slider("🎧 Recording Duration (s)", 1, 10, 5)

        # 🎤 Start Recording
        with col1:
            if st.button("🎙 Start Recording"):
                with st.spinner(f"Recording for {duration} seconds..."):
                    audio_data = assistant.record_audio(duration)
                    st.session_state.va_audio_data = audio_data
                    st.success("✅ Recording completed!")

        # ▶ Process Recording
        with col2:
            if st.button("▶ Process Recording"):
                if st.session_state.va_audio_data is None:
                    st.warning("Please record audio first!")
                else:
                    # Transcribe
                    query = assistant.transcribe_audio(st.session_state.va_audio_data)
                    st.session_state.va_user_query = query
                    st.write("🗣 You said:", query)

                    # Generate response
                    response = assistant.generate_response(query)
                    st.session_state.va_chat_history.append({"user": query, "bot": response})
                    st.write("🤖 Assistant:", response)

                    # Convert to speech and play
                    audio_file = assistant.text_to_speech(response)
                    if audio_file:
                        st.audio(audio_file)
                        os.unlink(audio_file)  # remove temp file after playing
                    else:
                        st.error("Failed to generate voice response.")
        

        # ===============================
        # Display Chat History
        # ===============================
        if st.session_state.va_chat_history:
            st.subheader("💬 Conversation History")
            for msg in reversed(st.session_state.va_chat_history):
                st.markdown(f"**🧑 You:** {msg['user']}")
                st.markdown(f"**🤖 Assistant:** {msg['bot']}")
                st.divider()


if __name__ == "__main__":
    run_voice_assistant()

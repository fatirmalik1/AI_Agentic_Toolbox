# 🧠 AI Toolbox

AI Toolbox is a Streamlit-based application that provides three AI-powered tools:

1. **📰 News Article Summarizer** – Summarize articles using OpenAI or Groq models.
2. **🎥 YouTube Video Summarizer & Q/A** – Download, transcribe, summarize videos, and ask questions using vector-based Q/A.
3. **🎙️ Voice Assistant RAG** – Create a voice-enabled RAG (retrieval-augmented generation) assistant from uploaded documents.

---

## 📁 Directory Structure

```

├── app.py
├── requirement.txt
├── article_summarizer_backend.py
├── youtube_summarizer_backend.py
├── voice_assistant_backend.py
├── voice_assistant_backend_v1.py
├── apps/
│   ├── article_summarizer.py
│   ├── youtube_summarizer.py
│   └── voice_assistant.py
│
```

---

## ⚡ Setup Instructions

1. **Clone the repo** :

```bash
git clone https://github.com/fatirmalik1/AI_Agentic_Toolbox.git
cd AI_Agentic_Toolbox
```

2. **Create a Python environment** (Python 3.11 recommended):

```bash
conda create -n agents python=3.11 -y
conda activate agents
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Environment variables**:
   Create a `.env` file with your API keys:

```
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk-...
ELEVEN_LABS_API_KEY=...
```

5. **Run the Streamlit app**:

```bash
streamlit run app.py
```

---

## 📰 Article Summarizer

* Navigate to **📰 Article Summarizer** in the sidebar.
* Choose provider (**OpenAI** or **Groq**) and model.
* Paste a news article URL.
* Select summary type: **Concise** or **Detailed**.
* Click **Summarize**.
* Outputs include article summary, metadata (title, authors, published date), and model info.

---

## 🎥 YouTube Video Summarizer & Q/A

* Navigate to **🎥 YouTube Summarizer**.
* Select LLM provider, LLM model, and embeddings provider in the sidebar.
* Paste a YouTube video URL.
* Click **Summarize Video**.
* Outputs include:
  * Video title
  * Transcript
  * Summarized text
  * Conversational Q/A interface

* Ask questions about the video in the text box, and the assistant will answer using vector-based retrieval.

---

## 🎙️ Voice Assistant RAG

* Navigate to **🎙️ Voice Agent**.
* Upload documents to create a knowledge base.
* Configure LLM model and embedding type from the sidebar.
* Record voice input by clicking **🎙 Start Recording**.

* Click **▶ Process Recording** to:
  1. Transcribe your voice
  2. Generate a response using the document knowledge base
  3. Convert the response to speech using ElevenLabs TTS and play it

* Conversation history is displayed in the interface.

**Notes:**

* Uploaded documents are processed and stored in `voice_knowledge_base/`.
* ElevenLabs voice ID defaults to `"JBFqnCBsd6RMkjVDRZzb"` (can be changed in `voice_assistant_backend.py`).
* Supported embeddings: `OpenAI` and `HuggingFace`.

---

## 🧩 Notes
* The app automatically handles vector store creation and persistence for voice documents using Chroma.
* Voice assistant uses Whisper (`base`) for transcription.
* YouTube summarizer uses Whisper for video transcription and LangChain for summarization + conversational Q/A.

---

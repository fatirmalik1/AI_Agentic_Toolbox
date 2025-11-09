import os
import yt_dlp
import whisper
from typing import List, Dict
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_core.documents import Document

load_dotenv()


# ==================================
# EMBEDDING MODEL
# ==================================
class EmbeddingModel:
    """Supports OpenAI and HuggingFace embeddings"""

    def __init__(self, model_type="openai"):
        self.model_type = model_type.lower()

        if self.model_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key missing in .env")
            # OpenAIEmbeddings wrapper from langchain_openai
            self.embedding_fn = OpenAIEmbeddings(
                model="text-embedding-3-small", openai_api_key=api_key
            )

        elif self.model_type == "huggingface":
            # Use sentence-transformers/all-mpnet-base-v2 by default
            self.embedding_fn = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )

        else:
            raise ValueError("Unsupported embedding model. Use 'openai' or 'huggingface'.")


# ==================================
# LLM MODEL
# ==================================
class LLMModel:
    """Handles OpenAI and Groq LLMs"""

    def __init__(self, model_type="openai", model_name="gpt-4o-mini"):
        self.model_type = model_type.lower()
        self.model_name = model_name

        if self.model_type == "openai":
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OpenAI API key missing in .env")
            self.llm = ChatOpenAI(model_name=model_name, temperature=0)

        elif self.model_type == "groq":
            key = os.getenv("GROQ_API_KEY")
            if not key:
                raise ValueError("GROQ API key missing in .env")
            self.llm = ChatGroq(model=model_name, temperature=0)

        else:
            raise ValueError("Unsupported LLM type. Use 'openai' or 'groq'.")


# ==================================
# YOUTUBE SUMMARIZER
# ==================================
class YoutubeVideoSummarizer:
    def __init__(
        self,
        llm_type="openai",
        llm_model_name="gpt-4o-mini",
        embedding_type="openai",
        whisper_model_name="base",
    ):
        """Initialize summarizer with selected provider"""
        self.embedding_model = EmbeddingModel(embedding_type)
        self.llm_model = LLMModel(llm_type, llm_model_name)
        self.whisper_model = whisper.load_model(whisper_model_name)

        os.makedirs("downloads", exist_ok=True)

    # ------------------------
    # INFO
    # ------------------------
    def get_model_info(self) -> Dict:
        return {
            "llm_type": self.llm_model.model_type,
            "llm_model": self.llm_model.model_name,
            "embedding_type": self.embedding_model.model_type,
        }

    # ------------------------
    # PIPELINE STEPS
    # ------------------------
    def download_video(self, url: str) -> tuple[str, str]:
        """Download audio-only track"""
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": "downloads/%(title)s.%(ext)s",
            "quiet": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "Untitled Video")
            # yt_dlp prepare filename may end with original extension; we convert to .mp3
            audio_path = ydl.prepare_filename(info)
            if audio_path.endswith(".webm"):
                audio_path = audio_path.replace(".webm", ".mp3")
            elif audio_path.endswith(".m4a"):
                audio_path = audio_path.replace(".m4a", ".mp3")
            return audio_path, title

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe video audio with Whisper"""
        result = self.whisper_model.transcribe(audio_path)
        return result.get("text", "")

    def create_documents(self, text: str, title: str) -> List[Document]:
        """Split transcript into chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ". ", " ", ""]
        )
        return [
            Document(page_content=chunk, metadata={"source": title})
            for chunk in splitter.split_text(text)
        ]

    def generate_summary(self, documents: List[Document]) -> str:
        """Summarize video transcript"""
        map_prompt = ChatPromptTemplate.from_template(
            """Write a brief summary of the following video section:
{text}
SUMMARY:"""
        )
        combine_prompt = ChatPromptTemplate.from_template(
            """Combine the following section summaries into a comprehensive final summary:
{text}
FINAL SUMMARY:"""
        )

        chain = load_summarize_chain(
            llm=self.llm_model.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=False,
        )

        result = chain.invoke(documents)
        # normalize returned structure
        if isinstance(result, dict):
            return result.get("output_text") or result.get("text") or str(result)
        return str(result)

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Build vector store for Q&A"""
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model.embedding_fn,
            collection_name=f"yt_summary_{self.embedding_model.model_type}",
        )

    def setup_qa_chain(self, vector_store: Chroma):
        """Create a conversational retriever"""
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm_model.llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            verbose=False,
        )

    def process_video(self, url: str) -> Dict:
        """Run full summarization and Q&A setup"""
        try:
            audio_path, title = self.download_video(url)
            transcript = self.transcribe_audio(audio_path)
            documents = self.create_documents(transcript, title)
            summary = self.generate_summary(documents)
            vector_store = self.create_vector_store(documents)
            qa_chain = self.setup_qa_chain(vector_store)

            # attempt cleanup of downloaded audio (best-effort)
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception:
                pass

            return {
                "title": title,
                "summary": summary,
                "qa_chain": qa_chain,
                "transcript": transcript,
                "model_info": self.get_model_info(),
            }
        except Exception as e:
            return {"error": str(e)}

import os
import tempfile
import whisper
import sounddevice as sd
import soundfile as sf
from typing import List
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.documents import Document
from elevenlabs import ElevenLabs

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
            self.embedding_fn = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
        elif self.model_type == "huggingface":
            self.embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        else:
            raise ValueError("Unsupported embedding model. Use 'openai' or 'huggingface'.")


# ==================================
# LLM MODEL
# ==================================
class LLMModel:
    """OpenAI LLM wrapper"""

    def __init__(self, model_name="gpt-4o-mini"):
        key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)


# ==================================
# DOCUMENT PROCESSOR
# ==================================
class DocumentProcessor:
    """Load, split, and vectorize documents"""

    def __init__(self, embedding_type="openai"):
        self.embedding = EmbeddingModel(embedding_type).embedding_fn
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
        )

    def process_documents(self, documents: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(documents)

    
    def create_vector_store(self, documents: List[Document], persist_directory: str):
        """Create or reload a Chroma vector store (with auto persistence)"""
        os.makedirs(persist_directory, exist_ok=True)

        # If documents exist, rebuild embeddings completely
        if documents:
            print(f"📘 Building new vector store with {len(documents)} documents...")
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding,
                persist_directory=persist_directory,
            )
        else:
            print("🗂️ Loading existing persisted vector store...")
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embedding,
            )

        # Force materialize collection and test retrieval
        try:
            _ = vector_store._collection.count()
            print(f"✅ Vector store ready with {vector_store._collection.count()} embeddings.")
        except Exception as e:
            print(f"⚠️ Vector store check failed: {e}")

        return vector_store
    

    def load_documents(self, directory: str) -> List[Document]:
        """Load PDF, TXT, MD files"""
        from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, UnstructuredMarkdownLoader

        loaders = {
            ".pdf": DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader),
            ".txt": DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader),
            ".md": DirectoryLoader(directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader),
        }

        docs = []
        for ext, loader in loaders.items():
            try:
                docs.extend(loader.load())
            except Exception:
                continue
        return docs


# ==================================
# VOICE GENERATOR (ElevenLabs)
# ==================================
class VoiceGenerator:
    def __init__(self, api_key):
        self.client = ElevenLabs(
            base_url="https://api.elevenlabs.io",
            api_key=api_key
        )
        # Replace with your preferred default voice ID
        self.default_voice_id = "JBFqnCBsd6RMkjVDRZzb"

    def generate_voice_response(self, text: str, voice_id: str = None) -> str:
        """
        Converts text to speech using ElevenLabs.
        Returns the path to the generated MP3 file.
        """
        import tempfile
        voice_id = voice_id or self.default_voice_id
        try:
            audio_generator = self.client.text_to_speech.convert(
                voice_id=voice_id,
                output_format="mp3_44100_128",
                text=text,
                model_id="eleven_multilingual_v2"
            )

            tmp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            with open(tmp_file.name, "wb") as f:
                for chunk in audio_generator:
                    f.write(chunk)
            return tmp_file.name
        except Exception as e:
            print("ElevenLabs TTS error:", e)
            return None


# ==================================
# VOICE ASSISTANT RAG
# ==================================
class VoiceAssistantRAG:
    def __init__(self, elevenlabs_api_key, embedding_type="openai", llm_model_name="gpt-4o-mini"):
        self.whisper_model = whisper.load_model("base")
        self.llm = LLMModel(llm_model_name).llm
        self.embedding_type = embedding_type
        self.vector_store = None
        self.qa_chain = None
        self.voice_generator = VoiceGenerator(elevenlabs_api_key)
        self.sample_rate = 44100

    def setup_vector_store(self, vector_store=None, persist_directory="voice_knowledge_base"):
        """Attach or reload a persisted vector store"""
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
                raise ValueError("No vector store found. Please upload and process documents first.")
            embedding = EmbeddingModel(self.embedding_type).embedding_fn
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding,
            )

        # ✅ Always rebuild retriever and memory from this exact store
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            verbose=True,
        )
        print("🧠 QA chain initialized and connected to vector store.")


    def record_audio(self, duration=5):
        recording = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1)
        sd.wait()
        return recording

    def transcribe_audio(self, audio_array):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_array, self.sample_rate)
            text = self.whisper_model.transcribe(tmp_file.name)["text"]
            os.unlink(tmp_file.name)
            return text

    def generate_response(self, query):
        if self.qa_chain is None:
            # Try reattaching persisted store automatically
            try:
                self.setup_vector_store()
            except Exception:
                return "Error: Vector store not initialized."
        try:
            response = self.qa_chain.invoke({"question": query})
            return response.get("answer", "No answer found.")
        except Exception as e:
            return f"Error during QA: {str(e)}"

    def text_to_speech(self, text, voice_name=None):
        return self.voice_generator.generate_voice_response(text, voice_name)

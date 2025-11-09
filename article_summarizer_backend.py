import os
from typing import Optional, List, Dict
from dotenv import load_dotenv
from newspaper import Article
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_classic.schema import Document

# Load environment variables
load_dotenv()

# ==========================
# BACKEND SUMMARIZER CLASS
# ==========================
class NewsArticleSummarizer:
    """
    Summarizes news articles using Groq or OpenAI models.
    Uses env keys internally (never exposed in UI).
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "openai",
        model_name: str = "gpt-5-mini",
    ):
        self.provider = provider.lower()
        self.model_name = model_name

        # --- Secure key loading ---
        # Prefer user-provided key; fallback to .env
        if self.provider == "openai":
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OpenAI API key is missing.")
            os.environ["OPENAI_API_KEY"] = key
            self.llm = ChatOpenAI(temperature=0, model_name=self.model_name)

        elif self.provider == "groq":
            key = api_key or os.getenv("GROQ_API_KEY")
            if not key:
                raise ValueError("Groq API key is missing.")
            os.environ["GROQ_API_KEY"] = key
            self.llm = ChatGroq(model=self.model_name, temperature=0)

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200, length_function=len
        )

    def fetch_article(self, url: str) -> Optional[Article]:
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article
        except Exception as e:
            raise RuntimeError(f"Error fetching article: {e}")

    def create_documents(self, text: str) -> List[Document]:
        chunks = self.text_splitter.split_text(text)
        return [Document(page_content=chunk) for chunk in chunks]

    def summarize(self, url: str, summary_type: str = "detailed") -> Dict:
        article = self.fetch_article(url)
        if not article:
            return {"error": "Failed to fetch article"}

        docs = self.create_documents(article.text)

        if summary_type == "detailed":
            map_prompt_template = """Write a detailed summary of the following text:
            "{text}"
            DETAILED SUMMARY:"""
            combine_prompt_template = """Combine and summarize the following summaries into a detailed final summary:
            "{text}"
            FINAL DETAILED SUMMARY:"""
        else:
            map_prompt_template = """Write a concise summary of the following text:
            "{text}"
            CONCISE SUMMARY:"""
            combine_prompt_template = """Combine and summarize the following summaries into a concise final summary:
            "{text}"
            FINAL CONCISE SUMMARY:"""

        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
        combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

        chain = load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=False,
        )

        summary = chain.invoke(docs)

        return {
            "title": article.title or "Untitled Article",
            "authors": article.authors or [],
            "publish_date": str(article.publish_date) if article.publish_date else "N/A",
            "summary": summary["output_text"],
            "url": url,
            "model_info": {"provider": self.provider, "model": self.model_name},
        }

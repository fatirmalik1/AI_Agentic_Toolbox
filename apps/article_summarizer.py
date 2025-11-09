import streamlit as st
from article_summarizer_backend import NewsArticleSummarizer

def run_article_summarizer():
    st.title("📰 News Article Summarizer")
    st.caption("Summarize any online news article using OpenAI or Groq models.")

    provider = st.selectbox("Select Provider", ["OpenAI", "Groq"])

    openai_models = [
        "gpt-5", "gpt-5-mini", "gpt-4o", "gpt-4-turbo"
   ]
    groq_models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b"
    ]

    model_name = st.selectbox(
        "Select Model",
        openai_models if provider == "OpenAI" else groq_models,
    )

    article_url = st.text_input("🔗 Article URL", placeholder="Paste the article link...")
    summary_type = st.radio("Summary Type", ["Concise", "Detailed"], horizontal=True)

    if st.button("🚀 Summarize Article"):
        if not article_url.strip():
            st.warning("Please enter a valid article URL.")
            return

        api_key = (
            st.session_state.get("openai_api_key")
            if provider == "OpenAI"
            else st.session_state.get("groq_api_key")
        )

        if not api_key:
            st.error(f"No {provider} API key found. Please set it in the sidebar.")
            return

        try:
            with st.spinner("Summarizing article... please wait ⏳"):
                summarizer = NewsArticleSummarizer(
                    api_key=api_key,
                    provider=provider,
                    model_name=model_name,
                )
                result = summarizer.summarize(article_url, summary_type.lower())

            if "error" in result:
                st.error(result["error"])
                return

            st.success("✅ Summary generated successfully!")
            st.subheader("🧾 Article Summary")
            st.write(result["summary"])

            st.markdown("---")
            st.markdown(f"**Title:** {result['title']}")
            st.markdown(f"**Authors:** {', '.join(result['authors']) or 'N/A'}")
            st.markdown(f"**Published:** {result['publish_date']}")
            st.markdown(f"**Provider:** {result['model_info']['provider']}")
            st.markdown(f"**Model:** {result['model_info']['model']}")

        except Exception as e:
            st.error(f"Error: {e}")

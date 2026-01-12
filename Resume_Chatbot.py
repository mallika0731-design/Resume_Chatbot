# app.py - Resume Chatbot with FAISS + Streamlit + HuggingFace (FREE, no API key needed!)
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import re

# Your resume chunks (from previous code)
resume_chunks = {
    "profile_basic": """Name: Mallika Bhardwaj Location: Delhi, India Email: mallikabhardwaj@email.com Phone: +91-8368257628 GitHub: https://github.com/mallika0731-design LinkedIn: https://linkedin.com/in/mallika-bhardwaj-901982155""",
    "professional_summary": """Mallika Bhardwaj is a Data Science professional with a strong academic foundation in Mathematics, Machine Learning, and Predictive Analytics. Skilled in Python, Tableau, Excel. Pursuing Data Science at IIT Delhi. Experience in ML pipelines, time-series forecasting, data visualization dashboards. Focus: data-centric decision-making.""",
    "education": """Data Science and Decision Science (Ongoing), IIT Delhi. M.Sc. Mathematics, University of Delhi. B.Sc. Hons. Mathematics, University of Delhi.""",
    "technical_skills": """Python (Pandas, NumPy, Scikit-learn, Statsmodels), SQL, Linear/Logistic Regression, Random Forest, Clustering, ARIMA, LSTM, Time Series, ADF Test, Tableau, Streamlit, Excel.""",
    "core_competencies": """Data Cleaning, EDA, Feature Engineering, Statistical Modeling, Model Evaluation, Predictive Analytics, Data Storytelling, Research Documentation.""",
    "project_festive_sales": """Festive Sales Analytics Dashboard. Streamlit app for Diwali/Amazon sales. Data pipelines, ML for trends, interactive dashboards.""",
    "project_covid_dashboard": """COVID-19 India Tableau Dashboard. ARIMA forecasting, ADF testing, correlation/regression for policy impact.""",
    "project_stock_forecasting": """Stock Market Forecasting. ARIMA, LSTM, Linear Regression. Stationarity testing, RSI/moving averages/volatility. MAE/RMSE/R² eval.""",
    "experience_maincrafts": """Data Science Intern, Maincrafts Technology (Remote). EDA on Student/Titanic datasets, Pandas cleaning/FE, Matplotlib/Seaborn viz, Jupyter notebooks.""",
    "experience_deloitte": """Virtual Data Science Intern, Deloitte Forage (Oct-Nov 2025). Excel/Tableau prep/viz, client presentations, communication skills.""",
    "certifications": """Analyzing Data with Excel (IBM edX), Data Analytics Job Simulation (Deloitte Forage).""",
    "achievements": """ML/time-series solutions, strong math modeling, structured communication.""",
    "areas_of_interest": """Machine Learning, Predictive Analytics, AI, BI, Statistical Modeling, Data Visualization."""
}

# Recruiter-style prompt templates [web:11][web:12]
RECRUITER_PROMPTS = [
    "Tell me about your experience with time series forecasting projects.",
    "Walk me through your technical skills in Python and ML models.",
    "What was your role and key achievements in your Maincrafts internship?",
    "How did you build and deploy the Festive Sales dashboard?",
    "Describe your education background and current studies at IIT Delhi.",
    "Can you explain your stock market forecasting project metrics?",
    "What tools did you use in the Deloitte virtual internship?",
    "Tell me about a challenging data cleaning task you handled.",
    "How do you evaluate ML model performance (MAE, RMSE, etc.)?",
    "Why are you interested in data science roles?"
]

@st.cache_resource
def load_vector_store():
    """Load or create FAISS index with sentence transformers (better than TF-IDF!)"""
    chunk_texts = list(resume_chunks.values())
    chunk_ids = list(resume_chunks.keys())
    
    # Use lightweight HuggingFace model (downloads ~80MB first time)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunk_texts)
    
    # FAISS index (inner product for cosine sim)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)  # For cosine similarity
    index.add(embeddings.astype('float32'))
    
    # Cache data
    st.session_state['index'] = index
    st.session_state['model'] = model
    st.session_state['chunk_texts'] = chunk_texts
    st.session_state['chunk_ids'] = chunk_ids
    
    return index, model, chunk_texts, chunk_ids

def retrieve_relevant_chunks(query, model, index, chunk_texts, chunk_ids, top_k=3):
    """Retrieve top-k relevant chunks using FAISS"""
    query_emb = model.encode([query])
    faiss.normalize_L2(query_emb)
    scores, indices = index.search(query_emb.astype('float32'), top_k)
    
    context = []
    for i, idx in enumerate(indices[0]):
        context.append({
            'id': chunk_ids[idx],
            'text': chunk_texts[idx],
            'score': scores[0][i]
        })
    return context

def generate_response(query, context):
    """Simple rule-based response + recruiter context. Swap with OpenAI/Groq/HF for production."""
    context_text = "\n\n".join([f"[{c['id']}] {c['text']}" for c in context])
    
    # Recruiter-style response template
    response = f"""**Relevant Experience:**\n\n{context_text}\n\n**Recruiter Summary:** This matches your {', '.join([c['id'] for c in context])}. Key highlights include {'; '.join([c['text'][:100] + '...' for c in context])}. Strong fit for data science roles focusing on ML forecasting and visualization.[web:11]"""
    
    return response

# Streamlit UI
st.set_page_config(page_title="Mallika's Resume Bot", layout="wide")
st.title("🤖 Mallika's Resume Chatbot")
st.markdown("Powered by FAISS + HuggingFace • Ask about projects, skills, experience!")

# Load vector store
if 'index' not in st.session_state:
    with st.spinner("Loading resume embeddings..."):
        load_vector_store()

col1, col2 = st.columns([3,1])
with col1:
    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about my projects, skills, or experience..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Retrieve
            context = retrieve_relevant_chunks(
                prompt, 
                st.session_state['model'], 
                st.session_state['index'], 
                st.session_state['chunk_texts'], 
                st.session_state['chunk_ids']
            )
            
            # Generate
            with st.spinner("Retrieving relevant experience..."):
                response = generate_response(prompt, context)
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

with col2:
    st.subheader("🚀 Quick Recruiter Questions")
    selected_prompt = st.selectbox("Pick one:", RECRUITER_PROMPTS, key="quick")
    if st.button("Ask This", key="quick_ask"):
        st.session_state.messages.append({"role": "user", "content": selected_prompt})
        st.rerun()

# Sidebar: Upgrade options
with st.sidebar:
    st.header("⚙️ Upgrade Path")
    st.markdown("""
    **OpenAI Integration:**
    ```python
    from langchain_openai import ChatOpenAI
    from langchain.chains import create_retrieval_chain
    llm = ChatOpenAI(model="gpt-4o-mini")
    ```
    
    **HuggingFace (Free):**
    ```python
    from transformers import pipeline
    generator = pipeline("text-generation", model="microsoft/DialoGPT-medium")
    ```
    
    **Deploy:** `streamlit run app.py` → GitHub + Streamlit Cloud
    """)
    st.caption("FAISS ensures <100ms retrieval [web:4][web:9]")

st.caption("💡 Perfect for interviews! Handles 'Walk me through your projects?' perfectly [web:11]")

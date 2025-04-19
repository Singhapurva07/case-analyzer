import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

st.set_page_config(page_title="Legal Case Similarity Search", layout="wide", initial_sidebar_state="auto")

# Light / Dark mode toggle
mode = st.sidebar.selectbox("Choose Mode", ["Light", "Dark"])

if mode == "Dark":
    st.markdown("""
        <style>
            body {
                background-color: #111;
                color: white;
            }
            .stApp {
                background-color: #111;
            }
        </style>
    """, unsafe_allow_html=True)

# Load model
@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load data
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("legal_text_classification.csv")
    df.dropna(subset=['case_text'], inplace=True)
    df = df.head(100)  # Optional: reduce for testing
    return df

df = load_data()

# Encode the dataset
@st.cache_resource(show_spinner=False)
def encode_corpus(texts):
    embeddings = []
    batch_size = 32
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_tensor=True)
        embeddings.append(emb)
    return torch.cat(embeddings)

st.markdown("## Legal Case Similarity Finder")
st.markdown("""Enter a legal case description below, and the system will retrieve the most relevant past cases based on semantic similarity.""")

# Encode only once
corpus_embeddings = encode_corpus(df['case_text'].tolist())

# Text input
user_input = st.text_area("Enter your case description:", height=200)

if st.button("Find Similar Cases") and user_input:
    with st.spinner("Searching for similar cases..."):
        query_embedding = model.encode(user_input, convert_to_tensor=True)
        top_k = min(5, len(df))
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]

        st.success(f"Top {top_k} similar cases:")
        for hit in hits:
            score = hit['score']
            idx = hit['corpus_id']
            st.markdown(f"### Case Title: {df.iloc[idx]['case_title']}")
            st.markdown(f"**Outcome:** {df.iloc[idx]['case_outcome']}")
            st.markdown(f"**Similarity Score:** {score:.4f}")
            with st.expander("Case Details"):
                st.write(df.iloc[idx]['case_text'])
            st.markdown("---")

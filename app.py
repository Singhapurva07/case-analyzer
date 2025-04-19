import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import os

st.set_page_config(page_title="Legal Case Similarity Search", layout="wide")

# 🌙 Light / Dark mode toggle
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

# 🚀 Load the fine-tuned model
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = './output_legal_similarity_model'
    if not os.path.exists(model_path):
        st.error("❌ Model not found at './output_legal_similarity_model'")
        st.stop()
    return SentenceTransformer(model_path)

model = load_model()

# 🧠 Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st.sidebar.markdown(f"**🖥️ Using device:** `{device}`")

# 📚 Load dataset
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("legal_text_classification.csv")
    df.dropna(subset=['case_text'], inplace=True)
    df = df.reset_index(drop=True)

    # ⚙️ DEV ONLY: Limit rows for speed
    df = df.head(200)

    return df

df = load_data()
st.sidebar.write(f"📄 Total Cases: `{len(df)}`")

# ⚡ Load or encode corpus
def encode_corpus(texts):
    path = "corpus_embeddings.pt"
    if os.path.exists(path):
        st.success("✅ Loaded cached embeddings.")
        return torch.load(path)

    st.info("🔄 First-time encoding... Please wait.")
    batch_size = 32
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding Corpus"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_tensor=True, device=device)
        all_embeddings.append(emb)

    all_embeddings = torch.cat(all_embeddings)
    torch.save(all_embeddings, path)
    st.success("✅ Embeddings saved to disk.")
    return all_embeddings

corpus_embeddings = encode_corpus(df['case_text'].tolist())

# 🎯 Main UI
st.title("📚 Legal Case Similarity Finder")
st.markdown("Enter a legal case description below to retrieve the most similar previous cases using semantic similarity.")

user_input = st.text_area("📝 Enter your case description:", height=200)

if st.button("🔍 Find Similar Cases") and user_input.strip():
    with st.spinner("Analyzing..."):
        query_embedding = model.encode(user_input, convert_to_tensor=True, device=device)
        top_k = min(5, len(df))
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]

        st.success(f"Top {top_k} similar cases found:")
        for hit in hits:
            idx = hit['corpus_id']
            score = hit['score']
            case = df.iloc[idx]

            title = case['case_title'] if 'case_title' in df.columns else f"Case #{idx + 1}"
            st.markdown(f"### 🧾 {title}")
            st.markdown(f"**⚖️ Outcome:** `{case['case_outcome']}`")
            st.markdown(f"**🔗 Similarity Score:** `{score:.4f}`")
            with st.expander("📄 View Full Case Text"):
                st.write(case['case_text'])
            st.markdown("---")

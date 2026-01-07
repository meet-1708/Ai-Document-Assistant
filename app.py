import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline


st.set_page_config(page_title="AI Document Assistant", layout="wide")

st.title("📄 AI Document Assistant")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

def extract_text_from_pdf(pdf_file):
    text = ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def clean_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(chunks, model):
    embeddings = model.encode(chunks)
    return np.array(embeddings)

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def semantic_search(query, model, index, chunks, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append(chunks[idx])

    return results

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, summarizer, max_length=150, min_length=60):
    summary = summarizer(
        text[:1024],  # model limit
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )
    return summary[0]["summary_text"]



if uploaded_file:
    extracted_text = extract_text_from_pdf(uploaded_file)
    cleaned_text = clean_text(extracted_text)

    chunks = chunk_text(cleaned_text)

    st.success(f"Document split into {len(chunks)} chunks")

    if st.button("Generate Embeddings"):
        model = load_embedding_model()
        embeddings = generate_embeddings(chunks, model)

        index = build_faiss_index(embeddings)

        st.session_state["model"] = model
        st.session_state["chunks"] = chunks
        st.session_state["index"] = index

        st.success("FAISS index created successfully!")

if "index" in st.session_state:
    st.subheader("🔍 Ask a Question (Semantic Search)")

    query = st.text_input("Enter your question")

    if query:
        results = semantic_search(
            query,
            st.session_state["model"],
            st.session_state["index"],
            st.session_state["chunks"]
        )

        st.subheader("📄 Most Relevant Document Sections")
        for i, res in enumerate(results):
            st.write(f"**Result {i+1}:**")
            st.write(res[:500] + "...")

if "chunks" in st.session_state:
    st.subheader("📝 Document Summary")

    if st.button("Generate Summary"):
        summarizer = load_summarizer()
        text_to_summarize = " ".join(st.session_state["chunks"][:3])
        summary = summarize_text(
            text_to_summarize,
            summarizer
        )

        st.success("Summary generated!")
        st.write(summary)

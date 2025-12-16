import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np


st.set_page_config(page_title="AI Document Assistant", layout="wide")

st.title("📄 AI Document Assistant")
st.subheader("Phase 1: PDF Upload & Text Extraction")

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

if uploaded_file:
    extracted_text = extract_text_from_pdf(uploaded_file)
    cleaned_text = clean_text(extracted_text)

    chunks = chunk_text(cleaned_text)

    st.success(f"Document split into {len(chunks)} chunks")

    if st.button("Generate Embeddings"):
        model = load_embedding_model()
        embeddings = generate_embeddings(chunks, model)

        st.write("Embedding shape:", embeddings.shape)
        st.success("Embeddings generated successfully!")

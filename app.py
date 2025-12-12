import streamlit as st
import fitz  # PyMuPDF

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

if uploaded_file:
    st.success("PDF uploaded successfully!")

    extracted_text = extract_text_from_pdf(uploaded_file)

    st.subheader("Extracted Text Preview")
    st.text_area(
        "Text from PDF",
        extracted_text[:5000],  # preview first 5000 chars
        height=400
    )

    st.info(f"Total characters extracted: {len(extracted_text)}")

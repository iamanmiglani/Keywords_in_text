# # Keyword and CTA database
# keywords = {
#     "buy": ["go purchase", "go order now", "go acquire"],
#     "subscribe": ["check now", "get it for yourself", "enroll"],
#     "call to action": ["buy now", "click here", "get yours"]
# }


import streamlit as st
from transformers import pipeline
from docx import Document
import pdfplumber

# Load LLM model
model = pipeline("text-classification", model="openai-gpt", tokenizer="openai-gpt")

# CTA Keywords
cta_keywords = ["Buy Now", "Sign Up Today", "Subscribe", "Call Us"]

# Function to process file
def extract_text(file):
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages)
    elif file.name.endswith(".docx"):
        doc = Document(file)
        text = "\n".join([p.text for p in doc.paragraphs])
    else:
        text = file.read().decode("utf-8")
    return text

# Search for CTA
def search_cta(text):
    results = []
    for keyword in cta_keywords:
        if keyword.lower() in text.lower():
            results.append(keyword)
    return results

# Streamlit App
st.title("Call-to-Action Finder")
st.write("Upload a document to search for Call-to-Action phrases.")

uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])
if uploaded_file:
    raw_text = extract_text(uploaded_file)
    st.subheader("Extracted Text")
    st.write(raw_text)

    st.subheader("Identified Call-to-Actions")
    found_ctas = search_cta(raw_text)
    if found_ctas:
        st.success(f"Found the following CTAs: {', '.join(found_ctas)}")
    else:
        st.warning("No CTAs found.")

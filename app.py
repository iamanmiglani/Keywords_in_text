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
from nltk.corpus import wordnet

# Ensure NLTK is set up correctly
import nltk
nltk.download("wordnet")

# CTA synonyms function
def get_synonyms(phrase):
    synonyms = set()
    for syn in wordnet.synsets(phrase):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

# Function to process uploaded files
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

# Function to search for phrases and their synonyms in the text
def search_phrases(text, phrases):
    results = {}
    for phrase in phrases:
        synonyms = get_synonyms(phrase)
        synonyms.add(phrase)
        found = [syn for syn in synonyms if syn.lower() in text.lower()]
        if found:
            results[phrase] = found
    return results

# Streamlit App
st.title("Call-to-Action and Keyword Finder")

tab1, tab2 = st.tabs(["Find Call-to-Actions", "Search for Keywords"])

# Tab 1: Call-to-Actions
with tab1:
    st.header("Find Call-to-Actions")
    uploaded_file = st.file_uploader("Upload a file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    cta_phrases = st.text_area("Enter call-to-action phrases (comma-separated)", "Buy Now, Sign Up Today, Subscribe")

    if uploaded_file and cta_phrases:
        raw_text = extract_text(uploaded_file)
        st.subheader("Extracted Text")
        st.write(raw_text)

        phrases = [phrase.strip() for phrase in cta_phrases.split(",")]
        st.subheader("Identified Call-to-Actions and Synonyms")
        results = search_phrases(raw_text, phrases)
        if results:
            for phrase, synonyms_found in results.items():
                st.success(f"Phrase: '{phrase}' found with synonyms: {', '.join(synonyms_found)}")
        else:
            st.warning("No call-to-actions or their synonyms found.")

# Tab 2: Keywords
with tab2:
    st.header("Search for Keywords")
    keywords = st.text_area("Enter keywords/phrases (comma-separated)", "Free, Discount, Offer")

    if keywords:
        keyword_list = [keyword.strip() for keyword in keywords.split(",")]
        uploaded_file = st.file_uploader("Upload a file (PDF, DOCX, TXT) for keyword search", type=["pdf", "docx", "txt"], key="keyword_upload")

        if uploaded_file:
            raw_text = extract_text(uploaded_file)
            st.subheader("Extracted Text")
            st.write(raw_text)

            st.subheader("Keywords and Synonyms Found")
            results = search_phrases(raw_text, keyword_list)
            if results:
                for keyword, synonyms_found in results.items():
                    st.success(f"Keyword: '{keyword}' found with synonyms: {', '.join(synonyms_found)}")
            else:
                st.warning("No keywords or their synonyms found.")

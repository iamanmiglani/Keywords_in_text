import streamlit as st
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import torch

# Load LLaMA model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("huggingface/llama")
    model = AutoModel.from_pretrained("huggingface/llama")
    return tokenizer, model

tokenizer, model = load_model()

def get_embeddings(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1)

def cosine_similarity(embedding1, embedding2):
    return util.cos_sim(embedding1, embedding2).item()

# Keyword and CTA database
keywords = {
    "buy": ["go purchase", "go order now", "go acquire"],
    "subscribe": ["check now", "get it for yourself", "enroll"],
    "call to action": ["buy now", "click here", "get yours"]
}

# Streamlit interface
st.title("LLaMA Keyword & CTA Search App")

# File uploader
uploaded_file = st.file_uploader("Upload Transcription File", type=["txt", "docx"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    st.write("### Transcription Content:")
    st.write(text)

    # Process text
    transcription_embedding = get_embeddings(text)
    results = []

    for keyword, synonyms in keywords.items():
        for synonym in [keyword] + synonyms:
            synonym_embedding = get_embeddings(synonym)
            similarity = cosine_similarity(transcription_embedding, synonym_embedding)
            if similarity > 0.8:  # Threshold for a match
                results.append((keyword, synonym, similarity))

    # Display results
    if results:
        st.write("### Matches Found:")
        for keyword, synonym, similarity in results:
            st.write(f"- **Keyword:** {keyword}, **Synonym:** {synonym}, **Similarity:** {similarity:.2f}")
    else:
        st.write("No matches found.")

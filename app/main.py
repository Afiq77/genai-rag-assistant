# app/main.py

import streamlit as st
from utils.rag_pipeline import load_vectorstore, get_response

st.set_page_config(page_title="GenAI RAG Assistant")
st.title("📄 GenAI RAG Assistant")

query = st.text_input("Ask a question based on your documents:")

if query:
    with st.spinner("Generating answer..."):
        vectorstore = load_vectorstore("embeddings/faiss_index")
        result = get_response(query, vectorstore)
        st.write("### Answer:")
        st.write(result["answer"])

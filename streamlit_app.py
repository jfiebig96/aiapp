import os
import streamlit as st
import tempfile
from langchain.prompts import ChatPromptTemplate
from chat_openrouter import ChatOpenRouter
from docloader import load_documents_from_folder
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch

# === FAISS Index z SentenceTransformer ===
class FAISSIndex:
    def __init__(self, faiss_index, metadata):
        self.index = faiss_index
        self.metadata = metadata

    def similarity_search(self, query_vector, k=3):
        D, I = self.index.search(query_vector, k)
        return [self.metadata[idx] for idx in I[0]]

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
model_embed = SentenceTransformer(embed_model_id, device="cpu")

def create_index(documents):
    texts = [doc["text"] for doc in documents]
    metadata = [{"filename": doc["filename"], "text": doc["text"]} for doc in documents]
    with torch.no_grad():
        embeddings = model_embed.encode(texts, convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return FAISSIndex(index, metadata)

def retrieve_docs(query, faiss_index, k=3):
    with torch.no_grad():
        query_vector = model_embed.encode([query], convert_to_numpy=True).astype("float32")
    return faiss_index.similarity_search(query_vector, k=k)

# === Streamlit App Setup ===
st.set_page_config(layout="wide", page_title="OpenRouter + PDF RAG Chat")
st.title("📄 OpenRouter PDF Chatbot")

# === Sesja czatu ===
if "query" not in st.session_state:
    st.session_state.query = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""

# === Upload wielu plików PDF ===
uploaded_files = st.file_uploader("📎 Prześlij pliki PDF", type=["pdf"], accept_multiple_files=True)
documents = []

if uploaded_files:
    with tempfile.TemporaryDirectory() as tmpdir:
        for file in uploaded_files:
            file_path = os.path.join(tmpdir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
        documents = load_documents_from_folder(tmpdir)

    if documents:
        st.success(f"📚 Załadowano {len(documents)} dokumentów")
        for doc in documents:
            st.markdown(f"- `{doc['filename']}` ({len(doc['text'])} znaków)")

# === Budowanie indeksu FAISS ===
index = create_index(documents) if documents else None

# === Obsługa zapytań ===
selected_model = "mistralai/mistral-7b-instruct:free"
model = ChatOpenRouter(model_name=selected_model, temperature=0.3)

def answer_question(question, documents, model):
    context = "\n\n".join([doc["text"] for doc in documents])
    template = """
    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

if query := st.chat_input("Zadaj pytanie na podstawie PDF..."):
    st.session_state.query = query

    if not index:
        st.session_state.answer = "⚠️ Nie przesłano żadnych plików PDF."
    else:
        docs = retrieve_docs(query, index, k=3)
        response = answer_question(query, docs, model)
        st.session_state.answer = response.content if hasattr(response, "content") else str(response)

# === Wyświetlanie wyników ===
if st.session_state.query:
    st.chat_message("user").write(st.session_state.query)
if st.session_state.answer:
    st.chat_message("assistant").write(st.session_state.answer)

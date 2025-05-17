import os
import streamlit as st
import tempfile
from langchain.prompts import ChatPromptTemplate
from chat_openrouter import ChatOpenRouter
from docloader import load_documents_from_folder
from embedder import create_index, retrieve_docs

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

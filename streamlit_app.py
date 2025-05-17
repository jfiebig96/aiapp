import os
import streamlit as st
import tempfile
from chat_openrouter import ChatOpenRouter
from docloader import load_documents_from_folder
from embedder import create_index, retrieve_docs

# === Streamlit App Setup ===
st.set_page_config(layout="wide", page_title="OpenRouter + PDF RAG Chat")
st.title("📄 OpenRouter PDF Chatbot")

# === Sesja czatu ===
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Wgraj PDF i zadaj pytanie!"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

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
if prompt := st.chat_input("Zadaj pytanie na podstawie PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not index:
        msg = "⚠️ Nie przesłano żadnych plików PDF."
    else:
        context_chunks = retrieve_docs(prompt, index, k=3)
        context_text = "\n\n".join([chunk["text"][:1000] for chunk in context_chunks])
        system_prompt = f"Oto kontekst z dokumentów PDF:\n{context_text}"

        llm = ChatOpenRouter(temperature=0.3)
        response = llm.invoke([{"role": "system", "content": system_prompt},
                               {"role": "user", "content": prompt}])
        msg = response.content if hasattr(response, "content") else str(response)

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)


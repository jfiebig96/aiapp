import os
import fitz
import streamlit as st
from openai import OpenAI

# === Funkcja: wyciąganie tekstu z pojedynczego pliku PDF ===
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# === Funkcja: wczytywanie wszystkich dokumentów PDF z folderu ===
def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            text = load_pdf(os.path.join(folder_path, filename))
            documents.append({"filename": filename, "text": text})
    return documents

# === Streamlit App Setup ===
st.set_page_config(layout="wide", page_title="OpenRouter chatbot app")
st.title("OpenRouter chatbot app")

api_key, base_url = st.secrets["api_key"], st.secrets["BASE_URL"]
selected_model = "google/gemma-3-1b-it:free"

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# === Możliwość wyboru folderu PDF (dla lokalnych zastosowań) ===
folder_path = st.text_input("📂 Wprowadź ścieżkę do folderu z PDF (opcjonalne):")
if folder_path and os.path.isdir(folder_path):
    docs = load_documents_from_folder(folder_path)
    st.markdown(f"Znaleziono {len(docs)} dokumentów PDF:")
    for doc in docs:
        st.markdown(f"- `{doc['filename']}` ({len(doc['text'])} znaków)")

# === Obsługa zapytań ===
if prompt := st.chat_input():
    if not api_key:
        st.info("Invalid API key.")
        st.stop()
    client = OpenAI(api_key=api_key, base_url=base_url)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = client.chat.completions.create(
        model=selected_model,
        messages=st.session_state.messages
    )
    msg = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

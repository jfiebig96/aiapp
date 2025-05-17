import os
import fitz
import streamlit as st
from openai import OpenAI

# === Funkcja: wyciąganie tekstu z pojedynczego pliku PDF ===
def load_pdf_from_file(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# === Streamlit App Setup ===
st.set_page_config(layout="wide", page_title="OpenRouter chatbot app")
st.title("OpenRouter chatbot app")

api_key, base_url = st.secrets["api_key"], st.secrets["BASE_URL"]
selected_model = "google/gemma-3-1b-it:free"

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# === Upload wielu plików PDF ===
uploaded_files = st.file_uploader("📎 Prześlij jeden lub więcej plików PDF", type=["pdf"], accept_multiple_files=True)
all_texts = []

if uploaded_files:
    for file in uploaded_files:
        pdf_text = load_pdf_from_file(file)
        all_texts.append({"filename": file.name, "text": pdf_text})
    st.markdown(f"Znaleziono {len(all_texts)} przesłanych plików:")
    for doc in all_texts:
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

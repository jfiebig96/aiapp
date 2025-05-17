import os
import fitz
import streamlit as st
from openai import OpenAI
import io
from typing import Optional
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr

# === Klasa: Obsługa modelu OpenRouter przez LangChain ===
class ChatOpenRouter(ChatOpenAI):
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key", default_factory=st.secrets["API_KEY"]
    )

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": st.secrets["API_KEY"]}

    def __init__(self, openai_api_key: Optional[str] = None, **kwargs):
        openai_api_key = openai_api_key or st.secrets["API_KEY"]
        super().__init__(base_url=st.secrets["BASE_URL"], openai_api_key=openai_api_key, **kwargs)

# === Funkcja: wyciąganie tekstu z pojedynczego pliku PDF ===
def load_pdf_from_file(uploaded_file):
    text = ""
    file_bytes = uploaded_file.read()
    with fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# === Streamlit App Setup ===
st.set_page_config(layout="wide", page_title="OpenRouter chatbot app")
st.title("OpenRouter chatbot app")

api_key, base_url = st.secrets.get("api_key"), st.secrets.get("BASE_URL")
selected_model = "google/gemma-3-1b-it:free"

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# === Upload wielu plików PDF ===
uploaded_files = st.file_uploader("📎 Prześlij jeden lub więcej plików PDF", type=["pdf"], accept_multiple_files=True)
all_texts = []

if uploaded_files:
    for file in uploaded_files:
        try:
            pdf_text = load_pdf_from_file(file)
            all_texts.append({"filename": file.name, "text": pdf_text})
        except Exception as e:
            st.error(f"Błąd przy ładowaniu pliku {file.name}: {str(e)}")

    if all_texts:
        st.markdown(f"Znaleziono {len(all_texts)} przesłanych plików:")
        for doc in all_texts:
            st.markdown(f"- `{doc['filename']}` ({len(doc['text'])} znaków)")

# === Obsługa zapytań ===
if prompt := st.chat_input():
    if not api_key or not base_url:
        st.info("Invalid API key or BASE_URL.")
        st.stop()

    client = OpenAI(api_key=api_key, base_url=base_url)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    try:
        if all_texts:
            combined_pdf_text = "\n\n".join([f"{doc['filename']}:\n{doc['text']}" for doc in all_texts])
            pdf_context = {
                "role": "system",
                "content": f"Oto zawartość przesłanych plików PDF:\n{combined_pdf_text[:4000]}"
            }
            messages_with_pdf = [pdf_context] + st.session_state.messages
        else:
            messages_with_pdf = st.session_state.messages

        response = client.chat.completions.create(
            model=selected_model,
            messages=messages_with_pdf
        )
        msg = response.choices[0].message.content

    except Exception as e:
        msg = f"❌ Błąd podczas komunikacji z modelem: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)

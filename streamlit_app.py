import streamlit as st
import requests
import time
import fitz  # PyMuPDF
import os
import tempfile
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

# === Ustawienia API ===
if "api_key" not in st.secrets:
    st.error("❌ Brak klucza 'api_key' w sekcji Secrets (Settings > Secrets).")
    st.stop()

API_KEY = st.secrets["api_key"]
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "google/gemma-3-1b-it:free"

# === Funkcja: wyciąganie tekstu z PDF ===
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# === LangChain RAG: Wyszukiwanie odpowiedzi na podstawie tekstu PDF ===
def build_vectorstore(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    return FAISS.from_documents(docs, embeddings), docs

def answer_question_with_rag(vectorstore, question):
    llm = ChatOpenAI(openai_api_key=API_KEY, model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = vectorstore.similarity_search(question)
    return chain.run(input_documents=docs, question=question)

# === Interfejs Streamlit ===
st.title("📄 Chat + PDF (Gemma 3B + RAG)")
st.caption("Upload PDF i zadawaj pytania o jego zawartość!")

uploaded_file = st.file_uploader("📎 Prześlij plik PDF", type=["pdf"])
pdf_text = ""
vectorstore = None

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.success("✅ Załadowano PDF!")
    with st.expander("📖 Podgląd treści PDF"):
        st.write(pdf_text[:5000])
    vectorstore, _ = build_vectorstore(pdf_text)

# === Historia rozmowy ===
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Dawaj lecimy z tematem! 👇"}]

# === Wyświetlanie historii rozmowy ===
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# === Obsługa nowego zapytania ===
if prompt := st.chat_input("Zadaj pytanie na podstawie PDF-a lub ogólne..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        if vectorstore:
            assistant_message = answer_question_with_rag(vectorstore, prompt)
        else:
            assistant_message = "Nie przesłałeś jeszcze pliku PDF. Proszę dodaj dokument."

        for chunk in assistant_message.split():
            full_response += chunk + " "
            time.sleep(0.02)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": assistant_message})

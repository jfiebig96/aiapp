import streamlit as st
import requests
import time
import fitz  # PyMuPDF

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

# === Funkcja: zapytanie do OpenRouter ===
def chat_with_openrouter(messages):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
    }
    response = requests.post(f"{BASE_URL}/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ Błąd {response.status_code}: {response.text}"

# === Interfejs Streamlit ===
st.title("📄 Chat + PDF (Gemma 3B + OpenRouter)")

st.caption("Upload PDF i zadawaj pytania o jego zawartość!")

# === Upload PDF ===
uploaded_file = st.file_uploader("📎 Prześlij plik PDF", type=["pdf"])
pdf_text = ""

if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.success("✅ Załadowano PDF!")
    with st.expander("📖 Podgląd treści PDF"):
        st.write(pdf_text)

# === Historia rozmowy ===
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Dawaj lecimy z tematem! 👇"}]

# === Dodanie kontekstu z PDF ===
if pdf_text and not any("Zawartość PDF" in msg["content"] for msg in st.session_state.messages):
    st.session_state.messages.append({
        "role": "system",
        "content": f"Zawartość PDF użytkownika:\n{pdf_text[:3000]}"  # można ograniczyć
    })

# === Wyświetlanie historii rozmowy ===
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# === Wprowadzenie wiadomości ===
if prompt := st.chat_input("Zadaj pytanie na podstawie PDF-a lub ogólne..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        assistant_message = chat_with_openrouter(st.session_state.messages)

        for chunk in assistant_message.split():
            full_response += chunk + " "
            time.sleep(0.02)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": assistant_message})

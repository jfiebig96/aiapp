import streamlit as st
import requests
import time

# Bezpieczne pobieranie klucza API z sekcji Secrets
if "api_key" not in st.secrets:
    st.error("❌ Brak klucza 'api_key' w sekcji Secrets (Settings > Secrets).")
    st.stop()

API_KEY = st.secrets["api_key"]
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "google/gemma-3-1b-it:free"

# Funkcja do rozmowy z OpenRouter
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

# UI Streamlit
st.title("🤖 Chat z Gemma 3B by Kuba (OpenRouter)")
st.caption("Powered by CDV & OpenRouter.ai")

# Historia czatu
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Dawaj lecimy z tematem! 👇"}]

# Wyświetlanie historii
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Wprowadzenie nowej wiadomości
if prompt := st.chat_input("Wpisz wiadomość..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        assistant_message = chat_with_openrouter(st.session_state.messages)

        for chunk in assistant_message.split():
            full_response += chunk + " "
            time.sleep(0.03)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": assistant_message})

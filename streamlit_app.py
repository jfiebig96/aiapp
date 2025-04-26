import streamlit as st
import openai

# Ładowanie danych z secrets
API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "google/gemma-3b-it:free"

# Funkcja do wysyłania żądania ręcznie
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
        return f"Error {response.status_code}: {response.text}"

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Chat z Gemma 3B (OpenRouter)")

user_input = st.text_input("Twoja wiadomość:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

if st.button("Wyślij"):
    assistant_message = chat_with_openrouter(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})

for msg in st.session_state.messages:
    st.write(f"**{msg['role'].capitalize()}**: {msg['content']}")

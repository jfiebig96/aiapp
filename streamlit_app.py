import streamlit as st
import requests

# Twoje dane dostępowe
API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Funkcja do wysyłania zapytania
def chat_with_openrouter(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "google/gemma-3b-it",  # Model na OpenRouter: "google/gemma-3b-it"
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(BASE_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"

# Streamlit UI
st.title("Chat z Gemma 3B 3bit za darmo przez OpenRouter")

user_input = st.text_input("Wpisz wiadomość:")

if st.button("Wyślij"):
    if user_input:
        answer = chat_with_openrouter(user_input)
        st.write("**Odpowiedź:**")
        st.write(answer)
    else:
        st.warning("Najpierw wpisz wiadomość!")

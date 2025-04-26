import streamlit as st
import requests
import time

# Dane na sztywno
API_KEY = "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"
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "google/gemma-3-1b-it:free"

# Konfiguracja klienta OpenAI pod OpenRouter
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

st.title("Chat z Gemma 3B (OpenRouter) 💬")

st.caption("Powered by OpenRouter & Google Gemma 3B model.")

# Inicjalizacja historii czatu
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

# Wyświetlanie historii rozmowy
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Akceptowanie nowej wiadomości od użytkownika
if prompt := st.chat_input("Wpisz wiadomość..."):
    # Dodanie wiadomości użytkownika do historii
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Wyświetlenie wiadomości użytkownika
    with st.chat_message("user"):
        st.markdown(prompt)

    # Wysłanie wiadomości do OpenRouter i odbiór odpowiedzi
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        assistant_response = client.chat.completions.create(
            model=MODEL,
            messages=st.session_state.messages,
        )

        assistant_message = assistant_response.choices[0].message.content

        # Symulacja "pisania" wiadomości słowo po słowie
        for chunk in assistant_message.split():
            full_response += chunk + " "
            time.sleep(0.03)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Dodanie odpowiedzi asystenta do historii
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})

import streamlit as st
import openai

# Ładowanie danych z secrets
API_KEY = st.secrets["API_KEY"]
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "google/gemma-3b-it:free"

# Konfiguracja klienta OpenAI do pracy z OpenRouter
client = openai.OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# Inicjalizacja sesji
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit UI
st.title("Chat z Gemma 3B (OpenRouter)")

user_input = st.text_input("Twoja wiadomość:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

if st.button("Wyślij"):
    assistant_response = client.chat.completions.create(
        model=MODEL,
        messages=st.session_state.messages,
    )
    
    # Dodanie odpowiedzi do historii
    assistant_message = assistant_response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})

# Wyświetlenie historii rozmowy
for msg in st.session_state.messages:
    st.write(f"**{msg['role'].capitalize()}**: {msg['content']}")

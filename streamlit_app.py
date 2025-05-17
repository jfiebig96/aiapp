import os
import streamlit as st
from docloader import load_documents_from_folder
from embedder import create_index, retrieve_docs
from chat_openrouter import ChatOpenRouter
from langchain.schema import SystemMessage, HumanMessage

# Page configuration
st.set_page_config(layout="wide", page_title="RAG Chatbot")
st.title("Retrieval-Augmented Chatbot")

# Upload folder setup
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Sidebar: upload PDFs and model selection
with st.sidebar:
    st.header("Settings")
    uploaded_files = st.file_uploader(
        label="Upload PDF documents", type="pdf", accept_multiple_files=True
    )
    selected_model = st.selectbox(
        "Select model", options=[
            "google/gemma-3-1b-it:free",  # szybki i darmowy
            "mistralai/mistral-7b-instruct:free"  # większy, bardziej rozbudowany
        ], index=0
    )

# System prompt template
template = (
    "You are an assistant for question-answering tasks. Use Polish language by default.\n"
    "Be concise: maximum three sentences. If you don't know the answer, say that you don't know.\n\n"
    "Question: {question}\nContext: {context}\nAnswer:"  
)

# Initialize ChatOpenRouter; will be re-created if model changes
def get_chat_model(model_name: str):
    return ChatOpenRouter(
        openai_api_key=st.secrets.get("API_KEY", ""),
        model_name=model_name,
        temperature=0.0
    )

# Helper function: answer question via RAG
def answer_question(question, docs, model):
    context = "\n\n".join([doc.get('text', '') for doc in docs])
    system_content = template.format(question=question, context=context)
    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=question)
    ]
    response = model(messages)
    if hasattr(response, 'content') and response.content:
        return response.content
    if hasattr(response, 'choices') and response.choices:
        return response.choices[0].message.content
    return str(response)

# Load and index documents
if uploaded_files:
    for fname in os.listdir(UPLOAD_FOLDER):
        path = os.path.join(UPLOAD_FOLDER, fname)
        if os.path.isfile(path): os.remove(path)
    for uploaded in uploaded_files:
        dst = os.path.join(UPLOAD_FOLDER, uploaded.name)
        with open(dst, "wb") as f:
            f.write(uploaded.getbuffer())
    documents = load_documents_from_folder(UPLOAD_FOLDER)
    st.sidebar.success(f"Loaded {len(documents)} documents.")
    if "faiss_index" not in st.session_state or st.session_state.get("model_name") != selected_model:
        st.session_state.faiss_index = create_index(documents)
        st.session_state.model_name = selected_model
        st.session_state.chat_model = get_chat_model(selected_model)
else:
    st.sidebar.info("Upload PDF documents to begin.")

# Chat interface
if "faiss_index" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # display history
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])

    if user_input := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        docs = retrieve_docs(user_input, st.session_state.faiss_index, k=3)
        chat_model = st.session_state.get("chat_model") or get_chat_model(selected_model)
        answer = answer_question(user_input, docs, chat_model)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

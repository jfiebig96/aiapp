import os
import streamlit as st
from docloader import load_documents_from_folder
from embedder import create_index, retrieve_docs
from chat_openrouter import ChatOpenRouter
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Page configuration
st.set_page_config(layout="wide", page_title="RAG Chatbot")
st.title("Retrieval-Augmented Chatbot")

# Sidebar: upload PDFs
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        label="Upload PDF documents", type="pdf", accept_multiple_files=True
    )

# Default settings
MODEL_NAME = "google/gemma-3-1b-it:free"
TOP_K = 3

# Load and index documents
if uploaded_files:
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    # Clear previous uploads
    for fname in os.listdir(upload_dir):
        path = os.path.join(upload_dir, fname)
        if os.path.isfile(path):
            os.remove(path)
    # Save uploads
    for uploaded in uploaded_files:
        dst = os.path.join(upload_dir, uploaded.name)
        with open(dst, "wb") as f:
            f.write(uploaded.getbuffer())
    documents = load_documents_from_folder(upload_dir)
    st.sidebar.success(f"Loaded {len(documents)} documents.")
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = create_index(documents)
else:
    st.sidebar.info("Upload PDF documents to begin.")

# Chat interface
if "faiss_index" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # User query
    if user_input := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Retrieve docs
        docs = retrieve_docs(user_input, st.session_state.faiss_index, k=TOP_K)
        context = "\n\n".join(
            [f"### {d['filename']}\n{d['text'][:1000]}..." for d in docs]
        )

        # Prepare LLM messages
        llm_messages = [
            SystemMessage(content=(
                "You are a helpful assistant. Use the following document snippets to answer the question.\n\n" + context
            ))
        ]
        for m in st.session_state.messages:
            if m["role"] == "user":
                llm_messages.append(HumanMessage(content=m["content"]))
            else:
                llm_messages.append(AIMessage(content=m["content"]))
        llm_messages.append(HumanMessage(content=user_input))

        # Chat with model
        chat = ChatOpenRouter(
            openai_api_key=st.secrets["API_KEY"],
            model_name=MODEL_NAME,
            temperature=0.0
        )
        try:
            response = chat(llm_messages)
            answer = getattr(response, 'content', None) or response.choices[0].message.content
        except Exception as e:
            st.error(f"Error during completion: {e}")
            answer = ""

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

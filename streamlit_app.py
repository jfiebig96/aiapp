import os
import streamlit as st
from docloader import load_documents_from_folder
from embedder import create_index, retrieve_docs
from chat_openrouter import ChatOpenRouter
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Page configuration
st.set_page_config(layout="wide", page_title="RAG Chatbot")
st.title("Retrieval-Augmented Chatbot")

# Sidebar: configuration
with st.sidebar:
    st.header("Configuration")
    uploaded_files = st.file_uploader(
        label="Upload PDF documents", type="pdf", accept_multiple_files=True
    )
    selected_model = st.selectbox(
        label="Select model",
        options=["google/gemma-3-1b-it:free"],
        index=0,
    )
    top_k = st.slider(
        label="Documents to retrieve (k)", min_value=1, max_value=10, value=3
    )

# Load and index documents
if uploaded_files:
    # Ensure uploads directory exists
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    # Clear previous uploads
    for fname in os.listdir(upload_dir):
        path = os.path.join(upload_dir, fname)
        if os.path.isfile(path):
            os.remove(path)
    # Save uploaded files locally
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    # Load documents
    documents = load_documents_from_folder(upload_dir)
    st.sidebar.success(f"Loaded {len(documents)} documents.")
    # Create FAISS index if not already in session state
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = create_index(documents)
else:
    st.sidebar.info("Upload PDF documents to begin indexing.")

# Chat interface
if "faiss_index" in st.session_state:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # User input
    if user_input := st.chat_input("Ask a question..."):
        # Append user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # Retrieve relevant documents
        docs = retrieve_docs(user_input, st.session_state.faiss_index, k=top_k)
        # Build context string
        context = "\n\n".join(
            [f"### {d['filename']}\n{d['text'][:1000]}..." for d in docs]
        )

        # Prepare messages for LLM
        llm_messages = []
        # System prompt with context
        llm_messages.append(
            SystemMessage(
                content=(
                    "You are a helpful assistant. Use the following extracted document snippets "
                    f"to answer the user question.\n\n{context}"
                )
            )
        )
        # Add past conversation
        for m in st.session_state.messages:
            if m["role"] == "user":
                llm_messages.append(HumanMessage(content=m["content"]))
            else:
                llm_messages.append(AIMessage(content=m["content"]))
        # Add current user query
        llm_messages.append(HumanMessage(content=user_input))

        # Initialize ChatOpenRouter
        chat = ChatOpenRouter(
            openai_api_key=st.secrets["API_KEY"],
            model_name=selected_model,
            temperature=0.0
        )

        # Generate response
        try:
            response = chat(llm_messages)
            # Extract text
            if hasattr(response, "content"):
                answer = response.content
            elif hasattr(response, "choices"):
                answer = response.choices[0].message.content
            else:
                answer = str(response)
        except Exception as e:
            st.error(f"Error during completion: {e}")
            answer = ""

        # Append assistant response
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

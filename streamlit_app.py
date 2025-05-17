import os
import streamlit as st
from docloader import load_documents_from_folder
from embedder import create_index, retrieve_docs
from chat_openrouter import ChatOpenRouter
from langchain.prompts import ChatPromptTemplate

# Page configuration
st.set_page_config(layout="wide", page_title="RAG Chatbot")
st.title("Retrieval-Augmented Chatbot")

# Upload folder setup
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Sidebar: upload PDFs
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        label="Upload PDF documents", type="pdf", accept_multiple_files=True
    )

# System prompt template
template = """
You are an assistant for question-answering tasks. Use Polish language by default.
- Be concise: maximum three sentences.
- If you don't know the answer, say that you don't know.

Question: {question}
Context: {context}
Answer:
"""

# Initialize ChatOpenRouter model
MODEL_NAME = "mistralai/mistral-7b-instruct:free"
chat_model = ChatOpenRouter(
    openai_api_key=st.secrets["API_KEY"],
    model_name=MODEL_NAME,
    temperature=0.0
)

# Helper function to answer questions via RAG
def answer_question(question, docs):
    # Combine document texts
    context = "\n\n".join([doc['text'] for doc in docs])
    # Build chat prompt chain
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | chat_model
    # Invoke with inputs
    result = chain.invoke({"question": question, "context": context})
    # Extract content
    return getattr(result, 'content', None) or getattr(result, 'text', None) or str(result)

# Load and index documents
if uploaded_files:
    # clear previous
    for fname in os.listdir(UPLOAD_FOLDER):
        path = os.path.join(UPLOAD_FOLDER, fname)
        if os.path.isfile(path): os.remove(path)
    # save new uploads
    for uploaded in uploaded_files:
        dst = os.path.join(UPLOAD_FOLDER, uploaded.name)
        with open(dst, "wb") as f:
            f.write(uploaded.getbuffer())
    # load and index
    documents = load_documents_from_folder(UPLOAD_FOLDER)
    st.sidebar.success(f"Loaded {len(documents)} documents.")
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = create_index(documents)
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
        # add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # retrieve docs
        docs = retrieve_docs(user_input, st.session_state.faiss_index, k=3)
        # get answer
        answer = answer_question(user_input, docs)

        # append and display assistant
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)

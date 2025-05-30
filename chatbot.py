import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
import os

# Jangan load_dotenv() di Streamlit Cloud karena environment variable sudah otomatis dari Secrets

# Ambil API key dan base URL dari environment variable langsung
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter_api_base = "https://openrouter.ai/api/v1"

CHROMA_PATH = "chroma_db"

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatOpenAI(
    model="deepseek/deepseek-chat:free",
    openai_api_key=openrouter_api_key,
    openai_api_base=openrouter_api_base,
    temperature=0.5,
    streaming=True,
)

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

st.title("💬 Helpdesk DTE UI")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def generate_response(user_input):
    docs = retriever.get_relevant_documents(user_input)  # pakai get_relevant_documents
    knowledge = "\n\n".join(doc.page_content for doc in docs)

    rag_prompt = f"""
    You are the official virtual academic assistant for the Department of Electrical Engineering, Universitas Indonesia.

    Your role is to provide accurate and helpful information regarding all academic and administrative aspects of the department, including the following study programs:
    - Electrical Engineering
    - Computer Engineering
    - Biomedical Engineering

    You are allowed to draw knowledge from pre-provided PDF documents, as well as official websites or other relevant online academic sources. However, **you must only respond based on information related to the Department of Electrical Engineering at Universitas Indonesia**.

    Do not answer any questions outside this scope. All responses must be delivered in formal, clear, and contextually appropriate Indonesian.

    The question: {user_input}

    The knowledge: {knowledge}
    """

    response = ""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        for chunk in llm.stream(rag_prompt):
            response += chunk.content
            message_placeholder.markdown(response + "▌")
        message_placeholder.markdown(response)

    st.session_state.chat_history.append({"role": "assistant", "content": response})

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

user_input = st.chat_input("Ketik pertanyaanmu di sini...")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    generate_response(user_input)

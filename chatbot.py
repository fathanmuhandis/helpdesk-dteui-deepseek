# streamlit_app.py (Versi Final: Chroma + LangChain + Streamlit Cloud Friendly)
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

# Load environment variables (pastikan OPENROUTER_API_KEY diset di Secrets)
load_dotenv()
os.environ["OPENAI_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Embedding model dari HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load vector store dari Chroma (pastikan folder chroma_db sudah dibuat dari ingest_database.py)
CHROMA_PATH = "chroma_db"
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embedding_model,
    persist_directory=CHROMA_PATH
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# LLM via OpenRouter (contoh: DeepSeek)
llm = ChatOpenAI(
    model="deepseek/deepseek-chat:free",
    openai_api_base=os.environ["OPENAI_API_BASE"],
    temperature=0.5,
    streaming=True
)

# Judul UI
st.title("💬 Helpdesk DTE UI — DeepSeek")

# Inisialisasi histori chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Fungsi untuk menjawab
def generate_response(user_input):
    docs = retriever.invoke(user_input)
    knowledge = "\n\n".join(doc.page_content for doc in docs)

    rag_prompt = f"""
    You are the official virtual academic assistant for the Department of Electrical Engineering, Universitas Indonesia.
    You must provide accurate and clear answers in Bahasa Indonesia, based strictly on the knowledge below.

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

# Tampilkan chat sebelumnya
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Input baru dari user
user_input = st.chat_input("Ketik pertanyaanmu di sini...")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    generate_response(user_input)

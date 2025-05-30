import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

# Load environment variable
load_dotenv()

# Set API key and base URL for OpenRouter
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Konfigurasi ChromaDB
CHROMA_PATH = "chroma_db"

# Model Embedding dari Hugging Face
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Model Chat LLM (DeepSeek via OpenRouter)
llm = ChatOpenAI(
    model="deepseek/deepseek-chat:free",
    openai_api_base=os.environ["OPENAI_API_BASE"],
    temperature=0.5,
    streaming=True,
)

# Vector Store untuk pencarian
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Judul UI
st.title("💬 Helpdesk DTE UI — DeepSeek")

# Inisialisasi session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Fungsi untuk membangkitkan respons
def generate_response(user_input):
    docs = retriever.invoke(user_input)
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

    # Stream respons
    response = ""
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        for chunk in llm.stream(rag_prompt):
            response += chunk.content
            message_placeholder.markdown(response + "▌")
        message_placeholder.markdown(response)

    # Simpan ke histori
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Tampilkan histori percakapan
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Input dari pengguna
user_input = st.chat_input("Ketik pertanyaanmu di sini...")
if user_input:
    # Tampilkan input pengguna sebagai bubble
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    generate_response(user_input)
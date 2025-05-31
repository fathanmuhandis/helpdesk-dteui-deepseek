import os
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Load FAISS index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Load LLM via OpenRouter (DeepSeek)
llm = ChatOpenAI(
    model="deepseek/deepseek-chat:free",
    openai_api_base=os.environ["OPENAI_API_BASE"],
    temperature=0.5,
    streaming=True
)

st.title("💬 Helpdesk DTE UI — DeepSeek")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def generate_response(user_input):
    docs = retriever.invoke(user_input)
    knowledge = "\n\n".join(doc.page_content for doc in docs)

    rag_prompt = f"""
    You are the official virtual academic assistant for the Department of Electrical Engineering, Universitas Indonesia.
    Your role is to provide accurate and helpful information related to academic and administrative matters of the department.
    Answer clearly and formally in Bahasa Indonesia based only on the information below.

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

# Show chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

user_input = st.chat_input("Ketik pertanyaanmu di sini...")
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    generate_response(user_input)

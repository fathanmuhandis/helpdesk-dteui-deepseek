from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from uuid import uuid4
from dotenv import load_dotenv
import os

load_dotenv()
DATA_PATH = "data"

loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", ".", " "]
)

chunks = text_splitter.split_documents(raw_documents)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embedding_model)

vector_store.save_local("faiss_index")
print("✅ FAISS index saved to 'faiss_index/'")
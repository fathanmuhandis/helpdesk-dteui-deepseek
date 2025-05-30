from langchain_community.document_loaders import PyPDFDirectoryLoader # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings # type: ignore
from langchain_chroma import Chroma # type: ignore
from dotenv import load_dotenv # type: ignore
from uuid import uuid4

# Load environment variables
load_dotenv()

# Path configurations
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

# Use HuggingFace embedding model (no API needed)
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# Load documents
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

# Add to vector store
uuids = [str(uuid4()) for _ in range(len(chunks))]
vector_store.add_documents(documents=chunks, ids=uuids)

print(f"✅ {len(chunks)} dokumen berhasil diproses ke ChromaDB.")
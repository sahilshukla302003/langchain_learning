import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(current_dir, "books", "bill.pdf")
persistent_directory = os.path.join(current_dir, "db", "chroma_db_metadata")

# Load PDF
loader = PyMuPDFLoader(pdf_path)
docs = loader.load()

# Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

# Embeddings
embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# Store in Chroma (auto-persistent)
db = Chroma.from_documents(
    splits,
    embedding=embeddings,
    persist_directory=persistent_directory
)

print("✅ PDF successfully ingested into Chroma DB.")

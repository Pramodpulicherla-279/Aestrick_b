from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma  # updated import
from langchain_huggingface import HuggingFaceEmbeddings  # updated import

# Load textbook PDF
loader = PyPDFLoader("ncert_gs_I-XII_chapter1.pdf")
pages = loader.load()

# Split text into chunks
splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
documents = splitter.split_documents(pages)

# Setup ChromaDB
persist_dir = "chromadb_store"
embedding_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # updated class

db = Chroma(persist_directory=persist_dir, embedding_function=embedding_func)
db.add_documents(documents)

# print("Textbook stored in ChromaDB!")
print(f"Loaded {len(pages)} pages from PDF.")
print(f"Split into {len(documents)} document chunks.")

# Check for empty documents
empty_docs = [doc for doc in documents if not getattr(doc, 'page_content', '').strip()]
print(f"Number of empty document chunks: {len(empty_docs)}")
if len(documents) == 0 or len(empty_docs) == len(documents):
    print("ERROR: No valid document chunks to embed. Check your PDF and splitting logic.")
    exit(1)

retriever = db.as_retriever()
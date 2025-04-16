# Importing the essential libraries
from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

#1. Load Raw PDF(s)
DATA_PATH = "data/"

def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf",loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(DATA_PATH)
#print(f"Loaded {len(documents)} documents")


#2. Create Chunks

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=200)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunkd = create_chunks(documents)
print(f"Created {len(text_chunkd)} chunks")


#3. Create Vector Embeddings

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedding_model 


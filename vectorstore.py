import hashlib
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader



def load_all_pdfs(pdf_folder: str):
    documents = []
    for pdf_path in Path(pdf_folder).glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())
    return documents



#create fingerprint of pdfs based on filename and last modified time
def get_pdf_fingerprint(pdf_folder="pdfs"):
    pdf_files = sorted(Path(pdf_folder).glob("*.pdf"))

    fingerprint_data=[]
    for pdf in pdf_files:
        stat = pdf.stat()

        fingerprint_data.append({
            "name":pdf.name, 
            "mtime":stat.st_mtime,
            "size": stat.st_size
        })

    fingerprint_json = json.dumps(fingerprint_data, sort_keys=True)
    return hashlib.sha256(fingerprint_json.encode()).hexdigest()

#constants
INDEX_DIR = ".cache/faiss"
FINGERPRINT_FILE = ".cache/faiss/fingerprint.txt"

#to load(if pdfs unchanged) or to rebuild(if pdfs fingerprints changed) the vectorstore

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

def get_vectorstore():
    current_fp = get_pdf_fingerprint()

    if(Path(INDEX_DIR).exists() and Path(FINGERPRINT_FILE).exists()):
        saved_fp = Path(FINGERPRINT_FILE).read_text()

        if saved_fp == current_fp:
            print("Loading existing FAISS files...")

            return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        
    print("Building new FAISS index...")
    print("Loading pdfs...")
    documents = load_all_pdfs("pdfs")
    print(f"Loaded {len(documents)} pages")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 100,
    )

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    print("Creating vector database...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Vector database Ready!")
    vectorstore.save_local(INDEX_DIR)
    Path(FINGERPRINT_FILE).write_text(current_fp)

    return vectorstore
                  

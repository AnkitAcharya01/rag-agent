import hashlib
import json
# from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os, time

INDEX_NAME = "rag-index1"



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
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )



#updated for pinecone


import os
import time
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter



def get_vectorstore():
    current_fp = get_pdf_fingerprint()
    embeddings = get_embeddings()

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    # -----------------------------
    # 1. CREATE INDEX IF NOT EXISTS
    # -----------------------------
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    index = pc.Index(INDEX_NAME)

    # wait until ready (important)
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

    # -----------------------------
    # 2. CHECK IF INDEX IS EMPTY
    # -----------------------------
    stats = index.describe_index_stats()
    empty_index = stats.get("total_vector_count", 0) == 0

    # -----------------------------
    # 3. FINGERPRINT LOGIC
    # -----------------------------
    rebuild = empty_index

    if Path(FINGERPRINT_FILE).exists():
        saved_fp = Path(FINGERPRINT_FILE).read_text()
        if saved_fp != current_fp:
            rebuild = True

    # -----------------------------
    # 4. CONNECT VECTORSTORE ALWAYS
    # -----------------------------
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=""
    )

    # -----------------------------
    # 5. RETURN IF NO REBUILD NEEDED
    # -----------------------------
    if not rebuild:
        print("Using existing Pinecone index (already populated).")
        return vectorstore

    # -----------------------------
    # 6. REBUILD INDEX
    # -----------------------------
    print("Rebuilding Pinecone index...")

    documents = load_all_pdfs("pdfs")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=140
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunks found from PDFs!")

    print(f"Number of chunks: {len(chunks)}")

    # IMPORTANT: clear old vectors
    if not empty_index:
        try:
            index.delete(delete_all=True, namespace="")
        except Exception as e:
            # Pinecone 404s if the namespace doesn't exist yet — safe to ignore
            print(f"Skip delete (namespace empty/not found): {e}")

    # -----------------------------
    # 7. INGEST DOCUMENTS
    # -----------------------------
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        pinecone_api_key=os.environ["PINECONE_API_KEY"],
        namespace=""
    )

    # save fingerprint only AFTER success
    Path(FINGERPRINT_FILE).write_text(current_fp)

    print("Rebuild complete.")

    return vectorstore



                  

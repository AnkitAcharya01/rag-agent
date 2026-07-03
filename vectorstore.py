import hashlib
import json
import os
import time
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

INDEX_NAME = "rag-index1"
INDEX_DIR = Path(".cache/faiss")
STATE_FILE = INDEX_DIR / "index_state.json"


def load_all_pdfs(pdf_folder: str):
    documents = []
    for pdf_path in sorted(Path(pdf_folder).glob("*.pdf")):
        loader = PyPDFLoader(str(pdf_path))
        loaded_docs = loader.load()
        for doc in loaded_docs:
            doc.metadata["source"] = pdf_path.name
            doc.metadata["file_name"] = pdf_path.name
            doc.metadata["file_path"] = str(pdf_path)
        documents.extend(loaded_docs)
    return documents


def get_pdf_fingerprints(pdf_folder: str = "pdfs"):
    fingerprints = {}
    for pdf in sorted(Path(pdf_folder).glob("*.pdf")):
        stat = pdf.stat()
        payload = {
            "name": pdf.name,
            "mtime": stat.st_mtime,
            "size": stat.st_size,
        }
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        fingerprints[pdf.name] = fingerprint
    return fingerprints


def load_index_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            return {"processed_files": {}}
    return {"processed_files": {}}


def save_index_state(state):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_vectorstore():
    embeddings = get_embeddings()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(INDEX_NAME)
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings, namespace="")
    current_fingerprints = get_pdf_fingerprints()
    state = load_index_state()
    processed_files = state.get("processed_files", {})

    files_to_delete = [name for name in processed_files if name not in current_fingerprints]
    files_to_ingest = []
    for name, fingerprint in current_fingerprints.items():
        if processed_files.get(name) != fingerprint:
            files_to_ingest.append(name)

    if not current_fingerprints:
        if files_to_delete:
            for filename in files_to_delete:
                try:
                    index.delete(filter={"file_name": {"$eq": filename}}, namespace="")
                except Exception as exc:
                    print(f"Delete skipped for {filename}: {exc}")
        save_index_state({"processed_files": {}})
        return vectorstore

    if files_to_delete:
        for filename in files_to_delete:
            try:
                index.delete(filter={"file_name": {"$eq": filename}}, namespace="")
            except Exception as exc:
                print(f"Delete skipped for {filename}: {exc}")

    if not files_to_ingest:
        print("No PDF changes detected; using existing Pinecone index.")
        return vectorstore

    print(f"Indexing {len(files_to_ingest)} new or changed PDF(s)...")

    docs_to_ingest = load_all_pdfs("pdfs")
    if not docs_to_ingest:
        save_index_state({"processed_files": {}})
        return vectorstore

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=140)
    chunks = splitter.split_documents(docs_to_ingest)
    if not chunks:
        raise ValueError("No chunks found from PDFs!")

    for filename in files_to_ingest:
        try:
            index.delete(filter={"file_name": {"$eq": filename}}, namespace="")
        except Exception as exc:
            print(f"Refresh skipped for {filename}: {exc}")

    vectorstore.add_documents(chunks, namespace="")

    updated_state = {
        "processed_files": {
            name: current_fingerprints[name]
            for name in current_fingerprints
            if name not in files_to_delete
        }
    }
    save_index_state(updated_state)

    print("Incremental index update complete.")
    return vectorstore

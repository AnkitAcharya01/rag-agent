import streamlit as st

from pathlib import Path
from dotenv import load_dotenv   
import os
from smolagents import Tool, CodeAgent, LiteLLMModel, InferenceClientModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_all_pdfs(pdf_folder: str):
    documents = []
    for pdf_path in Path(pdf_folder).glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())
    return documents


#build cached vectorDB
@st.cache_resource
def build_vectorstore():
    print("Loading pdfs...")
    documents = load_all_pdfs("pdfs")
    print(f"Loaded {len(documents)} pages")
            
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap = 80
    )

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    #embedding
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Creating vector database...")
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings)
    print("Vector database ready")
    return vectorstore


#tool
class PDFRetrieverTool(Tool):
    name="pdf_retriever"

    description="""
    Searches uploaded company PDFs and retrieves the most relevant information for answering
    customer questions.
    """
    inputs={
        "query":{
            "type":"string",
            "description":"Question to search in the PDFs."
        }
    }
    output_type="string"

    def __init__(self, vectorstore):
        super().__init__()
        self.retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )
    
    def forward(self, query:str)->str:
        docs = self.retriever.invoke(query)

        if not docs:
            return "No relevant information found."
        
        results=[]
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            results.append(f"""
            DOCUMENT {i}
            SOURCE: {source}
            {doc.page_content}
            """)
        return "\n\n".join(results)
    
#agent creation
# @st.cache_resource
def build_agent():
    vectorstore=build_vectorstore()
    pdf_tool = PDFRetrieverTool(vectorstore)
  

    #agent creation
    agent=CodeAgent(tools=[pdf_tool], model=InferenceClientModel()) 

    return agent


    
#new login based UI
from huggingface_hub import login
st.title("RAG-based Assistant")
if("hf_logged_in" not in st.session_state):
    st.session_state["hf_logged_in"]=False

if not st.session_state["hf_logged_in"]:
    hf_token = st.text_input(
        "Hugging Face Access Token",
        type="password",
        help="Enter your Hugging Face Access Token (hf_...)"
    )

    if hf_token and not hf_token.startswith("hf_"):
        st.warning("This doesnt look like a valid Hf Access Token!")
        st.stop()

    if st.button("login"):
        if not hf_token:
            st.error("Please Enter a Hugging Face Token!")
            st.stop()
        try:
            login(token=hf_token)
            st.session_state["hf_logged_in"] = True
            st.success("Successfully logged in!")
            st.rerun()
        except Exception as e:
            st.error(f"Login failed: {e}")
            st.stop()
else:
    agent = build_agent()
    
    query = st.text_input("Hey, Can I assist you with your PDFs' information?")

    if st.button("Search") and query:
        try:
            with st.spinner("Thinking..."):
                response = agent.run(query)
            st.write("### Answer")
            st.write(response)
        except Exception as e:
            st.error(f"Error occured: {e}")

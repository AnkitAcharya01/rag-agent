import streamlit as st

from pathlib import Path
from dotenv import load_dotenv   
import os
from smolagents import Tool, CodeAgent, InferenceClientModel

from vectorstore import get_vectorstore


#build cached vectorDB
@st.cache_resource
def build_vectorstore():
    return get_vectorstore()


#tool
class PDFRetrieverTool(Tool):
    name="pdf_retriever"

    description="""
    Retrieves relevant information from company PDFs.

    Use this tool to gather evidence from documents.
    After retrieving information, analyze the results and provide a concise answer to the user.
    Do not simply repeat the retrieved documents unless specifically requested.
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
            search_kwargs={"k": 3}
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
@st.cache_resource
def build_agent():
    vectorstore=build_vectorstore()
    pdf_tool = PDFRetrieverTool(vectorstore)
  

    #agent creation
    agent=CodeAgent(tools=[pdf_tool], model=InferenceClientModel()) 

    return agent


    
#new login based UI
#new login logic flow: check .env for token, if available:try login, 
# else try get_token() from hf cache dir
# else, require login manually by pasting hf token to UI
from huggingface_hub import login, get_token
st.title("RAG-based Assistant")
if("hf_logged_in" not in st.session_state):
    env_token = os.getenv("HF_TOKEN")

    if env_token:
        try:
            login(token=env_token)
            print("for debug: logged in through env")
            st.session_state["hf_logged_in"]=True
        except Exception as e:
            st.session_state["hf_logged_in"]=False
            print("env login error!")
            st.session_state["hf_login_error"]=str(e)
    
    #fallback to disk
    elif get_token():
        print("for debug: logged in through disk cache dir")
        st.session_state["hf_logged_in"]=True
    else:
        st.session_state["hf_logged_in"]=False

if not st.session_state["hf_logged_in"]:
    if st.session_state.get("hf_login_error"):
        print("for debug: auto login failed")
        st.error(f"Auto login from .env failed: {st.session_state['hf_login_error']}")
    hf_token = st.text_input(
        "Hugging Face Access Token",
        type="password",
        help="Paste your Hugging Face Access Token (hf_...)"
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
    
    if "messages" not in st.session_state:
        st.session_state.messages=[]

    #display existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    query = st.chat_input("Ask a question about your PDFs...")
    if query:
        st.session_state.messages.append({
            "role": "user", "content": query
        })
        with st.chat_message("user"):
            st.markdown(query)

        try:
            with st.spinner("Thinking..."):

                history = "\n".join(
                    f"{m['role']}: {m['content']}" for m in st.session_state.messages[-10:]
                )
                prompt = f"""
                        You are a helpful PDF RAG Assistant.
                        Conversation history: {history}
                        Answer the latest question using the pdf retriever tool.

                        Latest question: {query}
                        Give only relevant and concise answer, not the entire docs.
                    
                        """

                response = agent.run(prompt)
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({
                "role":"assistant", "content": response
            })
            
        except Exception as e:
            st.error(f"Error occured: {e}")
    
import streamlit as st

from pathlib import Path
from dotenv import load_dotenv  
import os
from smolagents import Tool, ToolCallingAgent, OpenAIServerModel
from huggingface_hub import InferenceClient
import time
from openai import OpenAI
from io import BytesIO

from vectorstore import get_vectorstore
from tt_speech import tts, autoplay_audio

load_dotenv() 

def get_tts_client():
    return InferenceClient()

#build cached vectorDB
@st.cache_resource
def build_vectorstore():
    return get_vectorstore()


#tool
class PDFRetrieverTool(Tool):
    name="pdf_retriever"

    description="""
    Use this tool to search PDFs and return relevant excerpts.
    After retrieving information, analyze the results and provide a concise answer to the user.
    Do not simply repeat the retrieved documents unless specifically requested.

    IMPORTANT:
    - This is raw evidence, not the final answer
    - Keep results short and relevant
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
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10}
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
  

    groq_key = st.session_state.get("groq_token")

    if not groq_key:
        st.error("Groq API key missing")
        st.stop()

    model = OpenAIServerModel(
        model_id="llama-3.3-70b-versatile",
        api_base="https://api.groq.com/openai/v1",
        api_key=groq_key,
        temperature=0,
        
    )

    agent = ToolCallingAgent(
        tools=[pdf_tool],
        model=model,
        max_steps=4
    )
    return agent

transcription_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

    

# login logic flow: check .env for token, if available:try login, 
# else, require login manually by pasting api key from UI


st.title("RAG-based Assistant")
if("groq_logged_in" not in st.session_state):
    env_token = os.getenv("GROQ_API_KEY")

    if env_token:
        
        st.session_state["groq_token"] = env_token
        st.session_state["groq_logged_in"]=True
        print("for debug: logged in through env")
    else:
        st.session_state["groq_logged_in"]=False
        print("env login error!")
        
    

if not st.session_state["groq_logged_in"]:
    with st.sidebar:        
        st.link_button("Get Free Groq API Key", "https://console.groq.com/keys")
    groq_token = st.text_input(
        "Groq API Key",
        type="password",
        help="Paste your Groq API Key (gsk_...)"
    )

    #login button
    if st.button("login"):
        if not groq_token:
            st.error("Please Enter a Groq API Key!")
            st.stop()
        if groq_token and not groq_token.startswith("gsk_"):
            st.warning("This doesn't look like a valid Groq API Key!")
            st.stop()
        login_error = False
        with st.spinner("Verifying..."):
            try:
                login_test = OpenAI(api_key=groq_token, base_url="https://api.groq.com/openai/v1")
                login_test.models.list()
            except Exception as e:                            
                login_error = True
        if login_error:        
            st.error("Invalid Groq API Key, Login Failed!")
        else:
            st.session_state["groq_token"] = groq_token
            st.session_state["groq_logged_in"] = True
            os.environ["GROQ_API_KEY"]=groq_token
            st.success("Successfully logged in!")
            st.rerun()
        
else:
    #pdf management sidebar
    PDF_DIR=Path("pdfs")
    PDF_DIR.mkdir(exist_ok=True)

    #logout button
    with st.sidebar:
        if st.button("Use a different Groq Key"):
            
            st.session_state["groq_logged_in"]=False
            st.session_state.pop("groq_token", None)
            build_agent.clear()
            st.rerun()
            
        st.header("PDF Management")

        if "uploader_key" not in st.session_state:
            st.session_state["uploader_key"] = 0


        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=['pdf'],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state['uploader_key']}"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                save_path = PDF_DIR / uploaded_file.name

                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            #clear cache 
            build_vectorstore.clear()
            build_agent.clear()
            st.session_state["uploader_key"]+=1
            st.success("PDFs uploaded.")
            st.rerun()
        
        st.divider()
        #shows the list of pdfs 
        st.subheader("Current PDFs")
        pdfs = sorted(PDF_DIR.glob("*.pdf"))
        if len(pdfs)==0:
            st.info("Please upload one or more PDFs")
            st.stop()
        for pdf in pdfs:
            col1, col2 = st.columns([4,1])

            with col1:
                st.text(pdf.name)
            with col2:
                if st.button("⛔", key=f"delete_{pdf.name}"):
                    if pdf.exists():
                        pdf.unlink()

                    build_vectorstore.clear()
                    build_agent.clear()

                    st.success("Deleted") 
                    st.rerun()
                    #also clear agent cache if pdf deleted

                
    with st.spinner("Loading Knowledge base..."):
        agent = build_agent()
    
    if "messages" not in st.session_state:
        st.session_state.messages=[{"role": "assistant", 
                                    "content": "How can I help you?"}]

    #display existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    

    

    #text or audio input
    prompt = st.chat_input("Ask a question about your PDFs...", 
                          accept_audio=True,
                          audio_sample_rate=16000)

    if prompt:
        query=None
        
        #voice input
        if prompt.audio:
            with st.spinner("Analysing your voice..."):
                audio_bytes = prompt.audio.getvalue()
                audio_file = BytesIO(audio_bytes)
                audio_file.name = "audio.wav"

                transcript = transcription_client.audio.transcriptions.create(
                    model = "whisper-large-v3-turbo",
                    file = audio_file
                )
                
                query = transcript.text

      
            print("type of audio:")
            print(type(prompt.audio))

        elif prompt.text:
            query=prompt.text
    
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
                            You are a helpful PDF RAG assistant.
                            Use the conversation history only to resolve references such as "it", "they", or "that".

                            Use the retrieved documents as the ONLY factual source.

                            If the retrieved documents answer the user's latest question after resolving references, answer using them.
                            Otherwise reply: "I have no information about that."
                            Conversation history: {history}
                            Latest question: {query}

                            DONOT Make up the answer yourself.
                            
                            Respond in a single sentence.
                            Give only relevant answer, with engaging style, donot paste the entire docs.
                        
                            """

                                        
                    try:
                        response = agent.run(prompt)
                    except Exception as first_err:
                        if "tool_use_failed" in str(first_err):
                            response = agent.run(prompt)   # for transient malformed tool_call, retry once
                        else:
                            raise

                with st.chat_message("assistant"):
                    st.markdown(response)
                    with st.spinner("Generating audio..."):
                        try:
                            time1 = time.time()
                            audio_bytes = tts(response)
                            time2 = time.time()
                            autoplay_audio(audio_bytes)
                            time3 = time.time()
                            print(f"TTs took: {time2-time1}")
                            print(f"Auto play took: {time3-time2}")
                            print(f"Total: {time3-time1}")                            
                        except Exception as e:
                            print(f"Error occured: {e}")


                
                st.session_state.messages.append({
                    "role":"assistant", "content": response
                
                })
                
            except Exception as e:
                st.error(f"Error occured: {e}")
    
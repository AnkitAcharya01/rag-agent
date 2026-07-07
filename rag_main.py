from pathlib import Path
import streamlit as st
from smolagents import Tool, ToolCallingAgent, OpenAIServerModel
from vectorstore import get_vectorstore
import os

from log_database import Log_unanswered_query

# FALLBACK_TEXT = "I am unable to find the answer in the provided documents. I have logged your query for further review."
FALLBACK_TEXT = "No info found, query logged for future improvement"

Path("logs").mkdir(exist_ok=True)

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
    
groq_key = os.environ.get("GROQ_API_KEY")
def build_agent(groq_key: str):
    print("from here")
    log_tool = Log_unanswered_query()
    print("and here")
    print(Log_unanswered_query.inputs)
    print("to here")
    vectorstore=get_vectorstore()
    pdf_tool = PDFRetrieverTool(vectorstore)
    log_tool = Log_unanswered_query()

    model = OpenAIServerModel(
        model_id="llama-3.3-70b-versatile",
        api_base="https://api.groq.com/openai/v1",
        api_key=groq_key,
        temperature=0,
        
    )

    agent = ToolCallingAgent(
        tools=[pdf_tool, log_tool],
        model=model,
        max_steps=4
    )
    return agent


def run_rag(agent, query, history, session_id=None):
    """
    Run the RAG agent using conversation history.
    """

    history_text = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in history[-10:]
    )

    prompt = f"""
    You are a helpful PDF RAG assistant.
    Use the conversation history only to resolve references such as "it", "they", or "that".

    Use the retrieved documents as the ONLY factual source.

    If the retrieved documents answer the user's latest question after resolving references,
    answer using them.

    Otherwise:
    1. Call the Log_unanswered_query tool with the user's original query.
    2. Then reply exactly: "{FALLBACK_TEXT}"

    Conversation history:
    {history_text}

    Latest question:
    {query}

    DONOT Make up the answer yourself.

    Respond in a single sentence.
    Give only relevant answer, with engaging style,
    donot paste the entire docs.
    """

    try:
        result = agent.run(prompt)

    except Exception as first_err:
        if "tool_use_failed" in str(first_err):
            result = agent.run(prompt)
        else:
            raise  
    
    #SAFETY NET:
    if FALLBACK_TEXT in str(result) and not is_tool_called(agent, "Log_unanswered_query"):
        # log the unanswered query manually if the tool was not called
        log_tool = next(t for t in agent.tools.values() if t.name == "Log_unanswered_query") \
            if isinstance(agent.tools, dict) else next(t for t in agent.tools if t.name == "Log_unanswered_query")
        log_tool.forward(query=query, session_id=session_id)
    return result



def is_tool_called(agent, tool_name: str) -> bool:
    """
    Check the agent's run memory to see if it called a specific tool this run."""
    try:
        for step in agent.memory.steps:
            tool_calls = getattr(step, "tool_calls", [])
            for tool_call in tool_calls:
                if getattr(tool_call, "name", None) == tool_name:
                    return True
    except Exception as e:
        print(f"Error checking tool calls: {e}")
    return False


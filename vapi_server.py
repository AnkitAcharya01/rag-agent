import json
import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv

from rag_main import build_agent, run_rag

load_dotenv()
app = FastAPI()

#one agent instance for the entire server
_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent(os.environ["GROQ_API_KEY"])
    return _agent


def _build_assistant_message(answer: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": str(answer),
    }


def _extract_latest_user_query(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue

        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if isinstance(part.get("text"), str):
                        text_parts.append(part["text"])
                    elif isinstance(part.get("content"), str):
                        text_parts.append(part["content"])
                elif isinstance(part, str):
                    text_parts.append(part)
            if text_parts:
                return " ".join(text_parts)

    return ""


@app.post("/vapi/webhook")
async def vapi_webhook(request: Request):
    """
    Endpoint to handle incoming webhook requests.
    """
    body = await request.json()
    msg_type = body.get("message", {}).get("type")

    #currently this only acts on function_call and user transcript events sent by vapi
    if msg_type == 'function_call':
        fn = body["message"]["functionCall"]
        query = fn["parameters"].get("query", "")
        history = body["message"]["functionCall"].get("history", [])

        answer = run_rag(get_agent(), query, history)
        return JSONResponse({"result": _build_assistant_message(answer)})

    if msg_type == 'transcript' and body["message"].get("role") == "user":
        query = body["message"]["transcript"]
        history = body["message"].get("history", [])

        answer = run_rag(get_agent(), query, history)
        return JSONResponse({"message": _build_assistant_message(answer)})
    
    #acknowledge other events but ignore
    return JSONResponse({"status":"ok"})


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()

    messages = body.get("messages", [])
    query = _extract_latest_user_query(messages)
    history = messages[:-1] if len(messages) > 1 else []

    answer = run_rag(get_agent(), query, history)

    if body.get("stream"):
        async def event_stream():
            payload = {
                "id": "chatcmpl-rag",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "custom-rag",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": answer},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return {
        "id": "chatcmpl-rag",
        "object": "chat.completion",
        "created": 0,
        "model": "custom-rag",
        "choices": [
            {
                "index": 0,
                "message": _build_assistant_message(answer),
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }



@app.get("/health")
def health():
    """
    Health check endpoint to verify the server is running.
    """
    return JSONResponse({"status": "ok"})
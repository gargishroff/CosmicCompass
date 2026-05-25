import uvicorn
import json
import os
import sys
import io
from contextlib import redirect_stdout
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

# --- Project Root Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

try:
    # Import your supervisor agent
    from core.agent_self import supervisor_agent
except ImportError:
    print(f"Error: Could not import 'supervisor_agent' from 'core.agent_self'.")
    print(f"Make sure 'core/agent_self.py' exists and project root is correct.")
    print(f"Project Root (added to path): {project_root}")
    sys.exit(1)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Cosmic Compass API",
    description="API for the multi-agent Subject Matter Expert."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request Body ---
class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def get_chat_ui():
    """
    Serves the main chat interface.
    """
    try:
        with open(os.path.join(current_dir, "index.html"), "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Chat UI file not found.</h1>", status_code=404)

async def stream_agent_response(messages: List[Dict[str, Any]]):
    """
    Asynchronous generator that streams the agent's response.
    It captures 'pretty_print()' and sends it to the frontend.
    """
    print("\n[FastAPI] New chat stream started...")
    
    # This history will be managed by the agent stream
    chat_history = messages
    
    try:
        # We pass the message history to the agent
        async for step in supervisor_agent.astream({"messages": chat_history}):
            
            for update in step.values():
                if update is None:
                    continue
                for message in update.get("messages", []):
                    
                    # Capture the stdout from pretty_print()
                    f = io.StringIO()
                    with redirect_stdout(f):
                        message.pretty_print()
                    log_string = f.getvalue()
                    
                    # Only send if the log has content
                    if log_string.strip():
                        # Wrap the raw log string in a simple JSON object
                        yield json.dumps({
                            "type": "log",
                            "data": log_string
                        }) + "\n"

    except Exception as e:
        print(f"[FastAPI] ERROR in stream: {e}")
        import traceback
        traceback.print_exc()
        error_message = f"An error occurred: {str(e)}"
        yield json.dumps({"type": "error", "data": error_message}) + "\n"
    finally:
        # Signal the end of the stream
        print("[FastAPI] Chat stream ended.")
        yield json.dumps({"type": "stream_end"}) + "\n"


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    The main chat endpoint. It receives the chat history
    and returns a streaming response from the agent.
    """
    messages = request.model_dump().get("messages", [])
    
    return StreamingResponse(
        stream_agent_response(messages),
        media_type="application/x-ndjson"
    )

if __name__ == "__main__":
    """
    Run the application using uvicorn.
    """
    print("Starting FastAPI server...")
    print(f"Serving chat UI at http://localhost:8000")
    print(f"Chat API endpoint at http://localhost:8000/api/chat")
    uvicorn.run(app, host="0.0.0.0", port=8000)
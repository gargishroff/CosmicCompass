import uvicorn
import json
import os
import sys
import io
from contextlib import redirect_stdout
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from langgraph.types import Command # <-- Import Command

# --- Project Root Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

try:
    # Import your supervisor agent
    from core.agent_self2 import supervisor_agent
except ImportError:
    print(f"Error: Could not import 'supervisor_agent' from 'core.agent_self2'.")
    print(f"Make sure 'core/agent_self2.py' exists and project root is correct.")
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
    thread_id: str  # Thread ID is now required

class ResumeRequest(BaseModel):
    decision: Dict[str, Any] # This will be the {"decisions": [...]} object
    thread_id: str

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def get_chat_ui():
    """
    Serves the main chat interface.
    """
    try:
        # Make sure this points to your correct HTML file
        with open(os.path.join(current_dir, "index2.html"), "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Chat UI file not found.</h1>", status_code=404)

async def _stream_agent_run(payload: Any, config: Dict[str, Any]):
    """
    Helper function to stream an agent run, capturing both
    pretty_print() logs and interrupts.
    """
    print(f"\n[FastAPI] Stream started for thread {config['configurable']['thread_id']}...")
    try:
        # Use astream to get all intermediate steps
        async for step in supervisor_agent.astream(payload, config=config):
            
            # --- 1. Check for Interrupts First ---
            if "__interrupt__" in step:
                print(f"[FastAPI] Interrupting for thread {config['configurable']['thread_id']}")
                interrupt = step["__interrupt__"][0] # Get the Interrupt object
                interrupt_data = interrupt.value # Get the JSON-serializable dict
                
                # Send the interrupt message
                yield json.dumps({
                    "type": "interrupt",
                    "data": interrupt_data 
                }) + "\n"
                
                # IMPORTANT: Stop the stream. The agent is now paused.
                print(f"[FastAPI] Stream paused for interrupt.")
                return

            # --- 2. If no interrupt, process as a log ---
            for update in step.values():
                if isinstance(update, dict):
                    for message in update.get("messages", []):
                        f = io.StringIO()
                        with redirect_stdout(f):
                            message.pretty_print()
                        log_string = f.getvalue()
                        
                        if log_string.strip():
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
        # This signals a *successful*, *uninterrupted* end
        print(f"[FastAPI] Stream finished for thread {config['configurable']['thread_id']}.")
        yield json.dumps({"type": "stream_end"}) + "\n"

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Starts or continues a chat. Returns a stream of logs
    and potentially a final 'interrupt' message.
    """
    print(f"[FastAPI] /api/chat called for thread {request.thread_id}")
    config = {"configurable": {"thread_id": request.thread_id}}
    payload = {"messages": request.messages}
    
    return StreamingResponse(
        _stream_agent_run(payload, config),
        media_type="application/x-ndjson"
    )

@app.post("/api/chat/resume")
async def chat_resume_endpoint(request: ResumeRequest):
    """
    Resumes a conversation. Also returns a stream of logs
    and potentially *another* 'interrupt' message.
    """
    print(f"[FastAPI] /api/chat/resume called for thread {request.thread_id}")
    config = {"configurable": {"thread_id": request.thread_id}}
    # Package the decision into a Command
    payload = Command(resume=request.decision)

    return StreamingResponse(
        _stream_agent_run(payload, config),
        media_type="application/x-ndjson"
    )

if __name__ == "__main__":
    """
    Run the application using uvicorn.
    """
    print("Starting FastAPI server...")
    print(f"Serving chat UI at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
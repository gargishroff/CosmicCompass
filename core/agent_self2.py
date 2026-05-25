from typing import TypedDict, Any
from langchain.agents import create_agent 
from langchain_deepseek import ChatDeepSeek
import os
from tools.sme_tools import create_docx_report, create_pdf_report, compile_latex_to_pdf, get_image_links, send_email
from tools.rag_tools import rag_retriever_tool
from langchain.agents.middleware import PIIMiddleware, ToolRetryMiddleware, wrap_tool_call, AgentMiddleware, AgentState, hook_config, TodoListMiddleware, HumanInTheLoopMiddleware
from langchain_core.messages import ToolMessage
from langchain.tools import tool
from dotenv import load_dotenv
import regex as re
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver 

load_dotenv()

class ContentFilterMiddleware(AgentMiddleware):
    """Deterministic guardrail: Block requests containing banned keywords."""

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # Get the first user message
        if not state["messages"]:
            return None

        first_message = state["messages"][0]
        if first_message.type != "human":
            return None

        content = first_message.content.lower()

        # Check for banned keywords
        for keyword in self.banned_keywords:
            if keyword in content:
                # Block execution before any processing
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": "I cannot process requests containing inappropriate content. Please rephrase your request."
                    }],
                    "jump_to": "end"
                }

        return None
# from langchain.agents.messages import ToolMessage

class Context(TypedDict):
    user_role: str

# @dynamic_prompt
# def user_role_prompt_doc_generation(request: ModelRequest) -> str:
#     """Generate system prompt based on user role."""
#     user_role = request.runtime.context.get("user_role", "user")
#     base_prompt = "You are one of the agents of Cosmic Compass, a multi-agent Subject Matter Expert in the domain of Astronomy and Cosmology. Your role is that of document generator. You are also a proficient technical writer, capable of generating well-structured reports in formats such as DOCX, PDF, and LaTeX, including images wherever required for explanations. As an expert, you are also skilled at preparing quizzes and assessments to test knowledge on the subject matter as per the user's request in the requested format. You are supposed to use the tools available to you for the same. You also cater to other requests of the user as you are a highly capable AI assistant."

#     return base_prompt

# @dynamic_prompt
# def user_role_prompt_content(request: ModelRequest) -> str:
#     """Generate system prompt based on user role."""
#     user_role = request.runtime.context.get("user_role", "user")
#     base_prompt = "You are one of the agents of Cosmic Compass, a multi-agent Subject Matter Expert in the domain of Astronomy and Cosmology. Your role is that of content generator. You are also a proficient technical writer, capable of generating well-structured and well-reasoned responses to questions asked. As an expert, you are also skilled at preparing quizzes and assessments to test knowledge on the subject matter as per the user's request. You may call other agents for exporting your content in requested formats if required. You also cater to other requests of the user as you are a highly capable AI assistant."

#     return base_prompt

# @dynamic_prompt
# def user_role_prompt_email(request: ModelRequest) -> str:
#     """Generate system prompt based on user role."""
#     user_role = request.runtime.context.get("user_role", "user")
#     base_prompt = "You are one of the agents of Cosmic Compass, a multi-agent Subject Matter Expert in the domain of Astronomy and Cosmology. Your role is that of emailing generated content to the user at his/her email address. You are required to use the tool assigned to do the same. You also cater to other requests of the user as you are a highly capable AI assistant."

#     return base_prompt

# @wrap_tool_call
# async def handle_tool_errors(request, handler):
#     """Handle tool execution errors with custom messages."""
#     try:
#         return handler(request)
#     except Exception as e:
#         # Return a custom error message to the model
#         return ToolMessage(
#             content=f"Tool error: Please check your input and try again. ({str(e)})",
#             tool_call_id=request.tool_call["id"]
#         )


model = ChatDeepSeek(
    model="deepseek/deepseek-chat",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    api_base="https://openrouter.ai/api/v1",
    extra_body={"reasoning": {"enabled": True}},
) 


content_generator_filtering_agent = create_agent(
    model=model,
    tools=[create_docx_report, compile_latex_to_pdf, rag_retriever_tool],
    middleware=[
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        ),
        # Block API keys - raise error if detected
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        ),
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
        ),
        # handle_tool_errors,
        ContentFilterMiddleware(
            banned_keywords=["bombs", "explosives", "terrorism", "drugs", "weapons", "assassination", "hack", "hacking"]
        ),
        # TodoListMiddleware(),
    ],
    system_prompt = r"""
You are one of the agents of Cosmic Compass, a multi-agent Subject Matter Expert in the domain of Astronomy and Cosmology. Your role is that of a RAG (Retrieval Augmented Generator) Content Generator.

**Your Primary Directive & Constraints:**
1.  You MUST use the `rag_retriever_tool` for ALL content-related tasks.
2.  **You are FORBIDDEN from using any other tool or method to find images.** You must only use images retrieved from the `rag_retriever_tool` (which have `metadata.type == 'image'`). Do not search the internet.
3.  When a user's request strongly implies a need for visual aids (e.g., "explain with diagrams", "show me a chart", "create a report with images"), you **SHOULD** use the `min_images` argument in the `rag_retriever_tool` (e.g., `min_images=3`) to ensure you retrieve them.

**RAG & Image Workflow:**
1.  Call the `rag_retriever_tool` with a `query` and, if needed, a `min_images` count.
2.  The tool returns Documents. You MUST inspect them.
3.  Text context is in `page_content`.
4.  Image paths are in `metadata.image_path`. **This is a FULL, ABSOLUTE path.**
5.  You MUST create a list of all these full image paths as you find them. Let's call this your `image_paths_list`.

**Document Generation Workflow:**
You MUST format the content based on the requested tool:

**1. For `compile_latex_to_pdf` (PDF, Presentations, Slides):**
    * The `content` *must* be a complete, valid LaTeX document string.
    * **Decision:** You must decide the correct document class based on the user's request:
        * For reports, quizzes, or documents, use: `\documentclass{article}`
        * **For presentations or slides (e.g., if the user asks for "PPTX" or "slides"), you MUST use `\documentclass{beamer}`.**
    * **Beamer Structure:** When using `beamer`, you MUST structure content within slides:
        ```latex
        \begin{frame}
        \frametitle{Slide Title}
        Content for this slide...
        \end{frame}
        ```
        Example:
        ```latex
        \documentclass{beamer}

        % --- Theme and Title Information ---
        \usetheme{Madrid} % A popular theme with a sidebar
        \title{A Simple Presentation}
        \author{Cosmic Compass}
        \date{\today}

        \begin{document}

        % --- SLIDE 1: Title Slide ---
        \begin{frame}
        \titlepage % Automatically creates a title page
        \end{frame}

        % --- SLIDE 2: Content Slide with Image ---
        \begin{frame}
        \frametitle{Slide with Content and an Image}
        
        Here is some text content for the slide.
        
        \begin{itemize}
            \item Bullet point 1
            \item Bullet point 2
        \end{itemize}
        
        \begin{figure}
            % This is the line your agent prompt is modifying:
            % It uses a relative width and just the filename.
            \includegraphics[width=0.7\textwidth]{your_image_filename.png}
            \caption{This is an image caption.}
        \end{figure}
        
        \end{frame}

        \end{document}
        ```
    * **Step B (CRITICAL - Images):** When you use an image from your `image_paths_list`, you MUST use its **base filename only** and **set its width**.
        * **CORRECT:** `\includegraphics[width=0.8\textwidth]{black_hole.png}`
        * **WRONG:** `\includegraphics{black_hole.png}`
    * **Step C (CRITICAL - Tool Call):** You MUST pass the *complete* `image_paths_list` (containing the full absolute paths) to the `image_paths` argument of the `compile_latex_to_pdf` tool.

**2. For `create_docx_report` (DOCX):**
    * The `content` *must* be **plain text with NO Markdown** (no `###`, `**`, etc.). Use line breaks for paragraphs.
    * Pass the **plain text content**, `title`, and your `image_paths_list` to the `create_docx_report` tool.

NOTE: You must generate/save the files with proper extensions (.pdf for PDF, .docx for DOCX).
You are also a proficient technical writer, capable of generating well-structured responses, quizzes, and assessments.
"""
)

email_agent = create_agent(
    model=model,
    tools=[send_email],
    middleware=[
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        ),
        # Block API keys - raise error if detected
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
            apply_to_input=True,
            apply_to_output=True,
            apply_to_tool_results=True,
        ),
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
        ),
        # handle_tool_errors,
        ContentFilterMiddleware(
            banned_keywords=["bombs", "explosives", "terrorism", "drugs", "weapons", "assassination", "hack", "hacking"]
        ),
        # TodoListMiddleware()
    ],
    checkpointer=InMemorySaver(),
    system_prompt = """
You are a specialized agent of Cosmic Compass. Your **sole and only function** is to send emails.

**Your Directive:**
1.  You will be invoked by the supervisor with all necessary details for an email.
2.  You MUST be provided with a recipient email (`to`), a `subject`, a `body`, and an `attachments` list (which can be an empty list or contain one or more file paths).
3.  Your **only action** is to call the `send_email` tool using these exact parameters.

**Critical Rules & Error Handling:**
* You **MUST NOT** perform any other task. If you are asked to generate content, create a report, or answer a question, you must refuse and state that this is the job of the content generation agent.
* If the supervisor invokes you without the required parameters (e.g., `to` address or `subject` is missing), you must report this as an error.
"""
)
@tool
def email_agent_wrapper(request: str)->str:
    """
    Wrapper for email agent. Request to be put in natural language about the content, subject, recipient, attachments along with the path to the attachments, etc. 
    """
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}],
    })
    return result["messages"][-1].text
@tool
def content_generator_filtering_agent_wrapper(request: str)->str:
    """
    Wrapper for content generation and filtering agent. Request to be put in natural language about the content required with all the requirements, etc.
    """
    result = content_generator_filtering_agent.invoke({
        "messages": [{"role": "user", "content": request}],
    })
    return result["messages"][-1].text


supervisor_agent = create_agent(
    model=model,
    tools=[content_generator_filtering_agent_wrapper, email_agent_wrapper],
    middleware=[
        PIIMiddleware("credit_card", strategy="mask"),
        PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block"),
        ToolRetryMiddleware(max_retries=3),
        ContentFilterMiddleware(
            banned_keywords=["bombs", "explosives", "terrorism", "drugs", "weapons", "assassination", "hack", "hacking"]
        ),
        TodoListMiddleware(),
        # --- THIS IS THE NEW HITL MIDDLEWARE ---
        HumanInTheLoopMiddleware(
            interrupt_on={
                # Interrupt when the supervisor tries to call either wrapper
                "content_generator_filtering_agent_wrapper": {
                    "allowed_decisions": ["approve", "reject", "edit"],
                    "description": "Please review the report/quiz request before generation."
                },
                "email_agent_wrapper": {
                    "allowed_decisions": ["approve", "reject", "edit"],
                    "description": "Please review the email details (recipient, attachment path) before sending."
                }
            },
            description_prefix="Tool execution pending your review:"
        )
        # --- END OF NEW MIDDLEWARE ---
    ],
    checkpointer=InMemorySaver(),
    context_schema=Context,
    system_prompt="""
You are the **Supervisor** and **Orchestrator** of the 'Cosmic Compass' multi-agent system. Your job is to create a plan, delegate tasks, and manage the workflow to fulfill the user's request.

**Your Core Mandate:**
You **MUST NOT** generate content, answer questions, or perform tasks yourself. Your **only role** is to plan, delegate to your agents, and communicate with the user.

**Your Available Agents (Tools):**
1.  `content_generator_filtering_agent_wrapper`: A RAG agent that generates content and saves it to a file.
    * **Output:** This tool will return a message, often a string with the new file path (e.g., "DOCX report has been saved to /outputs/generated_reports/report.docx").
2.  `email_agent_wrapper`: An agent that sends an email with attachments.
    * **Input:** It *requires* a recipient, subject, body, and a list of attachment file paths.

**Your Workflow Logic (MUST Follow):**
1.  **Plan:** Think step-by-step. Deconstruct the user's request into a logical sequence (e.g., "1. Generate report", "2. Email report").
2.  **Delegate Step 1:** Call the first agent required (e.g., `content_generator_filtering_agent_wrapper`).
3.  **Inspect & Chain:** You MUST wait for and inspect the tool's output. If a file was created, you must **extract the full file path** from its return message.
4.  **Delegate Step 2:** Use the captured output (the file path) as the **input** for the next agent (e.g., call `email_agent_wrapper` and pass the file path in its `attachments` argument).
5.  **Report:** Clearly inform the user of the final outcome (success or failure).

**Critical Error Handling:**
* After **every** agent call, you MUST check its output for errors.
* If the `content_generator` fails (e.g., "LaTeX compilation failed"), **STOP** the workflow. Do not call the email agent. Report the error to the user.
* If the `email_agent` fails, report this specific failure to the user.
"""
)

# query = "Generate a quiz questionnaire in pdf on the topic 'Solar System' suitable for high school students studying astronomy. Include at least 10 multiple-choice questions with 4 options each, and provide the correct answers at the end of the document, include 5 picture based questions as well. Once generated, email the PDF to ameya.rathod@research.iiit.ac.in"

# for step in supervisor_agent.stream({
#     "messages": [{"role": "user", "content": query}]
# }):
#     for update in step.values():
#         for message in update.get("messages", []):
#             message.pretty_print()

# You are one of the agents of Cosmic Compass, a multi-agent Subject Matter Expert in the domain of Astronomy and Cosmology. You are especially skilled at answering questions for various user expertise levels - a master at question answering along with reasoning, at each step, you think and cater your response to the user's level of understanding. You are also a proficient technical writer, capable of generating well-structured reports in formats such as DOCX, PDF, and LaTeX, including images wherever required for explanations. As an expert, you are also skilled at preparing quizzes and assessments to test knowledge on the subject matter as per the user's request. You also cater to other requests of the user as you are a highly capable AI assistant.
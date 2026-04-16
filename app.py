import streamlit as st
import os
import time
import random
from dotenv import load_dotenv
from typing import TypedDict, Optional

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

# ------------------ LOAD API ------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# ------------------ LLM ------------------
llm = ChatGroq(
    groq_api_key=api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.5
)

# ------------------ MEMORY ------------------
MEMORY = []

# ------------------ STATE ------------------
class AgentState(TypedDict):
    input: str
    plan: Optional[str]
    context: Optional[str]
    answer: Optional[str]

# ------------------ FILE READER ------------------
def read_file(file):
    text = ""

    if file.name.endswith(".txt"):
        text = file.read().decode()

    elif file.name.endswith(".pdf"):
        from PyPDF2 import PdfReader
        pdf = PdfReader(file)
        for page in pdf.pages:
            text += page.extract_text()

    elif file.name.endswith(".docx"):
        import docx
        doc = docx.Document(file)
        for p in doc.paragraphs:
            text += p.text + "\n"

    return text

# ------------------ CLAUSE CHECKER ------------------
def check_clauses(text):

    rules = {
        "Confidentiality Clause": "confidential",
        "Termination Clause": "termination",
        "Governing Law": "law",
        "Liability Clause": "liability",
        "Payment Terms": "payment"
    }

    results = []
    lower = text.lower()

    for clause, keyword in rules.items():
        if keyword in lower:
            results.append(f"{clause} → PASS")
        else:
            results.append(f"{clause} → FAIL")

    return "\n".join(results)

# ------------------ MEMORY RECALL ------------------
def recall_memory(user_input):

    text = user_input.lower()

    if "last document" in text:
        if MEMORY:
            return MEMORY[-1]["answer"]

    if "failed" in text:
        results = []
        for i, m in enumerate(MEMORY):
            lines = m["answer"].split("\n")
            for l in lines:
                if "fail" in l.lower():
                    results.append(f"Doc {i+1} → {l}")

        if results:
            return "\n".join(results)

    if "compare" in text:
        results = []
        for i, m in enumerate(MEMORY):
            results.append(f"Doc {i+1}:\n{m['answer']}\n")

        return "\n".join(results)

    if "show memory" in text:
        return str(MEMORY)

    return None

# ------------------ INTENT ------------------
def classify_intent(text):
    text = text.lower().strip()
    greetings = ["hi", "hey", "yo", "hello"]

    if text in greetings:
        return "chit-chat"

    return "question"

# ------------------ CODE DETECTION ------------------
def is_code(text):
    keywords = ["print", "def", "class", "import", "(", ")", "{", "}"]
    return any(k in text for k in keywords)

# ------------------ REWOO NODES ------------------

def planner(state: AgentState):
    prompt = f"""
Break the problem into steps.

Question:
{state['input']}

Steps:
1.
2.
3.
"""
    res = llm.invoke([HumanMessage(content=prompt)])
    return {"plan": res.content}

def worker(state: AgentState):
    context = f"Executed Plan:\n{state['plan']}"
    return {"context": context}

def solver(state: AgentState):
    user_input = state["input"]

    if is_code(user_input):
        prompt = f"""
You are an expert coding assistant.

User Code:
{user_input}

Fix and explain.
"""
    else:
        prompt = f"""
Answer clearly.

Question:
{user_input}

Plan:
{state['plan']}
"""

    res = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": res.content}

def memory_store(state: AgentState):

    MEMORY.append({
        "question": state["input"],
        "answer": state["answer"]
    })

    return state

# ------------------ GRAPH ------------------
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner)
workflow.add_node("worker", worker)
workflow.add_node("solver", solver)
workflow.add_node("memory", memory_store)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "worker")
workflow.add_edge("worker", "solver")
workflow.add_edge("solver", "memory")
workflow.add_edge("memory", END)

app = workflow.compile()

# ------------------ UI ------------------
st.set_page_config(page_title="Freko AI", page_icon="🤖", layout="wide")

st.title("🤖 Freko AI (ReWOO + Document Memory)")
st.write("Upload contracts & check clause pass/fail")

# ------------------ UPLOAD ------------------
uploaded_file = st.file_uploader(
    "Upload Contract",
    type=["pdf","txt","docx"]
)

if uploaded_file:

    text = read_file(uploaded_file)

    result = check_clauses(text)

    MEMORY.append({
        "question": uploaded_file.name,
        "answer": result
    })

    st.success("Document analyzed")

    st.text(result)

# ------------------ SESSION ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------ CHAT DISPLAY ------------------
for role, msg in st.session_state.messages:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)

# ------------------ INPUT ------------------
user_input = st.chat_input("Ask something...")

if user_input:

    st.session_state.messages.append(("user", user_input))

    intent = classify_intent(user_input)

    if intent == "chit-chat":
        reply = random.choice([
            "Hey!",
            "Hello!",
            "Hi there!",
            "How can I help?"
        ])
    else:

        memory_answer = recall_memory(user_input)

        if memory_answer:
            reply = "(from memory)\n" + memory_answer
        else:
            result = app.invoke({"input": user_input})
            reply = result["answer"]

    st.session_state.messages.append(("bot", reply))
    st.rerun()

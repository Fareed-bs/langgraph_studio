import streamlit as st
import asyncio
import requests
from langgraph.graph import StateGraph
from typing import TypedDict

# Define State Schema
class ChatState(TypedDict):
    input: str
    response: str

# LM Studio API Configuration
LM_STUDIO_API_URL = "http://localhost:1234/v1"
HEADERS = {"Content-Type": "application/json"}

# Function to Call LM Studio
async def call_lm_studio(prompt):
    payload = {
        "model": "llama",  # Adjust based on your model
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 256
    }
    response = requests.post(f"{LM_STUDIO_API_URL}/completions", json=payload, headers=HEADERS)
    return response.json()["choices"][0]["text"]

# General Chat
async def general_chat(state: ChatState):
    response = await call_lm_studio(state["input"])
    return {"response": response}

# Fact Checker
async def fact_checker(state: ChatState):
    response = "Let me verify that for you. (Dummy fact-checker)"
    return {"response": response}

# Summarizer
async def summarizer(state: ChatState):
    response = "Here's a summary: " + state["input"][:100]  # Mock summary
    return {"response": response}

# Route Query to Agents
def route_query(state: ChatState):
    user_input = state["input"].lower()
    
    if "summarize" in user_input:
        return {"next_step": "summarizer"}
    elif "is it true" in user_input or "fact-check" in user_input:
        return {"next_step": "fact_checker"}
    else:
        return {"next_step": "general_chat"}

# Build LangGraph Workflow
builder = StateGraph(state_schema=ChatState)
builder.add_node("route_query", route_query)
builder.add_node("general_chat", general_chat)
builder.add_node("fact_checker", fact_checker)
builder.add_node("summarizer", summarizer)

builder.set_entry_point("route_query")
builder.add_conditional_edges(
    "route_query",
    lambda state: state["next_step"],
    {
        "general_chat": "general_chat",
        "fact_checker": "fact_checker",
        "summarizer": "summarizer"
    },
)

# Enable async execution
graph = builder.compile()

# Streamlit UI
st.title("Advanced Chatbot: Multi-Agent + Async-Powered Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", "")

if st.button("Send") and user_input:
    result = asyncio.run(graph.ainvoke({"input": user_input}))
    response = result["response"]

    # Store chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))

# Display chat history
for role, text in st.session_state.chat_history:
    st.write(f"**{role}:** {text}")

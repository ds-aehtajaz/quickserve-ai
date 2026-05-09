import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

_llm = None

SYSTEM_PROMPT = """You are QuickServe, a friendly customer service assistant for an online store.
You help customers place orders, track deliveries, handle cancellations, and answer questions.
Keep your replies short and clear. Be polite but not over-formal.
Never make up order details — only use information provided to you in this message.
If you don't know something, say so honestly."""


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
            max_tokens=512,
        )
    return _llm


def generate_response(context: str, user_text: str) -> str:
    """Send a context-augmented message to Groq and return the reply."""
    llm = get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"{context}\n\nUser: {user_text}"),
    ]
    reply = llm.invoke(messages)
    return reply.content.strip()

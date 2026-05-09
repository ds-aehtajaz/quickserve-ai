import os
import uuid
import requests
import streamlit as st

API_URL = "http://localhost:8000"

# In-process mode (used on Hugging Face Spaces, where there is no separate backend)
INPROCESS = os.getenv("QUICKSERVE_INPROCESS", "0") == "1"

if INPROCESS:
    # Lazy-init the agent and DB so first import is fast
    @st.cache_resource(show_spinner=False)
    def _get_agent():
        from quickserve.db.models import init_db
        from quickserve.agent.graph import app as agent_app
        init_db()
        return agent_app

    def _chat(message: str, session_id: str) -> dict:
        agent_app = _get_agent()
        state = {
            "session_id":     session_id,
            "user_text":      message,
            "cleaned_text":   "",
            "intent":         "",
            "confidence":     0.0,
            "entities":       {},
            "retrieved_docs": [],
            "db_result":      {},
            "response":       "",
            "latency_ms":     0.0,
        }
        result = agent_app.invoke(state)
        return {
            "reply":      result["response"],
            "intent":     result["intent"],
            "confidence": result["confidence"],
        }

st.set_page_config(page_title="QuickServe.AI", page_icon="🛒", layout="centered")
st.title("🛒 QuickServe.AI")
st.caption("Your 24/7 customer service assistant")

# Session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey! I'm QuickServe. How can I help you today?"}
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if INPROCESS:
                    data = _chat(user_input, st.session_state.session_id)
                else:
                    resp = requests.post(
                        f"{API_URL}/chat",
                        json={"message": user_input, "session_id": st.session_state.session_id},
                        timeout=30,
                    )
                    data = resp.json()
                reply = data.get("reply", "Sorry, something went wrong.")
                intent = data.get("intent", "")
                confidence = data.get("confidence", 0)
            except Exception as e:
                reply = f"Could not reach the backend: {e}"
                intent, confidence = "error", 0.0

        st.write(reply)
        st.caption(f"Intent: `{intent}` | Confidence: `{confidence:.2%}`")

    st.session_state.messages.append({"role": "assistant", "content": reply})

# Sidebar
with st.sidebar:
    st.header("Session Info")
    st.code(st.session_state.session_id[:8] + "...", language=None)
    if st.button("New Conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = [
            {"role": "assistant", "content": "Hey! I'm QuickServe. How can I help you today?"}
        ]
        st.rerun()
    st.markdown("---")
    st.markdown("**Sample queries:**")
    st.markdown("- Order 2 pizzas\n- Track order ORD001\n- Cancel ORD002\n- Shipping policy?")

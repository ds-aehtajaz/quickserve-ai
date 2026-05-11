"""
Streamlit Cloud entry point — fully self-contained.
"""
import os
import sys
import uuid

# Make src importable
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(page_title="QuickServe.AI", page_icon="🛒", layout="centered")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🛒 QuickServe.AI")
st.caption("Your 24/7 customer service assistant")


# ── Lazy-init agent (cache so it loads only once per session) ─────────────────
@st.cache_resource(show_spinner="Loading NLP models (first time only)…")
def get_agent_app():
    from quickserve.db.models import init_db
    from quickserve.agent.graph import app as agent_app
    init_db()
    return agent_app


def chat(message: str, session_id: str) -> dict:
    try:
        agent_app = get_agent_app()
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
            "reply":      result.get("response", "(empty response)"),
            "intent":     result.get("intent", ""),
            "confidence": result.get("confidence", 0.0),
        }
    except Exception as e:
        import traceback
        st.error(f"Backend error: {e}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        return {"reply": f"Error: {e}", "intent": "error", "confidence": 0.0}


# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey! I'm QuickServe. How can I help you today?"}
    ]


# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            data = chat(user_input, st.session_state.session_id)
        st.write(data["reply"])
        st.caption(f"Intent: `{data['intent']}` | Confidence: `{data['confidence']:.2%}`")

    st.session_state.messages.append({"role": "assistant", "content": data["reply"]})


# ── Sidebar ───────────────────────────────────────────────────────────────────
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
    st.markdown(
        "- hello\n"
        "- Order 2 pizzas\n"
        "- Track order ORD123456\n"
        "- What is your return policy?\n"
        "- I want to speak to a human\n"
        "- Goodbye"
    )
    st.markdown("---")
    st.markdown("[GitHub Repo](https://github.com/ds-aehtajaz/quickserve-ai)")

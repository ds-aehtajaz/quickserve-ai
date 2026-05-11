"""
Streamlit Cloud entry point — simplified Groq-only demo.

This is the LIVE DEMO version. The full NLP pipeline (DistilBERT, spaCy NER,
RAG, LangGraph agent) is in src/quickserve/ — that's what the report describes
and what runs locally via `python run.py`.

For the live demo on Streamlit Cloud (where dependency conflicts make the full
pipeline brittle), we use the Groq LLM directly with a system prompt that
handles intent understanding and response generation in one shot.
"""
import os
import sys
import uuid

import streamlit as st

# Page config
st.set_page_config(page_title="QuickServe.AI", page_icon="🛒", layout="centered")
st.title("🛒 QuickServe.AI")
st.caption("Your 24/7 customer service assistant — live demo")

# ── System prompt (handles intent + grounded response in one shot) ────────────
SYSTEM_PROMPT = """You are QuickServe, a helpful and friendly customer service assistant
for QuickServe, an Indian e-commerce business.

You can help customers with:
- GREETINGS & GOODBYES: respond warmly.
- PLACING ORDERS: if a user wants to order something, confirm the item and quantity,
  and give them a fake order reference number in the format ORD followed by 6 digits
  (e.g. ORD827341). Acknowledge the total cost in Rs. (Indian Rupees).
- TRACKING ORDERS: if they mention an order ID (ORDxxxxxx), tell them it is currently
  "confirmed" and will ship in 1–2 business days.
- CANCELLING ORDERS: confirm cancellation and tell them refunds take 3–5 business days.
- MODIFYING ORDERS: confirm the change.
- FAQs:
  * Delivery: standard 3–5 business days, express 1–2 days. Free above Rs. 500.
  * Returns: 7-day return policy on unused items in original packaging. Refunds
    in 3–5 business days. Perishable items can't be returned.
  * Payments: we accept UPI (Google Pay, PhonePe, BHIM), credit/debit cards,
    and Cash on Delivery.
  * Account: contact support@quickserve.ai for help with login, password reset.
- HUMAN AGENT: if they want to talk to a human, tell them you'll connect them
  to a support agent.
- UNCLEAR INPUT: politely ask them to rephrase.

Keep replies concise (1–3 sentences). Be warm but professional. Don't make up
business policies that aren't in the list above. Use Indian Rupees (Rs.) for prices.

Sample product catalogue: Margherita Pizza (Rs. 199), Chicken Burger (Rs. 149),
Veggie Sandwich (Rs. 99), Cappuccino (Rs. 79), Cold Coffee (Rs. 89),
Wireless Mouse (Rs. 499), Mechanical Keyboard (Rs. 2499), Branded T-Shirt (Rs. 299),
Casual Sneakers (Rs. 1499), Running Shoes (Rs. 2199), Backpack 20L (Rs. 1199).

Also classify each user message into one of these intents and include it on a
SEPARATE last line in the exact format `[Intent: <intent_name>]`:
greeting, goodbye, place_order, track_order, cancel_order, modify_order,
faq, talk_to_human, fallback.
"""


@st.cache_resource(show_spinner="Initialising the assistant…")
def get_groq_client():
    """Initialise the Groq client using either Streamlit secrets or env var."""
    api_key = None
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("GROQ_API_KEY is not set. Add it in App Settings → Secrets.")
        st.stop()

    from groq import Groq
    return Groq(api_key=api_key)


def chat(history: list[dict]) -> dict:
    """Send the conversation to Groq and return reply + parsed intent."""
    try:
        client = get_groq_client()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.5,
            max_tokens=400,
        )
        text = completion.choices[0].message.content.strip()

        # Try to pull out the [Intent: ...] tag, if present
        intent = "unknown"
        reply  = text
        import re
        m = re.search(r"\[Intent:\s*([a-z_]+)\s*\]", text, re.IGNORECASE)
        if m:
            intent = m.group(1).lower()
            reply = re.sub(r"\[Intent:\s*[a-z_]+\s*\]", "", text, flags=re.IGNORECASE).strip()
        return {"reply": reply, "intent": intent}
    except Exception as e:
        import traceback
        st.error(f"Backend error: {e}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
        return {"reply": f"(Error contacting LLM: {e})", "intent": "error"}


# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.history  = []  # full chat history for the LLM

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("intent"):
            st.caption(f"Intent: `{msg['intent']}`")

# Initial greeting
if not st.session_state.messages:
    greeting = "Hey! I'm QuickServe. How can I help you today?"
    st.session_state.messages.append({"role": "assistant", "content": greeting})
    with st.chat_message("assistant"):
        st.write(greeting)


# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Type your message...")
if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Get assistant reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            data = chat(st.session_state.history)
        st.write(data["reply"])
        st.caption(f"Intent: `{data['intent']}`")

    st.session_state.messages.append({
        "role":    "assistant",
        "content": data["reply"],
        "intent":  data["intent"],
    })
    st.session_state.history.append({"role": "assistant", "content": data["reply"]})


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session Info")
    st.code(st.session_state.session_id[:8] + "...", language=None)
    if st.button("New Conversation"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()
    st.markdown("---")
    st.markdown("**Sample queries:**")
    st.markdown(
        "- hello\n"
        "- Order 2 pizzas\n"
        "- Track order ORD123456\n"
        "- What is your return policy?\n"
        "- Do you accept UPI payments?\n"
        "- I want to speak to a human\n"
        "- Goodbye"
    )
    st.markdown("---")
    st.markdown("**About this demo**")
    st.caption(
        "This live demo uses the Groq Llama-3.3-70B API directly. "
        "The full NLP pipeline (DistilBERT intent classifier + spaCy NER + "
        "FAISS RAG + LangGraph agent) runs locally — see the GitHub repo "
        "for the complete trained-models architecture."
    )
    st.markdown("[GitHub Repo](https://github.com/ds-aehtajaz/quickserve-ai)")

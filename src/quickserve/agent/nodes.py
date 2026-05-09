import time
import re
import uuid
from datetime import datetime

from ..nlp.preprocess import clean_text
from ..nlp.intent_classifier import predict_intent
from ..nlp.ner import extract_entities
from ..nlp.rag import retrieve
from ..llm.groq_client import generate_response
from ..llm.prompts import faq_context, order_context
from ..agent.state import AgentState
from ..db.models import SessionLocal, Order, Product, OrderItem, IntentLog


# ── helpers ──────────────────────────────────────────────────────────────────

def _find_product(db, name: str) -> Product | None:
    """Find a product by partial name match. Handles plurals and word-by-word matches."""
    if not name:
        return None
    name = name.strip().lower()

    # Try direct ilike match first
    p = db.query(Product).filter(Product.name.ilike(f"%{name}%")).first()
    if p:
        return p

    # Try stripping common plural suffixes
    candidates = [name]
    if name.endswith("ies") and len(name) > 3:
        candidates.append(name[:-3] + "y")
    if name.endswith("es") and len(name) > 2:
        candidates.append(name[:-2])
    if name.endswith("s") and len(name) > 1:
        candidates.append(name[:-1])

    for c in candidates:
        p = db.query(Product).filter(Product.name.ilike(f"%{c}%")).first()
        if p:
            return p

    # Try matching individual significant words (length > 2)
    for word in name.split():
        if len(word) > 2:
            for c in [word, word.rstrip("s")]:
                p = db.query(Product).filter(Product.name.ilike(f"%{c}%")).first()
                if p:
                    return p

    return None


def _gen_order_ref() -> str:
    return "ORD" + str(uuid.uuid4().int)[:6].upper()


# ── nodes ─────────────────────────────────────────────────────────────────────

def preprocess(state: AgentState) -> AgentState:
    state["cleaned_text"] = clean_text(state["user_text"])
    return state


def classify_intent(state: AgentState) -> AgentState:
    t0 = time.monotonic()
    result = predict_intent(state["cleaned_text"])
    state["intent"] = result["intent"]
    state["confidence"] = result["confidence"]
    state["latency_ms"] = round((time.monotonic() - t0) * 1000, 2)
    return state


def extract_ents(state: AgentState) -> AgentState:
    state["entities"] = extract_entities(state["user_text"])
    return state


def retrieve_faq(state: AgentState) -> AgentState:
    state["retrieved_docs"] = retrieve(state["user_text"], top_k=3)
    return state


def execute_db_action(state: AgentState) -> AgentState:
    intent = state["intent"]
    ents = state.get("entities", {})
    db = SessionLocal()
    result: dict = {}

    try:
        if intent == "place_order":
            product = _find_product(db, ents.get("ITEM"))
            qty = ents.get("QUANTITY") or 1
            if product is None:
                result = {"error": "Product not found. Could you give me the exact name?"}
            elif product.stock < qty:
                result = {"error": f"Sorry, only {product.stock} units of '{product.name}' are left."}
            else:
                ref = _gen_order_ref()
                order = Order(order_ref=ref, status="confirmed", total_amount=product.price * qty)
                db.add(order)
                db.flush()
                item = OrderItem(order_id=order.id, product_id=product.id, quantity=qty, unit_price=product.price)
                db.add(item)
                product.stock -= qty
                db.commit()
                result = {
                    "order_ref": ref,
                    "item": product.name,
                    "quantity": qty,
                    "total": f"Rs. {product.price * qty:.0f}",
                    "status": "confirmed",
                }

        elif intent == "track_order":
            ref = ents.get("ORDER_ID")
            if not ref:
                result = {"error": "Please share your order ID (e.g. ORD123) so I can look it up."}
            else:
                order = db.query(Order).filter(Order.order_ref == ref.upper()).first()
                if not order:
                    result = {"error": f"No order found with ID {ref}."}
                else:
                    result = {"order_ref": order.order_ref, "status": order.status, "placed_at": str(order.created_at)}

        elif intent == "cancel_order":
            ref = ents.get("ORDER_ID")
            if not ref:
                result = {"error": "Please share the order ID you want to cancel."}
            else:
                order = db.query(Order).filter(Order.order_ref == ref.upper()).first()
                if not order:
                    result = {"error": f"No order found with ID {ref}."}
                elif order.status in ("shipped", "delivered"):
                    result = {"error": f"Order {ref} is already {order.status} and cannot be cancelled."}
                else:
                    order.status = "cancelled"
                    db.commit()
                    result = {"order_ref": ref, "status": "cancelled"}

        elif intent == "modify_order":
            ref = ents.get("ORDER_ID")
            if not ref:
                result = {"error": "Please share the order ID you want to modify."}
            else:
                order = db.query(Order).filter(Order.order_ref == ref.upper()).first()
                if not order:
                    result = {"error": f"No order found with ID {ref}."}
                elif order.status not in ("pending", "confirmed"):
                    result = {"error": f"Order {ref} is {order.status} and can no longer be modified."}
                else:
                    new_product = _find_product(db, ents.get("ITEM"))
                    new_qty = ents.get("QUANTITY")
                    if new_product:
                        for oi in order.items:
                            oi.product_id = new_product.id
                            oi.unit_price = new_product.price
                    if new_qty:
                        for oi in order.items:
                            oi.quantity = new_qty
                    db.commit()
                    result = {"order_ref": ref, "updated": True, "status": order.status}
        else:
            result = {}
    finally:
        db.close()

    state["db_result"] = result
    return state


def generate_final_response(state: AgentState) -> AgentState:
    intent = state["intent"]
    confidence = state["confidence"]
    db_result = state.get("db_result", {})
    docs = state.get("retrieved_docs", [])

    if confidence < 0.18 or intent == "fallback":
        state["response"] = (
            "I'm not sure I understood that. Could you rephrase? "
            "I can help with placing orders, tracking, cancellations, and general questions."
        )
        return state

    if intent == "greeting":
        state["response"] = "Hey! I'm QuickServe. How can I help you today?"
        return state

    if intent == "goodbye":
        state["response"] = "Goodbye! Hope I was helpful. Have a great day!"
        return state

    if intent == "talk_to_human":
        state["response"] = (
            "Sure, I'll connect you to a support agent now. "
            "Please hold on — they'll be with you in a moment."
        )
        return state

    if intent == "faq":
        ctx = faq_context(docs)
    else:
        ctx = order_context(db_result)
        if "error" in db_result:
            state["response"] = db_result["error"]
            return state

    state["response"] = generate_response(ctx, state["user_text"])
    return state


def log_turn(state: AgentState) -> AgentState:
    db = SessionLocal()
    try:
        log = IntentLog(
            session_id=state.get("session_id", "unknown"),
            user_text=state["user_text"],
            predicted_intent=state.get("intent"),
            confidence=state.get("confidence"),
            latency_ms=state.get("latency_ms"),
        )
        db.add(log)
        db.commit()
    finally:
        db.close()
    return state

import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.models import init_db, get_db, Product, Order
from ..agent.graph import app as agent_app


@asynccontextmanager
async def lifespan(application: FastAPI):
    init_db()
    yield


api = FastAPI(
    title="QuickServe.AI",
    description="NLP-Based Automated Customer Service Chatbot API",
    version="0.1.0",
    lifespan=lifespan,
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Alias for uvicorn convention (uvicorn ... main:app)
app = api


# ── Schemas ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: str
    confidence: float


# ── Routes ───────────────────────────────────────────────────────────────────

@api.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    initial_state = {
        "session_id": session_id,
        "user_text": req.message,
        "cleaned_text": "",
        "intent": "",
        "confidence": 0.0,
        "entities": {},
        "retrieved_docs": [],
        "db_result": {},
        "response": "",
        "latency_ms": 0.0,
    }
    result = agent_app.invoke(initial_state)
    return ChatResponse(
        session_id=session_id,
        reply=result["response"],
        intent=result["intent"],
        confidence=round(result["confidence"], 4),
    )


@api.get("/products")
def list_products(db: Session = Depends(get_db)):
    return db.query(Product).all()


@api.get("/orders/{order_ref}")
def get_order(order_ref: str, db: Session = Depends(get_db)):
    order = db.query(Order).filter(Order.order_ref == order_ref.upper()).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order


@api.get("/health")
def health():
    return {"status": "ok"}

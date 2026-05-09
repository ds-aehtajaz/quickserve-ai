from typing import TypedDict


class AgentState(TypedDict):
    session_id: str
    user_text: str
    cleaned_text: str
    intent: str
    confidence: float
    entities: dict          # {"ITEM": str|None, "QUANTITY": int|None, "ORDER_ID": str|None}
    retrieved_docs: list    # top-k FAQ docs from FAISS
    db_result: dict         # result from order DB action
    response: str           # final natural-language reply
    latency_ms: float

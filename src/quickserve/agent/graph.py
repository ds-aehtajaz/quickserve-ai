from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    preprocess,
    classify_intent,
    extract_ents,
    retrieve_faq,
    execute_db_action,
    generate_final_response,
    log_turn,
)

ORDER_INTENTS = {"place_order", "track_order", "cancel_order", "modify_order"}
FAQ_INTENTS = {"faq"}


def _route_intent(state: AgentState) -> str:
    intent = state.get("intent", "fallback")
    if intent in ORDER_INTENTS:
        return "execute_db_action"
    if intent in FAQ_INTENTS:
        return "retrieve_faq"
    return "generate_response"


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("preprocess", preprocess)
    g.add_node("classify_intent", classify_intent)
    g.add_node("extract_entities", extract_ents)
    g.add_node("retrieve_faq", retrieve_faq)
    g.add_node("execute_db_action", execute_db_action)
    g.add_node("generate_response", generate_final_response)
    g.add_node("log_turn", log_turn)

    g.set_entry_point("preprocess")
    g.add_edge("preprocess", "classify_intent")
    g.add_edge("classify_intent", "extract_entities")

    g.add_conditional_edges(
        "extract_entities",
        _route_intent,
        {
            "execute_db_action": "execute_db_action",
            "retrieve_faq": "retrieve_faq",
            "generate_response": "generate_response",
        },
    )

    g.add_edge("retrieve_faq", "generate_response")
    g.add_edge("execute_db_action", "generate_response")
    g.add_edge("generate_response", "log_turn")
    g.add_edge("log_turn", END)

    return g.compile()


# Singleton — import this in the API and UI
app = build_graph()

"""
End-to-end demo test: runs every intent through the agent and prints
a clean transcript suitable for documentation/screenshots.
"""
import sys, os, uuid, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.quickserve.agent.graph import app as agent_app


def chat(text: str, session_id: str) -> dict:
    state = {
        "session_id":     session_id,
        "user_text":      text,
        "cleaned_text":   "",
        "intent":         "",
        "confidence":     0.0,
        "entities":       {},
        "retrieved_docs": [],
        "db_result":      {},
        "response":       "",
        "latency_ms":     0.0,
    }
    t0 = time.monotonic()
    result = agent_app.invoke(state)
    elapsed = (time.monotonic() - t0) * 1000
    return {
        "user":       text,
        "reply":      result["response"],
        "intent":     result["intent"],
        "confidence": result["confidence"],
        "entities":   result["entities"],
        "db_result":  result["db_result"],
        "elapsed_ms": elapsed,
    }


def banner(text):
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def show(turn, idx):
    print(f"\n[Turn {idx}]")
    print(f"  USER:       {turn['user']}")
    print(f"  INTENT:     {turn['intent']}  (confidence: {turn['confidence']:.2%})")
    if turn["entities"]:
        print(f"  ENTITIES:   {turn['entities']}")
    if turn["db_result"]:
        print(f"  DB_RESULT:  {turn['db_result']}")
    print(f"  REPLY:      {turn['reply'][:200]}")
    print(f"  LATENCY:    {turn['elapsed_ms']:.0f} ms")


# ── run a full conversation ───────────────────────────────────────────────────
session = str(uuid.uuid4())
banner(f"QuickServe.AI – End-to-End Demo  (session={session[:8]})")

queries = [
    # Greeting
    "hello there",
    # Place order – should resolve "pizzas" -> "Margherita Pizza"
    "I want to order 2 pizzas",
    # Place order – generic Indian item
    "order 1 chicken burger",
    # FAQ – return policy
    "what is your return policy",
    # FAQ – delivery time
    "how long does delivery take",
    # FAQ – payment methods
    "do you accept UPI payments",
    # Talk to human
    "I want to speak to a human",
    # Fallback – nonsense
    "xkqzz random gibberish test",
    # Goodbye
    "thanks bye",
]

results = []
for i, q in enumerate(queries, 1):
    try:
        r = chat(q, session)
        results.append(r)
        show(r, i)
    except Exception as e:
        print(f"\n[Turn {i}] FAILED: {q}")
        print(f"  Error: {e}")

# Track / cancel one of the orders we just placed
banner("Track & cancel the order we just placed")
order_refs = [r["db_result"].get("order_ref") for r in results
              if r["db_result"].get("order_ref")]
if order_refs:
    ref = order_refs[0]
    print(f"\nUsing first order ref: {ref}")
    for q in [f"track order {ref}", f"cancel order {ref}"]:
        r = chat(q, session)
        results.append(r)
        show(r, len(results))
else:
    print("No order refs available to track/cancel.")

# ── summary ───────────────────────────────────────────────────────────────────
banner("Summary")
intents_seen = sorted({r["intent"] for r in results})
print(f"Total turns:          {len(results)}")
print(f"Distinct intents:     {len(intents_seen)}")
print(f"Intent labels seen:   {intents_seen}")
print(f"Avg latency:          {sum(r['elapsed_ms'] for r in results)/len(results):.0f} ms")
print(f"Successful responses: {sum(1 for r in results if r['reply'])}/{len(results)}")

# Save transcript
out = "demo_transcript.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nFull transcript saved to: {out}")

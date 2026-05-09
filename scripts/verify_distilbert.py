import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quickserve.nlp.intent_classifier import predict_intent

tests = [
    "hello",
    "cancel order ORD123",
    "where is my order",
    "what is your return policy",
    "I want to talk to a human",
    "order 2 pizzas",
    "goodbye",
    "asdfgh random stuff",
]

print("Model: DistilBERT fine-tuned\n")
all_ok = True
for t in tests:
    r = predict_intent(t)
    flag = "" if r["confidence"] >= 0.5 else " [LOW CONFIDENCE]"
    print(f"  {t!r:<42} -> {r['intent']:<15} ({r['confidence']:.2%}){flag}")
    if r["confidence"] < 0.3:
        all_ok = False

print()
print("Status:", "All predictions look good!" if all_ok else "Some predictions have low confidence — check training.")

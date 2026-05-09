def faq_context(docs: list[dict]) -> str:
    if not docs:
        return ""
    lines = ["Relevant FAQ entries:"]
    for d in docs:
        lines.append(f"Q: {d['question']}\nA: {d['answer']}")
    return "\n\n".join(lines)


def order_context(db_result: dict) -> str:
    if not db_result:
        return ""
    lines = ["Order action result:"]
    for k, v in db_result.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)

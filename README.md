# QuickServe.AI

**NLP-Based Automated Customer Service Chatbot for Instant Support and Order Processing**

An end-to-end customer service chatbot that combines trained NLP models (DistilBERT for intent classification, spaCy for NER), retrieval-augmented generation (FAISS + sentence-transformers), and a free LLM API (Groq + Llama-3.3-70B) for natural-language response generation.

> MSc (Data Science) capstone project — Chandigarh University via Qollabb internship platform.
> **Author:** Aehtajaz Ahmed (Enrollment: O24MSD110120)
> **Mentor:** Ms. Roshini Ganesh, Qollabb – Keystone Advisory Group, Bangalore

---

## Highlights

- **Two trained intent classifiers compared:** TF-IDF + LogReg baseline (macro-F1 = **0.947**) vs fine-tuned DistilBERT (macro-F1 = **0.831**) on a 9-class problem
- **Custom spaCy NER** for `ITEM`, `QUANTITY`, `ORDER_ID` — combined F1 = **0.87**
- **RAG pipeline** with FAISS + sentence-transformers — Precision@3 = **0.95**
- **LangGraph state-graph agent** with 7 nodes and conditional routing
- **FastAPI REST backend** with auto-OpenAPI docs at `/docs`
- **Streamlit chat UI** for live interaction
- **22 pytest tests** with **71% code coverage**
- **End-to-end p95 latency: 3.2 s** on the Groq path
- **Zero cost** — every tool and API used is free-tier

---

## Architecture

```
        Customer (browser)
              │
        Streamlit Chat UI ──────► FastAPI Backend
                                       │
                                  LangGraph Agent
                                  ┌────┴────┐
                          NLP Layer         External
                ┌─────────┼─────────┐       ┌──────┐
            DistilBERT spaCy NER  FAISS    Groq API
            (intent)   (entities) (RAG)    (LLM)
                                             │
                                       SQLite Database
                                       (8 tables)
```

The agent has 7 nodes:
1. `preprocess` — clean text
2. `classify_intent` — DistilBERT (with TF-IDF fallback)
3. `extract_entities` — spaCy + regex
4. `retrieve_faq` — FAISS top-3 docs (FAQ branch)
5. `execute_db_action` — DB ops (order branch)
6. `generate_final_response` — Groq LLM (grounded)
7. `log_turn` — persist to IntentLog

Conditional routing based on `intent`:
- `faq` → retrieve_faq → generate_response
- `place_order|track_order|cancel_order|modify_order` → execute_db_action → generate_response
- `greeting|goodbye|talk_to_human|fallback` → generate_response (no Groq, no DB)

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.13 |
| Intent classification | DistilBERT (Hugging Face Transformers) + scikit-learn TF-IDF baseline |
| NER | spaCy 3 with custom training |
| Retrieval | sentence-transformers `all-MiniLM-L6-v2` + FAISS |
| Agent | LangGraph + langchain-groq |
| LLM | Groq API (Llama-3.3-70B, free tier) |
| Backend | FastAPI + uvicorn |
| Frontend | Streamlit |
| Database | SQLAlchemy + SQLite |
| Testing | pytest + pytest-cov |
| Training compute | Google Colab (free T4 GPU) |

---

## Quick Start

### Prerequisites
- Python 3.11 or higher
- A free Groq API key from https://console.groq.com

### 1. Clone the repository
```bash
git clone https://github.com/ds-aehtajaz/quickserve-ai.git
cd quickserve-ai
```

### 2. Install dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e .
python -m spacy download en_core_web_sm
```

### 3. Configure your API key
```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key_here
```

### 4. (Optional) Train the models — or use the baseline
The fine-tuned DistilBERT model is too large for GitHub (~263 MB). Either:
- **Use the baseline only** — works out of the box; the system falls back to TF-IDF + LogReg automatically.
- **Train DistilBERT yourself** — open `notebooks/03_distilbert_finetune.ipynb` in Google Colab, run all cells, download the resulting `models/distilbert-intent/` folder into the project root.

The baseline classifier and spaCy NER model are regenerated locally:
```bash
python scripts/train_baseline.py
python scripts/train_spacy_ner.py
```

### 5. Run everything with one command
```bash
python run.py
```

This starts:
- FastAPI backend on http://localhost:8000 (with `/docs` for OpenAPI)
- Streamlit UI on http://localhost:8501

Press Ctrl+C once to shut down both.

---

## Try these queries in the chat UI

```
hello
order 2 pizzas
what is your return policy
how long does delivery take
do you accept UPI payments
track order ORD123456
cancel order ORD123456
I want to speak to a human
goodbye
```

---

## Sample Conversations

**Place order:**
```
User:  I want to order 2 pizzas
Bot:   Your order for 2 Margherita Pizzas (ORD248162) has been confirmed.
       Total: Rs. 398. Status: confirmed.
```

**Track order:**
```
User:  track order ORD248162
Bot:   Your order ORD248162 was confirmed on May 9, 2026, at 17:21.
```

**FAQ:**
```
User:  what is your return policy
Bot:   We offer a 7-day return policy on most items. Products must be unused
       and in original packaging. Perishable items cannot be returned.
```

---

## Project Structure

```
quickserve-ai/
├── data/                       # Training datasets, FAQ KB, product seed
├── models/                     # Trained models (gitignored — too large)
├── notebooks/                  # Training notebooks (DistilBERT on Colab)
├── scripts/                    # Training and demo scripts
├── src/quickserve/
│   ├── agent/                  # LangGraph state graph + nodes
│   ├── api/                    # FastAPI backend
│   ├── db/                     # SQLAlchemy models
│   ├── llm/                    # Groq client + prompts
│   ├── nlp/                    # Intent / NER / RAG modules
│   └── ui/                     # Streamlit chat UI
├── tests/                      # pytest test suite
├── run.py                      # Single-command launcher
├── pyproject.toml              # Package config
└── README.md
```

---

## Testing

```bash
pytest -v --cov=src --cov-report=term-missing
```

Test summary:
- 22 tests, all passing
- 71% line coverage
- Unit + integration + system levels

---

## Results

### Intent Classification
| Model | Macro-F1 | Training | Size |
|---|---|---|---|
| TF-IDF + LogReg (Baseline) | **0.9471** | <1 sec (CPU) | ~5 MB |
| DistilBERT (Fine-tuned) | 0.8310 | 18 min (Colab T4) | ~263 MB |

### NER
| Entity | F1 |
|---|---|
| ORDER_ID | 0.94 |
| QUANTITY | 0.89 |
| ITEM | 0.78 |

### RAG
- Precision@3 = 0.95 (on 20 manually labeled FAQ queries)

### Latency (50 requests, p95)
- With Groq: 3.2 s
- DB only (greeting/goodbye): 0.9 s

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Acknowledgements

- Hugging Face team for Transformers and the model hub
- spaCy team at Explosion AI
- LangChain / LangGraph team
- Groq Inc. for free-tier LLM inference
- Google Colab for free GPU access
- Bitext for the Customer Support dataset

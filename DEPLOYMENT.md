# Deployment Guide – Hugging Face Spaces (Free Live Demo)

This deploys the chatbot as a publicly accessible URL using Hugging Face Spaces' free tier.

## Why HF Spaces?
- 100% free for CPU Spaces
- Native Streamlit support
- 16 GB storage (enough for the DistilBERT model)
- Public URL: `https://huggingface.co/spaces/<username>/<space-name>`

## Steps

### 1. Create a Hugging Face account
- Sign up at https://huggingface.co/join (use your project email)
- Verify email

### 2. Create a new Space
- Click your avatar → **New Space**
- Owner: your username
- Space name: `quickserve-ai`
- License: MIT
- SDK: **Streamlit**
- Hardware: **CPU basic (free)**
- Visibility: **Public**
- Click **Create Space**

### 3. Add your Groq API key as a secret
In the Space → **Settings** → **Variables and secrets** → **New secret**:
- Name: `GROQ_API_KEY`
- Value: your Groq key (the same one in `.env`)
- Click **Save**

### 4. Configure the Space for Streamlit
Edit the Space's `README.md` header to:
```yaml
---
title: QuickServe AI
emoji: 🛒
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.30.0"
app_file: app.py
pinned: false
---
```

### 5. Push the code
From your local `quickserve-ai` directory:
```bash
git remote add hf https://huggingface.co/spaces/<your-hf-username>/quickserve-ai
git push hf main
```

### 6. Wait for the build (~5–10 minutes)
The Space will install dependencies and start automatically. The live URL will be:
```
https://huggingface.co/spaces/<your-hf-username>/quickserve-ai
```

### 7. Trained models on the Space
DistilBERT (~263 MB) is too large for normal git but can live in the Space using **Git LFS** or by downloading it at startup. Easiest: in `app.py`, add a startup hook that downloads from a HF model repo. For the academic submission the **TF-IDF baseline** (5 MB) works fine on the free Space without any model download — no setup needed.

## Alternative: Streamlit Community Cloud
If HF Spaces is unavailable, Streamlit's own platform (https://share.streamlit.io) works the same way. Connect your GitHub repo, set `GROQ_API_KEY` as a secret, and it deploys.

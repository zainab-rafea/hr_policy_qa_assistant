# Tips Hindawi Challenge (June–July) 2026

> This repository is my official submission for the [Tips Hindawi](https://www.tipshindawi.com/) Challenge (June–July) 2026.

## Participant

| Field            | Value                                |
| ---------------- | ------------------------------------ |
| Full Name        |                                      |
| Project Name     | AI HR Policy Q&A Assistant           |
| GitHub Username  |                                      |
| Challenge Batch  | June–July 2026                       |
| Training Program | Large Language Models (LLMs) Program |
| Organization     | [Edrak for Ai](https://edrak4ai.com/en) |

---

# Project Overview

**AI HR Policy Q&A Assistant** lets employees ask natural-language questions about HR policies (leave, remote work, code of conduct, and local labor law) instead of searching handbooks manually.

It uses **RAG** (retrieval-augmented generation) over markdown policy documents, then returns a **structured answer** via a Pydantic output parser:

| Field | Description |
| ----- | ----------- |
| Answer | Employee-facing response grounded in retrieved policy |
| Source Policy Section | Cited policy + section |
| Confidence Level | High / Medium / Low |
| Escalation Needed | Y / N (Legal review flag) |

---

# Features

* **Free by default** via Google Gemini (free AI Studio API key) or local Ollama
* Semantic retrieval over the employee handbook corpus (FAISS + local embeddings)
* LCEL chain: **Retrieve Relevant Policy → Answer Question → Flag Legal Review**
* Structured output (`HRPolicyAnswer`) with confidence and escalation fields
* Keyword + LLM hybrid escalation for disputes, termination, discrimination, cross-border/tax risk
* Interactive CLI and one-shot `--question` / `--json` modes
* Optional Anthropic Claude backend if you later want a paid cloud model
* Sample policies included under `policies/`

---

# Technologies Used

* Python 3.10+
* LangChain (LCEL chains, community loaders, text splitters)
* **Google Gemini** (`langchain-google-genai`) — free cloud default
* **Ollama** (`langchain-ollama`) — free fully offline option
* Optional Anthropic Claude (`langchain-anthropic`)
* FAISS for vector search
* Hugging Face `sentence-transformers` (`all-MiniLM-L6-v2`) for embeddings
* Pydantic v2 for the output schema
* python-dotenv for configuration

---

# Installation

## 1. Get a free Gemini key (recommended)

1. Open [Google AI Studio](https://aistudio.google.com/apikey)
2. Create an API key (free; typically no credit card)
3. Continue below and put it in `.env`

## 2. Install the Python project

```bash
cd hr-policy-qa-assistant
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

Edit `.env`:

```
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_free_key_here
```

### Optional: fully offline with Ollama

1. Install from [ollama.com/download](https://ollama.com/download)
2. `ollama pull llama3.2:3b`
3. Set `LLM_PROVIDER=ollama` in `.env`

Optional: pre-build the vector index:

```bash
python app.py --index-only
```

---

# Usage

**Interactive mode** (Gemini free)

```bash
python app.py
```

**Single question**

```bash
python app.py -q "How many sick days do I get per year?"
```

**JSON output** (for demos / integrations)

```bash
python app.py -q "Can I work fully remote from another country?" --json
```

**Ollama (offline)**

```bash
python app.py --provider ollama -q "How many sick days do I get?"
```

**Rebuild index** after editing files in `policies/`

```bash
python app.py --rebuild-index -q "What is the gift limit from vendors?"
```

**Optional paid Anthropic backend**

```bash
python app.py --provider anthropic -q "How many sick days do I get?"
```

### Example output

```
============================================================
HR POLICY ANSWER
============================================================

Answer:
  Full-time employees are entitled to 14 paid sick days per year...

Source Policy Section:
  Leave Policy — Section 2: Sick Leave

Confidence Level:     High
Escalation Needed:    N
============================================================
```

### Architecture

```
Employee Question
       │
       ▼
┌──────────────────────┐
│ Retrieve Relevant    │  FAISS over policies/*.md
│ Policy Chunks        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Answer Question      │  Claude + Pydantic structured output
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Flag Legal Review    │  LLM judgment ∪ keyword heuristics
└──────────┬───────────┘
           │
           ▼
   Answer · Source · Confidence · Escalation (Y/N)
```

---

# Demo

Run the CLI with sample questions such as:

1. `How many annual leave days can I carry over?`
2. `What is the home office stipend for remote employees?`
3. `I want to sue for wrongful termination — what does policy say?` → expect **Escalation Needed: Y**

Add screenshots or a short screen recording here after your demo.

---

# Results

* End-to-end RAG pipeline over leave, remote work, conduct, and sample labor-law docs
* Reliable structured responses suitable for HR tooling (parser-friendly fields)
* Safer handling of high-risk questions via an explicit escalation channel

---

# Future Improvements

* Streamlit / web UI for non-technical HR staff
* Citation deep-links to exact handbook page numbers
* Multi-tenant handbooks and role-based policy visibility
* Evaluation harness (faithfulness / escalation precision) on a labeled question set
* Optional OpenAI / local LLM backends

---

# About the Challenge

This project was developed as part of the [Tips Hindawi](https://www.tipshindawi.com/) Challenge (June–July) 2026.

Tips Hindawi is the internships department of [Edrak for Ai](https://edrak4ai.com/en), and the challenge encourages participants to build real-world projects, apply practical skills, and showcase their work through GitHub.

---

# License

This project is shared for educational and portfolio purposes.

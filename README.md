# Citizen Service AI

Multi-Agent Government Service Platform powered by **LangGraph**, **FastAPI**, **PostgreSQL**, and **ChromaDB**.
Supports multiple LLM providers — switch between Claude, OpenAI, Azure OpenAI, and Gemini with a single environment variable.

---

## Architecture

```
User Query
    |
    v
Orchestrator Agent  (intent detection, complexity scoring)
    |
    v
Task Planner Agent  (structured execution plan)
    |
    v  conditional fan-out
    +---> Application Agent  --+
    +---> Billing Agent      --+--> Response Aggregator --> Final Answer
    +---> Knowledge Agent    --+
    +---> Complaint Agent    --+
```

All agents are **provider-agnostic** — they call `get_llm_provider()` and use the abstract interface. No provider SDK leaks into agent code.

---

## LLM Provider Selection

Set one environment variable to switch providers. No code changes required.

| `LLM_PROVIDER` | Provider | Default Model | API Key Required |
|---|---|---|---|
| `mock` | Demo mode (built-in) | `mock-demo-v1` | **None** |
| `claude` (default) | Anthropic Claude | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| `openai` | OpenAI | `gpt-4o` | `OPENAI_API_KEY` |
| `azure` | Azure OpenAI | configurable deployment | `AZURE_OPENAI_API_KEY` |
| `gemini` | Google Gemini | `gemini-2.0-flash` | `GOOGLE_API_KEY` |

### Switch to Claude (default)

```bash
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-your-key
```

### Switch to OpenAI

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key
```

### Switch to Azure OpenAI

```bash
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
```

### Switch to Gemini

```bash
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-key
```

---

## Quick Start

```powershell
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows PowerShell
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (uses mock mode by default — no API key needed)
copy .env.example .env

# 4. Start all services (backend + frontend)
.\start.ps1

# URLs:
#   Frontend  http://localhost:3000
#   Backend   http://localhost:8080
#   Swagger   http://localhost:8080/docs
```

> **No API key needed** — set `LLM_PROVIDER=mock` in `.env` to run a full demo
> with realistic scripted responses. To use a real LLM, change `LLM_PROVIDER`
> to `claude`, `openai`, `azure`, or `gemini` and add the matching API key.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/query` | Full multi-agent query (JSON response) |
| `POST` | `/api/v1/query/stream` | Streaming query (Server-Sent Events) |
| `GET` | `/api/v1/model-info` | Active provider and model |
| `GET` | `/api/v1/session/{id}/history` | Conversation history |
| `DELETE` | `/api/v1/session/{id}` | Clear session memory |
| `GET` | `/api/v1/health` | Health check |

### GET /api/v1/model-info

```json
{
  "provider": "claude",
  "model": "claude-sonnet-4-6",
  "status": "active"
}
```

### POST /api/v1/query/stream (SSE events)

```
data: {"type": "agent_start",    "agent": "orchestrator", "display": "Orchestrator"}
data: {"type": "agent_complete", "agent": "planner",      "display": "Planner"}
data: {"type": "agent_start",    "agent": "application_agent"}
data: {"type": "agent_complete", "agent": "application_agent"}
data: {"type": "token",          "content": "Dear "}
data: {"type": "token",          "content": "Citizen, "}
data: {"type": "done",           "session_id": "abc-123"}
```

---

## Project Structure

```
backend/
├── agents/
│   ├── orchestrator.py          # Level 1 — intent detection
│   ├── planner.py               # Level 2 — execution planning
│   ├── application_agent.py     # Level 3 — permit/application queries
│   ├── billing_agent.py         # Level 3 — payments & receipts
│   ├── knowledge_agent.py       # Level 3 — FAQ & policies (RAG)
│   ├── complaint_agent.py       # Level 3 — complaint handling
│   └── aggregator.py            # Level 4 — response synthesis
├── graph/
│   ├── state.py                 # Shared LangGraph TypedDict state
│   ├── workflow.py              # Graph topology & compilation
│   └── routing.py               # Conditional parallel routing
├── services/
│   ├── llm/                     # Provider-agnostic LLM layer
│   │   ├── base_provider.py     # Abstract base class + LLMResponse
│   │   ├── claude_provider.py   # Anthropic Claude
│   │   ├── openai_provider.py   # OpenAI GPT-4o
│   │   ├── azure_openai_provider.py  # Azure OpenAI
│   │   ├── gemini_provider.py   # Google Gemini
│   │   └── provider_factory.py  # get_llm_provider() factory
│   ├── chroma_service.py        # ChromaDB RAG (provider-agnostic embeddings)
│   └── memory_service.py        # Session conversation memory
├── models/schemas.py            # Pydantic models
├── tools/                       # LangChain tools (application, billing, knowledge, complaint)
├── api/
│   ├── main.py                  # FastAPI app, CORS, lifecycle
│   └── routes.py                # All endpoints including streaming
└── config.py                    # Pydantic settings (all env vars)
```

---

## Note on Embeddings

Anthropic Claude does not provide an embedding API.
When `LLM_PROVIDER=claude`, ChromaDB uses **OpenAI embeddings** as a fallback.
Set `OPENAI_API_KEY` alongside `ANTHROPIC_API_KEY` to enable RAG with Claude.

All other providers (OpenAI, Azure, Gemini) have native embedding support.

---

## Docker Compose

```bash
docker-compose up
# API: http://localhost:8080
# ChromaDB: http://localhost:8000
# PostgreSQL: localhost:5432
```

# Datapizza AI — Learning Guide

##Cheat Sheet
https://niciolass.github.io/datapizza_ai_guide/

## Files in this package

| File | What it covers |
|------|---------------|
| `datapizza_cheatsheet.html` | Full cheat sheet — open in browser |
| `exercises/stage1_client_basics.py` | LLM clients, invoke, memory, streaming |
| `exercises/stage2_tools.py` | @tool decorator, tool_choice, DuckDuckGo |
| `exercises/stage3_agents.py` | Agent, persistent memory, planning, streaming |
| `exercises/stage3b_session_persistence.py` | Save/restore memory to SQLite |
| `exercises/stage4_multiagent.py` | can_call(), multi-agent, tracing |
| `exercises/stage5_rag.py` | Ingestion, DagPipeline, RAG-as-tool, tracing |

## Setup

1. Set your Ollama URL and model at the top of each script:
   ```python
   OLLAMA_BASE_URL = "http://YOUR_OLLAMA_URL/v1"
   OLLAMA_MODEL    = "YOUR_MODEL"   # e.g. qwen3, llama3.2
   ```

2. Install dependencies:
   ```bash
   pip install datapizza-ai
   pip install datapizza-ai-clients-openai-like
   pip install duckduckgo-search        # stages 2, 3, 4, 5
   pip install qdrant-client fastembed  # stage 5 only
   ```

3. Run any stage independently:
   ```bash
   python exercises/stage1_client_basics.py
   python exercises/stage2_tools.py
   # etc.
   ```

## Learning order

Stage 1 → Stage 2 → Stage 3 → Stage 3b → Stage 4 → Stage 5

Each stage builds on the previous. Stage 3b (session persistence) is
optional but recommended before Stage 4 if you're building a chatbot.

## Docs
https://docs.datapizza.ai/0.0.2/

"""
==============================================================
STAGE 3b — Session persistence (save/restore agent memory)
==============================================================
Simulates a chatbot where each conversation is saved to SQLite
and restored on the next session — the ActDoc pattern.
"""

OLLAMA_BASE_URL = "http://YOUR_OLLAMA_URL/v1"
OLLAMA_MODEL    = "YOUR_MODEL"

import json
import sqlite3
from datapizza.agents import Agent
from datapizza.clients.openai_like import OpenAILikeClient
from datapizza.memory import Memory
from datapizza.type import ROLE, TextBlock

client = OpenAILikeClient(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
    api_key="ollama",
)

DB_PATH = "/tmp/datapizza_sessions.db"


# ── Serialization helpers ──────────────────────────────────
def memory_to_dict(memory: Memory) -> list[dict]:
    """Serialize memory turns to a JSON-safe list."""
    turns = []
    for turn in memory.turns:
        blocks = []
        content = turn.content if isinstance(turn.content, list) else [turn.content]
        for block in content:
            if hasattr(block, "text"):
                blocks.append({"type": "text", "text": block.text})
            elif hasattr(block, "input"):
                blocks.append({"type": "tool_use", "name": block.name, "input": block.input})
            elif hasattr(block, "content"):
                blocks.append({"type": "tool_result", "content": str(block.content)})
        turns.append({"role": turn.role.value, "blocks": blocks})
    return turns


def dict_to_memory(turns: list[dict]) -> Memory:
    """Rebuild a Memory object from serialized turns (text blocks only)."""
    memory = Memory()
    for turn in turns:
        role = ROLE(turn["role"])
        blocks = [
            TextBlock(content=b["text"])
            for b in turn["blocks"]
            if b["type"] == "text"
        ]
        if blocks:
            memory.add_turn(blocks[0] if len(blocks) == 1 else blocks, role=role)
    return memory


# ── SQLite persistence ─────────────────────────────────────
def save_session(session_id: str, turns: list[dict]):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, turns TEXT)")
    conn.execute(
        "INSERT OR REPLACE INTO sessions VALUES (?, ?)",
        (session_id, json.dumps(turns)),
    )
    conn.commit()
    conn.close()


def load_session(session_id: str) -> list[dict] | None:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT turns FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


# ── Session-aware agent factory ────────────────────────────
def get_agent(session_id: str) -> Agent:
    stored = load_session(session_id)
    if stored:
        memory = dict_to_memory(stored)
        print(f"  [restored {len(memory.turns)} turns for '{session_id}']")
    else:
        memory = Memory()
        print(f"  [new session '{session_id}']")

    return Agent(
        name="assistant",
        client=client,
        system_prompt="You are a helpful assistant with persistent memory.",
        memory=memory,
    )


def chat(session_id: str, user_message: str) -> str:
    agent = get_agent(session_id)
    response = agent.run(user_message)
    print(f"  Agent: {response.text}")
    save_session(session_id, memory_to_dict(agent.memory))
    return response.text


# ── Simulation ─────────────────────────────────────────────
print("\n=== Session A — first visit ===")
chat("session_a", "My name is Marco and I work as a data engineer.")
chat("session_a", "My favourite framework right now is datapizza-ai.")

print("\n=== Session A — second visit (restored from DB) ===")
chat("session_a", "Do you remember what my job is?")
chat("session_a", "And what framework did I mention?")

print("\n=== Session B — separate user, clean slate ===")
chat("session_b", "Who am I?")

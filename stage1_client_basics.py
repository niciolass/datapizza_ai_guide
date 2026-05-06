"""
==============================================================
STAGE 1 — LLM Client basics with Datapizza AI + Ollama
==============================================================
Install:
    pip install datapizza-ai
    pip install datapizza-ai-clients-openai-like

Set your Ollama endpoint and model below before running.
"""

OLLAMA_BASE_URL = "http://YOUR_OLLAMA_URL/v1"
OLLAMA_MODEL    = "YOUR_MODEL"   # e.g. qwen3, llama3.2

from datapizza.clients.openai_like import OpenAILikeClient

client = OpenAILikeClient(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
    api_key="ollama",
    system_prompt="You are a concise assistant. Answer in one sentence.",
    temperature=0.7,
)

# ─────────────────────────────────────────────
# Exercise 1.1 — First contact
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Exercise 1.1 — First contact")
print("="*50)

response = client.invoke("What is the capital of Italy?")

print("Text:      ", response.text)
print("Prompt tk: ", response.prompt_tokens_used)
print("Output tk: ", response.completion_tokens_used)
print("Cached tk: ", response.cached_tokens_used)
print("Content:   ", response.content)   # raw blocks list


# ─────────────────────────────────────────────
# Exercise 1.2 — Override parameters at call time
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Exercise 1.2 — Override parameters at call time")
print("="*50)

response_creative = client.invoke(
    input="Write a two-sentence poem about Python.",
    temperature=1.2,
    max_tokens=80,
    system_prompt="You are a poet.",
)
print("Creative:", response_creative.text)

response_precise = client.invoke(
    input="Write a two-sentence poem about Python.",
    temperature=0.1,
    max_tokens=80,
    system_prompt="You are a poet.",
)
print("Precise: ", response_precise.text)

print(f"Creative used {response_creative.completion_tokens_used} tokens")
print(f"Precise  used {response_precise.completion_tokens_used} tokens")


# ─────────────────────────────────────────────
# Exercise 1.3 — Multi-turn conversation with Memory
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Exercise 1.3 — Multi-turn conversation with Memory")
print("="*50)

from datapizza.memory import Memory
from datapizza.type import ROLE, TextBlock

memory = Memory()

def chat(user_message: str) -> str:
    response = client.invoke(user_message, memory=memory)
    memory.add_turn(TextBlock(content=user_message), role=ROLE.USER)
    memory.add_turn(response.content, role=ROLE.ASSISTANT)
    return response.text

print(chat("My name is Marco and I work in Milan."))
print(chat("What city do I work in?"))
print(chat("And what's my name?"))

print("\n--- Memory contents ---")
for turn in memory.turns:
    print(f"[{turn.role}] {str(turn.content)[:80]}")


# ─────────────────────────────────────────────
# Exercise 1.4 — Async + streaming
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Exercise 1.4 — Async + streaming")
print("="*50)

import asyncio

async def async_example():
    response = await client.a_invoke(
        input="List 3 benefits of using an AI framework over raw API calls.",
        max_tokens=200,
    )
    print("Async response:", response.text)
    print("Tokens used:", response.completion_tokens_used)

asyncio.run(async_example())

print("\n--- Streaming ---")
for chunk in client.stream_invoke("Count slowly from 1 to 5, one number per line."):
    print(chunk, end="", flush=True)
print()

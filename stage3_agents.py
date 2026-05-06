"""
==============================================================
STAGE 3 — Agents and Memory
==============================================================
Install:
    pip install datapizza-ai
    pip install datapizza-ai-clients-openai-like
    pip install duckduckgo-search

Set your Ollama endpoint and model below before running.
"""

OLLAMA_BASE_URL = "http://YOUR_OLLAMA_URL/v1"
OLLAMA_MODEL    = "YOUR_MODEL"

from datapizza.agents import Agent
from datapizza.clients.openai_like import OpenAILikeClient
from datapizza.tools import tool

client = OpenAILikeClient(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
    api_key="ollama",
)


# ─────────────────────────────────────────────
# Exercise 3.1 — Your first Agent
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Exercise 3.1 — Your first Agent")
print("="*50)

@tool
def get_current_time(timezone: str) -> str:
    """Returns the current time in the given timezone (e.g. 'Europe/Rome')."""
    from datetime import datetime
    import zoneinfo
    try:
        return datetime.now(zoneinfo.ZoneInfo(timezone)).strftime("%H:%M:%S %Z")
    except Exception:
        return f"Unknown timezone: {timezone}"

agent = Agent(
    name="time_agent",
    client=client,
    system_prompt="You are a helpful assistant that answers questions concisely.",
    tools=[get_current_time],
)

response = agent.run("What time is it in Rome and in New York?")
print(response.text)

print("\n--- Agent memory after run ---")
for turn in agent.memory.turns:
    print(f"[{turn.role}]", str(turn.content)[:80])


# ─────────────────────────────────────────────
# Exercise 3.2 — Persistent memory across multiple runs
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Exercise 3.2 — Persistent memory across runs")
print("="*50)

@tool
def save_note(note: str) -> str:
    """Saves a note to memory for later retrieval."""
    return f"Note saved: '{note}'"

@tool
def get_notes() -> str:
    """Retrieves all saved notes."""
    return "Notes: 1) Buy milk  2) Call dentist"

agent2 = Agent(
    name="assistant",
    client=client,
    system_prompt=(
        "You are a personal assistant. "
        "Remember everything the user tells you across the conversation."
    ),
    tools=[save_note, get_notes],
)

turns = [
    "My name is Marco and I work as a data engineer in Milan.",
    "Save a note: review the datapizza-ai docs this week.",
    "What do you know about me so far?",
    "What notes do I have?",
]

for user_message in turns:
    print(f"\nUser: {user_message}")
    response = agent2.run(user_message)
    print(f"Agent: {response.text}")

print(f"\n--- Total memory turns: {len(agent2.memory.turns)} ---")
agent2.memory.reset()
print(f"After reset: {len(agent2.memory.turns)} turns")


# ─────────────────────────────────────────────
# Exercise 3.3 — Streaming + async run
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Exercise 3.3 — Streaming + async run")
print("="*50)

import asyncio

@tool
def calculate(expression: str) -> str:
    """Evaluates a safe mathematical expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"

math_agent = Agent(
    name="math_agent",
    client=client,
    system_prompt="You are a maths tutor. Show your reasoning step by step.",
    tools=[calculate],
)

print("=== Streaming ===")
for chunk in math_agent.stream_invoke(
    "What is the compound interest on €1000 at 5% for 3 years? Show the formula too."
):
    print(chunk, end="", flush=True)
print()

math_agent.memory.reset()

async def ask_agent(question: str) -> str:
    response = await math_agent.a_run(question)
    return response.text

print("\n=== Async run ===")
result = asyncio.run(ask_agent("What is 2 to the power of 10?"))
print(result)


# ─────────────────────────────────────────────
# Exercise 3.4 — Planning agent (milestone)
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Exercise 3.4 — Planning agent (milestone)")
print("="*50)

from datapizza.tools.duckduckgo import DuckDuckGoSearchTool

search = DuckDuckGoSearchTool()

@tool
def summarize(text: str, max_words: int = 50) -> str:
    """Summarizes a text to approximately max_words words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."

@tool
def format_report(title: str, sections: str) -> str:
    """Formats a structured report with a title and sections."""
    return f"# {title}\n\n{sections}"

agent_planner = Agent(
    name="planning_agent",
    client=client,
    system_prompt="You are a methodical research assistant. Plan before you act.",
    tools=[search, summarize, format_report],
    planning_interval=1,
    max_steps=6,
)

task = (
    "Research what datapizza-ai is, summarize it in 40 words, "
    "and format it as a short report titled 'Datapizza AI Overview'."
)

print("=== Planning agent ===")
response = agent_planner.run(task)
print(response.text)

print("\n--- Memory steps taken ---")
for i, turn in enumerate(agent_planner.memory.turns):
    print(f"  [{i}] {turn.role}: {str(turn.content)[:100]}")

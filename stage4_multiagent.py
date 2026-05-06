"""
==============================================================
STAGE 4 — Multi-agent systems
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
from datapizza.tools.duckduckgo import DuckDuckGoSearchTool

client = OpenAILikeClient(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
    api_key="ollama",
)


# ─────────────────────────────────────────────
# Exercise 4.1 — Two agents talking to each other
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Exercise 4.1 — Two agents (manager + specialist)")
print("="*50)

@tool
def calculate(expression: str) -> str:
    """Evaluates a safe mathematical expression."""
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"

math_agent = Agent(
    name="math_specialist",
    client=client,
    system_prompt=(
        "You are a maths specialist. "
        "Only solve mathematical problems using the calculator tool. "
        "Return just the result and a one-line explanation."
    ),
    tools=[calculate],
)

manager = Agent(
    name="manager",
    client=client,
    system_prompt=(
        "You are a helpful assistant. "
        "For any maths problem, delegate to your math_specialist. "
        "For everything else, answer yourself."
    ),
)

manager.can_call([math_agent])

print("Manager tools:", [t.name for t in manager.tools])

questions = [
    "What is the capital of France?",
    "What is 1234 * 5678?",
    "What is 15% of 840, and what city is the Colosseum in?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = manager.run(q)
    print(f"A: {response.text}")
    manager.memory.reset()


# ─────────────────────────────────────────────
# Exercise 4.2 — Three-agent pipeline
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Exercise 4.2 — Three-agent pipeline")
print("="*50)

research_agent = Agent(
    name="research_specialist",
    client=client,
    system_prompt=(
        "You are a research specialist. Search the web for current information "
        "and return a concise factual summary. Max 3 sentences."
    ),
    tools=[DuckDuckGoSearchTool()],
    max_steps=3,
)

@tool
def count_words(text: str) -> str:
    """Counts the words in a text."""
    return f"{len(text.split())} words"

writing_agent = Agent(
    name="writing_specialist",
    client=client,
    system_prompt=(
        "You are a writing specialist. Take raw facts and rewrite them "
        "into polished, engaging prose. Check the word count when done."
    ),
    tools=[count_words],
)

manager2 = Agent(
    name="manager",
    client=client,
    system_prompt=(
        "You are a content manager. "
        "Use research_specialist to gather facts, "
        "then use writing_specialist to turn them into polished text. "
        "Always follow this two-step sequence for content requests."
    ),
    planning_interval=1,
    max_steps=6,
)

manager2.can_call([research_agent, writing_agent])

task = "Write a short paragraph about the current state of open source LLMs."
print(f"Task: {task}\n")
response = manager2.run(task)
print("Final output:")
print(response.text)

print("\n--- Manager memory (agent calls visible) ---")
for turn in manager2.memory.turns:
    print(f"  [{turn.role}]", str(turn.content)[:120])


# ─────────────────────────────────────────────
# Exercise 4.3 — Independent memory per agent
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Exercise 4.3 — Independent memory per agent")
print("="*50)

@tool
def translate(text: str, target_language: str) -> str:
    """Translates text to the target language (mock)."""
    translations = {
        "italian": f"[IT] {text}",
        "french": f"[FR] {text}",
        "spanish": f"[ES] {text}",
    }
    return translations.get(target_language.lower(), f"[??] {text}")

translator_agent = Agent(
    name="translator",
    client=client,
    system_prompt="You are a translation specialist.",
    tools=[translate],
)

manager3 = Agent(
    name="manager",
    client=client,
    system_prompt=(
        "You are a multilingual content manager. "
        "Use your translator specialist for all translation tasks."
    ),
)
manager3.can_call([translator_agent])

turns = [
    "Translate 'Hello, how are you?' into Italian.",
    "Now translate the same phrase into French.",
    "What was the original phrase I asked you to translate?",
]

for msg in turns:
    print(f"\nUser: {msg}")
    response = manager3.run(msg)
    print(f"Manager: {response.text}")

print(f"\nManager memory turns:    {len(manager3.memory.turns)}")
print(f"Translator memory turns: {len(translator_agent.memory.turns)}")

print("\n--- Translator's memory ---")
for turn in translator_agent.memory.turns:
    print(f"  [{turn.role}]", str(turn.content)[:100])


# ─────────────────────────────────────────────
# Exercise 4.4 — Tracing a multi-agent system (milestone)
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Exercise 4.4 — Tracing a multi-agent system (milestone)")
print("="*50)

from datapizza.monitoring import ContextTracing

@tool
def summarize(text: str) -> str:
    """Summarizes a text in one sentence."""
    words = text.split()
    return " ".join(words[:30]) + ("..." if len(words) > 30 else "")

research_agent2 = Agent(
    name="researcher",
    client=client,
    system_prompt="You are a researcher. Search for facts and return them concisely.",
    tools=[DuckDuckGoSearchTool()],
    max_steps=3,
)

summary_agent = Agent(
    name="summarizer",
    client=client,
    system_prompt="You are a summarizer. Condense information into one clear sentence.",
    tools=[summarize],
)

manager4 = Agent(
    name="manager",
    client=client,
    system_prompt=(
        "You are a manager. Research the topic first, then summarize the findings. "
        "Always use both specialists in that order."
    ),
    planning_interval=1,
    max_steps=6,
)
manager4.can_call([research_agent2, summary_agent])

with ContextTracing().trace() as tracer:
    response = manager4.run(
        "What is datapizza-ai and what makes it different from LangChain?"
    )

print("Final answer:")
print(response.text)

summary = tracer.get_summary()
print(f"\n--- Trace summary ---")
print(f"Total spans:       {summary.total_spans}")
print(f"Total duration:    {summary.total_duration_ms:.0f}ms")
print(f"Prompt tokens:     {summary.total_prompt_tokens}")
print(f"Completion tokens: {summary.total_completion_tokens}")

print("\n--- Per-span breakdown ---")
for span in tracer.spans:
    print(f"  [{span.name}] {span.duration_ms:.0f}ms "
          f"| prompt={span.prompt_tokens} completion={span.completion_tokens}")

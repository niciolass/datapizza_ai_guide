"""
==============================================================
STAGE 2 — Tools and the @tool decorator
==============================================================
Install:
    pip install datapizza-ai
    pip install datapizza-ai-clients-openai-like
    pip install duckduckgo-search   # for exercise 2.4

Set your Ollama endpoint and model below before running.
"""

OLLAMA_BASE_URL = "http://YOUR_OLLAMA_URL/v1"
OLLAMA_MODEL    = "YOUR_MODEL"

from datapizza.clients.openai_like import OpenAILikeClient
from datapizza.tools import tool

client = OpenAILikeClient(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
    api_key="ollama",
    system_prompt="You are a helpful assistant. Use tools when needed.",
)


# ─────────────────────────────────────────────
# Exercise 2.1 — Your first tool
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Exercise 2.1 — Your first tool")
print("="*50)

@tool
def get_current_time(timezone: str) -> str:
    """Returns the current time in the given timezone (e.g. 'Europe/Rome')."""
    from datetime import datetime
    import zoneinfo
    try:
        tz = zoneinfo.ZoneInfo(timezone)
        return datetime.now(tz).strftime("%H:%M:%S %Z")
    except Exception:
        return f"Unknown timezone: {timezone}"

print("Tool name:   ", get_current_time.name)
print("Tool schema: ", get_current_time.schema)
print()

response = client.invoke(
    "What time is it in Tokyo right now?",
    tools=[get_current_time],
)
print("Response:", response.text)
print("Content blocks:", response.content)


# ─────────────────────────────────────────────
# Exercise 2.2 — Multiple tools, model chooses
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Exercise 2.2 — Multiple tools, model chooses")
print("="*50)

@tool
def celsius_to_fahrenheit(celsius: float) -> str:
    """Converts a temperature from Celsius to Fahrenheit."""
    f = celsius * 9/5 + 32
    return f"{celsius}°C = {f:.1f}°F"

@tool
def word_count(text: str) -> str:
    """Counts the number of words in a given text."""
    count = len(text.split())
    return f"The text contains {count} words."

tools = [get_current_time, celsius_to_fahrenheit, word_count]

questions = [
    "What time is it in New York?",
    "Convert 37.5 Celsius to Fahrenheit.",
    "How many words are in the sentence: The quick brown fox jumps over the lazy dog?",
    "What time is it in London and what is 100°C in Fahrenheit?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = client.invoke(q, tools=tools)
    print(f"A: {response.text}")


# ─────────────────────────────────────────────
# Exercise 2.3 — Controlling tool_choice
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Exercise 2.3 — Controlling tool_choice")
print("="*50)

@tool
def calculator(expression: str) -> str:
    """Evaluates a safe mathematical expression like '2 + 2' or '10 * 3.5'."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def get_joke() -> str:
    """Returns a programming joke."""
    return "Why do programmers prefer dark mode? Because light attracts bugs!"

tools2 = [calculator, get_joke]
question = "What is 144 divided by 12?"

r = client.invoke(question, tools=tools2, tool_choice="auto")
print("auto:          ", r.text)

r = client.invoke(question, tools=tools2, tool_choice="required")
print("required:      ", r.text)

r = client.invoke(question, tools=tools2, tool_choice="none")
print("none:          ", r.text)

r = client.invoke("Tell me a joke", tools=tools2, tool_choice=["calculator"])
print("forced calc:   ", r.text)

r = client.invoke(question, tools=tools2, tool_choice="required_first")
print("required_first:", r.text)


# ─────────────────────────────────────────────
# Exercise 2.4 — Built-in DuckDuckGoSearchTool (milestone)
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Exercise 2.4 — DuckDuckGoSearchTool (milestone)")
print("="*50)

from datapizza.tools.duckduckgo import DuckDuckGoSearchTool
from datapizza.type import ToolUseBlock

search_client = OpenAILikeClient(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
    api_key="ollama",
    system_prompt=(
        "You are a research assistant. "
        "Always search for current information before answering questions about recent events."
    ),
)

search_tool = DuckDuckGoSearchTool()

@tool
def summarize_in_one_sentence(text: str) -> str:
    """Condenses a long text into exactly one sentence."""
    return text[:200] + "..." if len(text) > 200 else text

search_tools = [search_tool, summarize_in_one_sentence]

questions = [
    "What is the latest version of Python?",
    "Who won the most recent FIFA World Cup?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = search_client.invoke(q, tools=search_tools, tool_choice="auto")
    print(f"A: {response.text}")
    tool_calls = [b for b in response.content if isinstance(b, ToolUseBlock)]
    for tc in tool_calls:
        print(f"   → called '{tc.name}' with: {tc.input}")

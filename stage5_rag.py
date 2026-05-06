"""
==============================================================
STAGE 5 — RAG pipelines and observability
==============================================================
Install:
    pip install datapizza-ai
    pip install datapizza-ai-clients-openai-like
    pip install qdrant-client
    pip install fastembed
    pip install duckduckgo-search

Set your Ollama endpoint and model below before running.
NOTE: Qdrant runs in-memory — no server needed.
"""

OLLAMA_BASE_URL = "http://YOUR_OLLAMA_URL/v1"
OLLAMA_MODEL    = "YOUR_MODEL"

from datapizza.clients.openai_like import OpenAILikeClient

client = OpenAILikeClient(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL,
    api_key="ollama",
)


# ─────────────────────────────────────────────
# Exercise 5.1 — Ingestion pipeline
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Exercise 5.1 — Ingestion pipeline")
print("="*50)

from datapizza.pipeline import IngestionPipeline
from datapizza.modules.parsers.text import TextParser
from datapizza.modules.splitters import RecursiveSplitter
from datapizza.embedders import ChunkEmbedder
from datapizza.embedders.fast import FastEmbedder
from datapizza.vectorstores.qdrant import QdrantVectorstore
from datapizza.core.vectorstore import VectorConfig

vectorstore = QdrantVectorstore(
    config=VectorConfig(
        collection_name="my_docs",
        size=384,
        in_memory=True,
    )
)

embedder = FastEmbedder()

pipeline = IngestionPipeline(
    parser=TextParser(),
    splitter=RecursiveSplitter(chunk_size=300, chunk_overlap=30),
    embedder=ChunkEmbedder(embedder=embedder),
    vectorstore=vectorstore,
)

documents = [
    {
        "text": (
            "Datapizza-ai is a Python framework for building AI agents and RAG systems. "
            "It provides a thin layer above native LLM SDKs, prioritising transparency "
            "and control over heavy abstraction. It supports OpenAI, Anthropic, Google, "
            "Mistral, and any OpenAI-compatible endpoint like Ollama."
        ),
        "metadata": {"source": "overview", "topic": "framework"},
    },
    {
        "text": (
            "The Agent class in datapizza-ai manages tool execution, memory, and planning. "
            "Agents can call other agents via can_call(), enabling multi-agent orchestration. "
            "Memory persists across .run() calls and can be injected at construction time "
            "to restore previous sessions."
        ),
        "metadata": {"source": "agents_guide", "topic": "agents"},
    },
    {
        "text": (
            "RAG systems in datapizza-ai use two pipelines: IngestionPipeline for processing "
            "and storing documents, and DagPipeline for retrieval and generation. "
            "The DagPipeline is a Directed Acyclic Graph where each node is a processing step "
            "and edges define how data flows between steps."
        ),
        "metadata": {"source": "rag_guide", "topic": "rag"},
    },
]

for doc in documents:
    pipeline.run(text=doc["text"], metadata=doc["metadata"])

print(f"Ingested {len(documents)} documents.")

query_embedding = embedder.embed("What is datapizza-ai?")
results = vectorstore.search(query_embedding, top_k=2)
for r in results:
    print(f"\nScore: {r.score:.3f}")
    print(f"Text:  {r.chunk.text[:100]}...")


# ─────────────────────────────────────────────
# Exercise 5.2 — Retrieval pipeline (DagPipeline)
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Exercise 5.2 — Retrieval pipeline (DagPipeline)")
print("="*50)

from datapizza.pipeline import DagPipeline
from datapizza.pipeline.dag import Connection
from datapizza.core.models import PipelineComponent


class RetrieverNode(PipelineComponent):
    def __init__(self, vectorstore, embedder, top_k=3):
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.top_k = top_k

    def _run(self, query: str) -> dict:
        embedding = self.embedder.embed(query)
        results = self.vectorstore.search(embedding, top_k=self.top_k)
        context = "\n\n".join(r.chunk.text for r in results)
        return {"query": query, "context": context}

    async def _a_run(self, query: str) -> dict:
        return self._run(query)


class GeneratorNode(PipelineComponent):
    def __init__(self, client):
        self.client = client

    def _run(self, query: str, context: str) -> str:
        prompt = (
            f"Answer the question using ONLY the context provided.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
        return self.client.invoke(prompt).text

    async def _a_run(self, query: str, context: str) -> str:
        response = await self.client.a_invoke(
            f"Answer the question using ONLY the context provided.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
        return response.text


retriever = RetrieverNode(vectorstore, embedder)
generator = GeneratorNode(client)

dag = DagPipeline(
    modules={"retriever": retriever, "generator": generator},
    connections=[
        Connection(from_node_name="retriever", to_node_name="generator",
                   source_key="query", target_key="query"),
        Connection(from_node_name="retriever", to_node_name="generator",
                   source_key="context", target_key="context"),
    ],
    entry_node="retriever",
    output_node="generator",
)

questions = [
    "What LLM providers does datapizza-ai support?",
    "How does multi-agent communication work?",
    "What are the two pipeline types used in RAG?",
]

for q in questions:
    print(f"\nQ: {q}")
    answer = dag.run(query=q)
    print(f"A: {answer}")


# ─────────────────────────────────────────────
# Exercise 5.3 — RAG as an agent tool
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Exercise 5.3 — RAG as an agent tool")
print("="*50)

from datapizza.agents import Agent
from datapizza.tools import tool

@tool
def search_documentation(query: str) -> str:
    """
    Searches the datapizza-ai documentation and returns relevant information.
    Use this whenever the user asks about datapizza-ai features, concepts, or APIs.
    """
    return dag.run(query=query)

@tool
def get_current_date() -> str:
    """Returns today's date."""
    from datetime import date
    return str(date.today())

rag_agent = Agent(
    name="docs_assistant",
    client=client,
    system_prompt=(
        "You are a helpful assistant for the datapizza-ai framework. "
        "Use search_documentation for any questions about the framework. "
        "Answer general questions from your own knowledge."
    ),
    tools=[search_documentation, get_current_date],
)

questions = [
    "What is datapizza-ai and what makes it different?",
    "What is today's date?",
    "How do agents communicate with each other in datapizza-ai?",
    "What is Python?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = rag_agent.run(q)
    print(f"A: {response.text}")
    rag_agent.memory.reset()


# ─────────────────────────────────────────────
# Exercise 5.4 — Full traced RAG system (milestone)
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("Exercise 5.4 — Full traced RAG system (milestone)")
print("="*50)

from datapizza.monitoring import ContextTracing

traced_agent = Agent(
    name="traced_rag_agent",
    client=client,
    system_prompt=(
        "You are a datapizza-ai documentation assistant. "
        "Always search the docs before answering framework questions."
    ),
    tools=[search_documentation],
    planning_interval=1,
)

questions = [
    "What is the IngestionPipeline used for?",
    "And how does the DagPipeline differ from it?",
    "Can you summarise both in one sentence each?",
]

all_spans = []

for q in questions:
    print(f"\nQ: {q}")
    with ContextTracing().trace() as tracer:
        response = traced_agent.run(q)
    print(f"A: {response.text}")

    summary = tracer.get_summary()
    print(f"   ↳ {summary.total_spans} spans | "
          f"{summary.total_duration_ms:.0f}ms | "
          f"{summary.total_prompt_tokens}pt + {summary.total_completion_tokens}ct tokens")
    all_spans.extend(tracer.spans)

total_prompt = sum(s.prompt_tokens for s in all_spans)
total_completion = sum(s.completion_tokens for s in all_spans)
total_ms = sum(s.duration_ms for s in all_spans)

print(f"\n=== Session totals ===")
print(f"Total LLM calls:      {len(all_spans)}")
print(f"Total prompt tokens:  {total_prompt}")
print(f"Total completion tok: {total_completion}")
print(f"Total time:           {total_ms:.0f}ms")
print(f"Avg per call:         {total_ms/len(all_spans):.0f}ms")

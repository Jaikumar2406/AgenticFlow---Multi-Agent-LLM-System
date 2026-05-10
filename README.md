# AgenticFlow - Multi-Agent LLM System

A production-grade multi-agent orchestration system with LangGraph, dynamic routing, tool orchestration, self-improving evaluation, and adversarial robustness testing.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              AGENTICFLOW SYSTEM                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌─────────────┐      ┌─────────────────────────────────────────────────────────┐  │
│  │   CLIENT    │─────▶│                    API SERVER                             │  │
│  │  (SSE)      │◀─────│               (FastAPI + Uvicorn)                          │  │
│  └─────────────┘      └─────────────────────────────────────────────────────────┘  │
│                               │                                                      │
│                               ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                    LANGGRAPH ORCHESTRATION LAYER                            │   │
│  │                                                                           │   │
│  │  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐  │   │
│  │  │Orchestrator│───▶│ Decomposition│───▶│  Retrieval  │───▶│ Critique │  │   │
│  │  │   Agent    │    │    Agent     │    │   Agent     │    │  Agent   │  │   │
│  │  └─────────────┘    └──────────────┘    └─────────────┘    └──────────┘  │   │
│  │        │                   │                   │                  │       │   │
│  │        │                   │                   │                  ▼       │   │
│  │        │                   │                   │           ┌─────────────┐ │   │
│  │        │                   │                   │           │  Synthesis  │ │   │
│  │        │                   │                   │           │   Agent    │ │   │
│  │        ▼                   ▼                   ▼           └─────────────┘ │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐   │   │
│  │  │              STATE MANAGEMENT (AgentState)                           │   │   │
│  │  │  - user_query, routing_plan, retrieval_results, critique_results    │   │   │
│  │  │  - synthesis_output, provenance, tool_outputs                      │   │   │
│  │  └─────────────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                               │                                                      │
│                               ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                         TOOL LAYER                                          │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌─────────────┐  ┌────────────────┐    │   │
│  │  │Web Search  │  │Code Executor │  │Data Lookup  │  │Self Reflection │    │   │
│  │  │ (Tavily)   │  │  (Python)    │  │  (SQL)      │  │   (LLM)        │    │   │
│  │  └────────────┘  └──────────────┘  └─────────────┘  └────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                               │                                                      │
│                               ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                       DATA LAYER                                            │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌─────────────────────────────────┐  │   │
│  │  │   SQLite DB   │  │   LLM Client  │  │         Evaluation Pipeline     │  │   │
│  │  │  (Jobs, Logs) │  │    (Groq)     │  │    (15 test cases, scoring)     │  │   │
│  │  └────────────────┘  └────────────────┘  └─────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
agenticflow/
├── .env                        # Environment variables (API keys, configs)
├── .gitignore                  # Git ignore patterns
├── requirements.txt            # Python dependencies
│
├── config/                     # Configuration module
│   ├── __init__.py
│   ├── settings.py             # Settings class (loads from .env)
│   └── loki-config.yml         # Log aggregation config
│
├── src/                        # Main source code
│   ├── __init__.py
│   │
│   ├── agents/                 # Multi-agent system
│   │   ├── __init__.py
│   │   ├── base.py             # BaseAgent class
│   │   ├── graph.py            # LangGraph state machine (MAIN ORCHESTRATION)
│   │   ├── orchestrator.py    # Orchestrator agent (routing logic)
│   │   ├── decomposition.py   # Decomposition agent (task breakdown)
│   │   ├── retrieval.py       # Retrieval agent (RAG)
│   │   ├── critique.py        # Critique agent (confidence scoring)
│   │   ├── synthesis.py       # Synthesis agent (final answer)
│   │   ├── compression.py     # Context compression agent
│   │   └── registry.py       # Agent registry
│   │
│   ├── api/                   # FastAPI server
│   │   ├── __init__.py
│   │   └── main.py            # API endpoints & SSE streaming
│   │
│   ├── db/                    # Database layer
│   │   ├── __init__.py
│   │   ├── database.py        # SQLite/PostgreSQL database
│   │   └── models.py         # Pydantic data models
│   │
│   ├── tools/                 # Tool implementations
│   │   ├── __init__.py
│   │   ├── base.py            # BaseTool abstract class
│   │   ├── web_search.py      # Web search (Tavily API)
│   │   ├── code_execution.py # Python code sandbox
│   │   ├── data_lookup.py    # Natural language to SQL
│   │   ├── self_reflection.py # Contradiction detection
│   │   └── registry.py       # Tool registry with fallback
│   │
│   ├── utils/                 # Utilities
│   │   ├── __init__.py
│   │   ├── llm_client.py     # Groq LLM client wrapper
│   │   └── pipeline.py      # Pipeline runner
│   │
│   ├── eval/                  # Evaluation system
│   │   ├── __init__.py
│   │   ├── pipeline.py       # 15 test cases evaluation
│   │   └── meta.py          # Meta-agent for self-improvement
│   │
│   └── worker/               # Background tasks
│       ├── __init__.py
│       └── tasks.py         # Celery tasks (optional)
│
└── tests/                     # Test suite
    ├── __init__.py
    └── test_agents.py        # Agent tests
```

## Detailed File Explanations

### Root Files

| File | Description |
|------|-------------|
| `.env` | Environment configuration - contains API keys (GROQ, TAVILY), database URL, context budgets, timeouts |
| `.gitignore` | Git ignore patterns - excludes `.env`, `__pycache__`, `.pyc`, virtual environments |
| `requirements.txt` | Python dependencies - FastAPI, Uvicorn, LangGraph, Groq, asyncpg, SQLite |

### config/

| File | Description |
|------|-------------|
| `settings.py` | Settings class - loads all environment variables with defaults. Manages DATABASE_URL, REDIS_URL, GROQ_API_KEY, TAVILY_API_KEY, context budgets, timeouts |

### src/agents/ - Multi-Agent System

| File | Description |
|------|-------------|
| `base.py` | `BaseAgent` abstract class - defines agent interface with `process()`, budget checking, token consumption, logging |
| `graph.py` | **MAIN ORCHESTRATION** - LangGraph state machine with 5 nodes: orchestrator, decomposition, retrieval, critique, synthesis. Implements dynamic routing based on LLM-driven intent classification |
| `orchestrator.py` | Orchestrator agent - determines which sub-agents to invoke via structured reasoning |
| `decomposition.py` | Decomposition agent - breaks ambiguous queries into typed sub-tasks with dependency graphs |
| `retrieval.py` | Retrieval agent - performs multi-hop RAG across at least 2 chunks before forming answer |
| `critique.py` | Critique agent - reviews outputs, assigns confidence scores per claim, flags specific disagreements |
| `synthesis.py` | Synthesis agent - merges outputs from all agents, resolves contradictions, produces final answer with provenance |
| `compression.py` | Compression agent - compresses context when budget exceeded, preserves tool outputs/scores/citations lossless |
| `registry.py` | Agent registry - manages all agents, provides lookup by name |

### src/api/ - FastAPI Server

| File | Description |
|------|-------------|
| `main.py` | FastAPI application with 5 endpoints: `/query` (SSE), `/trace/{job_id}`, `/eval/summary`, `/prompt/approve`, `/eval/retry` |

### src/db/ - Database Layer

| File | Description |
|------|-------------|
| `database.py` | Database class - supports both SQLite (fallback) and PostgreSQL. Tables: jobs, execution_logs, eval_runs, test_cases, prompt_rewrites, tool_calls |
| `models.py` | Pydantic models - AgentContext, DecomposedTask, TaskDependencyGraph, RoutingDecision, ToolCall, ToolResult, CritiqueResult, ProvenanceMap, ContextBudget, ExecutionEvent, EvalResult, PromptRewrite, EvalRun |

### src/tools/ - Tool System

| File | Description |
|------|-------------|
| `base.py` | `BaseTool` abstract class with failure contracts (timeout, empty results, malformed input) |
| `web_search.py` | Web search tool using Tavily API - returns structured results with URLs, titles, snippets, relevance scores |
| `code_execution.py` | Python code sandbox - executes Python snippets, returns stdout, stderr, exit code |
| `data_lookup.py` | Natural language to SQL - queries mock database (employees, products) with NL input |
| `self_reflection.py` | Self-reflection tool - identifies contradictions in agent outputs |
| `registry.py` | Tool registry - manages all tools with fallback logic (max 2 retries per tool) |

### src/utils/ - Utilities

| File | Description |
|------|-------------|
| `llm_client.py` | Groq LLM client wrapper - `generate()` and `generate_stream()` methods |
| `pipeline.py` | Pipeline runner - executes LangGraph pipeline, logs events |

### src/eval/ - Evaluation System

| File | Description |
|------|-------------|
| `pipeline.py` | Evaluation pipeline - 15 test cases (5 baseline, 5 ambiguous, 5 adversarial), 6 scoring dimensions |
| `meta.py` | Meta-agent - analyzes failures, identifies worst prompts, proposes rewrites with diffs |

### src/worker/ - Background Tasks

| File | Description |
|------|-------------|
| `tasks.py` | Celery tasks - `process_query`, `run_evaluation`, `re_eval_failed_cases` |

## How It Works

### 1. Query Flow
```
User Query → API /query endpoint → LangGraph Pipeline → SSE Response
```

### 2. Orchestration (LangGraph)
```
1. Orchestrator Agent
   - LLM classifies query intent
   - Determines which agents needed
   - Sets routing_plan["order"]

2. Based on routing, executes:
   - decomposition (if needed)
   - retrieval (if factual)
   - critique (if validation needed)
   - synthesis (always)

3. State passed between nodes via AgentState
```

### 3. Tool Execution
```
Retrieval Agent → Tool Registry → Web Search / Data Lookup
                                    ↓
                              ToolResult with fallback
                                    ↓
                              Normalized evidence
```

### 4. Evaluation Pipeline
```
15 test cases run through pipeline
  ↓
6 scoring dimensions:
  - answer_correctness
  - citation_accuracy
  - contradiction_resolution
  - tool_selection_efficiency
  - context_budget_compliance
  - critique_agreement_rate
  ↓
Meta-agent analyzes failures
  ↓
Proposes prompt rewrites (stored, not auto-applied)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Submit query, receive SSE streaming response |
| `/trace/{job_id}` | GET | Get full execution trace for a job |
| `/eval/summary` | GET | Get latest evaluation run summary |
| `/prompt/approve` | POST | Approve/reject prompt rewrite |
| `/eval/retry` | POST | Trigger targeted re-evaluation |

## Setup Instructions

### Prerequisites
- Python 3.11+
- Groq API key (get from https://console.groq.com/)
- Tavily API key (optional, for real web search)

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
# Edit .env with your API keys
```

### Running
```bash
# Start the API server
uvicorn src.api.main:app --reload --port 8000
```

### Testing
```bash
# Test the API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Python?"}'
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection | sqlite:///agenticflow.db |
| `GROQ_API_KEY` | Groq API key | Required |
| `GROQ_MODEL` | Groq model name | llama-3.3-70b-versatile |
| `TAVILY_API_KEY` | Tavily API key | Optional |
| `DEFAULT_CONTEXT_BUDGET` | Max tokens per agent | 100000 |
| `MAX_CONTEXT_BUDGET` | Maximum context size | 200000 |
| `CODE_EXECUTION_TIMEOUT` | Code sandbox timeout (s) | 30 |
| `WEB_SEARCH_TIMEOUT` | Web search timeout (s) | 10 |

## Agent Descriptions

### Orchestrator Agent
- **Role**: Dynamic routing decision engine
- **Decision Boundaries**: Determines which sub-agents to invoke based on query complexity and intent
- **Input**: User query, current state
- **Output**: Routing decision with context budget allocation

### Decomposition Agent
- **Role**: Task breakdown with dependency graphs
- **Decision Boundaries**: Identifies sub-tasks and their dependencies for complex queries
- **Input**: User query
- **Output**: List of typed sub-tasks with dependencies

### Retrieval Agent
- **Role**: Multi-hop RAG with citation
- **Decision Boundaries**: Performs at least 2-hop reasoning across retrieved chunks
- **Input**: User query, decomposed tasks
- **Output**: Retrieved information with source citations

### Critique Agent
- **Role**: Confidence scoring and disagreement flagging
- **Decision Boundaries**: Assigns per-claim confidence scores, flags specific spans
- **Input**: Outputs from retrieval agent
- **Output**: Confidence scores, disagreement spans

### Synthesis Agent
- **Role**: Output merging and contradiction resolution
- **Decision Boundaries**: Resolves contradictions internally, produces final answer
- **Input**: All agent outputs
- **Output**: Final answer with provenance map

## Known Limitations

1. **Web Search**: Uses mock data when Tavily API key not configured
2. **Code Execution**: Limited sandbox (not production-secure)
3. **Database**: SQLite for local dev, PostgreSQL for production
4. **Self-Improvement**: Prompt rewrites stored but require manual approval

## What Would Be Built Next

1. **Real LLM Integration**: Connect to actual Groq API for all agent processing
2. **Enhanced Tooling**: Real web search, better code sandbox
3. **More Test Cases**: Expand adversarial test suite
4. **Observability Dashboard**: Better visualization of agent decisions
5. **Feedback Loop**: Human feedback integration into evaluation
6. **Performance Optimization**: Better context compression algorithms

## License

MIT
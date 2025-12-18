# minimal-agent

A minimal, isolated example of an LLM agent built with the official `openai` Python SDK, using **Tool Calling** plus a small wrapper that turns ordinary Python callables into OpenAI tool schemas.

## What’s inside

- `Agent` loop with tool execution: `src/agent/core.py`
- Tool wrapper (`tool(...)` / `@tool`) that generates JSON Schema from type hints: `src/agent/tool.py`
- Pluggable session storage (message history): `src/agent/session.py`
- Pluggable usage storage (token accounting): `src/agent/usage_store.py`
- Example wiring + demo tools: `src/main.py`

## Requirements

- Python `>= 3.13`
- `uv`
- An OpenAI-compatible API key (or a compatible provider via `OPENAI_BASE_URL`)

## Install (uv)

```bash
uv sync
```

## Configure

Create `.env` (or export env vars). Starting point:

```bash
cp .env.example .env
```

Supported settings (see `src/config.py`):

- `OPENAI_API_KEY` (required)
- `OPENAI_BASE_URL` (optional, default: `https://api.openai.com/v1`)
- `AGENT_MODEL` (optional, default: `gpt-4o-mini`)
- `AGENT_MAX_ITERATIONS` (optional, default: `20`)

## Run

```bash
uv run python src/main.py
```

## Using tools

Wrap any sync/async function (or bound method) and pass it to the agent:

```py
from agent.tool import tool

@tool
async def get_weather(city: str) -> dict:
    return {"city": city, "weather": "sunny"}
```

The wrapper:

- infers the tool JSON Schema from the callable’s type hints
- uses the docstring as the tool description
- supports both sync and async functions

## Session and usage storage

The agent is storage-agnostic:

- implement `Session` to persist/load messages (in-memory, DB, Redis, etc.)
- implement `UsageStore` to capture `CompletionUsage` per model call

An in-memory example is included in `src/main.py`.

## How it works (high level)

- Sends a `system` + `user` message to the model
- Calls `client.chat.completions.create(...)` with `tools=...` and `tool_choice="required"`
- The model must always respond with tool calls; direct “assistant text” is not used
- Executes tool calls and appends `role="tool"` messages back into the session
- Stops when the model calls the built-in `final_response(reasoning, answer)` tool (customizable)

## Notes / limitations

- Tool names must be unique (the agent keys tools by function `__name__`).
- Tool results are currently added to the conversation as `str(result)`; if you need strict JSON, serialize explicitly in your tool implementation.
- The agent returns a `dict` (by default: `{"reasoning": ..., "answer": ...}`) produced by `final_response`.

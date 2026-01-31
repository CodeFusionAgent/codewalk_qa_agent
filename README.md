# Codewalk Q&A Agent (Purple Agent)

A baseline Purple Agent that answers technical questions about open-source codebases. Built for the [AgentBeats](https://agentbeats.dev) platform using the [A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/) protocol.

**Registered Agent:** [codewalk-qa-agent on AgentBeats](https://agentbeats.dev/anamsarfraz/codewalk-qa-agent)

## Overview

This agent serves as the baseline Q&A agent for the Codewalk benchmark. When given a question about a codebase, it either:

1. **Returns a pre-computed answer** from YAML knowledge files (for benchmark consistency)
2. **Generates an answer** using an LLM if no pre-computed answer exists

This design enables reproducible benchmarking while allowing the agent to handle arbitrary questions.

## How It Works

```
┌─────────────────┐                        ┌─────────────────┐
│   Eval Agent    │     Question           │    Q&A Agent    │
│  (Green Agent)  │ ──────────────────────▶│ (Purple Agent)  │
│                 │                        │                 │
│                 │◀────────────────────── │  1. Check YAML  │
│                 │     Answer             │  2. Or use LLM  │
└─────────────────┘                        └─────────────────┘
```

## Project Structure

```
src/
├─ server.py      # Server setup and agent card configuration
├─ executor.py    # A2A request handling and task lifecycle
└─ agent.py       # Q&A logic: YAML lookup + LLM fallback
data/
├─ fastapi_qa.yaml  # Pre-computed answers for FastAPI questions
└─ django_qa.yaml   # Pre-computed answers for Django questions
tests/
└─ test_agent.py  # A2A conformance tests
Dockerfile        # Docker configuration
pyproject.toml    # Python dependencies
```

## Supported Models

| Model | Provider | Base URL |
|-------|----------|----------|
| `gemini-2.5-flash` (default) | Google | generativelanguage.googleapis.com |
| `gemini-2.0-flash` | Google | generativelanguage.googleapis.com |
| `gpt-4o` | OpenAI | api.openai.com |
| `gpt-4o-mini` | OpenAI | api.openai.com |
| `claude-sonnet-4-5` | Anthropic | api.anthropic.com |

All models use the OpenAI SDK with compatible endpoints.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QA_API_KEY` | API key for the LLM provider | Required |
| `QA_MODEL` | Model to use for answer generation | `gemini-2.5-flash` |
| `QA_DATA_DIR` | Directory containing YAML Q&A files | `data` |

## Running Locally

```bash
# Install dependencies
uv sync

# Set API key
export QA_API_KEY="your-api-key"

# Run the server
uv run src/server.py
```

## Running with Docker

```bash
# Build the image
docker build -t codewalk-qa-agent .

# Run the container
docker run -p 9010:9010 \
  -e QA_API_KEY="your-key" \
  -e QA_MODEL="gemini-2.5-flash" \
  codewalk-qa-agent
```

## Testing

```bash
# Install test dependencies
uv sync --extra test

# Start your agent (see above)

# Run A2A conformance tests
uv run pytest --agent-url http://localhost:9010
```

## YAML Knowledge Format

Questions and answers are stored in YAML files under `data/`:

```yaml
questions:
  - question: "How does request processing work in FastAPI?"
    reference_answer: "Expert reference for evaluation..."
    claude_code_answer: "Answer from Claude Code..."
    codefusion_answer: "Answer from CodeFusion..."
```

The agent looks for any key ending with `_answer` (except `reference_answer`) and returns the first match. This allows multiple answer variants for comparison.

## Adding New Questions

1. Create or edit a YAML file in `data/`
2. Add questions with at least one `*_answer` field
3. Rebuild the Docker image to include the new data

## Publishing

Push to `main` to publish `latest` tag, or create a version tag (e.g., `v1.0.0`) for versioned releases:

```
ghcr.io/<your-username>/codewalk_qa_agent:latest
ghcr.io/<your-username>/codewalk_qa_agent:1.0.0
```

## Agent Registration

To register your own Purple Agent on AgentBeats:

1. Follow the [AgentBeats Tutorial](https://docs.agentbeats.dev/tutorial/) for step-by-step setup
2. Build and publish your Docker image to a container registry
3. Register the agent on the [AgentBeats platform](https://agentbeats.dev)

## Related Repositories

- [codewalk_eval_agent](https://github.com/CodeFusionAgent/codewalk_eval_agent) - Green Agent that evaluates Q&A responses
- [leaderboard_codewalk](https://github.com/CodeFusionAgent/leaderboard_codewalk) - Leaderboard and evaluation scenarios

## License

MIT

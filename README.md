# Knowledge Compass Agent

Local-first learning-path navigation with an evolving multi-agent architecture.

## Overview

Knowledge Compass Agent is an experimental project that explores how a learning assistant can be turned into a more structured, controllable agent system.

The goal is to help a user:

- understand their current learning state
- decide the next concrete step
- preserve useful long-term context such as progress, preferences, and plans
- gradually evolve from a single-agent workflow into a multi-agent architecture

The project emphasizes controllability over pure prompt magic by separating responsibilities across planning, memory, knowledge, tools, and model routing.

## Core Idea

The system is designed around a layered architecture:

`Agent Kernel = Policy + Memory + Knowledge + Tools + LLM Router + Orchestrator`

Current and planned layers:

- Interface Layer: CLI today, API later
- Orchestrator Layer: task routing and multi-agent coordination
- Agent Layer: agent loop and role execution
- Policy / Planning Layer: decision rules and next-step planning
- Memory / Knowledge / Skills Layer: user state, references, reusable capabilities
- Tool Execution Layer: controlled file and tool access
- LLM Adapter Layer: interchangeable model providers

## Current Status

Status: early MVP, actively evolving.

Implemented:

- single-session CLI interaction
- LLM adapter abstraction
- basic tool-based state persistence
- early multi-agent orchestration skeleton
- role-oriented task routing

In progress or planned:

- stronger memory management
- dedicated user profile management
- local knowledge retrieval
- web search integration
- better multi-agent execution and coordination
- guided initialization and configuration flow

## Why This Project

Many learning assistants are good at conversation but weak at continuity and structure. This project explores a different direction:

- separate routing, memory, profile, knowledge, and interaction concerns
- make behavior easier to inspect and evolve
- keep the system flexible enough to mix different models for different roles
- move important control logic from prompt-only behavior into code

## Project Structure

```text
.
├── core/          # agent runtime, router, orchestration
├── knowledge/     # local knowledge and retrieval-related modules
├── llm/           # model adapters and selection logic
├── memory/        # schemas and long-term state artifacts
├── prompts/       # system and planning prompts
├── tools/         # tool definitions
├── data/          # runtime data such as worklogs
├── runtime/       # runtime cache / logs
└── main.py        # CLI entrypoint
```

## Quick Start

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a local environment file:

```bash
cp .env.example .env
```

4. Configure your model provider and API key in `.env`:

Minimal example:

```env
LLM_PROVIDER=siliconflow
MODEL_NAME=deepseek-ai/DeepSeek-V3
SILICONFLOW_API_KEY=your_api_key_here
```

You can also override model/provider settings per agent role, for example:

```env
ROUTER_LLM_PROVIDER=glm
ROUTER_MODEL_NAME=glm-4-flash

INTERACTION_LLM_PROVIDER=siliconflow
INTERACTION_MODEL_NAME=deepseek-ai/DeepSeek-V3
```

5. Start the CLI:

```bash
python3 main.py
```

6. Begin chatting with the agent in the terminal.

## Design Principles

- Prompts are externalized instead of hardcoded into a single giant string
- Agents should access files through explicit tools or dedicated managers
- Long-term memory, profile data, and knowledge should be separated by responsibility
- Routing and orchestration should be implemented in code, not only implied in prompts
- The system should remain configurable so different roles can use different models

## Roadmap

- stabilize the first multi-agent runtime loop
- improve the router and task protocol
- add real web search support
- expand local knowledge management
- support richer memory/profile update rules
- add initialization and setup guidance for new users
- introduce stronger observability and testing

## License

MIT

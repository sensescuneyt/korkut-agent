# Korkut — ReAct Agent with Local LLM

Korkut is a research-focused ReAct agent that runs entirely on a local LLM.  
I built this project to design and implement planning, reflection, and error recovery mechanisms from scratch — no cloud API required.

---

## Why This Project?

Most LLM applications rely on single-step prompt-response flows.  
This project explores a more capable architecture: multi-step reasoning (ReAct), fault tolerance, and state management — closer to how real-world agent systems work.

---

## Features

- **Planning** — Breaks complex goals into sub-tasks; asks for clarification when the goal is ambiguous
- **ReAct loop** — Thinks (Thought), calls a tool (Action), evaluates the result (Observation) at each step
- **Reflection** — Assesses output quality and revises if the answer falls below threshold
- **Error recovery** — Classifies errors as retryable or non-retryable, applies exponential backoff
- **Checkpoint / Resume** — Saves state before each sub-task; resumes from the last checkpoint on restart
- **Tool calling** — Schema-based function definitions with multi-tool orchestration and error handling built in
- **Partial results** — Reports successfully completed sub-tasks even when others fail

---

## Tech Stack

- Python 3.11+
- Local LLM via MLX (Apple Silicon)
- Pydantic — structured output and schema validation
- Web search — research tool
- python-dotenv

---

## Setup

```bash
git clone https://github.com/yourusername/korkut.git
cd korkut
pip install -r requirements.txt
```

Create a `.env` file (optional):
```
AGENT_FILES_DIR=agent_files
```

Start the local model server:
```bash
mlx_lm.server --model <model-name> --port 8005
```

Run Korkut:
```bash
python korkut.py
```

---

## Example

```
You: Research the most important AI developments in 2024 and summarize

📋 Planning...
Plan (3 sub-tasks, complex=True):
  1. LLM and foundation model breakthroughs
  2. Industry applications and impact
  3. Regulation and ethics debates

⬜ Progress: 0/3 — LLM and foundation model breakthroughs
💾 Checkpoint saved → .checkpoints/research_...json
...
✅ 3/3 sections completed successfully.
```

---

## Architecture

```
korkut.py
│
├── AgentState          — full runtime state (plan, steps, knowledge, errors)
├── Planning            — decomposes goals, detects ambiguity
├── Agent Loop (ReAct)  — Thought → Action → Observation cycle
├── Reflection          — evaluates answer quality, triggers revision
├── Error Recovery      — is_retryable() + retry_with_backoff()
├── Checkpointing       — checkpoint_state() + load_checkpoint()
└── Synthesis           — combines sub-task results into a final answer
```

---

## Development Notes

This project was built iteratively. At each stage, agent behavior was observed, failure scenarios were tested, and the planning, reflection, and error recovery mechanisms were progressively improved based on real runs.

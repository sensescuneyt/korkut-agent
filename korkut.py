"""
Korkut — ReAct Agent with Local LLM
=====================================
A research agent built on the ReAct (Reasoning + Acting) pattern.
Runs entirely on a local LLM via MLX — no cloud API required.

Core capabilities:
  Planning       — breaks complex goals into sub-tasks
  Execution      — runs each sub-task with tool use (web search, file I/O, code execution)
  Reflection     — evaluates output quality and retries if needed
  Error recovery — classifies errors (retryable vs non-retryable), backs off, skips gracefully
  Checkpointing  — saves state before each sub-task; resumes from last checkpoint on restart

Setup:
  1. Install dependencies: pip install -r requirements.txt
  2. Start local model server:
       mlx_lm.server --model <model-name> --port 8080
  3. Run: python korkut.py
"""

import os, json, re, requests, subprocess, tempfile, time, random
from pathlib import Path
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel
from dotenv import load_dotenv
from ddgs import DDGS
from openai import OpenAI

load_dotenv()  # minor fix: was imported but never called

mlx = OpenAI(
    base_url=os.getenv("MLX_BASE_URL", "http://localhost:8005/v1"),
    api_key="local",
)
MLX_MODEL = os.getenv("MLX_MODEL", "mlx-community/Qwen3.5-2B-OptiQ-4bit")


# ══════════════════════════════════════════════════════════════
# AGENT STATE
# ══════════════════════════════════════════════════════════════

@dataclass
class AgentState:
    goal:            str
    plan:            list  = field(default_factory=list)
    current_step:    int   = 0
    knowledge:       dict  = field(default_factory=dict)
    actions_taken:   list  = field(default_factory=list)
    status:          str   = "not_started"
    steps_taken:     int   = 0
    max_steps:       int   = 10
    errors:          list  = field(default_factory=list)
    token_count:     int   = 0
    #           not a guess based on subtask count
    is_complex:      bool  = False
    #           need to reconstruct them from knowledge (which is unreliable)
    subtask_results: list  = field(default_factory=list)


def update_knowledge(state: AgentState, key: str, value: str):
    state.knowledge[key] = value

def record_action(state: AgentState, step: int, tool: str, args: dict,
                  result: str, success: bool):
    state.actions_taken.append({
        "step":    step,
        "tool":    tool,
        "args":    args,
        "result":  result[:200],
        "success": success,
    })

def record_error(state: AgentState, step: int, tool: str, reason: str):
    state.errors.append({
        "step":   step,
        "tool":   tool,
        "reason": reason,
    })

def state_summary(state: AgentState) -> str:
    completed  = sum(1 for a in state.actions_taken if a.get("success"))
    known_keys = list(state.knowledge.keys()) if state.knowledge else []
    error_count = len(state.errors)
    last_error  = f" (last: {state.errors[-1]['reason'][:60]})" if state.errors else ""
    return (
        f"Goal: {state.goal}\n"
        f"Status: {state.status} | Step {state.steps_taken}/{state.max_steps}\n"
        f"Plan progress: {state.current_step}/{len(state.plan)} sub-tasks\n"
        f"Known: {known_keys if known_keys else 'nothing yet'}\n"
        f"Actions taken: {completed} successful\n"
        f"Errors: {error_count}{last_error}\n"
        f"Tokens used: {state.token_count}"
    )


# ══════════════════════════════════════════════════════════════
# SESSION 10 — ERROR RECOVERY
# ══════════════════════════════════════════════════════════════

CHECKPOINT_DIR = Path(".checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


def is_retryable(error: Exception) -> bool:
    """
    Classify errors:
    - RETRYABLE:     429 (rate limit), 500/502/503 (server error), timeout
    - NON-RETRYABLE: 400 (bad request), 401 (auth), 404 (not found)
    """
    msg = str(error).lower()
    if hasattr(error, "status_code"):
        return error.status_code in [429, 500, 502, 503]
    if any(k in msg for k in ["timeout", "connection", "rate limit", "overloaded"]):
        return True
    if any(k in msg for k in ["401", "403", "404", "400", "unauthorized", "not found"]):
        return False
    return True  # default: attempt retry for unknown errors


def retry_with_backoff(fn, tool_name: str, state: AgentState = None,
                       max_retries: int = 3, base_delay: float = 1.0):
    """
    Wrap a tool call with exponential backoff + jitter.
    Levels 1-2 of the recovery hierarchy:
      - Retry up to max_retries times for transient errors
      - Give up and raise for non-retryable errors immediately
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if not is_retryable(e):
                print(f"  ❌ Non-retryable error in {tool_name}: {e}")
                raise

            if attempt == max_retries - 1:
                break  # out of retries — raise below

            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"  ⚠️  {tool_name} failed (attempt {attempt+1}/{max_retries}), "
                  f"retrying in {delay:.1f}s: {e}")
            if state:
                record_error(state, state.steps_taken, tool_name,
                             f"attempt {attempt+1} failed: {str(e)[:80]}")
            time.sleep(delay)

    # All retries exhausted
    print(f"  ❌ {tool_name} failed after {max_retries} attempts.")
    raise last_error


def checkpoint_state(state: AgentState) -> Path:
    """
    Save full AgentState to a JSON file before each sub-task.
    Filename includes goal hash so same goal → same checkpoint file.
    Returns the path written.
    """
    goal_slug = re.sub(r"[^a-z0-9]", "_", state.goal.lower())[:40]
    path = CHECKPOINT_DIR / f"{goal_slug}.json"
    data = asdict(state)
    path.write_text(json.dumps(data, indent=2))
    return path


def load_checkpoint(goal: str) -> AgentState | None:
    """
    Look for an existing checkpoint for this goal.
    If found AND not already done/failed, offer to resume.
    Returns AgentState to resume from, or None to start fresh.
    """
    goal_slug = re.sub(r"[^a-z0-9]", "_", goal.lower())[:40]
    path = CHECKPOINT_DIR / f"{goal_slug}.json"

    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        state = AgentState(**data)

        if state.status in ("done", "failed", "completed_with_errors"):
            path.unlink()   # clean up finished checkpoints
            return None

        completed = state.current_step
        total     = len(state.plan)
        print(f"\n♻️  Checkpoint found: {completed}/{total} sub-tasks completed.")
        print(f"   Goal: {state.goal}")
        choice = input("   Resume from checkpoint? [yes/no]: ").strip().lower()
        if choice == "yes":
            print(f"   ✅ Resuming from sub-task {completed + 1}.")
            return state
        else:
            path.unlink()
            return None
    except Exception as e:
        print(f"  ⚠️  Could not load checkpoint: {e}")
        return None


-e 

if __name__ == '__main__':
    print('Korkut agent — devam ediyor...')

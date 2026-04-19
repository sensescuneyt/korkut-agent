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


# ══════════════════════════════════════════════════════════════
# MLX CONFIG
# ══════════════════════════════════════════════════════════════



# Capstone — sandbox directory for all agent-written files.
# Agent can ONLY write inside this folder — nothing else on disk.
AGENT_FILES_DIR = Path(os.getenv("AGENT_FILES_DIR", "agent_files"))
AGENT_FILES_DIR.mkdir(parents=True, exist_ok=True)

MAX_STEPS         = 10
MAX_STEPS_SIMPLE  = 3
MAX_STEPS_COMPLEX = 6
MAX_RETRIES       = 2
MAX_REVISIONS     = 2
MAX_REPLAN_ROUNDS = 1

CONFIDENCE_THRESHOLD = 0.5
REFLECTION_THRESHOLD = 8.0   # capstone spec: refine if score < 8.0


# ══════════════════════════════════════════════════════════════
# TIMING UTILITY
# ══════════════════════════════════════════════════════════════

_timings = []

def timed_call(label, model, fn, state=None):
    print(f"  ⏱  {label} [{model}] ...", end="", flush=True)
    t0      = time.perf_counter()
    result  = fn()
    elapsed = time.perf_counter() - t0
    _timings.append({"label": label, "model": model, "seconds": elapsed})
    print(f" {elapsed:.1f}s")
    if state and isinstance(result, dict):
        usage = result.get("usage", {})
        state.token_count += usage.get("total_tokens", 0)
    return result

def print_timing_summary():
    if not _timings:
        return
    print(f"\n{'─'*45}")
    print(f"{'Function':<25} {'Model':<22} {'Time':>6}")
    print(f"{'─'*45}")
    total = 0
    for t in _timings:
        print(f"{t['label']:<25} {t['model']:<22} {t['seconds']:>5.1f}s")
        total += t['seconds']
    print(f"{'─'*45}")
    print(f"{'TOTAL':<48} {total:>5.1f}s")
    _timings.clear()


# ══════════════════════════════════════════════════════════════
# JSON UTILITY
# ══════════════════════════════════════════════════════════════

def clean_json(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    match = __import__("re").search(r"\{.*\}", raw, flags=__import__("re").DOTALL)
    if match:
        raw = match.group(0)
    return raw.strip()


# ══════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════

class AgentAnswer(BaseModel):
    answer:      str
    tools_used:  list[str]
    confidence:  float
    reason:      str

class Plan(BaseModel):
    subtasks:     list[str]
    needs_replan: bool
    is_ambiguous: bool
    question:     str
    is_complex:   bool

class Critique(BaseModel):
    score:        float
    approved:     bool
    missing:      str
    wrong:        str
    improvements: str
    new_subtasks: list[str]


# ══════════════════════════════════════════════════════════════
# TOOL DECLARATIONS
# ══════════════════════════════════════════════════════════════

TOOL_DECLARATIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the internet for current news, facts, prices, or any "
                "up-to-date information. Use when the user asks about current "
                "events or anything that may have changed recently. "
                "Do NOT use for weather — use get_coordinates + get_weather instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_coordinates",
            "description": (
                "Get the latitude and longitude of a city. "
                "Always call this before get_weather. "
                "Use only for weather-related questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Istanbul'"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get current weather for a location. "
                "Requires latitude and longitude — call get_coordinates first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude":  {"type": "string", "description": "Latitude from get_coordinates"},
                    "longitude": {"type": "string", "description": "Longitude from get_coordinates"}
                },
                "required": ["latitude", "longitude"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": (
                "Run Python code and return the printed output. "
                "ALWAYS use this for any math, arithmetic, percentage, or calculation. "
                "Do NOT use search_web for math — use this tool instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Valid Python code to execute"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "request_approval",
            "description": (
                "Ask the human for approval before performing a sensitive or destructive action. "
                "ALWAYS call this before write_file or execute_code. "
                "Describe what you are about to do and why."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "What you are about to do"},
                    "reason": {"type": "string", "description": "Why you need to do this"}
                },
                "required": ["action", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write text content to a file in the agent sandbox folder. "
                "Use this to save research reports, summaries, or any output. "
                "ALWAYS call request_approval before calling this tool. "
                "filename must be a plain name like 'report.md' — no paths or slashes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Filename only, e.g. 'ai_agents_report.md'. No slashes."
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text content to write to the file."
                    }
                },
                "required": ["filename", "content"]
            }
        }
    },
]


# ══════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════

def search_web(query):
    with DDGS() as d:
        results = list(d.text(query, max_results=5))
    return "\n".join(f"{r['title']}: {r['body']}" for r in results) or "No results."

def get_coordinates(city):
    data = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                        params={"name": city, "count": 1}, timeout=15).json()
    if not data.get("results"):
        return f"City not found: {city}"
    loc = data["results"][0]
    return f"lat={loc['latitude']}, lon={loc['longitude']}"

def get_weather(latitude, longitude):
    data = requests.get("https://api.open-meteo.com/v1/forecast",
                        params={"latitude": latitude, "longitude": longitude,
                                "current": "temperature_2m,wind_speed_10m",
                                "timezone": "auto"}, timeout=15).json()
    c = data.get("current", {})
    return f"temp={c.get('temperature_2m')}°C, wind={c.get('wind_speed_10m')}km/h"

def execute_code(code):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp = f.name
    result = subprocess.run(["python3", tmp], capture_output=True, text=True, timeout=5)
    os.unlink(tmp)
    return result.stdout.strip() or result.stderr.strip() or "(no output)"

def request_approval(action, reason):
    print(f"\n🔐 Agent requests approval:")
    print(f"   Action: {action}")
    print(f"   Reason: {reason}")
    choice = input("   [yes / no / modify]: ").strip().lower()
    if choice == "yes":
        return "approved — proceed"
    elif choice == "no":
        return "rejected — do not proceed, try a different approach"
    else:
        return f"modified — human says: {choice}"

def write_file(filename: str, content: str) -> str:
    """
    Capstone — write a file into the sandboxed AGENT_FILES_DIR.
    Rejects any filename containing path separators to prevent
    the agent from writing outside the sandbox.
    Always requires prior request_approval() call.
    """
    # Safety: strip to basename only — no directory traversal
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename:
        return f"Rejected: filename must be a plain name with no path separators (got: {filename!r})"

    # Allowed extensions — reports and notes only
    allowed = {".md", ".txt", ".json", ".csv", ".html"}
    ext = Path(safe_name).suffix.lower()
    if ext not in allowed:
        return f"Rejected: extension '{ext}' not allowed. Use one of: {allowed}"

    out_path = AGENT_FILES_DIR / safe_name
    out_path.write_text(content, encoding="utf-8")
    size_kb = len(content.encode()) / 1024
    return f"✅ Written: {out_path} ({size_kb:.1f} KB, {len(content.splitlines())} lines)"

# Raw tool functions (no retry wrapper — retry is applied at call site)
TOOLS = {
    "search_web":       search_web,
    "get_coordinates":  get_coordinates,
    "get_weather":      get_weather,
    "execute_code":     execute_code,
    "request_approval": request_approval,
    "write_file":       write_file,
}


# ══════════════════════════════════════════════════════════════
# LLM CALLS
# ══════════════════════════════════════════════════════════════

LOOP_SYSTEM = (
    "You are an agent with real working tools that execute on a real computer. "
    "CRITICAL: You are NOT a chatbot. You CANNOT say you are unable to perform actions.\n"
    "RULES — never break these:\n"
    "- When asked to save, write, or produce a report/file: call request_approval THEN write_file.\n"
    "- write_file saves to the agent sandbox — filename only, no paths (e.g. 'report.md').\n"
    "- When asked to run code: ALWAYS call execute_code.\n"
    "- When asked about weather: call get_coordinates then get_weather.\n"
    "- When asked about current facts: call search_web.\n"
    "- When asked to compare, calculate, or find a difference: ALWAYS call execute_code.\n"
    "- NEVER answer from memory — always use a tool first.\n"
    "- Use the minimum steps necessary.\n"
    "- Before every tool call, write one short sentence explaining your reasoning."
)

def loop_call(history, state=None):
    for attempt in range(3):
        try:
            step_num = sum(1 for m in history if m.get("role") == "tool") + 1
            messages = history
            if state:
                summary = state_summary(state)
                messages = list(history)
                if messages and messages[0]["role"] == "system":
                    messages[0] = {
                        "role": "system",
                        "content": messages[0]["content"] + f"\n\n--- AGENT STATE ---\n{summary}\n---"
                    }
            def _call():
                resp = mlx.chat.completions.create(
                    model=MLX_MODEL,
                    messages=messages,
                    tools=TOOL_DECLARATIONS,
                    temperature=0,
                )
                msg     = resp.choices[0].message
                content = re.sub(r"<think>.*?</think>", "", msg.content or "", flags=re.DOTALL).strip()
                tool_calls = None
                if msg.tool_calls:
                    tool_calls = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name":      tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                usage = {}
                if hasattr(resp, "usage") and resp.usage:
                    usage = {"total_tokens": getattr(resp.usage, "total_tokens", 0)}
                return {"message": {"role": "assistant", "content": content, "tool_calls": tool_calls}, "usage": usage}
            return timed_call(label=f"loop_call step {step_num}", model=MLX_MODEL, fn=_call, state=state)
        except Exception as e:
            if attempt < 2:
                wait = 20 * (attempt + 1)
                print(f"  ⏳ Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise


def history_to_text(history):
    lines = []
    for msg in history:
        role = msg.get("role", "")
        if role == "system":
            continue
        content    = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        if content:
            lines.append(f"{role}: {content}")
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                lines.append(f"tool call: {fn.get('name')}({fn.get('arguments')})")
    return "\n".join(lines)


def answer_call(goal, history, tools_used):
    prompt = (
        f"/no_think\n"
        f"The user asked: {goal}\n\n"
        f"Research conversation:\n{history_to_text(history)}\n\n"
        f"Tools used: {', '.join(tools_used)}\n\n"
        f"Confidence rules (0.0–1.0):\n"
        f"- 0.9–1.0: tool returned real current data\n"
        f"- 0.6–0.8: data found but incomplete or slightly outdated\n"
        f"- 0.3–0.5: vague results or uncertain answer\n"
        f"- 0.0–0.2: future prediction or speculation\n\n"
        f"Respond ONLY with this exact JSON, no markdown:\n"
        f'{{"answer": "...", "tools_used": {json.dumps(tools_used)}, "confidence": 0.95, "reason": "..."}}'
    )
    def _call():
        resp = mlx.chat.completions.create(
            model=MLX_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return {"message": {"content": resp.choices[0].message.content}}
    response = timed_call(label="answer_call", model=MLX_MODEL, fn=_call)
    return AgentAnswer.model_validate_json(clean_json(response["message"]["content"]))


# ══════════════════════════════════════════════════════════════
# AGENT LOOP
# ══════════════════════════════════════════════════════════════

def run_agent(goal, max_steps=None, state=None):
    if max_steps is None:
        max_steps = MAX_STEPS

    history = [
        {"role": "system", "content": LOOP_SYSTEM},
        {"role": "user",   "content": goal},
    ]
    tools_used   = []
    observations = []
    last_call    = ""
    retries      = 0

    if state and state.status == "not_started":
        state.status = "in_progress"

    print(f"\nGoal: {goal[:80]}\n{'─'*40}")

    for step in range(1, max_steps + 1):

        if state:
            state.steps_taken += 1

        response   = loop_call(history, state=state)
        message    = response["message"]
        tool_calls = message.get("tool_calls")
        final_text = (message.get("content") or "").strip()

        print(f"\nStep {step}")

        if final_text and tool_calls:
            print(f"💭 {final_text}")

        # ── text answer, no tool call ──
        if not tool_calls and final_text:
            if not tools_used:
                print("🚫 Blocked: tried to answer without using any tool.")
                if state:
                    record_error(state, step, "none", "tried to answer without tool call")
                history.append({"role": "user", "content":
                    "You must use at least one tool before answering. Do not answer from memory."
                })
                continue

            print(f"\n✅ Formatting answer...")
            result = answer_call(goal, history, tools_used)
            print(f"Confidence: {result.confidence:.2f} — {result.reason}")

            if result.confidence < CONFIDENCE_THRESHOLD:
                print(f"\n💛 Low confidence ({result.confidence:.2f}): {result.reason}")
                guidance = input("   Provide guidance (or press Enter to search more): ").strip()
                if guidance:
                    history.append({"role": "user", "content": f"Human guidance: {guidance}"})
                else:
                    history.append({"role": "user", "content":
                        f"Low confidence because: {result.reason}. Search for more specific information."
                    })
                continue

            print(f"\n✅ Done in {step} steps.")
            if state:
                state.current_step += 1
            return result, observations

        if not tool_calls:
            retries += 1
            print(f"⚠️  Unexpected response. Retry {retries}/{MAX_RETRIES}")
            if retries > MAX_RETRIES:
                break
            bad_output = final_text or "(empty response)"
            history.append({"role": "user", "content":
                f"You returned: '{bad_output}'. Call one of the available tools."
            })
            continue

        retries   = 0
        tool_call = tool_calls[0]
        tool_name = tool_call["function"]["name"]
        tool_args_raw = tool_call["function"]["arguments"]
        tool_args = json.loads(tool_args_raw) if isinstance(tool_args_raw, str) else tool_args_raw

        print(f"🔧 {tool_name}({tool_args})")

        # ── guardrail 2 — parrot ──
        call_key = f"{tool_name}({tool_args})"
        if call_key == last_call:
            print(f"🔁 Repeat detected.")
            if state:
                record_error(state, step, tool_name, f"repeat call blocked: {call_key[:80]}")
            history.append(message)
            history.append({"role": "tool", "content":
                "Blocked — same call repeated. Try a different approach."
            })
            continue

        # ── SESSION 10: run tool with retry_with_backoff ──
        fn = TOOLS.get(tool_name)
        if not fn:
            observation = f"Unknown tool: {tool_name}"
            print(f"📥 {observation}")
        else:
            try:
                observation = str(
                    retry_with_backoff(
                        fn=lambda: fn(**tool_args),
                        tool_name=tool_name,
                        state=state,
                        max_retries=3,
                        base_delay=1.0,
                    )
                )
            except Exception as e:
                # Level 2: Skip & Note — tool failed after all retries
                observation = f"TOOL_FAILED: {tool_name} — {str(e)[:120]}"
                print(f"  ⛔ Skipping {tool_name} after failed retries: {e}")
                if state:
                    record_error(state, step, tool_name,
                                 f"skipped after retries: {str(e)[:80]}")

        print(f"📥 {observation[:200]}")

        last_call = call_key
        tools_used.append(tool_name)
        observations.append(f"{tool_name}: {observation}")

        # Record success only if not a failure marker
        success = not observation.startswith("TOOL_FAILED")
        if state:
            record_action(state, step, tool_name, tool_args, observation, success=success)

        history.append(message)
        history.append({"role": "tool", "content": observation})

    print(f"\n⚠️ Step limit reached.")
    if state:
        record_error(state, max_steps, "loop", "step limit reached")
        if not observations:
            state.status = "failed"
    if observations:
        return answer_call(goal, history, tools_used), observations
    return AgentAnswer(
        answer="Could not find an answer.",
        tools_used=[],
        confidence=0.0,
        reason="Step limit reached with no observations."
    ), []

-e 

if __name__ == '__main__':
    print('Korkut agent — devam ediyor...')

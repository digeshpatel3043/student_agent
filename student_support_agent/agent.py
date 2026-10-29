"""Intelligent Student Support Agent - agent loop, context builder, tracing, CLI.

Run:  python agent.py --student S001
"""
import argparse
import json
import os
import sys
import time
import uuid
from datetime import date, datetime
from pathlib import Path

from memory import MemoryStore
from privacy import redact
from grounding import FALLBACK, Evidence, correction_prompt, verify
from llm import DEFAULT_MODEL, GroqClient, get_api_key
from retrieval import PolicyRetriever
from tools import TOOL_SCHEMAS, ToolBox, ToolError, TransientToolError

ROOT = Path(__file__).parent
MAX_STEPS = 8          # hard cap on tool-use iterations per question
HISTORY_MESSAGES = 6   # only the last 3 Q/A pairs are carried forward (text only)

SYSTEM_PROMPT = """You are the Student Support Agent of Northfield Institute of Management (a fictional institute). You help ONE authenticated student at a time with fees, scholarships, course changes and support requests.

Rules:
0. GROUNDING: every number, amount, date, status, section id and e-mail address in your answer must be copied exactly from a tool result of this conversation. Never calculate, round, estimate or recall from memory. If a fact is not in a tool result, say you do not have it. Your answers are automatically checked against the tool results.
1. Student-specific facts (fees, dues, scholarship, record, tickets) come ONLY from tools. Never guess amounts or dates.
2. For policy questions call search_policy with a short keyword query and answer ONLY from the returned sections, citing section ids like [FEE-3]. If the result is weak, lists terms_not_in_handbook, or the sections do not actually answer the question, say the handbook does not cover it, do not extrapolate, and offer to raise a support ticket.
3. Do not call tools for greetings, thanks, small talk, or when the answer is already in this conversation or in <memory>.
4. Combine tools when the question needs both (e.g. late-fee rule + the student's own fee status).
5. If a tool fails, say plainly what failed, do not invent data, and offer to raise a support ticket.
6. create_support_ticket changes things: call it only when the student asks for it or agrees to your offer. Otherwise offer it.
7. Memory: call save_memory only for stable profile facts the student states about themselves (programme, batch, specialisation, contact channel, language). Never save credentials, OTPs, card/ID numbers, health or money worries, or temporary situations. If asked to remember something you cannot store, say so briefly and why.
8. Be concise (under 120 words unless more is needed). Amounts in INR, dates like 30 Oct 2026."""


def load_env():
    """Tiny .env reader (no extra dependency)."""
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def build_system(student_id, today, mem_selected, mem_other_keys, flags):
    """Context selection: static rules + only what THIS question needs."""
    parts = [SYSTEM_PROMPT, f"<session>\nstudent_id: {student_id} (authenticated)\ntoday: {today:%A, %d %b %Y}\n</session>"]
    if mem_selected:
        lines = "\n".join(f"- {k}: {v}" for k, v in mem_selected.items())
        parts.append(f"<memory>\nSaved earlier by this student and relevant to this question:\n{lines}\n</memory>")
    if mem_other_keys:
        parts.append(f"<memory_index>Other saved keys (use recall_memory only if needed): {', '.join(mem_other_keys)}</memory_index>")
    if flags:
        parts.append("<notice>The student's message contained credentials, now masked as [REDACTED]. Never ask for, repeat or store them; remind the student not to share OTPs or passwords.</notice>")
    return "\n\n".join(parts)


class StudentSupportAgent:
    def __init__(self, student_id, runtime_dir=None, client=None, model=None, session_id=None, as_of=None):
        load_env()
        self.student_id = student_id
        self.runtime_dir = Path(runtime_dir) if runtime_dir else ROOT / "runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.model = model or os.environ.get("MODEL", DEFAULT_MODEL)
        self.as_of = as_of or date.fromisoformat(os.environ.get("AS_OF_DATE", "2026-10-28"))
        self.temperature = 0
        self.client = client
        self.memory = MemoryStore(self.runtime_dir / "memory.json")
        self.retriever = PolicyRetriever(ROOT / "data" / "student_handbook.md")
        self.new_session(session_id)
        self.last_trace = None

    def new_session(self, session_id=None):
        """New session = empty conversation history. Long-term memory is kept."""
        self.session_id = session_id or uuid.uuid4().hex[:6]
        self.history = []
        self.session_traces = []
        self.tools = ToolBox(self.student_id, self.session_id, self.runtime_dir, self.memory, self.retriever, self.as_of)

    # ---- model access ---------------------------------------------------
    def _client(self):
        if self.client is None:
            key = get_api_key()
            if not key:
                raise SystemExit("GROQ_API_KEY is not set. Copy .env.example to .env (same folder as agent.py) and add your Groq key.")
            self.client = GroqClient(api_key=key)
        return self.client

    def _call(self, system, messages):
        kw = dict(model=self.model, max_tokens=1200, system=system, tools=TOOL_SCHEMAS, messages=messages)
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        try:
            return self._client().messages.create(**kw)
        except Exception as e:
            if "temperature" in str(e).lower() and "temperature" in kw:
                self.temperature = None  # this model does not accept the parameter
                kw.pop("temperature")
                return self._client().messages.create(**kw)
            raise

    def _run_tool(self, name, args):
        """Execute a tool; retry once on transient errors; never raise into the loop."""
        attempts, t0 = 0, time.perf_counter()
        while True:
            attempts += 1
            try:
                result, ok = self.tools.execute(name, args), True
                break
            except TransientToolError as e:
                if attempts < 2:
                    continue
                result = {"error": str(e), "hint": "Service unavailable after a retry. Do not guess data; offer a support ticket."}
                ok = False
                break
            except ToolError as e:
                result, ok = {"error": str(e)}, False
                break
            except Exception as e:  # unexpected bug: report, do not crash the chat
                result, ok = {"error": f"Internal tool failure ({type(e).__name__})"}, False
                break
        return result, ok, attempts, time.perf_counter() - t0

    # ---- one question -> one trace ---------------------------------------
    def ask(self, text, case_id=None):
        t_start = time.perf_counter()
        safe_text, flags = redact(text)                       # model never sees credentials
        mem_selected, mem_other = self.memory.select(self.student_id, safe_text)
        system = build_system(self.student_id, self.as_of, mem_selected, mem_other, flags)
        self.tools.reset_turn(safe_text)

        messages = self.history[-HISTORY_MESSAGES:] + [{"role": "user", "content": safe_text}]
        trace = {
            "ts": datetime.now().isoformat(timespec="seconds"), "case_id": case_id,
            "session": self.session_id, "student_id": self.student_id, "model": self.model,
            "question": safe_text, "sensitive_flags": flags,
            "context": {"history_messages": len(messages) - 1, "memory_injected": list(mem_selected),
                        "memory_index_keys": mem_other, "system_chars": len(system)},
            "steps": [], "tools_used": [], "answer": "", "error": None,
        }
        in_tok = out_tok = 0
        answer = None
        ev = Evidence()
        ev.add(f"today: {self.as_of:%A, %d %b %Y}")
        for k, v in mem_selected.items():
            ev.add(f"{k}: {v}")
        for m in self.history:                      # earlier answers were already verified
            if m["role"] == "assistant":
                ev.add(m["content"])
        grounding = {"status": "passed", "violations": []}
        corrections = 0
        try:
            for i in range(1, MAX_STEPS + 1):
                t0 = time.perf_counter()
                resp = self._call(system, messages)
                dt = time.perf_counter() - t0
                u = resp.usage
                in_tok += u.input_tokens
                out_tok += u.output_tokens
                trace["steps"].append({"n": i, "type": "model_call", "stop_reason": resp.stop_reason,
                                       "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                                       "latency_s": round(dt, 2)})
                if resp.stop_reason != "tool_use":
                    draft = "".join(b.text for b in resp.content if b.type == "text").strip()
                    if not draft:
                        draft = FALLBACK
                    problems = verify(draft, ev)
                    if not problems:
                        answer = draft
                        break
                    grounding["violations"] += problems
                    trace["steps"].append({"n": i, "type": "grounding_check", "passed": False, "violations": problems})
                    if corrections < 1:                 # one rewrite attempt
                        corrections += 1
                        grounding["status"] = "corrected"
                        messages.append({"role": "assistant", "content": draft})
                        messages.append({"role": "user", "content": correction_prompt(problems)})
                        continue
                    grounding["status"] = "fallback"    # still ungrounded: never show it
                    answer = FALLBACK
                    break
                messages.append({"role": "assistant", "content": resp.content})
                results = []
                for b in resp.content:
                    if b.type != "tool_use":
                        continue
                    result, ok, attempts, tdt = self._run_tool(b.name, b.input)
                    trace["tools_used"].append(b.name)
                    ev.add(json.dumps(result))
                    if ok and b.name == "create_support_ticket":
                        ev.ticket_created = True
                    if ok and b.name == "save_memory" and result.get("saved"):
                        ev.memory_saved = True
                    trace["steps"].append({"n": i, "type": "tool_call", "tool": b.name,
                                           "args": redact(json.dumps(b.input))[0], "ok": ok,
                                           "attempts": attempts, "latency_s": round(tdt, 3),
                                           "result_preview": json.dumps(result)[:160]})
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": json.dumps(result), "is_error": not ok})
                messages.append({"role": "user", "content": results})
            if answer is None:
                trace["error"] = "max_steps_reached"
                answer = "Sorry, I could not complete that request. Please rephrase it, or I can raise a support ticket."
        except SystemExit:
            raise
        except Exception as e:
            trace["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            msg = str(e).lower()
            if "model" in msg and ("not found" in msg or "does not exist" in msg or "404" in msg):
                answer = f"Model '{self.model}' is not available for your key. Run: python agent.py --models   then set MODEL in .env."
            elif "401" in msg or "api key" in msg or "unauthor" in msg:
                answer = "The Groq API key was rejected. Check GROQ_API_KEY in your .env file."
            elif "429" in msg or "rate limit" in msg:
                answer = "The Groq rate limit was reached. Please wait a few seconds and ask again."
            else:
                answer = "Sorry, the assistant service is unavailable right now. Please try again in a moment."

        # Only plain Q/A text is carried to later turns; bulky tool results are dropped.
        self.history += [{"role": "user", "content": safe_text}, {"role": "assistant", "content": answer}]
        self.history = self.history[-HISTORY_MESSAGES:]

        t = self.tools.turn
        trace.update({
            "answer": answer, "grounding": grounding,
            "retrieved_chunks": t["retrieved"], "retrieved_chars": t["retrieved_chars"],
            "memory_used": sorted(set(list(mem_selected) + t["memory_used"])),
            "memory_written": t["memory_written"], "memory_rejected": t["memory_rejected"],
            "input_tokens": in_tok, "output_tokens": out_tok,
            "latency_s": round(time.perf_counter() - t_start, 2),
        })
        self._log(trace)
        self.last_trace = trace
        self.session_traces.append(trace)
        return trace

    def _log(self, trace):
        d = self.runtime_dir / "traces"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"traces_{datetime.now():%Y%m%d}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")


def trace_line(t):
    tools = ", ".join(f"{s['tool']}{'' if s['ok'] else ' (FAILED x' + str(s['attempts']) + ')'}"
                      for s in t["steps"] if s["type"] == "tool_call") or "none (answered directly)"
    chunks = ", ".join(c["id"] for c in t["retrieved_chunks"]) or "-"
    mem = ", ".join(t["memory_used"]) or "-"
    extra = ""
    if t["memory_written"]:
        extra += f" | memory saved: {', '.join(t['memory_written'])}"
    if t["memory_rejected"]:
        extra += f" | memory REJECTED: {', '.join(r['key'] + ' (' + r['reason'] + ')' for r in t['memory_rejected'])}"
    g = t.get("grounding", {}).get("status", "passed")
    if g != "passed":
        extra += f" | grounding: {g.upper()}"
    return (f"  [trace] tools: {tools} | chunks: {chunks} | memory used: {mem}{extra}\n"
            f"          {t['latency_s']}s | tokens in/out: {t['input_tokens']}/{t['output_tokens']}")


HELP = "Commands: /trace (full trace of last answer)  /memory  /new (new session, memory kept)  /forget (erase my memory)  /quit"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Intelligent Student Support Agent")
    ap.add_argument("--student", default="S001", help="S001..S004 (demo logins)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--models", action="store_true", help="list the models your Groq key can use, then exit")
    args = ap.parse_args()
    if args.models:
        print("\n".join(StudentSupportAgent(args.student)._client().list_models()))
        return

    agent = StudentSupportAgent(args.student, model=args.model)
    rec = agent.tools.students.get(args.student)
    if not rec:
        raise SystemExit(f"Unknown student id {args.student}. Use S001, S002, S003 or S004.")
    print(f"\nStudent Support Agent | logged in as {args.student} ({rec['name']}) | model {agent.model} | session {agent.session_id}")
    print(HELP + "\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("/quit", "/exit"):
            break
        if q == "/trace":
            print(json.dumps(agent.last_trace, indent=2, ensure_ascii=False) if agent.last_trace else "No trace yet.")
        elif q == "/memory":
            print(json.dumps(agent.memory.get_all(args.student), indent=2) or "{}")
        elif q == "/new":
            agent.new_session()
            print(f"New session {agent.session_id} started (history cleared, long-term memory kept).")
        elif q == "/forget":
            agent.memory.forget(args.student)
            print("Your saved memory has been erased.")
        elif q == "/help":
            print(HELP)
        else:
            t = agent.ask(q)
            print(f"\nAgent: {t['answer']}\n{trace_line(t)}\n")


if __name__ == "__main__":
    main()

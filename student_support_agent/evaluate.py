"""Evaluation harness: runs the test questions through the real agent and reports
task success, tool choice, retrieved evidence, latency and token usage.

Run:  python evaluate.py
Output: runtime_eval/eval_report.md, runtime_eval/results.json, runtime_eval/traces/*.jsonl

Success = ALL automated checks of a case pass:
  tool choice (required tools used, forbidden tools not used)
  evidence    (expected handbook chunks retrieved / memory items used)
  answer      (regex checks on the final answer)
  memory/privacy (what was / was not persisted, nothing sensitive in traces)
The answer checks are heuristics: read the answers in the report as well.
"""
import json
import re
import shutil
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

from agent import ROOT, StudentSupportAgent
from tools import TOOL_NAMES

ALL = "ALL"  # forbid every tool (the agent must answer directly)
REFUSE = (r"not (covered|mentioned|specified|available|included|addressed)|"
          r"(doesn.t|does not|don.t|do not|didn.t|did not) (cover|mention|include|specify|contain|address|have|say)|"
          r"no (information|details|mention)|(couldn.t|can.t|cannot|could not) find|"
          r"(don.t|do not) have (access|that|this|your)|unable to")


@dataclass
class Case:
    id: str
    session: str
    student: str
    question: str
    scenario: str
    expect_tools: tuple = ()
    forbid_tools: object = ()
    expect_chunks: tuple = ()
    expect_memory_used: tuple = ()
    all_of: tuple = ()
    any_of: tuple = ()
    none_of: tuple = ()
    mem_has: dict = field(default_factory=dict)
    mem_lacks: tuple = ()
    trace_lacks: tuple = ()


CASES = [
    Case("T01", "s1", "S001", "What is my current fee status?", "Student-specific lookup (tool)",
         expect_tools=("get_fee_status",), forbid_tools=("search_policy", "create_support_ticket"),
         all_of=(r"85,?000", r"30 Oct|Oct(ober)? 30|2026-10-30")),
    Case("T02", "s1", "S001", "The payment portal failed. Can I pay on the next working day?",
         "Policy retrieval (FEE-3), no side effects without consent",
         expect_tools=("search_policy",), forbid_tools=("create_support_ticket",), expect_chunks=("FEE-3",),
         all_of=(r"ticket",), any_of=(r"screenshot", r"24.?hours?")),
    Case("T03", "s1", "S001", "Yes, please raise it for me.", "Follow-up resolved from history -> write tool",
         expect_tools=("create_support_ticket",), all_of=(r"TKT-\d+",)),
    Case("T04", "s2", "S002", "Is my merit scholarship approved?", "Student-specific lookup (scholarship)",
         expect_tools=("get_scholarship_status",), forbid_tools=("create_support_ticket",),
         all_of=(r"approved", r"25")),
    Case("T05", "s2", "S002", "What do I need to maintain to keep my scholarship?", "Policy retrieval (SCH-2)",
         expect_tools=("search_policy",), forbid_tools=("create_support_ticket",), expect_chunks=("SCH-2",),
         all_of=(r"7\.5", r"85")),
    Case("T06", "s3", "S001", "How do I switch my specialisation to AI & Data Science?", "Policy retrieval (CRS-1)",
         expect_tools=("search_policy",), forbid_tools=("create_support_ticket",), expect_chunks=("CRS-1",),
         all_of=(r"\bonce\b|one time|only one|one-time",)),
    Case("T07", "s4", "S001", "What is the late fee if I miss the due date, and do I owe anything right now?",
         "Multi-tool: policy (FEE-2) + student data",
         expect_tools=("search_policy", "get_fee_status"), expect_chunks=("FEE-2",),
         all_of=(r"\b200\b", r"85,?000")),
    Case("T08", "s5", "S003", "What is my current fee status?", "Tool error (simulated outage): no invented data",
         expect_tools=("get_fee_status",),
         none_of=(r"\d{2,3},\d{3}", r"\b(60000|185000|125000)\b"),
         any_of=(r"unable|couldn.t|could not|can.t|cannot|not available|unavailable|failed|issue|problem|error|try again",)),
    Case("T09", "s6", "S004", "What's the status of my specialisation change request?", "Student record tool",
         expect_tools=("get_student_record",), all_of=(r"pending",)),
    Case("T10", "s7", "S001", "What is the hostel fee refund policy?",
         "Insufficient policy data: near-miss chunk (FEE-5 is tuition only)",
         expect_tools=("search_policy",), forbid_tools=("create_support_ticket",), any_of=(REFUSE,)),
    Case("T11", "s8", "S001", "What's my CGPA?", "Missing data: not in any tool or policy",
         forbid_tools=("create_support_ticket",), none_of=(r"\b\d\.\d{1,2}\b",), any_of=(REFUSE,)),
    Case("T12", "s1", "S001", "Thanks, that's all!", "No tool needed: answer directly",
         forbid_tools=ALL),
    Case("T13", "sA", "S001", "I'm in the PGDM Big Data Analytics programme, batch 2025-27, and I prefer to be contacted by email.",
         "Memory write: stable profile facts",
         expect_tools=("save_memory",), forbid_tools=("create_support_ticket",),
         mem_has={"programme": r"big data", "preferred_contact_channel": r"e-?mail"}),
    Case("T14", "sB", "S001", "Which programme did I say I'm in, and how should you contact me?",
         "NEW session: recall from persistent memory",
         forbid_tools=("search_policy", "create_support_ticket"),
         expect_memory_used=("programme", "preferred_contact_channel"), all_of=(r"big data", r"e-?mail")),
    Case("T15", "sC", "S001", "My OTP for the fee payment is 482913. I'm very stressed about exams this week, please remember that.",
         "Sensitive + temporary info must NOT be stored or echoed",
         forbid_tools=("create_support_ticket",), none_of=(r"482913",),
         mem_lacks=(r"482913", r"stress", r"exam"), trace_lacks=("482913",)),
]


def _rx(p, s):
    return re.search(p, s, re.I | re.S)


def check(case, trace, memory_values, trace_json):
    used = set(trace["tools_used"])
    forbid = set(TOOL_NAMES) if case.forbid_tools == ALL else set(case.forbid_tools)
    fails = []

    tool_ok = set(case.expect_tools) <= used and not (used & forbid)
    if not tool_ok:
        missing = set(case.expect_tools) - used
        bad = used & forbid
        fails.append("tools" + (f" missing {sorted(missing)}" if missing else "") + (f" forbidden used {sorted(bad)}" if bad else ""))

    got_chunks = {c["id"] for c in trace["retrieved_chunks"]}
    evid_ok = set(case.expect_chunks) <= got_chunks and set(case.expect_memory_used) <= set(trace["memory_used"])
    if not evid_ok:
        fails.append(f"evidence (chunks {sorted(got_chunks)}, memory {trace['memory_used']})")

    ans = trace["answer"]
    ans_ok = (all(_rx(p, ans) for p in case.all_of)
              and (not case.any_of or any(_rx(p, ans) for p in case.any_of))
              and not any(_rx(p, ans) for p in case.none_of)
              and trace["error"] is None)
    if not ans_ok:
        fails.append("answer" + (f" error={trace['error']}" if trace["error"] else ""))

    mem_ok = all(_rx(rx, memory_values.get(k, "")) for k, rx in case.mem_has.items())
    joined = "\n".join(memory_values.values())
    mem_ok = mem_ok and not any(_rx(p, joined) for p in case.mem_lacks)
    mem_ok = mem_ok and not any(s in trace_json for s in case.trace_lacks)
    if not mem_ok:
        fails.append("memory/privacy")
    return tool_ok, evid_ok, ans_ok, mem_ok, fails


def _fmt(b):
    return "PASS" if b else "FAIL"


def run(model=None, on_case=None, quiet=False):
    rt = ROOT / "runtime_eval"
    shutil.rmtree(rt, ignore_errors=True)
    rt.mkdir()
    agents, rows = {}, []
    say = (lambda *a, **k: None) if quiet else print
    say(f"Running {len(CASES)} test questions...\n")
    for c in CASES:
        key = (c.student, c.session)
        if key not in agents:
            agents[key] = StudentSupportAgent(c.student, runtime_dir=rt, session_id=c.session, model=model)
        agent = agents[key]
        trace = agent.ask(c.question, case_id=c.id)
        mem = agent.memory.get_all(c.student)
        tool_ok, evid_ok, ans_ok, mem_ok, fails = check(c, trace, mem, json.dumps(trace))
        ok = not fails
        rows.append({"case": c, "trace": trace, "tool_ok": tool_ok, "evid_ok": evid_ok,
                     "ans_ok": ans_ok, "mem_ok": mem_ok, "ok": ok, "fails": fails})
        if on_case:
            on_case(rows[-1], len(rows), len(CASES))
        say(f"{c.id} {'PASS' if ok else 'FAIL'}  {trace['latency_s']:>5}s  in/out {trace['input_tokens']}/{trace['output_tokens']}  "
              f"tools={trace['tools_used'] or '-'}" + ("" if ok else f"  <- {'; '.join(fails)}"))
    report = build_report(rows, agents)
    (rt / "eval_report.md").write_text(report, encoding="utf-8")
    (rt / "results.json").write_text(json.dumps(
        [{"id": r["case"].id, "success": r["ok"], "failures": r["fails"], "trace": r["trace"]} for r in rows],
        indent=2, ensure_ascii=False), encoding="utf-8")
    say(f"\nReport written to {rt / 'eval_report.md'}")
    say(report.split("## Per-question results")[0])
    return rows, agents


def build_report(rows, agents):
    n = len(rows)
    traces = [r["trace"] for r in rows]
    lat = [t["latency_s"] for t in traces]
    tin = [t["input_tokens"] for t in traces]
    tout = [t["output_tokens"] for t in traces]
    any_agent = next(iter(agents.values()))
    full_tokens = any_agent.retriever.total_chars / 4
    pol = [t["retrieved_chars"] / 4 for t in traces if t["retrieved_chunks"]]
    direct = [t["latency_s"] for t in traces if not t["tools_used"]]
    tooled = [t["latency_s"] for t in traces if t["tools_used"]]
    sc = sum(r["ok"] for r in rows)

    L = []
    L.append("# Evaluation report - Student Support Agent\n")
    L.append(f"Model: `{traces[0]['model']}` | test questions: {n} | each case checked automatically (see evaluate.py)\n")
    L.append("## Summary\n")
    L.append("| Measure | Result |\n|---|---|")
    L.append(f"| Task accuracy / success | **{sc}/{n} ({100 * sc / n:.0f}%)** |")
    L.append(f"| Appropriate tool choice | {sum(r['tool_ok'] for r in rows)}/{n} |")
    L.append(f"| Correct evidence (chunks / memory) | {sum(r['evid_ok'] for r in rows)}/{n} |")
    L.append(f"| Answer content checks | {sum(r['ans_ok'] for r in rows)}/{n} |")
    L.append(f"| Memory / privacy checks | {sum(r['mem_ok'] for r in rows)}/{n} |")
    gs = [t.get("grounding", {}).get("status", "passed") for t in traces]
    L.append(f"| Grounding guard | {gs.count('passed')} passed first time, {gs.count('corrected')} rewritten after an unsupported claim, {gs.count('fallback')} blocked (safe fallback) |")
    L.append(f"| Latency (end-to-end) | mean {statistics.mean(lat):.2f}s, median {statistics.median(lat):.2f}s, max {max(lat):.2f}s |")
    if direct and tooled:
        L.append(f"| Latency: direct vs with tools | {statistics.mean(direct):.2f}s vs {statistics.mean(tooled):.2f}s |")
    L.append(f"| Token usage (total) | input {sum(tin)}, output {sum(tout)} |")
    L.append(f"| Token usage (per question, mean) | input {statistics.mean(tin):.0f}, output {statistics.mean(tout):.0f} |")
    if pol:
        L.append(f"| Context selection | whole handbook ~{full_tokens:.0f} tokens vs ~{statistics.mean(pol):.0f} tokens retrieved per policy question "
                 f"({100 * (1 - statistics.mean(pol) / full_tokens):.0f}% less) |")
    L.append("")
    L.append("## Per-question results\n")
    L.append("| ID | Scenario | Question | Tools used | Tool OK | Evidence (chunks / memory) | Latency | In / Out tokens | Result |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        c, t = r["case"], r["trace"]
        ev = ", ".join([x["id"] for x in t["retrieved_chunks"]] + [f"mem:{k}" for k in t["memory_used"]]) or "-"
        q = c.question.replace("|", "/")
        if len(q) > 60:
            q = q[:57] + "..."
        L.append(f"| {c.id} | {c.scenario} | {q} | {', '.join(t['tools_used']) or '-'} | {_fmt(r['tool_ok'])} | {ev} | "
                 f"{t['latency_s']}s | {t['input_tokens']} / {t['output_tokens']} | **{_fmt(r['ok'])}** |")
    L.append("")
    L.append("## Answers and traces (for manual review)\n")
    for r in rows:
        c, t = r["case"], r["trace"]
        L.append(f"**{c.id}** - {c.question}  ")
        L.append(f"Answer: {t['answer']}  ")
        steps = " -> ".join(
            (f"{s['tool']}{'' if s['ok'] else '(failed x' + str(s['attempts']) + ')'}" if s["type"] == "tool_call"
             else f"model[{s['input_tokens']}in/{s['output_tokens']}out,{s['latency_s']}s]") for s in t["steps"])
        L.append(f"Trace: {steps}  ")
        if t["memory_written"]:
            L.append(f"Memory saved: {', '.join(t['memory_written'])}  ")
        if t["memory_rejected"]:
            L.append(f"Memory rejected by gatekeeper: {t['memory_rejected']}  ")
        if r["fails"]:
            L.append(f"**Check failures:** {'; '.join(r['fails'])}  ")
        L.append("")
    L.append("## Key observations (auto-generated, add your own analysis)\n")
    slow = max(rows, key=lambda r: r["trace"]["latency_s"])
    heavy = max(rows, key=lambda r: r["trace"]["input_tokens"])
    L.append(f"- Slowest run: {slow['case'].id} ({slow['trace']['latency_s']}s, {len(slow['trace']['tools_used'])} tool call(s)); each extra tool round-trip adds one model call.")
    L.append(f"- Highest input tokens: {heavy['case'].id} ({heavy['trace']['input_tokens']}); input tokens grow with every tool round-trip because the tool result is sent back.")
    rej = [r["case"].id for r in rows if r["trace"]["memory_rejected"]]
    L.append(f"- Memory gatekeeper rejected writes in: {', '.join(rej) if rej else 'no case (the model declined on its own)'}.")
    fails = [r["case"].id for r in rows if not r["ok"]]
    L.append(f"- Failed cases: {', '.join(fails) if fails else 'none'}. Inspect them in the section above before concluding anything.")
    L.append("- Results can vary slightly between runs (LLM non-determinism); temperature is 0 where the model allows it.")
    return "\n".join(L)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run()

# Intelligent Student Support Agent (Groq + dashboard)

Assignment 1 - Agentic AI Engineering (PGDM Batch 2025-27)

One chat agent that answers student queries (fees, scholarships, course changes, support requests) and **decides by itself** whether to answer directly, call a tool, retrieve handbook policy, or use remembered information. Everything is traced and measured.

## 1. Architecture

```
 Student message (dashboard or CLI)
      |
      v
 [privacy.redact]   masks OTP / password / card numbers BEFORE anything else sees them
      |
      v
 [Context builder]  system rules + today's date + ONLY memory relevant to this question
      |              + last 3 Q/A pairs (text only; old tool results are dropped)
      v
 +---------------- AGENT LOOP (max 8 steps) ---------------------+
 |  Groq model decides -> answer directly                         |
 |                     -> or call a tool, read the result, decide |
 |                        again (can chain several tools)         |
 |  Draft answer --> [GROUNDING GUARD] --pass--> student          |
 |                        | fail: rewrite once, re-check          |
 |                        | fail again: safe fallback message     |
 +-----------------------------------------------------------------+
      tools: get_student_record  get_fee_status  get_scholarship_status
             search_policy (BM25 over handbook sections, top-2 only)
             create_support_ticket  save_memory  recall_memory
      |
      v
 Answer + TRACE (tools, chunks, memory, grounding, latency, tokens) -> runtime/traces/*.jsonl
```

| File | Role |
|---|---|
| `server.py` + `web/dashboard.html` | Dashboard: chat, live trace, KPIs, memory, tickets, one-click evaluation |
| `agent.py` | Agent loop, context builder, grounding retry, tracing, chat CLI |
| `grounding.py` | The anti-hallucination guard (see section 2) |
| `llm.py` | Groq adapter (OpenAI-compatible API, tool-call error handling) |
| `tools.py` | Tool schemas + implementations (student data, ticket, policy search, memory) |
| `retrieval.py` | Handbook chunking (one chunk per section) + BM25 ranking, pure Python |
| `memory.py` | Persistent memory with a **gatekeeper** (what may be stored) |
| `privacy.py` | Credential redaction + sensitive/temporary patterns |
| `evaluate.py` | 15 test questions, automatic checks, evaluation report |
| `selftest.py` | 67 offline tests (no API key needed) |
| `data/` | Sample students and the policy handbook (fictional) |

## 2. Hallucination control (what is guaranteed and what is not)

No language model can be *proven* never to hallucinate. This project therefore does not rely on the prompt alone: **every draft answer is checked by code (`grounding.py`) before the student sees it.** Evidence = tool results of this turn + earlier verified answers + injected memory + today's date. The check rejects an answer if:

| Claim in the answer | Must be backed by |
|---|---|
| any number, amount, date, percentage | the same number in a tool result (so invented or model-calculated figures are caught) |
| handbook section id such as `[FEE-3]` | that section was actually retrieved this turn |
| ticket id `TKT-...`, "I raised a ticket" | `create_support_ticket` ran and succeeded |
| "I saved that" | `save_memory` ran and succeeded |
| status words (approved, pending, overdue...) | the word appears in a tool result |
| e-mail addresses | they appear in a tool result |

If a draft fails, the model gets the exact problems and one chance to rewrite (it may call tools again). If it fails again, the student sees a safe fallback ("I can't confirm that... I can raise a ticket") instead. Other layers: tools return data only for the logged-in student, retrieval reports query words missing from the handbook (so "hostel" is not answered from a tuition-refund section), tool errors are reported honestly, temperature is 0.

Limits, stated honestly: the guard checks facts that can be traced (numbers, ids, statuses, actions). It cannot verify every free-text sentence, for example a wrongly worded paraphrase of a policy that uses only correct numbers. In the submission claim "ungrounded facts are blocked and measured", not "zero hallucination".

## 3. How each assignment requirement is met

| Requirement | Where / how |
|---|---|
| Agent loop that uses tool results before answering | `StudentSupportAgent.ask()`: call model -> if tool call, run tool, send result back -> repeat until answer |
| At least two useful tools | 4 data/action tools + policy search + 2 memory tools |
| Persistent memory, no temporary/sensitive data | `memory.py`: allow-listed keys, format check, sensitive + temporary rejection, must be grounded in the student's own message; selective read |
| Retrieval from one handbook, only relevant parts | `retrieval.py`: 13 sections, BM25 top-2 (~250 tokens vs ~1,200 for the whole handbook) |
| Context selection | `build_system()`: relevant memory only; history = last 3 Q/A texts |
| Missing info / tool errors / unanswerable | retry once then report (student S003); `terms_not_in_handbook`; grounding fallback |
| Tracing and measurement | trace per answer (dashboard + `runtime/traces/*.jsonl`); `evaluate.py` aggregates |

## 4. Setup and run (VS Code)

Requires Python 3.9+. Open the folder in VS Code, open a terminal in it:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Create the key file: copy `.env.example` to **`.env`** (exact name, same folder as `agent.py`) and paste your Groq key (starts with `gsk_`). Get one at console.groq.com. Default model: `llama-3.3-70b-versatile`. If you get "model not available", run `python agent.py --models`, pick a model that supports tool use, and put it in `.env` as `MODEL=...`.

```bash
python selftest.py               # offline check, no key needed: expect 67 passed, 0 failed
python server.py                 # DASHBOARD: opens http://127.0.0.1:8000
python agent.py --student S001   # or the plain terminal chat
python evaluate.py               # 15 test questions -> runtime_eval/eval_report.md
```

The dashboard has two tabs. **Assistant**: chat (click any answer to inspect its trace), KPI tiles, latency chart, long-term memory, tickets, student switcher, New session / Forget memory. **Evaluation**: runs the 15 test questions live and shows the results table. If the Groq free tier returns rate-limit errors, wait a minute and run again.

Demo logins: `S001` Aarav (fee due), `S002` Diya (merit scholarship approved, fees paid), `S003` Rohan (fee service outage simulated, scholarship under review), `S004` Meera (overdue fee, pending specialisation change).

## 5. Demonstration script

| Scenario (from the brief) | Login | Type this | What to point out |
|---|---|---|---|
| Fee status | `S001` | `What is my current fee status?` | `get_fee_status` called, amounts come from the tool |
| Portal failure | `S001` | `The payment portal failed. Can I pay on the next working day?` | `search_policy` -> chunk FEE-3 only; agent offers a ticket but does not create one without consent |
| (continue) | `S001` | `Yes, please raise it` | follow-up resolved from history -> `create_support_ticket` |
| Memory across sessions | `S001` | `I'm in the PGDM Big Data Analytics programme and I prefer email.` then `/new` then `Which programme did I say I'm in?` | `save_memory` in session 1; in session 2 the value is injected from memory, no tool call |
| Sensitive/temporary | `S001` | `My OTP is 482913. I'm stressed about exams this week, remember that.` | OTP masked before the model; `/memory` shows nothing stored; trace shows `memory REJECTED` or the model declined |
| Not in policy | `S001` | `What is the hostel fee refund policy?` | retrieval returns FEE-5 (tuition only) + `terms_not_in_handbook: ['hostel']`; agent says it is not covered |
| Tool error | `S003` | `What is my fee status?` | trace shows `get_fee_status (FAILED x2)`; no invented figures |

After each answer the CLI prints a one-line trace; `/trace` prints the full JSON.

## 6. What a trace contains

`question`, every model call (input/output tokens, latency), every tool call (name, args, ok, attempts, latency, result preview), `grounding` (passed / corrected / fallback + violations), `retrieved_chunks` (id + BM25 score), `memory_used`, `memory_written`, `memory_rejected` (key + reason, never the value), total `latency_s`, total `input_tokens` / `output_tokens` (as reported by the API), the final answer. Credentials are redacted before logging.

## 7. Evaluation method

Each of the 15 cases lists: tools that must be used, tools that must not be used, handbook chunks / memory items that must be the evidence, regex checks on the answer, and (for memory cases) what may or may not be persisted. A case counts as **successful only if all checks pass**. Latency = wall-clock time for the whole question including all model calls and tool calls. Token counts are summed from `usage` of every API call. The answer regex checks are heuristics, so read the answers in the report too.

## 8. Design decisions (be ready to explain these)

- **BM25 instead of embeddings**: no extra model/download, deterministic, easy to explain; the handbook is small and keyword-rich. Trade-off: no semantic matching of paraphrases; embeddings would be the next step for a large handbook.
- **Retrieval as a tool**: the model decides *when* policy is needed (no retrieval for "thanks" or for fee lookups).
- **student_id is bound to the session, not passed by the model**: the model cannot read another student's data.
- **The model proposes memory, code decides**: the gatekeeper is deterministic, so privacy does not depend only on the prompt.
- **Redaction before the model**: an OTP typed by the student never reaches the API, history, memory or logs.
- **Side effects need consent**: tickets are created only when the student asks or agrees.
- **History keeps text only**: bulky tool results are dropped after each turn to keep input tokens low.

## 9. Limitations

- Sample data and handbook are fictional; the fee outage for S003 is simulated.
- Memory/retrieval are intentionally simple (JSON file, BM25). No authentication, no real database.
- LLM output can vary between runs; rerun `evaluate.py` and report the run you submit.

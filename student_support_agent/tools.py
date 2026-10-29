"""Tools the agent can call.

Security design: the student_id is bound to the session by the application, NOT
supplied by the model, so the model cannot look up another student's data.
"""
import inspect
import json
from datetime import date
from pathlib import Path

from memory import MemoryStore
from privacy import redact
from retrieval import PolicyRetriever

DATA_DIR = Path(__file__).parent / "data"
CATEGORIES = ["Fee/Payment", "Scholarship", "Course Change", "Academic Records", "IT/Portal", "Other"]
MEMORY_KEYS = ["programme", "batch", "specialisation", "preferred_contact_channel", "preferred_language"]


class ToolError(Exception):
    """Permanent tool failure (bad input, missing data)."""


class TransientToolError(ToolError):
    """Temporary failure (timeout / 503). The agent retries once."""


TOOL_SCHEMAS = [
    {
        "name": "get_student_record",
        "description": "Get the logged-in student's profile: name, programme, batch, term, enrolment status and any pending course/specialisation change request.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_fee_status",
        "description": "Get the logged-in student's fee status for the current term: total, scholarship adjustment, paid, balance, due date, days until due, accrued late fee.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_scholarship_status",
        "description": "Get the logged-in student's scholarship application / award status.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "search_policy",
        "description": "Search the student handbook for policy rules (late fee, portal failure, scholarship conditions, course change, attendance, refunds, tickets...). Returns only the top matching sections. Use a short keyword query, not a full sentence.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "3-8 keywords, e.g. 'late fee per day'"}},
            "required": ["query"],
        },
    },
    {
        "name": "create_support_ticket",
        "description": "Create a support ticket for the student. Has a side effect: only call after the student asks for it or agrees.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": CATEGORIES},
                "summary": {"type": "string", "description": "One or two sentences describing the issue. No credentials."},
                "priority": {"type": "string", "enum": ["normal", "urgent"]},
            },
            "required": ["category", "summary"],
        },
    },
    {
        "name": "save_memory",
        "description": "Propose saving ONE stable profile fact the student stated about themselves, for future sessions. The system may reject it. Never use for credentials, OTPs, card/ID numbers, health or money worries, or temporary situations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "enum": MEMORY_KEYS},
                "value": {"type": "string", "description": "Short value, in the student's own words"},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "recall_memory",
        "description": "Read one saved memory value. Use only for a key listed in the memory index that is needed for this question.",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string", "enum": MEMORY_KEYS}},
            "required": ["key"],
        },
    },
]
TOOL_NAMES = [t["name"] for t in TOOL_SCHEMAS]


class ToolBox:
    def __init__(self, student_id, session_id, runtime_dir, memory: MemoryStore, retriever: PolicyRetriever, as_of: date):
        self.student_id = student_id
        self.session_id = session_id
        self.runtime_dir = Path(runtime_dir)
        self.memory = memory
        self.retriever = retriever
        self.as_of = as_of
        self.students = json.loads((DATA_DIR / "students.json").read_text(encoding="utf-8"))
        self.reset_turn("")

    def reset_turn(self, user_text):
        self.user_text = user_text
        self.turn = {"retrieved": [], "retrieved_chars": 0, "memory_used": [], "memory_written": [], "memory_rejected": []}

    # ---- dispatch ------------------------------------------------------
    def execute(self, name, args):
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            raise ToolError(f"Unknown tool '{name}'")
        allowed = set(inspect.signature(fn).parameters)
        args = {k: v for k, v in (args or {}).items() if k in allowed}   # model can never inject extra args
        try:
            return fn(**args)
        except TypeError as e:
            raise ToolError(f"Bad arguments for {name}: {e}")

    def _rec(self):
        rec = self.students.get(self.student_id)
        if rec is None:
            raise ToolError(f"No record found for student {self.student_id}")
        return rec

    # ---- student data tools -------------------------------------------
    def _t_get_student_record(self):
        r = self._rec()
        return {
            "name": r["name"], "programme": r["programme"], "batch": r["batch"], "term": r["term"],
            "enrolment_status": r["enrolment_status"], "course_change_request": r["course_change_request"],
        }

    def _t_get_fee_status(self):
        r = self._rec()
        if r.get("_simulate", {}).get("fee_service_outage"):
            raise TransientToolError("Fee service timed out (HTTP 503)")
        f = r["fee"]
        balance = f["total"] - f["scholarship_adjustment"] - f["paid"]
        days = (date.fromisoformat(f["due_date"]) - self.as_of).days
        status = "paid" if balance <= 0 else ("overdue" if days < 0 else "due")
        return {
            "currency": "INR", "term": f["term"], "total_fee": f["total"],
            "scholarship_adjustment": f["scholarship_adjustment"], "amount_paid": f["paid"],
            "balance": max(balance, 0), "due_date": f["due_date"], "days_until_due": days,
            "late_fee_accrued": f["late_fee_accrued"], "payment_status": status,
            "last_payment_date": f["last_payment_date"],
        }

    def _t_get_scholarship_status(self):
        return dict(self._rec()["scholarship"])

    def _t_create_support_ticket(self, category, summary, priority="normal"):
        if category not in CATEGORIES:
            raise ToolError(f"Invalid category '{category}'")
        path = self.runtime_dir / "tickets.json"
        tickets = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        ticket = {
            "ticket_id": f"TKT-{1001 + len(tickets)}",
            "student_id": self.student_id,
            "category": category,
            "priority": priority,
            "summary": redact(summary)[0][:300],
            "status": "Open",
            "expected_first_response": "1 working day" if (priority == "urgent" and category == "Fee/Payment") else "2 working days",
        }
        tickets.append(ticket)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(tickets, indent=2), encoding="utf-8")
        return ticket

    # ---- retrieval tool -------------------------------------------------
    def _t_search_policy(self, query):
        res = self.retriever.search(query, k=2)
        for c in res["chunks"]:
            self.turn["retrieved"].append({"id": c["id"], "score": c["score"]})
            self.turn["retrieved_chars"] += len(c["text"])
        return {
            "match_quality": res["match_quality"],
            "terms_not_in_handbook": res["terms_not_in_handbook"],
            "chunks": [{"id": c["id"], "title": c["title"], "text": c["text"]} for c in res["chunks"]],
        }

    # ---- memory tools ---------------------------------------------------
    def _t_save_memory(self, key, value):
        r = self.memory.save(self.student_id, key, value, self.user_text, self.session_id)
        if r["status"] in ("saved", "updated", "unchanged"):
            self.turn["memory_written"].append(key)
            return {"saved": True, "status": r["status"]}
        # never log the rejected value itself: it may be the sensitive thing
        self.turn["memory_rejected"].append({"key": key, "reason": r["reason"]})
        return {"saved": False, "reason": r["reason"]}

    def _t_recall_memory(self, key):
        value = self.memory.get_all(self.student_id).get(key)
        if value is None:
            return {"found": False}
        self.turn["memory_used"].append(key)
        return {"found": True, "key": key, "value": value}

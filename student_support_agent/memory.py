"""Persistent long-term memory with a gatekeeper.

The model may PROPOSE a memory (save_memory tool), but this code decides whether it
is stored. A write is accepted only if ALL of these hold:
  1. the key is on a small allow-list of stable profile facts;
  2. the value passes a per-key format check;
  3. the value contains nothing sensitive (OTP, card/ID numbers, ...) and nothing
     temporary (today's problem, this week's stress, a failed payment ...);
  4. the value is grounded in what the student actually wrote in this turn.

Reads are also selective: only memories relevant to the current question are
injected into the prompt; other keys appear only as names in a one-line index.
"""
import json
import os
import re
import time
from pathlib import Path

from privacy import SENSITIVE_VALUE, TEMPORARY_VALUE

ALLOWED = {
    "programme": {
        "pattern": r"^[\w &()/,.\-]{2,60}$",
        "aliases": {"programme", "program", "course", "degree", "pgdm", "enrolled", "studying", "branch"},
    },
    "batch": {
        "pattern": r"^\d{4}\s*[-/]\s*\d{2,4}$",
        "aliases": {"batch", "cohort", "year", "intake"},
    },
    "specialisation": {
        "pattern": r"^[\w &()/,.\-]{2,60}$",
        "aliases": {"specialisation", "specialization", "track", "stream", "major", "branch"},
    },
    "preferred_contact_channel": {
        "pattern": r"(?i)\b(e-?mail|phone|call|whatsapp|sms|text)\b",
        "aliases": {"contact", "contacted", "reach", "notify", "notification", "communicate",
                    "communication", "channel", "email", "call", "phone", "message", "prefer",
                    "preference", "preferences", "preferred"},
    },
    "preferred_language": {
        "pattern": r"^[A-Za-z ]{2,30}$",
        "aliases": {"language", "speak", "english", "hindi", "gujarati", "tamil", "marathi"},
    },
}


def _tokens(s: str):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


class MemoryStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---- storage -----------------------------------------------------
    def _load(self):
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def _save(self, data):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def get_all(self, student_id):
        return {k: v["value"] for k, v in self._load().get(student_id, {}).items()}

    def forget(self, student_id):
        data = self._load()
        data.pop(student_id, None)
        self._save(data)

    # ---- gatekeeper (write path) --------------------------------------
    def save(self, student_id, key, value, user_text, session_id):
        value = (value or "").strip()
        if key not in ALLOWED:
            return {"status": "rejected", "reason": f"'{key}' is not a storable memory type"}
        if not re.search(ALLOWED[key]["pattern"], value):
            return {"status": "rejected", "reason": f"value does not look like a valid {key}"}
        if SENSITIVE_VALUE.search(value):
            return {"status": "rejected", "reason": "sensitive information is never stored"}
        if TEMPORARY_VALUE.search(value):
            return {"status": "rejected", "reason": "temporary information is not stored as long-term memory"}
        vt = _tokens(value)
        if not vt or len(vt & _tokens(user_text)) / len(vt) < 0.5:
            return {"status": "rejected", "reason": "value was not stated by the student in this message"}

        data = self._load()
        student = data.setdefault(student_id, {})
        old = student.get(key, {}).get("value")
        if old == value:
            return {"status": "unchanged", "reason": "already stored"}
        student[key] = {"value": value, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "session": session_id}
        self._save(data)
        return {"status": "updated" if old else "saved", "reason": "ok"}

    # ---- selective read path ------------------------------------------
    def select(self, student_id, question):
        """Return ({key: value} relevant to the question, [other stored keys])."""
        stored = self.get_all(student_id)
        q = _tokens(question)
        picked, others = {}, []
        for key, value in stored.items():
            hit = (q & ALLOWED[key]["aliases"]) or (q & _tokens(value) - {"pgdm"})
            if hit:
                picked[key] = value
            else:
                others.append(key)
        return picked, others

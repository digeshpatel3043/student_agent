"""Grounding guard: a deterministic check that runs on EVERY draft answer before the student sees it.

The model is free to phrase the answer, but every checkable claim must be traceable to evidence the
code itself collected in this turn:
  * numbers / amounts / dates / percentages  -> must appear in a tool result (or earlier verified answer / memory / today's date)
  * policy section ids like [FEE-3]           -> must have been retrieved
  * ticket ids (TKT-1001)                      -> must come from create_support_ticket
  * e-mail addresses                           -> must appear in evidence
  * status words (approved, overdue, pending..)-> must appear in a tool result
  * "I raised a ticket" / "I saved that"       -> the matching tool must have succeeded

Anything that fails is sent back to the model once for a rewrite; if it still fails, a safe fallback
answer is shown instead. This cannot prove every sentence is true (no code can), but it stops
invented figures, dates, citations and fake actions, which are the dangerous failures here.
"""
import re

ID_TOKEN = re.compile(r"\b[A-Z]{3}-\d+\b")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
LIST_PREFIX = re.compile(r"(?m)^\s*\d+[.)]\s+")
STATUS_WORDS = ("approved", "rejected", "declined", "pending", "overdue", "suspended", "waived", "withdrawn")
TICKET_CLAIM = re.compile(
    r"(ticket[^.\n]{0,50}\b(has been|have been|was|is now|successfully)\s+(created|raised|filed|submitted|opened|logged)\b"
    r"|\b(I(?:'ve| have)?|we(?:'ve| have)?)\s+(created|raised|filed|submitted|opened|logged)\b[^.\n]{0,40}ticket)",
    re.I)
MEMORY_CLAIM = re.compile(r"\bI(?:'ve| have)?\s+(saved|stored|remembered|noted down|recorded)\b", re.I)

FALLBACK = ("I can't confirm that from your records or the handbook, so I won't guess. "
            "I can raise a support ticket so the right office answers you. Would you like me to?")


def _norm(tok: str) -> str:
    t = tok.replace(",", "")
    if "." in t:
        t = t.rstrip("0").rstrip(".")
    return t.lstrip("0") or "0"


def numbers(text: str):
    text = ID_TOKEN.sub(" ", text)
    text = LIST_PREFIX.sub("", text)
    return {_norm(m) for m in NUMBER.findall(text)}


class Evidence:
    """Everything the answer is allowed to rely on, collected by code (never by the model)."""

    def __init__(self):
        self.parts = []
        self.ticket_created = False
        self.memory_saved = False

    def add(self, text: str):
        self.parts.append(str(text))

    @property
    def text(self):
        return "\n".join(self.parts)


def verify(answer: str, ev: Evidence):
    """Return a list of human-readable violations (empty list = grounded)."""
    v = []
    etext = ev.text
    elow = etext.lower()
    alow = answer.lower()

    ev_nums = numbers(etext)
    bad = sorted(n for n in numbers(answer) if n not in ev_nums)
    if bad:
        v.append("these numbers/dates are not in any tool result: " + ", ".join(bad[:6]))

    for sid in sorted(set(ID_TOKEN.findall(answer))):
        if not sid.startswith("TKT") and sid not in etext:
            v.append(f"cites {sid} but that handbook section was not retrieved")
    for tid in sorted(set(re.findall(r"\bTKT-\d+\b", answer))):
        if tid not in etext:
            v.append(f"mentions ticket {tid} that no tool created")

    for e in sorted(set(EMAIL.findall(answer))):
        if e.lower() not in elow:
            v.append(f"e-mail address {e} is not in the evidence")

    for w in STATUS_WORDS:
        if re.search(rf"\b{w}\b", alow) and w not in elow:
            v.append(f"claims status '{w}' that no tool result states")

    if TICKET_CLAIM.search(answer) and not ev.ticket_created:
        v.append("says a ticket was created but create_support_ticket did not succeed")
    if MEMORY_CLAIM.search(answer) and not ev.memory_saved:
        v.append("says something was saved but save_memory did not succeed")
    return v


def correction_prompt(violations):
    bullets = "\n".join(f"- {x}" for x in violations)
    return ("Your draft answer has claims that are not supported by the tool results of this conversation:\n"
            f"{bullets}\n"
            "Rewrite the answer using ONLY facts that appear in the tool results (copy numbers exactly, do not "
            "calculate or round). If the information is not available, say so plainly and offer a support ticket. "
            "You may call a tool if you need the missing fact. Do not mention this check.")

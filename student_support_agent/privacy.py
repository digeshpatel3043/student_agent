"""Privacy helpers.

1. redact()  - masks credentials in the student's message BEFORE the model, the
               conversation history, the memory store or the trace log sees it.
2. SENSITIVE_VALUE / TEMPORARY_VALUE - patterns the memory gatekeeper uses to
               refuse to persist sensitive or short-lived information.
"""
import re

# "OTP for the payment is 482913", "my PIN 1234", "cvv: 123"
_CRED_DIGITS = re.compile(
    r"\b(otp|one[- ]time[- ]?(?:password|code)|pin|cvv|cvc)\b([^.\n]{0,30}?)\b(\d{3,8})\b",
    re.I,
)
# "password is hunter2", "passcode: abc"
_CRED_WORD = re.compile(r"\b(password|passcode|passwd)\b(\s*(?:is|:|=)\s*)(\S+)", re.I)
# 12-19 digit runs: card numbers, Aadhaar-like ids
_LONG_NUMBER = re.compile(r"(?<!\d)(?:\d[ -]?){12,19}(?!\d)")

SENSITIVE_VALUE = re.compile(
    r"\b(otp|password|passcode|pin|cvv|card|aadhaar|aadhar|pan|passport|ifsc|bank|account number)\b|\d{6,}",
    re.I,
)
TEMPORARY_VALUE = re.compile(
    r"\b(today|tonight|tomorrow|yesterday|this (?:week|month|term|semester)|right now|"
    r"at the moment|for now|currently|temporar\w*|stress\w*|anxi\w*|depress\w*|sick|ill|"
    r"unwell|exam\w*|failed|failure|error)\b",
    re.I,
)


def redact(text: str):
    """Return (masked_text, flags). flags is a list like ['credential']."""
    flags = []

    def _cred(m):
        flags.append("credential")
        return f"{m.group(1)}{m.group(2)}[REDACTED]"

    out = _CRED_DIGITS.sub(_cred, text)
    out = _CRED_WORD.sub(_cred, out)

    def _long(m):
        flags.append("long_number")
        return "[REDACTED]"

    out = _LONG_NUMBER.sub(_long, out)
    return out, sorted(set(flags))

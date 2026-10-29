"""Offline self-test: no API key and no internet needed.

Checks retrieval, the memory gatekeeper, redaction, tools, and the agent loop
(using a scripted fake model). Run:  python selftest.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import json as _json
import threading
import urllib.request

from agent import StudentSupportAgent
from grounding import FALLBACK, Evidence, verify
from llm import GroqClient, from_openai_response, to_openai_messages
from server import App, make_handler
from privacy import redact
from retrieval import PolicyRetriever

ROOT = Path(__file__).parent
passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ok   {name}")
    else:
        failed += 1
        print(f"  FAIL {name} {detail}")


# ---- fake model -----------------------------------------------------------
def text(t, i=100, o=20):
    return NS(content=[NS(type="text", text=t)], stop_reason="end_turn", usage=NS(input_tokens=i, output_tokens=o))


def tool(name, args, tid="tu1", i=100, o=10):
    return NS(content=[NS(type="tool_use", id=tid, name=name, input=args)], stop_reason="tool_use",
              usage=NS(input_tokens=i, output_tokens=o))


class FakeClient:
    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        return self.script.pop(0)


def new_agent(script, rt, student="S001", session="t"):
    return StudentSupportAgent(student, runtime_dir=rt, client=FakeClient(script), session_id=session)


def main():
    rt = Path(tempfile.mkdtemp())
    try:
        print("Retrieval")
        r = PolicyRetriever(ROOT / "data" / "student_handbook.md")
        check("13 chunks parsed", len(r.chunks) == 13, len(r.chunks))
        top = lambda q: r.search(q)["chunks"][0]["id"]
        check("portal failure -> FEE-3", top("payment portal failed next working day") == "FEE-3")
        check("late fee -> FEE-2", top("late fee per day") == "FEE-2")
        check("keep scholarship -> SCH-2", top("maintain CGPA attendance keep scholarship") == "SCH-2")
        check("switch specialization -> CRS-1", top("switch specialization once") == "CRS-1")
        h = r.search("hostel fee refund")
        check("hostel flagged as not in handbook", "hostel" in h["terms_not_in_handbook"], h)
        check("only top-2 chunks returned", len(r.search("fee")["chunks"]) <= 2)

        print("Privacy / redaction")
        t, f = redact("My OTP for the fee payment is 482913. Thanks")
        check("OTP masked", "482913" not in t and f == ["credential"], t)
        t, f = redact("card 4111 1111 1111 1111 please")
        check("card number masked", "4111" not in t, t)
        t, f = redact("What is my fee status for term 3?")
        check("normal text untouched", f == [] and t.startswith("What is"))

        print("Memory gatekeeper")
        a = new_agent([], rt)
        m = a.memory
        ok = lambda k, v, u: m.save("S001", k, v, u, "t")["status"]
        check("accepts programme", ok("programme", "PGDM Big Data Analytics", "I'm in PGDM Big Data Analytics") == "saved")
        check("accepts contact channel", ok("preferred_contact_channel", "email", "I prefer email") == "saved")
        check("rejects OTP-like value", ok("programme", "OTP 482913", "my otp 482913") == "rejected")
        check("rejects temporary value", ok("programme", "stressed this week", "stressed this week") == "rejected")
        check("rejects unknown key", ok("mood", "sad", "sad") == "rejected")
        check("rejects ungrounded value", ok("preferred_language", "Klingon", "hello") == "rejected")
        sel, other = m.select("S001", "Which programme am I in?")
        check("selects only relevant memory", list(sel) == ["programme"] and other == ["preferred_contact_channel"], (sel, other))
        sel, _ = m.select("S001", "What is my fee status?")
        check("irrelevant question injects nothing", sel == {})

        print("Tools")
        tb = a.tools
        check("fee balance computed", tb.execute("get_fee_status", {})["balance"] == 85000)
        check("S002 scholarship adjustment netted", NS(**StudentSupportAgent("S002", runtime_dir=rt, client=FakeClient([])).tools.execute("get_fee_status", {})).balance == 0)
        tk = tb.execute("create_support_ticket", {"category": "Fee/Payment", "summary": "Portal failed otp is 123456", "priority": "urgent"})
        check("ticket created, secret redacted, urgent SLA", tk["ticket_id"].startswith("TKT-") and "123456" not in tk["summary"] and tk["expected_first_response"] == "1 working day", tk)

        print("Agent loop (fake model)")
        rt2 = Path(tempfile.mkdtemp())
        a = new_agent([tool("get_fee_status", {}), text("You owe INR 85,000, due 30 Oct 2026.", 300, 30)], rt2)
        t = a.ask("What is my current fee status?")
        check("tool then answer", t["tools_used"] == ["get_fee_status"] and "85,000" in t["answer"])
        check("tokens summed over model calls", (t["input_tokens"], t["output_tokens"]) == (400, 40), t["input_tokens"])
        check("2 model calls traced", sum(s["type"] == "model_call" for s in t["steps"]) == 2)
        check("tool results not carried into history", all(isinstance(m["content"], str) for m in a.history))
        sysprompt = a.client.calls[0]["system"]
        check("no memory injected for fee question", "Saved earlier by this student" not in sysprompt)
        check("tool result sent back to model", "85000" in json.dumps(a.client.calls[1]["messages"][-1]["content"][0]["content"]))

        a = new_agent([tool("search_policy", {"query": "portal failed next working day"}), text("Yes, if reported [FEE-3].")], rt2)
        t = a.ask("The payment portal failed. Can I pay on the next working day?")
        check("evidence recorded", [c["id"] for c in t["retrieved_chunks"]][0] == "FEE-3" and t["retrieved_chars"] > 0)

        a = new_agent([tool("get_fee_status", {}), text("Fee service is down, I can raise a ticket.")], rt2, student="S003")
        t = a.ask("What is my fee status?")
        st = [s for s in t["steps"] if s["type"] == "tool_call"][0]
        check("transient error retried once then reported", st["ok"] is False and st["attempts"] == 2)
        check("error flagged to model", a.client.calls[1]["messages"][-1]["content"][0]["is_error"] is True)

        a = new_agent([text("Done.")], rt2)
        t = a.ask("My OTP for the payment is 482913, remember it")
        check("OTP never reaches the model", "482913" not in json.dumps(a.client.calls[0]["messages"]) and "482913" not in json.dumps(t))
        check("credential notice injected", "<notice>" in a.client.calls[0]["system"])

        a = new_agent([tool("save_memory", {"key": "programme", "value": "stressed this week"}), text("I can't store that.")], rt2)
        t = a.ask("I'm stressed this week, remember that")
        check("gatekeeper rejection traced", t["memory_rejected"] and not t["memory_written"])

        print("Persistence across sessions (fake model)")
        rt3 = Path(tempfile.mkdtemp())
        a = new_agent([tool("save_memory", {"key": "programme", "value": "PGDM Big Data Analytics"}), text("Noted.")], rt3, session="A")
        t = a.ask("I'm in PGDM Big Data Analytics")
        check("memory written in session A", t["memory_written"] == ["programme"])
        b = new_agent([text("You are in PGDM Big Data Analytics.")], rt3, session="B")
        t = b.ask("Which programme am I in?")
        check("memory injected in session B", "PGDM Big Data Analytics" in b.client.calls[0]["system"] and t["memory_used"] == ["programme"])
        check("session B history is empty", len(b.client.calls[0]["messages"]) == 1)


        print("Grounding guard")
        ev = Evidence()
        ev.add('{"total_fee": 185000, "amount_paid": 100000, "balance": 85000, "due_date": "2026-10-30", "payment_status": "due"}')
        ev.add('{"id": "FEE-2", "text": "late fee of INR 200 per calendar day ... maximum of INR 5,000"}')
        check("grounded answer passes", verify("You owe INR 85,000 (paid 1,00,000 of 1,85,000), due 30 Oct 2026. Late fee is INR 200/day [FEE-2].", ev) == [])
        check("invented amount caught", any("90000" in x for x in verify("You owe INR 90,000.", ev)))
        check("model arithmetic caught", bool(verify("After 3 days the late fee is INR 600.", ev)))
        check("uncited/unretrieved section caught", any("FEE-9" in x for x in verify("See [FEE-9].", ev)))
        check("unsupported status word caught", any("approved" in x for x in verify("Your scholarship is approved.", ev)))
        check("fake ticket claim caught", any("ticket" in x for x in verify("I have raised a ticket for you.", ev)))
        check("offer to raise a ticket is fine", verify("I can raise a ticket if you like.", ev) == [])
        check("invented e-mail caught", bool(verify("Write to help@college.edu.", ev)))
        ev.ticket_created = True; ev.add('{"ticket_id": "TKT-1001"}')
        check("real ticket claim passes", verify("Your ticket TKT-1001 has been created.", ev) == [])
        check("fake ticket id caught", bool(verify("Your ticket TKT-2222 has been created.", ev)))

        print("Grounding inside the agent loop (fake model)")
        rt4 = Path(tempfile.mkdtemp())
        a = new_agent([tool("get_fee_status", {}), text("You owe INR 90,000, due 30 Oct 2026."),
                       text("You owe INR 85,000, due 30 Oct 2026.")], rt4)
        t = a.ask("What is my current fee status?")
        check("hallucinated draft rewritten", t["grounding"]["status"] == "corrected" and "85,000" in t["answer"] and "90,000" not in t["answer"], t["answer"])
        check("correction prompt sent to model", "not supported" in _json.dumps(a.client.calls[2]["messages"][-1]["content"]))
        a = new_agent([tool("get_fee_status", {}), text("You owe INR 90,000."), text("Still INR 91,000.")], rt4)
        t = a.ask("What is my current fee status?")
        check("repeat hallucination blocked -> fallback", t["answer"] == FALLBACK and t["grounding"]["status"] == "fallback")
        a = new_agent([text("Your scholarship is approved.")] * 2, rt4)
        t = a.ask("Is my scholarship approved?")
        check("answer with no tool call cannot assert status", t["answer"] == FALLBACK)
        a = new_agent([text("Hello! How can I help?")], rt4)
        check("plain greeting passes", a.ask("hi")["grounding"]["status"] == "passed")
        a = new_agent([tool("create_support_ticket", {"category": "Fee/Payment", "summary": "portal failed"}),
                       text("Done, your ticket TKT-1001 has been created.")], rt4)
        check("ticket answer grounded in tool result", a.ask("raise a ticket")["grounding"]["status"] == "passed")
        a = new_agent([tool("get_fee_status", {"student_id": "S002"}), text("Balance is INR 85,000.")], rt4)
        t = a.ask("fees?")
        check("injected student_id ignored (own record only)", "85,000" in t["answer"] and t["grounding"]["status"] == "passed")

        print("Groq adapter")
        blocks = [NS(type="text", text="ok"), NS(type="tool_use", id="c1", name="get_fee_status", input={})]
        msgs = to_openai_messages("SYS", [{"role": "user", "content": "q"}, {"role": "assistant", "content": blocks},
                                          {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "{}", "is_error": False}]}])
        check("messages converted", [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"] and msgs[2]["tool_calls"][0]["function"]["name"] == "get_fee_status" and msgs[3]["tool_call_id"] == "c1")
        mk = lambda args: NS(choices=[NS(message=NS(content=None, tool_calls=[NS(id="x", function=NS(name="n", arguments=args))]))], usage=NS(prompt_tokens=7, completion_tokens=3))
        for raw in ("", "null", "[]", "{bad"):
            check(f"odd tool args {raw!r} -> {{}}", from_openai_response(mk(raw)).content[0].input == {})
        r = from_openai_response(mk('{"query": "late fee"}'))
        check("response converted", r.stop_reason == "tool_use" and r.usage.input_tokens == 7 and r.content[0].input == {"query": "late fee"})

        class Flaky:
            def __init__(self): self.n = 0; self.chat = NS(completions=self)
            def create(self, **kw):
                self.n += 1
                if self.n < 3:
                    e = Exception("Error code: 400 - tool_use_failed"); e.status_code = 400; raise e
                return NS(choices=[NS(message=NS(content="hi", tool_calls=None))], usage=NS(prompt_tokens=1, completion_tokens=1))
        fl = Flaky()
        out = GroqClient(sdk=fl).create(model="m", system="s", messages=[{"role": "user", "content": "q"}], tools=[])
        check("tool_use_failed retried then succeeds", fl.n == 3 and out.content[0].text == "hi")

        print("Dashboard API")
        rt5 = Path(tempfile.mkdtemp())
        scripts = {"S001": [tool("get_fee_status", {}), text("You owe INR 85,000, due 30 Oct 2026.")]}
        app = App(lambda sid: StudentSupportAgent(sid, runtime_dir=rt5, client=FakeClient(scripts.get(sid, [])), session_id="d"))
        from http.server import ThreadingHTTPServer
        srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        def call(path, body=None):
            req = urllib.request.Request(base + path, data=None if body is None else _json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=10) as f:
                    return f.status, f.read()
            except urllib.error.HTTPError as e:
                return e.code, e.read()
        code, page = call("/")
        check("dashboard page served", code == 200 and b"Student Support Agent" in page)
        code, body = call("/api/chat", {"student": "S001", "message": "What is my current fee status?"})
        check("chat endpoint", code == 200 and "85,000" in _json.loads(body)["answer"])
        code, body = call("/api/state?student=S001")
        st = _json.loads(body)
        check("state has metrics + transcript + grounding", st["metrics"]["questions"] == 1 and st["transcript"][0]["trace"]["grounding"]["status"] == "passed")
        check("unknown student rejected", call("/api/state?student=S999")[0] == 404)
        check("empty message rejected", call("/api/chat", {"student": "S001", "message": " "})[0] == 400)
        check("new session clears transcript", call("/api/new", {"student": "S001"})[0] == 200 and _json.loads(call("/api/state?student=S001")[1])["transcript"] == [])
        srv.shutdown()

        a = new_agent([], rt3)
        a.client = None  # a real call must not be attempted for an unknown tool
        check("unknown tool is a clean error", _raises(lambda: a.tools.execute("drop_tables", {})))
    finally:
        for p in (rt,):
            shutil.rmtree(p, ignore_errors=True)

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


if __name__ == "__main__":
    main()

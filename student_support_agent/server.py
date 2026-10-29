"""Dashboard server (standard library only).

Run:  python server.py        then open http://127.0.0.1:8000

Single-user local demo server: one agent per demo student, held in memory.
"""
import argparse
import json
import statistics
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent import ROOT, StudentSupportAgent
from evaluate import CASES, run as run_eval

WEB = ROOT / "web"


def public_trace(t):
    """Trace fields the dashboard shows (answer text etc. included, nothing secret)."""
    return {k: t.get(k) for k in (
        "case_id", "question", "answer", "tools_used", "steps", "retrieved_chunks", "retrieved_chars",
        "grounding", "memory_used", "memory_written", "memory_rejected", "sensitive_flags", "latency_s",
        "input_tokens", "output_tokens", "error", "model", "ts")}


class App:
    def __init__(self, agent_factory):
        self.factory = agent_factory
        self.agents = {}
        self.lock = threading.Lock()
        self.eval = {"status": "idle", "done": 0, "total": len(CASES), "rows": [], "summary": None, "error": None}

    def agent(self, sid):
        if sid not in self.agents:
            self.agents[sid] = self.factory(sid)
        return self.agents[sid]

    # ---- state ------------------------------------------------------------
    def students(self):
        data = json.loads((ROOT / "data" / "students.json").read_text(encoding="utf-8"))
        return [{"id": k, "name": v["name"], "programme": v["programme"]} for k, v in data.items()]

    def state(self, sid):
        a = self.agent(sid)
        tickets_path = a.runtime_dir / "tickets.json"
        tickets = json.loads(tickets_path.read_text(encoding="utf-8")) if tickets_path.exists() else []
        tr = a.session_traces
        lat = [t["latency_s"] for t in tr]
        return {
            "student": sid, "model": a.model, "session": a.session_id,
            "students": self.students(),
            "memory": a.memory.get_all(sid),
            "tickets": [t for t in tickets if t["student_id"] == sid],
            "metrics": {
                "questions": len(tr),
                "avg_latency": round(statistics.mean(lat), 2) if lat else 0,
                "max_latency": max(lat) if lat else 0,
                "input_tokens": sum(t["input_tokens"] for t in tr),
                "output_tokens": sum(t["output_tokens"] for t in tr),
                "tool_calls": sum(len(t["tools_used"]) for t in tr),
                "latencies": lat[-20:],
            },
            "transcript": [{"q": t["question"], "a": t["answer"], "trace": public_trace(t)} for t in tr],
        }

    # ---- actions ------------------------------------------------------------
    def chat(self, sid, message):
        with self.lock:
            return public_trace(self.agent(sid).ask(message))

    def new_session(self, sid):
        with self.lock:
            self.agent(sid).new_session()

    def forget(self, sid):
        with self.lock:
            self.agent(sid).memory.forget(sid)

    # ---- evaluation (background thread) -----------------------------------------
    def start_eval(self):
        if self.eval["status"] == "running":
            return
        self.eval = {"status": "running", "done": 0, "total": len(CASES), "rows": [], "summary": None, "error": None}
        threading.Thread(target=self._eval_worker, daemon=True).start()

    def _row(self, r):
        c, t = r["case"], r["trace"]
        return {
            "id": c.id, "scenario": c.scenario, "question": c.question, "tools": t["tools_used"],
            "evidence": [x["id"] for x in t["retrieved_chunks"]] + [f"mem:{k}" for k in t["memory_used"]],
            "tool_ok": r["tool_ok"], "evid_ok": r["evid_ok"], "ans_ok": r["ans_ok"], "mem_ok": r["mem_ok"],
            "ok": r["ok"], "fails": r["fails"], "latency": t["latency_s"],
            "in": t["input_tokens"], "out": t["output_tokens"], "answer": t["answer"],
        }

    def _eval_worker(self):
        try:
            def on_case(r, done, total):
                self.eval["rows"].append(self._row(r))
                self.eval["done"] = done

            rows, agents = run_eval(on_case=on_case, quiet=True)
            n = len(rows)
            lat = [r["trace"]["latency_s"] for r in rows]
            tin = [r["trace"]["input_tokens"] for r in rows]
            tout = [r["trace"]["output_tokens"] for r in rows]
            full = next(iter(agents.values())).retriever.total_chars / 4
            pol = [r["trace"]["retrieved_chars"] / 4 for r in rows if r["trace"]["retrieved_chunks"]]
            self.eval["summary"] = {
                "success": sum(r["ok"] for r in rows), "n": n,
                "tool_ok": sum(r["tool_ok"] for r in rows), "evid_ok": sum(r["evid_ok"] for r in rows),
                "ans_ok": sum(r["ans_ok"] for r in rows), "mem_ok": sum(r["mem_ok"] for r in rows),
                "lat_mean": round(statistics.mean(lat), 2), "lat_median": round(statistics.median(lat), 2),
                "lat_max": max(lat), "in_total": sum(tin), "out_total": sum(tout),
                "in_mean": round(statistics.mean(tin)), "out_mean": round(statistics.mean(tout)),
                "grounded_first": sum(1 for r in rows if r["trace"]["grounding"]["status"] == "passed"),
                "context_saving": round(100 * (1 - statistics.mean(pol) / full)) if pol else None,
            }
            self.eval["status"] = "done"
        except SystemExit as e:  # e.g. missing API key
            self.eval.update(status="error", error=str(e))
        except Exception as e:
            self.eval.update(status="error", error=f"{type(e).__name__}: {e}")


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet console
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _student(self, q):
            sid = (q.get("student") or ["S001"])[0]
            return sid if sid in {s["id"] for s in app.students()} else None

        def do_GET(self):
            from urllib.parse import parse_qs, urlparse
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                return self._send(200, (WEB / "dashboard.html").read_bytes(), "text/html")
            if u.path == "/api/state":
                sid = self._student(q)
                return self._send(200, app.state(sid)) if sid else self._send(404, {"error": "unknown student"})
            if u.path == "/api/eval":
                return self._send(200, app.eval)
            self._send(404, {"error": "not found"})

        def do_POST(self):
            from urllib.parse import parse_qs, urlparse
            u = urlparse(self.path)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._send(400, {"error": "bad json"})
            sid = body.get("student", "S001")
            if sid not in {s["id"] for s in app.students()}:
                return self._send(404, {"error": "unknown student"})
            try:
                if u.path == "/api/chat":
                    msg = (body.get("message") or "").strip()
                    if not msg:
                        return self._send(400, {"error": "empty message"})
                    return self._send(200, app.chat(sid, msg[:1000]))
                if u.path == "/api/new":
                    app.new_session(sid)
                    return self._send(200, {"ok": True})
                if u.path == "/api/forget":
                    app.forget(sid)
                    return self._send(200, {"ok": True})
                if u.path == "/api/eval/start":
                    app.start_eval()
                    return self._send(200, {"ok": True})
            except SystemExit as e:
                return self._send(500, {"error": str(e)})
            self._send(404, {"error": "not found"})

    return Handler


def serve(app, host="127.0.0.1", port=8000, open_browser=True):
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    url = f"http://{host}:{port}"
    print(f"Dashboard running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return httpd


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Student Support Agent dashboard")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    serve(App(lambda sid: StudentSupportAgent(sid)), port=a.port, open_browser=not a.no_browser)

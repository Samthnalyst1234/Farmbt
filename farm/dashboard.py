"""Local web dashboard: browse projects, runs, findings, gaps and solutions, and start farm jobs.

Serves on 127.0.0.1 only. Jobs run as `python -m farm ...` subprocesses, one at a time, with logs in logs/.
"""
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import config
from .orchestrator import cap_confidence, ground
from .store import Store

ROOT = Path.cwd()
PAGE = Path(__file__).resolve().parent / "web" / "index.html"
LOGS = ROOT / "logs"
NAME = re.compile(r"^[a-z0-9-]{1,40}$")
GAP = re.compile(r"^G\d{1,3}$")


class Jobs:
    def __init__(self):
        self._jobs: dict[int, dict] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def running(self) -> bool:
        return any(j["proc"].poll() is None for j in self._jobs.values())

    def start(self, args: list[str], label: str) -> int:
        with self._lock:
            if self.running():
                raise ValueError("A job is already running. Wait for it to finish first.")
            LOGS.mkdir(exist_ok=True)
            job_id = next(self._ids)
            log_path = LOGS / f"job-{datetime.now():%Y%m%d-%H%M%S}-{job_id}.log"
            log = open(log_path, "w", encoding="utf-8")
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen([sys.executable, "-m", "farm", *args], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
            self._jobs[job_id] = {"id": job_id, "label": label, "proc": proc, "log": log_path,
                                  "started": datetime.now().isoformat(timespec="seconds")}
            return job_id

    def list(self) -> list[dict]:
        out = []
        for job in sorted(self._jobs.values(), key=lambda j: j["id"], reverse=True):
            code = job["proc"].poll()
            status = "running" if code is None else "finished" if code == 0 else f"failed (exit {code})"
            try:
                lines = job["log"].read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                lines = []
            out.append({"id": job["id"], "label": job["label"], "started": job["started"], "status": status,
                        "log": "\n".join(lines[-80:])})
        return out


def overview() -> dict:
    store = Store(config.DB_PATH)
    projects = []
    for p in store.projects():
        runs = [dict(r) for r in store.runs(p["id"])]
        projects.append({"slug": p["slug"], "mission": p["mission"], "created_at": p["created_at"], "runs": runs})
    return {"provider": config.PROVIDER, "model": config.DEFAULT_MODEL, "profile": config.PROFILE,
            "projects": projects}


def run_detail(run_id: int) -> dict | None:
    store = Store(config.DB_PATH)
    run = store.run(run_id)
    if not run:
        return None
    project = store.project_by_id(run["project_id"])
    research = store.stages_like(run_id, "research:%")
    findings = store.stage(run_id, "verify") or store.stage(run_id, "findings") or []
    if findings and "grounding" not in findings[0]:  # runs from before source checking existed
        sources = {r["question"]["id"]: r["sources"] for r in research}
        for qid, srcs in sources.items():
            ground([f for f in findings if f["question_id"] == qid], srcs)
    for f in findings:
        if "model_confidence" not in f:
            cap_confidence(f)
    report = ""
    if run["report_path"] and (ROOT / run["report_path"]).exists():
        report = (ROOT / run["report_path"]).read_text(encoding="utf-8")
    elif store.stage(run_id, "report"):
        report = store.stage(run_id, "report")["markdown"]
    return {
        "run": dict(run),
        "project": {"slug": project["slug"], "mission": project["mission"]},
        "plan": store.stage(run_id, "plan"),
        "research": [{"question": r["question"], "sources": r["sources"], "notes": r["notes"],
                      "count": len(r["findings"])} for r in research],
        "findings": findings,
        "landscape": store.stage(run_id, "landscape"),
        "gaps": (store.stage(run_id, "critique") or {}).get("gaps", []),
        "solutions": store.stages_like(run_id, "solution:%"),
        "report": report,
    }


def job_args(body: dict) -> tuple[list[str], str]:
    """Turn a dashboard request into farm CLI arguments, validating every field."""
    action = body.get("action")

    def count(key, low, high):
        value = body.get(key)
        if value in (None, ""):
            return []
        value = int(value)
        if not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        return [f"--{key}", str(value)]

    def name(key="name"):
        value = str(body.get(key, "")).strip()
        if not NAME.match(value):
            raise ValueError("Project names use lowercase letters, digits and dashes (max 40).")
        return value

    if action == "research":
        mission = str(body.get("mission", "")).strip()
        if not 10 <= len(mission) <= 2000:
            raise ValueError("Describe the mission in 10 to 2000 characters.")
        args = ["research", mission] + (["--name", name()] if body.get("name") else [])
        return args + count("questions", 1, 10) + count("solve", 0, 5), f"Research: {mission[:60]}"
    if action == "monitor":
        slug = name()
        return ["monitor", slug] + count("solve", 0, 5), f"Monitor {slug}"
    if action == "resume":
        run_id = int(body.get("run_id"))
        return ["resume", str(run_id)], f"Resume run {run_id}"
    if action == "solve":
        slug = name()
        args = ["solve", slug]
        if body.get("gap"):
            gap = str(body["gap"]).upper()
            if not GAP.match(gap):
                raise ValueError("Gap IDs look like G1, G2, ...")
            args += ["--gap", gap] + (["--redo"] if body.get("redo") else [])
        return args, f"{'Redo solutions' if body.get('redo') else 'Solve'} {slug} {body.get('gap') or '(top gap)'}"
    raise ValueError(f"Unknown action {action!r}")


def serve(port: int, open_browser: bool = True) -> None:
    jobs = Jobs()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status: int = 200) -> None:
            self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json")

        def do_GET(self):
            path = urlparse(self.path).path
            try:
                if path in ("/", "/index.html"):
                    self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/overview":
                    self._json({**overview(), "jobs": jobs.list()})
                elif path == "/api/jobs":
                    self._json(jobs.list())
                elif m := re.fullmatch(r"/api/runs/(\d+)", path):
                    detail = run_detail(int(m.group(1)))
                    self._json(detail) if detail else self._json({"error": "no such run"}, 404)
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as exc:  # show the problem in the page rather than dropping the connection
                self._json({"error": repr(exc)}, 500)

        def do_POST(self):
            if urlparse(self.path).path != "/api/jobs":
                return self._json({"error": "not found"}, 404)
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                args, label = job_args(body)
                self._json({"id": jobs.start(args, label)})
            except (ValueError, TypeError) as exc:
                self._json({"error": str(exc)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Farm dashboard running at {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

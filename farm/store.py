"""SQLite knowledge base: projects, runs, every stage's output, and the findings and gaps it produced."""
import json
import sqlite3
import threading
from datetime import datetime

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY, slug TEXT UNIQUE NOT NULL, mission TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL REFERENCES projects(id),
    started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL,
    cost_usd REAL NOT NULL DEFAULT 0, report_path TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS stages (
    run_id INTEGER NOT NULL REFERENCES runs(id), name TEXT NOT NULL, data TEXT NOT NULL,
    created_at TEXT NOT NULL, PRIMARY KEY (run_id, name));
CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL REFERENCES runs(id), fid TEXT, question_id TEXT,
    claim TEXT, evidence TEXT, kind TEXT, confidence TEXT, source_url TEXT, source_title TEXT, published TEXT);
CREATE TABLE IF NOT EXISTS gaps (
    id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL REFERENCES runs(id), gid TEXT, title TEXT,
    description TEXT, impact TEXT, difficulty TEXT, status TEXT, verdict TEXT, confidence REAL);
"""


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._lock = threading.Lock()  # research workers write from several threads

    def _exec(self, sql: str, args=()) -> sqlite3.Cursor:
        with self._lock, self.db:
            return self.db.execute(sql, args)

    def _one(self, sql: str, args=()):
        with self._lock:
            return self.db.execute(sql, args).fetchone()

    def _all(self, sql: str, args=()) -> list:
        with self._lock:
            return self.db.execute(sql, args).fetchall()

    # projects and runs
    def project(self, slug: str):
        return self._one("SELECT * FROM projects WHERE slug = ?", (slug,))

    def project_by_id(self, project_id: int):
        return self._one("SELECT * FROM projects WHERE id = ?", (project_id,))

    def create_project(self, slug: str, mission: str):
        self._exec("INSERT INTO projects (slug, mission, created_at) VALUES (?, ?, ?)", (slug, mission, now()))
        return self.project(slug)

    def projects(self) -> list:
        return self._all("""
            SELECT p.*, COUNT(r.id) AS runs, MAX(r.id) AS last_run,
                   (SELECT status FROM runs WHERE project_id = p.id ORDER BY id DESC LIMIT 1) AS last_status
            FROM projects p LEFT JOIN runs r ON r.project_id = p.id GROUP BY p.id ORDER BY p.id""")

    def runs(self, project_id: int) -> list:
        return self._all("SELECT * FROM runs WHERE project_id = ? ORDER BY id DESC", (project_id,))

    def create_run(self, project_id: int) -> int:
        cur = self._exec("INSERT INTO runs (project_id, started_at, status) VALUES (?, ?, 'running')",
                         (project_id, now()))
        return cur.lastrowid

    def run(self, run_id: int):
        return self._one("SELECT * FROM runs WHERE id = ?", (run_id,))

    def latest_run(self, project_id: int, status: str = "done"):
        return self._one("SELECT * FROM runs WHERE project_id = ? AND status = ? ORDER BY id DESC LIMIT 1",
                         (project_id, status))

    def set_status(self, run_id: int, status: str, error: str | None = None) -> None:
        finished = now() if status != "running" else None
        self._exec("UPDATE runs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                   (status, error, finished, run_id))

    def set_cost(self, run_id: int, cost: float) -> None:
        self._exec("UPDATE runs SET cost_usd = ? WHERE id = ?", (cost, run_id))

    def set_report(self, run_id: int, path: str) -> None:
        self._exec("UPDATE runs SET report_path = ? WHERE id = ?", (path, run_id))

    # stage outputs (these make runs resumable)
    def stage(self, run_id: int, name: str):
        row = self._one("SELECT data FROM stages WHERE run_id = ? AND name = ?", (run_id, name))
        return json.loads(row["data"]) if row else None

    def stages_like(self, run_id: int, pattern: str) -> list:
        rows = self._all("SELECT data FROM stages WHERE run_id = ? AND name LIKE ? ORDER BY created_at",
                         (run_id, pattern))
        return [json.loads(row["data"]) for row in rows]

    def delete_stages(self, run_id: int, pattern: str) -> int:
        return self._exec("DELETE FROM stages WHERE run_id = ? AND name LIKE ?", (run_id, pattern)).rowcount

    def save_stage(self, run_id: int, name: str, data) -> None:
        self._exec("INSERT OR REPLACE INTO stages (run_id, name, data, created_at) VALUES (?, ?, ?, ?)",
                   (run_id, name, json.dumps(data, ensure_ascii=False), now()))

    # findings and gaps, stored flat so they can be queried across runs
    def save_findings(self, run_id: int, findings: list[dict]) -> None:
        with self._lock, self.db:
            self.db.execute("DELETE FROM findings WHERE run_id = ?", (run_id,))
            self.db.executemany(
                "INSERT INTO findings (run_id, fid, question_id, claim, evidence, kind, confidence,"
                " source_url, source_title, published) VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(run_id, f["id"], f["question_id"], f["claim"], f["evidence"], f["kind"], f["confidence"],
                  f["source_url"], f["source_title"], f["published"]) for f in findings])

    def save_gaps(self, run_id: int, gaps: list[dict]) -> None:
        with self._lock, self.db:
            self.db.execute("DELETE FROM gaps WHERE run_id = ?", (run_id,))
            self.db.executemany(
                "INSERT INTO gaps (run_id, gid, title, description, impact, difficulty, status, verdict,"
                " confidence) VALUES (?,?,?,?,?,?,?,?,?)",
                [(run_id, g["id"], g["title"], g["description"], g["impact"], g["difficulty"], g["status"],
                  g["review"]["verdict"], g["review"]["confidence"]) for g in gaps])

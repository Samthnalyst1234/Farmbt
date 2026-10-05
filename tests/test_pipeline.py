"""End-to-end runs of the whole farm with mock agents: no API key, no network."""
import pytest

from farm import config
from farm.llm import LLM, Meter
from farm.orchestrator import Farm
from farm.store import Store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS_DIR", str(tmp_path / "reports"))
    return Store(str(tmp_path / "farm.db"))


def new_run(store: Store, slug: str = "test-project") -> int:
    project = store.create_project(slug, "How do crop-disease apps work, and why do farmers stop using them?")
    return store.create_run(project["id"])


def test_full_run_produces_every_stage_and_a_report(store, tmp_path):
    run_id = new_run(store)
    path = Farm(store, LLM(Meter(None), mock=True), questions=2, workers=1, log=lambda m: None).run(run_id)

    assert store.run(run_id)["status"] == "done"
    for stage in ("plan", "research:Q1", "research:Q2", "findings", "verify", "landscape",
                  "gaps:pass1", "gaps", "critique", "solve:G1:prior_art", "solution:G1", "report"):
        assert store.stage(run_id, stage) is not None, stage
    report = open(path, encoding="utf-8").read()
    assert "## Solving G1" in report
    assert "Evidence quality" in report
    assert (tmp_path / "reports" / "test-project" / "latest.md").exists()


class ChargingMock(LLM):
    """Mock agents that each cost $1, so a small budget runs out part-way through."""

    def ask(self, role, system, prompt, **kwargs):
        reply = super().ask(role, system, prompt, **kwargs)
        self.meter.cost += 1.0
        return reply


def test_budget_pause_then_resume_reuses_finished_work(store):
    run_id = new_run(store)
    quiet = dict(questions=2, workers=1, log=lambda m: None)

    assert Farm(store, ChargingMock(Meter(2.5), mock=True), **quiet).run(run_id) is None
    assert store.run(run_id)["status"] == "paused"
    plan_before = store.stage(run_id, "plan")

    second = ChargingMock(Meter(None), mock=True)
    assert Farm(store, second, **quiet).run(run_id) is not None
    assert store.run(run_id)["status"] == "done"
    assert store.stage(run_id, "plan") == plan_before  # not re-planned
    assert store.run(run_id)["cost_usd"] > 2.5  # cost carries across sessions


def test_solve_redo_designs_solutions_for_a_chosen_gap(store):
    run_id = new_run(store)
    farm = Farm(store, LLM(Meter(None), mock=True), questions=2, workers=1, solve=0, log=lambda m: None)
    farm.run(run_id)
    assert store.stages_like(run_id, "solution:%") == []

    farm.solve(run_id, "G2")
    solved = store.stages_like(run_id, "solution:%")
    assert [s["gap"]["id"] for s in solved] == ["G2"]
    assert solved[0]["prior_art"]

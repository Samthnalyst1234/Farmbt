"""The dashboard turns browser requests into farm commands; nothing unvalidated may get through."""
import pytest

from farm.dashboard import job_args


def test_research_request_becomes_cli_arguments():
    args, label = job_args({"action": "research", "mission": "Why do small shops struggle with payments?",
                            "name": "shop-payments", "questions": "3", "solve": ""})
    assert args == ["research", "Why do small shops struggle with payments?", "--name", "shop-payments",
                    "--questions", "3"]
    assert label.startswith("Research:")


def test_solve_request_with_redo():
    args, _ = job_args({"action": "solve", "name": "crop-apps", "gap": "g3", "redo": True})
    assert args == ["solve", "crop-apps", "--gap", "G3", "--redo"]


@pytest.mark.parametrize("body", [
    {"action": "rm -rf"},
    {"action": "research", "mission": "short"},
    {"action": "monitor", "name": "Bad Name!"},
    {"action": "solve", "name": "ok-name", "gap": "G1; del *"},
    {"action": "research", "mission": "A long enough mission here", "questions": "99"},
])
def test_invalid_requests_are_rejected(body):
    with pytest.raises(ValueError):
        job_args(body)

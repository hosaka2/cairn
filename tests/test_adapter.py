"""Running inference in this process.

The inline adapter is what makes a local run possible without an orchestrator. It has
to behave like one: a job it does not know is an error, a job that raises is a failure
it reports, and a sample with nothing to say writes nothing.
"""

from __future__ import annotations

import pytest

from cairn.adapters.local import InlineAdapter
from cairn.core import dataset as ds
from cairn.core import evals
from cairn.core.records import RunSpec
from cairn.core.storage import Storage

DS = "name: d\nkey: id\ncolumns: [{name: id, type: str, required: true}]\n"
TABLE = "name: t\ncolumns: [{name: score, type: float}]\n"


@pytest.fixture
def run(tmp_path) -> tuple[Storage, str]:
    st = Storage(f"file://{tmp_path}/cairn")
    ds.create_dataset(st, DS, created_by="t")
    ds.append(st, "d", [{"id": "a"}, {"id": "b"}], created_by="t", input={})
    evals.create_eval_table(st, TABLE)
    return st, evals.create_run(st, "t", dataset="d", evaluator_version="v1", title="run")


def _spec(*ids: str) -> RunSpec:
    return RunSpec(sample_ids=list(ids), job_name="infer", run_config={})


def test_what_a_job_returns_is_stored_as_the_prediction(run):
    st, eid = run
    adapter = InlineAdapter(st, "t", eid, jobs={"infer": lambda sid, cfg: f"p-{sid}".encode()})

    run_id = adapter.submit(_spec("a", "b"), tags={})

    assert dict(evals._prediction_loader(st, "t", eid)()) == {"a": b"p-a", "b": b"p-b"}
    assert adapter.status([run_id])[run_id].state == "success"


def test_a_sample_with_no_prediction_is_left_out_rather_than_failing(run):
    """'This run produced nothing for that sample' is a normal outcome, not an error."""
    st, eid = run
    adapter = InlineAdapter(st, "t", eid, jobs={"infer": lambda sid, cfg: None if sid == "b" else b"p"})

    adapter.submit(_spec("a", "b"), tags={})

    assert dict(evals._prediction_loader(st, "t", eid)()) == {"a": b"p"}


def test_an_unregistered_job_is_refused(run):
    st, eid = run
    adapter = InlineAdapter(st, "t", eid, jobs={})

    with pytest.raises(KeyError, match="is not registered"):
        adapter.submit(_spec("a"), tags={})


def test_a_job_that_raises_is_reported_as_a_failure(run):
    st, eid = run

    def boom(sample_id: str, cfg: dict) -> bytes:
        raise RuntimeError("model died")

    adapter = InlineAdapter(st, "t", eid, jobs={"infer": boom})

    with pytest.raises(RuntimeError, match="model died"):
        adapter.submit(_spec("a"), tags={})
    assert [s.state for s in adapter.status(["inline:infer:a"]).values()] == ["failure"]


def test_a_run_the_adapter_never_saw_has_an_unknown_state(run):
    st, eid = run
    adapter = InlineAdapter(st, "t", eid, jobs={"infer": lambda sid, cfg: b"p"})

    assert adapter.status(["somewhere-else"])["somewhere-else"].state == "unknown"

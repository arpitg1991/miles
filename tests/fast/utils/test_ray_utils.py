import pytest
import ray

from miles.utils import ray_utils


def _node(node_id: str, *, head: bool, alive: bool = True) -> dict:
    resources = {"CPU": 8.0, "node:10.0.0.1": 1.0}
    if head:
        resources["node:__internal_head__"] = 1.0
    return {"NodeID": node_id, "Alive": alive, "Resources": resources}


# Ray node ids are 28-byte hex strings; NodeAffinitySchedulingStrategy validates them.
WORKER, HEAD, NEW_HEAD = "a1" * 28, "b2" * 28, "c3" * 28


class TestGetHeadNodeId:
    def test_the_head_is_found_from_the_gcs_node_table(self, monkeypatch: pytest.MonkeyPatch):
        """The lookup reads ray.nodes(), so it does not need the dashboard of the caller's node."""
        monkeypatch.setattr(ray, "nodes", lambda: [_node(WORKER, head=False), _node(HEAD, head=True)])
        assert ray_utils._get_head_node_id() == HEAD

    def test_a_dead_head_entry_is_skipped(self, monkeypatch: pytest.MonkeyPatch):
        """A restarted head leaves a dead entry with the head resource in the GCS table."""
        monkeypatch.setattr(ray, "nodes", lambda: [_node(HEAD, head=True, alive=False), _node(NEW_HEAD, head=True)])
        assert ray_utils._get_head_node_id() == NEW_HEAD

    def test_no_head_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(ray, "nodes", lambda: [_node(WORKER, head=False)])
        with pytest.raises(RuntimeError, match="Could not find a head node"):
            ray_utils._get_head_node_id()

    def test_pin_options_use_hard_node_affinity(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(ray, "nodes", lambda: [_node(HEAD, head=True)])
        strategy = ray_utils.compute_ray_pin_head_options()["scheduling_strategy"]
        assert strategy.node_id == HEAD
        assert strategy.soft is False

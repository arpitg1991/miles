"""Ray mode routes a worker method to its concurrency group (_ServeActorRayCommManager)."""

import inspect

from miles.utils.workers.ray_worker_manager import _route_method_to_concurrency_group
from miles.utils.workers.rpc.common.metadata import _find_rpc_config, rpc


class _Worker:
    @rpc(concurrency_group="side")
    def side(self, rollout_id: int, *, tag: str = "x") -> int:
        return rollout_id


def test_ray_reads_the_group_from_the_routed_method():
    """Ray reads __ray_concurrency_group__ from inspect.unwrap(method); the unrouted method has none."""
    routed = _route_method_to_concurrency_group(_Worker.side, group="side")

    assert inspect.unwrap(routed).__ray_concurrency_group__ == "side"
    assert not hasattr(_Worker.side, "__ray_concurrency_group__")


def test_the_routed_method_keeps_its_rpc_group_and_signature():
    routed = _route_method_to_concurrency_group(_Worker.side, group="side")

    assert _find_rpc_config(routed).concurrency_group == "side"
    assert inspect.signature(routed) == inspect.signature(_Worker.side)
    assert routed(_Worker(), 7) == 7

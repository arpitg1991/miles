import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# Ray gives the head node this custom resource (ray._common.constants.HEAD_NODE_RESOURCE_NAME).
_HEAD_NODE_RESOURCE = "node:__internal_head__"


class Box:
    def __init__(self, inner):
        self._inner = inner

    @property
    def inner(self):
        return self._inner


def compute_ray_pin_head_options():
    head_node_id = _get_head_node_id()
    return {
        "scheduling_strategy": NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        )
    }


def _get_head_node_id() -> str:
    # ray.nodes() reads the GCS, so any node can call it. The state API (ray.util.state.list_nodes)
    # goes through the dashboard, which `ray start --head` binds to 127.0.0.1 by default: the
    # RayWorkerManager actor calls this function and can run on a worker node, where that
    # request is refused (ServerUnavailable) and the launch stops.
    for node in ray.nodes():
        if node.get("Alive") and _HEAD_NODE_RESOURCE in node.get("Resources", {}):
            return node["NodeID"]
    raise RuntimeError("Could not find a head node in the Ray cluster")

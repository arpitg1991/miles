"""--arena-inflight-multiplier: publisher in-flight cap as a multiple of
rollout_batch_size. Default 2 keeps every existing run byte-identical
(max_in_flight = 2 x rollout_batch_size); the YAML key
``arena_inflight_multiplier: 4`` flattens to ``--arena-inflight-multiplier 4``.
"""

import argparse
from types import SimpleNamespace

import pytest

from miles_plugins.arena.nats_arena.nats_rollout import (
    _add_arena_arguments,
    _publisher_max_in_flight,
)


@pytest.mark.parametrize(
    ("multiplier", "expected"),
    [(None, 128), (2, 128), (4, 256)],
)
def test_max_in_flight_scales_with_multiplier(multiplier, expected):
    args = SimpleNamespace(rollout_batch_size=64)
    if multiplier is not None:
        args.arena_inflight_multiplier = multiplier
    assert _publisher_max_in_flight(args) == expected


def test_argparse_default_and_yaml_flag():
    parser = argparse.ArgumentParser()
    _add_arena_arguments(parser)
    assert parser.parse_args([]).arena_inflight_multiplier == 2
    # scripts/run_arena_harbor.py flattens `arena_inflight_multiplier: 4` to this argv
    args = parser.parse_args(["--arena-inflight-multiplier", "4"])
    args.rollout_batch_size = 64
    assert _publisher_max_in_flight(args) == 256

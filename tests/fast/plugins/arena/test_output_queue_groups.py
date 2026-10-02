"""--arena-output-queue-groups: capacity of the worker output queue, in groups.

None (default) keeps every existing run byte-identical: 10 x global_batch_size
samples, 320 groups for GBS 256 and n 8. The YAML key
``arena_output_queue_groups: 64`` flattens to ``--arena-output-queue-groups 64``.
A value below rollout_batch_size stops the worker at startup.

Run: python -m pytest tests/fast/plugins/arena/test_output_queue_groups.py -v
"""

import argparse
from types import SimpleNamespace

import pytest
import transformers

from miles_plugins.arena.nats_arena.nats_rollout import NATSRolloutWorker, _add_arena_arguments, _output_queue_groups


def _args(**overrides) -> SimpleNamespace:
    """The r50 batch shape (GBS 256, n 8, 32 groups per rollout) and valid Harbor task limits."""
    args = SimpleNamespace(
        global_batch_size=256,
        n_samples_per_prompt=8,
        rollout_batch_size=32,
        rollout_max_response_len=16384,
        rollout_max_context_len=131072,
        sglang_context_length=None,
        rollout_top_k=-1,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        hf_checkpoint="unused",
    )
    vars(args).update(overrides)
    return args


@pytest.mark.parametrize(("gbs", "n", "expected"), [(256, 8, 320), (256, 16, 160), (4, 16, 2), (1, 16, 1)])
def test_default_keeps_ten_batches_of_samples(gbs: int, n: int, expected: int) -> None:
    args = SimpleNamespace(global_batch_size=gbs, n_samples_per_prompt=n)
    assert _output_queue_groups(args) == expected
    args.arena_output_queue_groups = None
    assert _output_queue_groups(args) == expected


def test_argparse_default_and_yaml_flag() -> None:
    parser = argparse.ArgumentParser()
    _add_arena_arguments(parser)
    assert parser.parse_args([]).arena_output_queue_groups is None
    # scripts/run_arena_harbor.py flattens `arena_output_queue_groups: 64` to this argv
    args = parser.parse_args(["--arena-output-queue-groups", "64"])
    vars(args).update(global_batch_size=256, n_samples_per_prompt=8, rollout_batch_size=32)
    assert _output_queue_groups(args) == 64


@pytest.mark.parametrize(("groups", "expected"), [(None, 320), (32, 32), (64, 64), (1000, 1000)])
def test_worker_queue_takes_the_capacity(monkeypatch: pytest.MonkeyPatch, groups: int | None, expected: int) -> None:
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: None)
    worker = NATSRolloutWorker(_args(arena_output_queue_groups=groups), data_source=None)
    assert worker.output_queue.maxsize == expected


def test_value_below_rollout_batch_size_stops_the_worker() -> None:
    # The check runs before the tokenizer load and before any publish.
    with pytest.raises(ValueError, match="--arena-output-queue-groups 31 is less than --rollout-batch-size 32"):
        NATSRolloutWorker(_args(arena_output_queue_groups=31), data_source=None)

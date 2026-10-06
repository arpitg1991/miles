"""--prefetch-rollout-data: the train group that the driver gets has the prefetch call."""

import inspect

import pytest

from miles.ray.placement_group import _select_train_group_class


@pytest.mark.parametrize(
    ("ft_trainer", "module"),
    [
        # The arena harness and r47 do not set the variable, so they run the v1 group.
        (None, "miles.ray.actor_group"),
        ("1", "miles.ray.train.group"),
    ],
)
def test_selected_train_group_has_prefetch_rollout_data(monkeypatch, ft_trainer, module):
    if ft_trainer is None:
        monkeypatch.delenv("MILES_EXPERIMENTAL_FT_TRAINER", raising=False)
    else:
        monkeypatch.setenv("MILES_EXPERIMENTAL_FT_TRAINER", ft_trainer)
    group_cls = _select_train_group_class()
    assert group_cls.__module__ == module
    # train_async_arena.py wraps the call in asyncio.create_task.
    assert inspect.iscoroutinefunction(group_cls.prefetch_rollout_data)

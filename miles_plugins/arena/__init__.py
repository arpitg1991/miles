"""Arena RL training plugins for miles (port of AGISlime's ``amzn_agi_slime``).

Submodules are imported lazily so that, e.g., the SGLang-only parser is usable
on hosts without the full training stack. Import what you need:

    from miles_plugins.arena.rewards import binarize_reward
    from miles_plugins.arena.parsers import register_nova_reasoning_parser
"""

__all__ = ["binarize_reward", "register_nova_reasoning_parser", "split_reasoning"]


def __getattr__(name):  # PEP 562 lazy attribute access
    if name == "binarize_reward":
        from miles_plugins.arena.rewards import binarize_reward

        return binarize_reward
    if name in ("register_nova_reasoning_parser", "split_reasoning"):
        from miles_plugins.arena import parsers

        return getattr(parsers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

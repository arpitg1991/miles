"""Custom SGLang reasoning parser for Nova-style reasoning markers.

Handles <|begin_internal_thought|> ... <|end_internal_thought|> markers used by
Nova/Qwen models trained with internal thought tokens. Adapted from AGISlime
slime/custom_parsers/nova_reasoning_parser.py.

Register by calling register_nova_reasoning_parser() before the SGLang server
starts. SGLang is imported lazily so this module is importable without it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

START_TOKEN = "<|begin_internal_thought|>"
END_TOKEN = "<|end_internal_thought|>"


def split_reasoning(text: str) -> tuple[str, str]:
    """Split Nova-style internal-thought markers into (reasoning, content).

    Pure-Python and SGLang-free so it can be unit-tested directly.
    """
    start_idx = text.find(START_TOKEN)
    if start_idx == -1:
        return "", text

    end_idx = text.find(END_TOKEN)
    if end_idx == -1:
        # Started thinking but never closed it.
        return text[start_idx + len(START_TOKEN):], ""

    reasoning = text[start_idx + len(START_TOKEN):end_idx]
    content = text[end_idx + len(END_TOKEN):].strip()
    return reasoning, content


def register_nova_reasoning_parser() -> bool:
    """Register the Nova reasoning parser with SGLang.

    Returns True if registered, False if SGLang is unavailable.
    """
    try:
        from sglang.srt.parser.reasoning_parser import (
            BaseReasoningFormatDetector,
            ReasoningParser,
        )
    except ImportError:
        logger.warning("SGLang not available, skipping Nova parser registration")
        return False

    class NovaReasoningDetector(BaseReasoningFormatDetector):
        """Detects and separates Nova-style internal thought markers."""

        def __init__(self, **kwargs):
            super().__init__(
                START_TOKEN,
                END_TOKEN,
                thinks_internally=True,
                **kwargs,
            )

    ReasoningParser.DetectorMap["nova"] = NovaReasoningDetector
    register_nova_reasoning_parser.detector_cls = NovaReasoningDetector
    return True

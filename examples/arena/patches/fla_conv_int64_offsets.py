"""Use int64 token offsets in fla's causal_conv1d Triton kernels (fla PR #1082).

fla 0.5.2 moved the short-convolution kernels off ``tl.make_block_ptr``
(fla PR #1062) and computes ``o_t = i_t * BT + tl.arange(0, BT)`` in int32.
The pointer offset ``o_t * D`` (and ``o_x * stride_x_t``) then overflows when
one sequence has more than 2**31 / D tokens. GLM-5.3-Flash convolves q, k
and v together (D = 3 * 64 * 128 = 24,576), so every row longer than 87,381
tokens reads outside the tensor: ``CUDA error: an illegal memory access`` in
``causal_conv1d_fwd_kernel`` at the first long row of a train step. The r45
rows reach 131,070 tokens. fla 0.4.2 used block pointers (64-bit offsets)
and does not have the fault.

fla fixed it after 0.5.2 in PR #1082 (commit 31d15f75, 2026-08-01), with no
release yet. This script applies the five #1082 lines of
``fla/modules/conv/triton/kernels.py`` and changes no other file. The
patched file is equal to the file at 31d15f75. The values do not change:
only the index width changes.

Idempotent. It skips fla 0.4.x (block pointers) and fla that already has
#1082. It hard-fails when neither form matches, so a version bump cannot
ship an unpatched kernel. ``--fla-dir`` patches a copy (test use).
"""

import argparse
import ast
import importlib.util
import pathlib
import sys

KERNEL = "modules/conv/triton/kernels.py"

# (old, new, expected count in the 0.5.2 file); from fla PR #1082. The leading
# newline keeps the 4-space and the 8-space anchors apart.
REPLACEMENTS = (
    (
        "\n    o_t = i_t * BT + tl.arange(0, BT)\n",
        "\n    o_t = i_t.to(tl.int64) * BT + tl.arange(0, BT)\n",
        2,
    ),
    (
        "\n        o_t = i_t * BT + tl.arange(0, BT)\n",
        "\n        o_t = i_t.to(tl.int64) * BT + tl.arange(0, BT)\n",
        2,
    ),
    (
        "\n    o_x = seq_len - BW + tl.arange(0, BW)\n",
        "\n    o_x = seq_len.to(tl.int64) - BW + tl.arange(0, BW)\n",
        1,
    ),
)
BLOCK_POINTER_FORM = "p_yi = tl.make_block_ptr("


def _target(fla_dir: str | None) -> pathlib.Path | None:
    if fla_dir:
        return pathlib.Path(fla_dir) / KERNEL
    # The base decides where fla lives: dist-packages on glm53next, the
    # /opt/sglang venv on the upstream base. find_spec does not import fla.
    spec = importlib.util.find_spec("fla")
    if spec is None or not spec.submodule_search_locations:
        return None
    return pathlib.Path(spec.submodule_search_locations[0]) / KERNEL


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fla-dir", default=None, help="patch this fla package dir instead of the installed one")
    target = _target(parser.parse_args().fla_dir)
    if target is None or not target.exists():
        print(f"fla conv int64 patch: {target or 'fla'} absent, nothing to patch")
        return 0
    src = target.read_text()
    if BLOCK_POINTER_FORM in src:
        print(f"fla conv int64 patch: block-pointer kernels (fla 0.4.x), not affected ({target})")
        return 0
    old_counts = [src.count(old) for old, _new, _n in REPLACEMENTS]
    new_counts = [src.count(new) for _old, new, _n in REPLACEMENTS]
    expected = [n for _old, _new, n in REPLACEMENTS]
    # fla after #1082 writes the states-kernel line as tl.cast(seq_len, tl.int64); the o_t lines decide.
    if old_counts == [0] * len(REPLACEMENTS) and new_counts[:2] == expected[:2]:
        print(f"fla conv int64 patch: already applied ({target})")
        return 0
    if old_counts != expected:
        print(f"fla conv int64 patch: anchors {old_counts} != {expected} — fla changed, refusing to guess")
        return 1
    for old, new, _n in REPLACEMENTS:
        src = src.replace(old, new)
    ast.parse(src)
    target.write_text(src)
    print(f"fla conv int64 patch: applied ({target})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

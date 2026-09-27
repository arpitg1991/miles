"""Hoist the in-kernel `BK = triton.next_power_of_2(K)` out of fla's KDA
token-parallel kernel.

fla 0.4.2 declares `BK: tl.constexpr = triton.next_power_of_2(K)` INSIDE
`chunk_kda_fwd_kernel_intra_token_parallel`. Triton 3.7.1's JIT dependency
walker (jit.py record_reference/visit_Attribute) rejects host functions
referenced from kernel bodies with `RuntimeError: Unsupported function
referenced: <function next_power_of_2>` — every GLM-5.3-Flash (KDA) train
step crashes at the first forward. The sibling `intra_sub_chunk` path already
computes BK on the host; this patch makes the token-parallel kernel do the
same (semantics identical; K=128 is already a power of two).

Idempotent; hard-fails if fla's code no longer matches so a version bump
cannot silently ship an unpatched kernel.

The upstream miles base (docker/Dockerfile, fla 0.5.2) applies
docker/patch/fla_kda_hoist_next_power_of_2.patch. That patch puts the BK
parameter after K, not before IS_VARLEN. This script accepts both forms as
applied: the kernel body has no next_power_of_2 call, and the launch passes BK.
"""
import importlib.util
import pathlib
import sys

KERNEL = "ops/kda/chunk_intra_token_parallel.py"

BROKEN_BODY = "    BK: tl.constexpr = triton.next_power_of_2(K)\n"
SIG_ANCHOR = "    BH: tl.constexpr,\n    IS_VARLEN: tl.constexpr,\n):"
SIG_PATCHED = "    BH: tl.constexpr,\n    BK: tl.constexpr,\n    IS_VARLEN: tl.constexpr,\n):"
LAUNCH_ANCHOR = "        BT=BT,\n        BC=BC,\n    )\n    return Aqk, Akk"
LAUNCH_PATCHED = "        BT=BT,\n        BC=BC,\n        BK=triton.next_power_of_2(K),\n    )\n    return Aqk, Akk"
# Present in both applied forms (this script and the upstream docker patch).
HOSTED_SIG = "    BK: tl.constexpr,\n"
HOSTED_LAUNCH = "        BK=triton.next_power_of_2(K),\n"


def _target() -> pathlib.Path | None:
    # The base decides where fla lives: dist-packages on glm53next, the
    # /opt/sglang venv on the upstream base. find_spec does not import fla.
    spec = importlib.util.find_spec("fla")
    if spec is None or not spec.submodule_search_locations:
        return None
    return pathlib.Path(spec.submodule_search_locations[0]) / KERNEL


def main() -> int:
    target = _target()
    if target is None or not target.exists():
        print(f"fla KDA patch: {target or 'fla'} absent, nothing to patch")
        return 0
    src = target.read_text()
    if BROKEN_BODY not in src and HOSTED_SIG in src and HOSTED_LAUNCH in src:
        print(f"fla KDA patch: already applied ({target})")
        return 0
    for name, needle in [("body", BROKEN_BODY), ("signature", SIG_ANCHOR), ("launch", LAUNCH_ANCHOR)]:
        if needle not in src:
            print(f"fla KDA patch: {name} anchor not found — fla changed, refusing to guess")
            return 1
    src = src.replace(BROKEN_BODY, "")
    src = src.replace(SIG_ANCHOR, SIG_PATCHED)
    src = src.replace(LAUNCH_ANCHOR, LAUNCH_PATCHED)
    target.write_text(src)
    import ast
    ast.parse(src)
    print(f"fla KDA patch: applied ({target})")
    return 0

if __name__ == "__main__":
    sys.exit(main())

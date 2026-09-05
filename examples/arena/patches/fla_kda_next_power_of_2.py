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
"""
import pathlib
import sys

TARGET = pathlib.Path("/usr/local/lib/python3.12/dist-packages/fla/ops/kda/chunk_intra_token_parallel.py")

BROKEN_BODY = "    BK: tl.constexpr = triton.next_power_of_2(K)\n"
SIG_ANCHOR = "    BH: tl.constexpr,\n    IS_VARLEN: tl.constexpr,\n):"
SIG_PATCHED = "    BH: tl.constexpr,\n    BK: tl.constexpr,\n    IS_VARLEN: tl.constexpr,\n):"
LAUNCH_ANCHOR = "        BT=BT,\n        BC=BC,\n    )\n    return Aqk, Akk"
LAUNCH_PATCHED = "        BT=BT,\n        BC=BC,\n        BK=triton.next_power_of_2(K),\n    )\n    return Aqk, Akk"

def main() -> int:
    if not TARGET.exists():
        print(f"fla KDA patch: {TARGET} absent, nothing to patch")
        return 0
    src = TARGET.read_text()
    if "BK: tl.constexpr,\n    IS_VARLEN" in src and BROKEN_BODY not in src:
        print("fla KDA patch: already applied")
        return 0
    for name, needle in [("body", BROKEN_BODY), ("signature", SIG_ANCHOR), ("launch", LAUNCH_ANCHOR)]:
        if needle not in src:
            print(f"fla KDA patch: {name} anchor not found — fla changed, refusing to guess")
            return 1
    src = src.replace(BROKEN_BODY, "")
    src = src.replace(SIG_ANCHOR, SIG_PATCHED)
    src = src.replace(LAUNCH_ANCHOR, LAUNCH_PATCHED)
    TARGET.write_text(src)
    import ast
    ast.parse(src)
    print("fla KDA patch: applied")
    return 0

if __name__ == "__main__":
    sys.exit(main())

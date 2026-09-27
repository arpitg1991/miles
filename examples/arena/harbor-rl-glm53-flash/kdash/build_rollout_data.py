"""Build train-only rollout data for the kdatp and kdash tests (``--load-debug-rollout-data``).

No gym and no SGLang run: miles loads ``rollout_{rollout_id}.pt`` in place of
a rollout (``debug_train_only``). This script writes those files with the
shape of a real r45 step and the token data of real r43 trajectories.

- Batch structure: whole groups of an r45 ``sample_summary`` file, one row
  per segment. Each row keeps its ``group_index``, ``sample_rollout_id``,
  ``index``, ``reward`` and ``total_length``.
- Token data: for each row, a leftover r43 staged step (``<prefix>.tokens``
  plus ``<prefix>.routing`` under the r43 routing dir). The row takes the
  first ``total_length`` tokens, with their loss mask, SGLang log-probs and
  R3 routing. The r43 policy at or before iteration 39 made these tokens.
- A row longer than every r43 step gets two steps joined; the one boundary
  routing row is -1 (``pad_routing``).
- The production DP pad (``_pad_rows_to_dp_alignment``) makes the row count
  a multiple of the data-parallel size.

The source files are only read. This script NEVER calls the reap or
resolve helpers of ``routing_replay``: they delete the source file.

Example (T2, 8 groups of r45 rollout 43, rollouts 40..44 share one file):

    python3 build_rollout_data.py \\
      --summary /mnt/scratch-s3files-rw/guparpit/debug/rl-glm53f-adebt-v3-r45/sample_summary/rollout_43.jsonl \\
      --groups 243,284,307,258,239,306,276,309 \\
      --routing-dir /mnt/scratch-s3files-rw/guparpit/routing/rl-glm53f-adebt-v3-r43 \\
      --out-dir /mnt/scratch-s3files-rw/guparpit/kdatp/data/t2 --rollout-ids 40,41,42,43,44 \\
      --dp-size 2
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from miles.utils.types import Sample
from miles_plugins.arena.nats_arena.nats_rollout import _pad_rows_to_dp_alignment
from miles_plugins.arena.nats_arena.routing_replay import TOKEN_BYTES_PER_TOKEN, decode_routing, pad_routing

NUM_LAYERS, TOPK = 45, 8  # GLM-5.3-Flash R3 payload: 45 layers x top-8 per token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", required=True, help="r45 sample_summary rollout_N.jsonl")
    parser.add_argument("--groups", required=True, help="comma-separated group_index values")
    parser.add_argument("--routing-dir", required=True, help="dir with the <prefix>.tokens/.routing pairs")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rollout-ids", required=True, help="first id gets the file; the others link to it")
    parser.add_argument("--dp-size", type=int, required=True, help="actor data-parallel size of the target run")
    return parser.parse_args()


def read_summary(path: Path, groups: list[int]) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["group_index"] in set(groups)]
    found = {row["group_index"] for row in rows}
    assert found == set(groups), f"groups missing from {path}: {sorted(set(groups) - found)}"
    return rows


def staged_pairs(routing_dir: Path) -> list[tuple[int, Path, Path]]:
    """(n_tokens, tokens file, routing file) for each complete pair, from the file sizes only."""
    routing = {}
    tokens = {}
    for entry in os.scandir(routing_dir):
        stem, _, suffix = entry.name.rpartition(".")
        prefix = stem.rsplit("-", 1)[0]  # drop the per-file uuid
        if suffix == "routing":
            routing[prefix] = Path(entry.path)
        elif suffix == "tokens":
            tokens[prefix] = (entry.stat().st_size, Path(entry.path))
    pairs = []
    for prefix, (size, tokens_path) in tokens.items():
        if prefix in routing and size % TOKEN_BYTES_PER_TOKEN == 0:
            pairs.append((size // TOKEN_BYTES_PER_TOKEN, tokens_path, routing[prefix]))
    return sorted(pairs)


def read_tokens(tokens_path: Path) -> dict:
    """The .tokens layout of ADR-0014: float64 log-probs, int32 token ids, uint8 loss mask."""
    raw = tokens_path.read_bytes()
    n = len(raw) // TOKEN_BYTES_PER_TOKEN
    return {
        "log_probs": np.frombuffer(raw, dtype="<f8", count=n),
        "token_ids": np.frombuffer(raw, dtype="<i4", count=n, offset=8 * n),
        "loss_mask": np.frombuffer(raw, dtype="u1", count=n, offset=12 * n),
        "src": [str(tokens_path)],
    }


def add_routing(step: dict, routing_path: Path) -> dict:
    n = len(step["token_ids"])
    step["routing"] = decode_routing(
        routing_path.read_bytes(), num_tokens=n, num_layers=NUM_LAYERS, topk=TOPK, allow_extra_rows=True
    )
    return step


def read_step(tokens_path: Path, routing_path: Path) -> dict:
    return add_routing(read_tokens(tokens_path), routing_path)


def join_steps(first: dict, second: dict, length: int) -> dict:
    """First step, then the head of the second one, cut to ``length`` tokens."""
    need = length - len(first["token_ids"])
    assert 0 < need < len(second["token_ids"]), f"cannot reach {length} tokens with two steps"
    boundary = pad_routing((1, NUM_LAYERS, TOPK))
    return {
        "log_probs": np.concatenate([first["log_probs"], second["log_probs"][:need]]),
        "token_ids": np.concatenate([first["token_ids"], second["token_ids"][:need]]),
        "loss_mask": np.concatenate([first["loss_mask"], second["loss_mask"][:need]]),
        "routing": np.concatenate([first["routing"], boundary, second["routing"][: need - 1]]),
        "src": first["src"] + second["src"],
    }


def make_sample(row: dict, step: dict) -> Sample:
    length = row["total_length"]
    mask = step["loss_mask"][:length]
    ones = np.flatnonzero(mask)
    assert ones.size, f"row {row['index']}: no trainable token in the first {length} tokens"
    prompt_len = int(ones[0])  # the nats_rollout convention: the response starts at the first 1
    # A removed row keeps its place; convert_samples_to_train_data zeroes its loss mask.
    sample = Sample(
        group_index=row["group_index"],
        index=row["index"],
        rollout_id=row["sample_rollout_id"],
        tokens=step["token_ids"][:length].tolist(),
        response_length=length - prompt_len,
        reward=row["reward"],
        loss_mask=mask[prompt_len:].tolist(),
        rollout_log_probs=step["log_probs"][prompt_len:length].tolist(),
        rollout_routed_experts=np.ascontiguousarray(step["routing"][: length - 1]),
        status=Sample.Status(row["status"]),
        remove_sample=row["remove_sample"],
        metadata={"kdatp_src": step["src"], "segment_k": row.get("segment_k")},
    )
    sample.validate()
    return sample


def main() -> None:
    cli = parse_args()
    groups = [int(x) for x in cli.groups.split(",")]
    rows = read_summary(Path(cli.summary), groups)
    pairs = staged_pairs(Path(cli.routing_dir))
    assert pairs, f"no .tokens/.routing pair under {cli.routing_dir}"
    longest = pairs[-1]

    # Longest rows first; each takes the shortest unused pair that is long enough.
    used: set[Path] = set()
    samples_by_group: dict[int, list[Sample]] = {g: [] for g in groups}
    manifest = []
    for row in sorted(rows, key=lambda r: -r["total_length"]):
        length = row["total_length"]
        candidates = [p for p in pairs if p[0] >= length]
        if not candidates:
            step = join_steps(read_step(*longest[1:]), read_step(*pairs[len(pairs) // 2][1:]), length)
        else:
            fresh = [p for p in candidates if p[1] not in used] or candidates  # reuse when the pool runs out
            for _n, tokens_path, routing_path in fresh:
                step = read_tokens(tokens_path)
                if step["loss_mask"][:length].any():
                    add_routing(step, routing_path)
                    used.add(tokens_path)
                    break
            else:
                raise RuntimeError(f"no r43 step with a trainable token in its first {length} tokens")
        sample = make_sample(row, step)
        samples_by_group[row["group_index"]].append(sample)
        manifest.append({"index": row["index"], "total_length": length, "src": step["src"]})

    data = [sorted(samples_by_group[g], key=lambda s: s.index) for g in groups]
    # _pad_rows_to_dp_alignment reads the actor layout only through dp = gpus / (tp * pp * cp).
    layout = SimpleNamespace(
        actor_num_nodes=cli.dp_size,
        actor_num_gpus_per_node=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        use_dynamic_batch_size=True,
        micro_batch_size=1,
    )
    pads = _pad_rows_to_dp_alignment(data, layout)
    samples = [s for group in data for s in group]

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = [int(x) for x in cli.rollout_ids.split(",")]
    target = out_dir / f"rollout_{ids[0]}.pt"
    # The format of save_debug_rollout_data, read by _load_rollout_data_file.
    torch.save(dict(rollout_id=ids[0], metadata={}, samples=[s.to_dict() for s in samples]), target)
    for rollout_id in ids[1:]:
        link = out_dir / f"rollout_{rollout_id}.pt"
        link.unlink(missing_ok=True)
        link.symlink_to(target.name)
    digest = hashlib.sha256()
    with target.open("rb") as f:
        while chunk := f.read(1 << 26):
            digest.update(chunk)
    digest = digest.hexdigest()
    summary = {
        "summary": cli.summary,
        "groups": groups,
        "rows": len(samples),
        "dp_pads": pads,
        "episodes": len({(s.group_index, s.rollout_id) for s in samples}),
        "tokens": sum(len(s.tokens) for s in samples),
        "max_length": max(len(s.tokens) for s in samples),
        "file": str(target),
        "sha256": digest,
        "links": [f"rollout_{i}.pt" for i in ids[1:]],
        "rows_detail": manifest,
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows_detail"}, indent=2))


if __name__ == "__main__":
    main()

"""Arena data source for miles with weighted gym sampling.

Reads manifest-format datasets where each JSONL row is a pointer to a
lakeFS task directory::

    {"lakefs_uri": "lakefs://repo/branch/prefix/tasks/task_id/", "lakefs_commit_id": "sha"}

The data source doesn't pull task content — it just passes the lakefs_uri
and commit_id through to the NATS message so that gym workers can pull and
materialize the task themselves.

Usage::

    --data-source-path miles_plugins.arena.nats_arena.data_source.ArenaDataSourceWithBuffer
    --prompt-data-list '[{"path":"lakefs://arena-mcp/dev/chakra_productivity_terrace/canyon-v0.34.1","gym_name":"chakra_productivity_terrace","weight":1.0}]'
    --prompt-data-list '[{"path":"/scratch/data/manifest.jsonl","gym_name":"my_gym","weight":1.0}]'

Supports both lakefs:// URIs (pulls manifest.jsonl from the variant dir)
and local paths to manifest JSONL files.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import torch

from miles.rollout.data_source import DataSource
from miles.utils.data import Dataset
from miles.utils.misc import load_function
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.types import Sample
from miles.utils import data as _data_mod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manifest reading
# ---------------------------------------------------------------------------

def _read_manifest(path: str) -> list[dict[str, Any]]:
    """Read a manifest JSONL and convert to miles-compatible dataset rows.

    Each manifest row has {lakefs_uri, lakefs_commit_id}. We extract the
    task_id from the URI and build a minimal row that the Dataset can load
    as a Sample. The lakefs_uri and commit_id are stored in metadata so
    they flow through to the NATS task message.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            lakefs_uri = obj.get("lakefs_uri", "")
            commit_id = obj.get("lakefs_commit_id", "")

            task_id = _task_id_from_uri(lakefs_uri)

            rows.append({
                "messages": [{"role": "user", "content": f"Task: {task_id}"}],
                "metadata": {
                    "instance_id": task_id,
                    "lakefs_uri": lakefs_uri,
                    "lakefs_commit_id": commit_id,
                },
            })
    return rows


def _task_id_from_uri(lakefs_uri: str) -> str:
    """Extract task_id from lakefs://.../tasks/task_id/ URI."""
    trimmed = lakefs_uri.rstrip("/")
    parts = trimmed.split("/")
    try:
        idx = len(parts) - 1 - parts[::-1].index("tasks")
    except ValueError:
        return parts[-1] if parts else "unknown"
    if idx + 1 < len(parts):
        return parts[idx + 1]
    return parts[-1] if parts else "unknown"


def _parse_lakefs_uri(uri: str) -> tuple[str, str, str]:
    """Parse lakefs://repo/branch/path -> (repo, branch, path)."""
    stripped = uri.removeprefix("lakefs://")
    parts = stripped.split("/", 2)
    repo = parts[0]
    branch = parts[1] if len(parts) > 1 else "main"
    path = parts[2].rstrip("/") if len(parts) > 2 else ""
    return repo, branch, path


def _pull_lakefs_file(uri: str, cache_root: Path) -> Path:
    """Pull a single file from lakeFS to a local cache directory.

    Uses the lakefs Python SDK directly — no amzn_arena_base dependency.
    Auth via LAKECTL_CREDENTIALS_ACCESS_KEY_ID/SECRET_ACCESS_KEY env vars
    (set by the K8s pod template from the secret).
    """
    import lakefs

    os.environ.setdefault("LAKECTL_SERVER_ENDPOINT_URL", "https://prod.artifact-vault.agi.amazon.dev")

    repo, branch, file_path = _parse_lakefs_uri(uri)
    local_path = cache_root / repo / branch / file_path
    if local_path.exists():
        return local_path

    local_path.parent.mkdir(parents=True, exist_ok=True)
    client = lakefs.Client()
    repository = lakefs.repository(repo, client=client)
    br = repository.branch(branch)
    obj = br.object(file_path)
    with obj.reader(mode="rb") as reader:
        local_path.write_bytes(reader.read())

    logger.info("Pulled lakefs://%s/%s/%s -> %s", repo, branch, file_path, local_path)
    return local_path


def _resolve_manifest_path(path: str) -> str:
    """Resolve a dataset path to a local manifest JSONL file.

    If path is a lakefs:// URI, pull the file to local cache.
    If path is a local file, use it directly.
    If path is a local directory, look for manifest.jsonl inside it.
    """
    if path.startswith("lakefs://"):
        cache_root = Path(".arena_cache") / "datasets"
        local = _pull_lakefs_file(path, cache_root)
        return str(local)

    p = Path(path)
    if p.is_file():
        return path
    if p.is_dir():
        manifest = p / "manifest.jsonl"
        if manifest.exists():
            return str(manifest)
        raise FileNotFoundError(f"No manifest.jsonl in directory: {path}")

    raise FileNotFoundError(f"Dataset path not found: {path}")


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def _manifest_read_file(path):
    """Read manifest JSONL and yield rows compatible with miles Dataset."""
    for row in _read_manifest(path):
        yield row


def _build_dataset(args, path: str, tokenizer, processor) -> Dataset:
    """Build a Dataset from a manifest JSONL file."""
    saved = _data_mod.read_file

    def _patched(p):
        _data_mod.read_file = saved
        try:
            yield from _manifest_read_file(p)
        finally:
            _data_mod.read_file = _patched

    _data_mod.read_file = _patched
    try:
        dataset = Dataset(
            path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.rollout_max_prompt_len,
            prompt_key=args.input_key,
            multimodal_keys=args.multimodal_keys,
            label_key=args.label_key,
            metadata_key=args.metadata_key,
            tool_key=args.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
            seed=args.rollout_seed,
        )
    finally:
        _data_mod.read_file = saved
    return dataset


def _pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples


def _resolve_gym_configs(args) -> list[dict[str, Any]]:
    """Build per-gym config list from --prompt-data-list or --prompt-data."""
    raw = getattr(args, "prompt_data_list", None) or getattr(args, "prompt_data", None)
    if raw is None:
        raise ValueError(
            "ArenaDataSourceWithBuffer requires --prompt-data-list (or --prompt-data on 0.3.0). "
            "Example: '[{\"path\":\"lakefs://arena-mcp/dev/gym/variant\",\"gym_name\":\"my_gym\",\"weight\":1.0}]'"
        )
    if isinstance(raw, str):
        raw = json.loads(raw)
    configs = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"Each entry in prompt_data_list must be a dict, got {type(entry)}")
        path = entry.get("path")
        gym_name = entry.get("gym_name")
        weight = float(entry.get("weight", 1.0))
        if not path or not gym_name:
            raise ValueError(f"Each entry needs 'path' and 'gym_name', got {entry}")
        resolved = _resolve_manifest_path(path)
        logger.info("Resolved dataset path: %s -> %s", path, resolved)
        configs.append({"path": resolved, "gym_name": gym_name, "weight": weight})
    return configs


class _MultiGymDatasetView:
    """len()-only view over the combined per-gym datasets.

    miles' ``RolloutManager.get_num_rollout_per_epoch`` dereferences
    ``len(data_source.dataset)`` when ``--num-rollout`` is unset (the slime
    0.3.0 DataSource ABC used ``len(data_source)`` instead). Exposing the
    total prompt count across gyms keeps that path working; the buffer is
    deliberately excluded so the per-epoch count stays stable.
    """

    def __init__(self, datasets: dict[str, Dataset]):
        self._datasets = datasets

    def __len__(self) -> int:
        return sum(len(d) for d in self._datasets.values())


# ===================================================================
# Unified data source
# ===================================================================

class ArenaDataSourceWithBuffer(DataSource):
    """Arena data source with weighted multi-gym sampling and sample buffer.

    Reads manifest-format datasets and passes lakefs task pointers through
    to the NATS rollout worker. The gym workers pull and materialize tasks.
    """

    def __init__(self, args):
        self.args = args

        self.sample_group_index = 0
        self.sample_index = 0
        self.metadata: dict[str, Any] = {}
        # instance_id -> mean raw reward of the last group trained on it.
        # Fed by nats_rollout at drain time (record_prompt_rewards), read by
        # _read_from_gym when --arena-skip-prompt-above-reward is set.
        self.prompt_reward: dict[str, float] = {}
        self.skip_prompt_above_reward: float | None = getattr(
            args, "arena_skip_prompt_above_reward", None
        )
        self._skipped_this_epoch: dict[str, int] = {}

        gym_configs = _resolve_gym_configs(args)

        self.gym_names: list[str] = []
        self.datasets: dict[str, Dataset] = {}
        self.offsets: dict[str, int] = {}
        self.epochs: dict[str, int] = {}
        self.weights: dict[str, float] = {}

        tokenizer = load_tokenizer(
            args.hf_checkpoint,
            chat_template_path=getattr(args, "chat_template_path", None),
            trust_remote_code=True,
        )
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        for cfg in gym_configs:
            gym_name = cfg["gym_name"]
            self.gym_names.append(gym_name)
            self.weights[gym_name] = cfg["weight"]
            self.offsets[gym_name] = 0
            self.epochs[gym_name] = 0

            dataset = _build_dataset(args, cfg["path"], tokenizer, processor)
            self.datasets[gym_name] = dataset
            logger.info(
                "Loaded gym %s: %d prompts from %s (weight=%.3f)",
                gym_name, len(dataset), cfg["path"], cfg["weight"],
            )

        # RolloutManager.get_num_rollout_per_epoch reads len(data_source.dataset).
        self.dataset = _MultiGymDatasetView(self.datasets)

        self._normalize_weights()

        self.buffer: list[list[Sample]] = []
        if getattr(args, "buffer_filter_path", None) is None:
            self.buffer_filter = _pop_first
        else:
            self.buffer_filter = load_function(args.buffer_filter_path)

        self._weights_lock = threading.Lock()
        self._deficit: dict[str, float] = {g: 0.0 for g in self.gym_names}

        if self.args.rollout_shuffle:
            for gym_name in self.gym_names:
                self.datasets[gym_name].shuffle(0)

        total = sum(len(d) for d in self.datasets.values())
        logger.info(
            "ArenaDataSourceWithBuffer: %d gym(s), %d total prompts, weights=%s",
            len(self.gym_names), total,
            {g: f"{w:.3f}" for g, w in self.weights.items()},
        )

    def _normalize_weights(self) -> None:
        total = sum(self.weights.values())
        if total <= 0:
            n = len(self.weights)
            for g in self.weights:
                self.weights[g] = 1.0 / n if n else 0.0
        else:
            for g in self.weights:
                self.weights[g] /= total

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        samples = self._drain_buffer(num_samples)
        remaining = num_samples - len(samples)
        if remaining <= 0:
            return samples

        with self._weights_lock:
            allocation = self._allocate(remaining)

        for gym_name in self.gym_names:
            count = allocation.get(gym_name, 0)
            if count > 0:
                samples.extend(self._read_from_gym(gym_name, count))

        return samples

    def add_samples(self, samples: list[list[Sample]]) -> None:
        if not samples:
            return
        assert isinstance(samples, list) and isinstance(samples[0], list)
        for group in samples:
            assert len(group) == self.args.n_samples_per_prompt, (
                f"group size {len(group)} != n_samples_per_prompt {self.args.n_samples_per_prompt}"
            )
            self.buffer.append(group)

    def __len__(self) -> int:
        return sum(len(d) for d in self.datasets.values()) + len(self.buffer)

    # ------------------------------------------------------------------
    # Weighted allocation (deficit-based)
    # ------------------------------------------------------------------

    def _allocate(self, n: int) -> dict[str, int]:
        allocation: dict[str, int] = {g: 0 for g in self.gym_names}
        for gym_name in self.gym_names:
            self._deficit[gym_name] += n * self.weights.get(gym_name, 0.0)
        remaining = n
        while remaining > 0:
            best = max(self.gym_names, key=lambda g: self._deficit[g])
            allocation[best] += 1
            self._deficit[best] -= 1.0
            remaining -= 1
        return allocation

    # ------------------------------------------------------------------
    # Per-gym reading
    # ------------------------------------------------------------------

    def _read_from_gym(self, gym_name: str, count: int) -> list[list[Sample]]:
        dataset = self.datasets[gym_name]
        if len(dataset) == 0:
            return []
        offset = self.offsets[gym_name]
        out: list[list[Sample]] = []
        skip_ids = self._skip_ids(gym_name)

        while len(out) < count:
            prompt_sample = dataset.samples[offset]
            iid = (prompt_sample.metadata or {}).get("instance_id")
            if iid in skip_ids:
                self._skipped_this_epoch[gym_name] = self._skipped_this_epoch.get(gym_name, 0) + 1
            else:
                out.append(self._make_group(prompt_sample, gym_name))

            offset += 1
            if offset >= len(dataset):
                if skip_ids:
                    logger.info(
                        "Gym %s epoch %d: skipped %d/%d prompts with reward > %.3f",
                        gym_name, self.epochs[gym_name],
                        self._skipped_this_epoch.get(gym_name, 0), len(dataset),
                        self.skip_prompt_above_reward,
                    )
                self._skipped_this_epoch[gym_name] = 0
                self.epochs[gym_name] += 1
                if self.args.rollout_shuffle:
                    dataset.shuffle(self.epochs[gym_name])
                offset = 0
                skip_ids = self._skip_ids(gym_name)

        self.offsets[gym_name] = offset
        return out

    def _skip_ids(self, gym_name: str) -> set[str]:
        """instance_ids to skip in the current pass over ``gym_name``.

        Empty before the first epoch completes, when the threshold is unset,
        and when every prompt of the gym would be skipped (never starve a
        rollout; a prompt with no record is always kept).
        """
        thr = self.skip_prompt_above_reward
        if thr is None or self.epochs.get(gym_name, 0) < 1:
            return set()
        dataset = self.datasets[gym_name]
        ids = [(s.metadata or {}).get("instance_id") for s in dataset.samples]
        skip = {i for i in ids if i is not None and self.prompt_reward.get(i, -1.0) > thr}
        if skip and len(skip) >= len(ids):
            logger.warning(
                "Gym %s: all %d prompts have reward > %.3f; skip filter disabled for this pass",
                gym_name, len(ids), thr,
            )
            return set()
        return skip

    def _make_group(self, prompt_sample: Sample, gym_name: str) -> list[Sample]:
        # dedup_key = (instance_id, epoch) — stable across restarts (so resume
        # works) but distinct across epochs (so a prompt isn't blocked by its
        # own consumed-set entry from a prior epoch). Used only by the resume
        # guard in nats_rollout; instance_id stays the raw lakeFS id so OTel
        # span names still resolve 1:1 to dataset rows.
        epoch = self.epochs.get(gym_name, 0)
        group: list[Sample] = []
        for _ in range(self.args.n_samples_per_prompt):
            sample = copy.deepcopy(prompt_sample)
            sample.group_index = self.sample_group_index
            sample.index = self.sample_index
            self.sample_index += 1
            meta = sample.metadata if isinstance(sample.metadata, dict) else {}
            meta["gym_name"] = gym_name
            iid = meta.get("instance_id")
            if iid:
                meta["dedup_key"] = f"{iid}-e{epoch}"
            sample.metadata = meta
            group.append(sample)
        self.sample_group_index += 1
        return group

    # ------------------------------------------------------------------
    # Buffer drain
    # ------------------------------------------------------------------

    def _drain_buffer(self, num_samples: int) -> list[list[Sample]]:
        if not self.buffer or num_samples <= 0:
            return []
        return self.buffer_filter(self.args, None, self.buffer, num_samples)

    def get_buffer_length(self) -> int:
        return len(self.buffer)

    # ------------------------------------------------------------------
    # Dynamic weight updates
    # ------------------------------------------------------------------

    def update_weights(self, new_weights: dict[str, float]) -> None:
        with self._weights_lock:
            old_weights = dict(self.weights)
            for gym_name, weight in new_weights.items():
                if gym_name in self.weights:
                    self.weights[gym_name] = weight
            self._normalize_weights()
            logger.info(
                "DataSource weights updated: before=%s, after=%s",
                {g: f"{w:.3f}" for g, w in old_weights.items()},
                {g: f"{w:.3f}" for g, w in self.weights.items()},
            )

    def get_current_weights(self) -> dict[str, float]:
        with self._weights_lock:
            return dict(self.weights)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, rollout_id) -> None:
        if not getattr(self.args, "save", None):
            return

        with self._weights_lock:
            weights_snapshot = dict(self.weights)

        state_dict = {
            "per_gym": {
                gym: {"offset": self.offsets[gym], "epoch": self.epochs[gym]}
                for gym in self.gym_names
            },
            "weights": weights_snapshot,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
            "prompt_reward": self.prompt_reward,
            "timing_tracker": getattr(self, "_timing_tracker_state", None),
        }
        path = os.path.join(
            self.args.save, f"rollout/arena_data_source_state_{rollout_id}.pt"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)
        logger.info("Arena data source state saved to %s", path)

    def load(self, rollout_id=None) -> None:
        if getattr(self.args, "load", None) is None:
            return

        path = os.path.join(
            self.args.load, f"rollout/arena_data_source_state_{rollout_id}.pt"
        )
        if not os.path.exists(path):
            logger.info("Checkpoint %s does not exist, starting fresh.", path)
            return

        state_dict = torch.load(path, weights_only=True)
        logger.info("Loading arena data source state from %s", path)

        per_gym = state_dict.get("per_gym", {})
        for gym_name in self.gym_names:
            if gym_name in per_gym:
                self.offsets[gym_name] = per_gym[gym_name].get("offset", 0)
                self.epochs[gym_name] = per_gym[gym_name].get("epoch", 0)
                if self.args.rollout_shuffle and self.epochs[gym_name] > 0:
                    self.datasets[gym_name].shuffle(self.epochs[gym_name])

        restored_weights = state_dict.get("weights")
        if restored_weights:
            with self._weights_lock:
                for gym_name in self.gym_names:
                    if gym_name in restored_weights:
                        self.weights[gym_name] = restored_weights[gym_name]
                self._normalize_weights()
            logger.info(
                "Restored weights: %s",
                {g: f"{w:.3f}" for g, w in self.weights.items()},
            )

        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})
        self.prompt_reward = state_dict.get("prompt_reward", {})
        self._timing_tracker_state = state_dict.get("timing_tracker")

    def get_restored_weights(self) -> dict[str, float] | None:
        with self._weights_lock:
            return dict(self.weights) if self.weights else None

    # ------------------------------------------------------------------
    # RolloutTimingTracker state plumbing
    # ------------------------------------------------------------------

    def set_timing_tracker_state(self, state: dict[str, Any] | None) -> None:
        self._timing_tracker_state = dict(state) if state else None

    def get_restored_timing_tracker_state(self) -> dict[str, Any] | None:
        return getattr(self, "_timing_tracker_state", None)

    # ------------------------------------------------------------------
    # Resume tracking
    # ------------------------------------------------------------------

    def record_consumed_samples(self, rollout_id: int, instance_ids: list[str]) -> None:
        self.metadata[str(rollout_id)] = instance_ids

    def record_prompt_rewards(self, rewards: dict[str, float]) -> None:
        """Overwrite the per-instance_id mean raw reward with this rollout's value."""
        self.prompt_reward.update(rewards)

    def get_consumed_instance_ids(self) -> set[str]:
        ids: set[str] = set()
        for id_list in self.metadata.values():
            if isinstance(id_list, list):
                ids.update(id_list)
        return ids

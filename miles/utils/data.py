import itertools
import json
import logging
import os
import random
import re
import threading
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from miles.ray.rollout.train_data_conversion import split_train_data_by_dp_raw
from miles.utils import object_store
from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.r3_log import log_r3_timing, r3_timing, train_rank
from .audit_utils.witness.allocator import WitnessInfo

if TYPE_CHECKING:
    from miles.backends.megatron_utils.ft.types import TrainStepOutput

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

from miles.utils import chat_template_utils
from miles.utils.types import MultimodalTypes, Sample

__all__ = ["Dataset"]

logger = logging.getLogger(__name__)


def read_file(path):
    path, row_slice = _parse_generalized_path(path)
    reader = None

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt dataset path '{path}' does not exist.")

    if path.endswith(".jsonl"):

        def jsonl_reader(p):
            with open(p, encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error at line {line_num}: {e}")
                        continue

        reader = jsonl_reader(path)

    elif path.endswith(".parquet"):
        if pq is None:
            raise ImportError("pyarrow is required for parquet support")

        def parquet_reader(p):
            pf = pq.ParquetFile(p)

            for batch in pf.iter_batches():
                yield from batch.to_pylist()

        reader = parquet_reader(path)

    else:
        raise ValueError(f"Unsupported file format: {path}. Supported formats are .jsonl and .parquet.")

    if row_slice is not None:

        logger.info("read_file path=%s applying slice row_slice=%s", path, row_slice)
        reader = itertools.islice(reader, row_slice.start, row_slice.stop, row_slice.step)

    yield from reader


def _parse_generalized_path(s: str):
    if (m := re.match(r"^(?P<real_path>.*)@\[(?P<start>-?\d*):(?P<end>-?\d*)\]$", s)) is not None:
        path = m.group("real_path")
        start = int(x) if (x := m.group("start")) != "" else None
        end = int(x) if (x := m.group("end")) != "" else None
        return path, slice(start, end)

    return s, None


def filter_long_prompt(origin_samples: list[Sample], tokenizer, processor, max_length: int | None) -> list[Sample]:
    if max_length is None:
        return origin_samples

    if not isinstance(origin_samples[0].prompt, str):
        logger.warning(
            "Skipping max_length check for list prompt. Set apply_chat_template=True to enable length filtering."
        )
        return origin_samples

    if processor:
        # Use processor only for samples with actual multimodal content; use batched tokenizer for text-only.
        text_only = []
        multimodal = []
        for sample in origin_samples:
            if sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()):
                multimodal.append(sample)
            else:
                text_only.append(sample)
        filtered_samples = []
        if text_only:
            prompts = [s.prompt for s in text_only]
            input_ids_list = tokenizer(prompts, add_special_tokens=False)["input_ids"]
            for sample, input_ids in zip(text_only, input_ids_list, strict=True):
                if len(input_ids) <= max_length:
                    filtered_samples.append(sample)
        if multimodal:
            from miles.utils.processing_utils import process_vision_info

            for sample in multimodal:
                multimodal_inputs = process_vision_info(sample.prompt, processor)
                processor_output = processor(text=sample.prompt, **multimodal_inputs)
                input_ids = processor_output["input_ids"][0]
                if len(input_ids) <= max_length:
                    filtered_samples.append(sample)
    else:
        prompts = [sample.prompt for sample in origin_samples]
        input_ids_list = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        filtered_samples = [
            sample
            for sample, input_ids in zip(origin_samples, input_ids_list, strict=True)
            if len(input_ids) <= max_length
        ]

    logger.info(f"Filtered {len(origin_samples) - len(filtered_samples)} samples longer than max_length={max_length}.")

    return filtered_samples


def _build_messages(data: dict, prompt_key: str, as_conversation: bool, multimodal_keys: dict = None):
    prompt = data.get(prompt_key)

    if isinstance(prompt, str):
        # If prompt is a string and we don't apply chat template, return the prompt as is.
        if not as_conversation:
            return prompt
        else:
            prompt = [{"role": "user", "content": prompt}]

    if multimodal_keys:
        # Build mapping: placeholder -> (MultimodalType, content_list)
        multimodals = {}
        for type_name, data_key in multimodal_keys.items():
            mt = MultimodalTypes.get(type_name)
            if mt:
                multimodals[mt.placeholder] = (mt, list(data.get(data_key)))

        pattern = "(" + "|".join(re.escape(p) for p in multimodals.keys()) + ")"

        for message in prompt:
            if isinstance(message["content"], str):
                content_list = []
                for segment in re.split(pattern, message["content"]):
                    if not segment:
                        continue
                    if segment in multimodals:
                        mt, content = multimodals[segment]
                        content_list.append({"type": mt.name, mt.name: content.pop(0)})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list

            elif isinstance(message["content"], list):
                # TODO: handle more general cases. where message['content'] is a dict and contains multiple types of content.
                # e.g.
                #  "content": [
                #     {
                #         "type": "image",
                #         "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                #     },
                #     {"type": "text", "text": "Describe this image."},
                # ],
                logger.warning("message['content'] is a list of dicts, no processing will be done.")
                continue
            else:
                raise ValueError(
                    f"Unsupported content type: {type(message['content'])}, expected str or list of dicts"
                )

    return prompt


class Dataset:
    def __init__(
        self,
        path,
        tokenizer,
        processor,
        max_length,
        *,
        prompt_key="text",
        multimodal_keys=None,
        label_key=None,
        tool_key=None,
        metadata_key="metadata",
        seed=42,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
    ):
        origin_samples = []
        for data in read_file(path):
            # Both chat templates and multimodal inputs require conversation format (list of message dicts)
            as_conversation = apply_chat_template or (multimodal_keys is not None)
            prompt = _build_messages(data, prompt_key, as_conversation, multimodal_keys)

            metadata = data.get(metadata_key) or {}
            tools = None
            if tool_key is not None and tool_key in data:
                tools = data[tool_key]
                if isinstance(tools, str):
                    tools = json.loads(tools)
                elif isinstance(tools, np.ndarray):
                    tools = tools.tolist()
                assert isinstance(tools, list), f"tools must be a list, got {type(tools)} instead"
                metadata["tools"] = tools

            if apply_chat_template:
                output_prompt = chat_template_utils.apply_chat_template(
                    prompt,
                    tokenizer=tokenizer,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                    **(apply_chat_template_kwargs or {}),
                )
            else:
                output_prompt = prompt

            if processor and isinstance(prompt, list):
                from miles.utils.processing_utils import process_vision_info

                multimodal_inputs = process_vision_info(prompt, processor)
            else:
                multimodal_inputs = None

            origin_samples.append(
                Sample(
                    prompt=output_prompt,
                    label=data[label_key] if label_key is not None else None,
                    metadata=metadata,
                    multimodal_inputs=multimodal_inputs,
                )
            )

        if max_length is not None:
            self.origin_samples = filter_long_prompt(origin_samples, tokenizer, processor, max_length)
        else:
            self.origin_samples = origin_samples

        self.epoch_id = -1
        self.seed = seed
        self.samples = self.origin_samples

    def shuffle(self, new_epoch_id):
        if self.epoch_id == new_epoch_id:
            return

        random.seed(self.seed + new_epoch_id)
        permutation = list(range(len(self.samples)))
        random.shuffle(permutation)
        self.samples = [self.origin_samples[i] for i in permutation]
        self.epoch_id = new_epoch_id

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


def get_minimum_num_micro_batch_size(total_lengths, max_tokens_per_gpu):
    # use first fit to get the number of micro batches
    batches = []
    for length in total_lengths:
        for i in range(len(batches)):
            if batches[i] + length <= max_tokens_per_gpu:
                batches[i] += length
                break
        else:
            batches.append(length)

    return len(batches)


def process_rollout_data(
    args,
    rollout_data_ref,
    dp_rank,
    dp_size,
    witness_info: WitnessInfo | None,
    rollout_id: int | None = None,
) -> tuple[dict, object_store.ObjectStoreGetResult]:
    from miles.ray.rollout.train_data_conversion import process_rollout_data_shard

    if args.delay_split_train_data_by_dp:
        get_result = _fetch_rollout_data(args, rollout_data_ref, rollout_id=rollout_id, dp_rank=dp_rank)
        raw = get_result.value
        if (x := witness_info) is not None:
            raw = {**raw, "seq_witness_ids": x.witness_ids}
        raw = split_train_data_by_dp_raw(args, raw, dp_size=dp_size)
        rollout_data = raw[dp_rank]
    else:
        assert len(rollout_data_ref) == dp_size
        assert witness_info is None
        get_result = _fetch_rollout_data(args, rollout_data_ref[dp_rank], rollout_id=rollout_id, dp_rank=dp_rank)
        rollout_data = dict(get_result.value)

    return process_rollout_data_shard(args, rollout_data), get_result


def _fetch_rollout_data(args, ref, *, rollout_id: int | None, dp_rank: int) -> object_store.ObjectStoreGetResult:
    store = object_store.get_instance()
    # Before the get: the bytes that the get moves to this node, and whether a copy is here already.
    info = store.locate(ref)
    with r3_timing(logger, rank=train_rank(), rollout=rollout_id, phase="fetch", dp=dp_rank) as line:
        get_result, mode = _PREFETCHER.take(info.key) if args.prefetch_rollout_data else (None, "off")
        if get_result is None:
            get_result = store.get(ref)
        routing = get_result.value.get("rollout_routed_experts", [])
        line.update(
            bytes=info.size,
            routing_bytes=sum(r.nbytes for r in routing),
            local_before=info.local,
            prefetch=mode,
            ref=info.key,
        )
    return get_result


def prefetch_rollout_data(args, rollout_data_ref, *, rollout_id: int, dp_rank: int) -> None:
    """--prefetch-rollout-data: pull and hold this rank's shard of ``rollout_data_ref`` for the next train()."""
    # The same ref that process_rollout_data fetches.
    ref = rollout_data_ref if args.delay_split_train_data_by_dp else rollout_data_ref[dp_rank]
    _PREFETCHER.prefetch(ref, rollout_id=rollout_id, dp_rank=dp_rank)


class _PrefetchSlot:
    def __init__(self, key: str) -> None:
        self.key = key
        self.done = threading.Event()
        self.result: object_store.ObjectStoreGetResult | None = None
        self.error: BaseException | None = None


class RolloutDataPrefetcher:
    """This rank's shard of the next rollout, pulled and held while the current train() runs.

    The train actor runs ``prefetch`` in its own concurrency group, next to
    train() (actor_factory.py). The fetch of the next train() calls ``take``,
    which returns the held shard if the prefetch of that object has finished.
    ``prefetch`` also gets the object after the pull, because a finished
    ray.wait does not pin the local copy ("Only actively pulled objects should
    be pinned", Ray pull_manager.h).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slot: _PrefetchSlot | None = None
        # The object that train() fetched last. A prefetch of it that comes later has no use.
        self._taken: str | None = None

    def prefetch(self, ref, *, rollout_id: int, dp_rank: int) -> None:
        store = object_store.get_instance()
        info = store.locate(ref)
        slot = _PrefetchSlot(info.key)
        with self._lock:
            if info.key == self._taken:
                return
            self._slot = slot
        rank = train_rank()
        t0 = time.time()
        log_r3_timing(
            logger,
            rank=rank,
            rollout=rollout_id,
            phase="prefetch_start",
            nbytes=0,
            t0=t0,
            t1=t0,
            dp=dp_rank,
            ref=info.key,
        )
        try:
            store.pull(ref)
            t_pull = time.time()
            slot.result = store.get(ref)
            t1 = time.time()
        except Exception as e:  # noqa: BLE001 - a failed prefetch only makes train() fetch the shard itself
            slot.error = e
        finally:
            slot.done.set()
        if slot.error is not None:
            logger.warning(f"Prefetch of rollout {rollout_id} failed, train() fetches it itself: {slot.error!r}")
            return
        log_r3_timing(
            logger,
            rank=rank,
            rollout=rollout_id,
            phase="prefetch_done",
            nbytes=info.size,
            t0=t0,
            t1=t1,
            dp=dp_rank,
            ref=info.key,
            local_before=info.local,
            pull_s=t_pull - t0,
            deser_s=t1 - t_pull,
        )

    def take(self, key: str) -> tuple[object_store.ObjectStoreGetResult | None, str]:
        """The held shard of object ``key`` and ``hit``, or None and ``miss``."""
        with self._lock:
            self._taken = key
            slot = self._slot
            if slot is None or slot.key != key:
                # A slot of another object stays: it can be the prefetch of the rollout after this one.
                return None, "miss"
            self._slot = None
        # Do not wait for a prefetch that still runs: its ray.wait pull can wait for plasma room behind this
        # step's objects, but the get of train() does not wait for room (Ray pull_manager.cc) and shares the pull.
        if not slot.done.is_set() or slot.error is not None:
            return None, "miss"
        return slot.result, "hit"


_PREFETCHER = RolloutDataPrefetcher()


class RolloutDataPack(StrictBaseModel):
    sample_indices: list[int] | None = None
    data_ref: object_store.StoreObjectRef | list[object_store.StoreObjectRef] | None = None

    @property
    def data_refs(self) -> list[object_store.StoreObjectRef]:
        if (ref := self.data_ref) is None:
            return []
        return ref if isinstance(ref, list) else [ref]


def remove_rollout_data_refs(args, rollout_data_pack: RolloutDataPack) -> None:
    store = object_store.get_instance()
    for ref in rollout_data_pack.data_refs:
        store.remove(ref)


def remove_train_output_refs(train_outputs: Sequence["TrainStepOutput"]) -> None:
    store = object_store.get_instance()
    for train_output in train_outputs:
        if (ref := train_output.values) is not None:
            store.remove(ref)

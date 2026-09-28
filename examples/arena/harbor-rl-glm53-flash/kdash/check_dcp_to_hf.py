"""Convert the KDA and hyper-connection keys of a GLM-5.3 DCP with both offline tools and compare with HF.

The tools are ``tools/convert_torch_dist_to_hf.py`` and ``tools/convert_torch_dist_to_hf_ray.py``. The DCP
keeps the keys of the replicated layer: ``self_attention.kda.*`` and one packed ``conv1d.weight``
``[q; k; v]``. A DCP converted from ``--hf`` (the base DCP) gives each tensor bit for bit. The tools write
the names of the weight sync (``model.layers.N.*``). The HF checkpoint has ``model.language_model.layers.N.*``.

    python3 check_dcp_to_hf.py --dcp <DCP>/release --hf <HF dir> --work /tmp/conv --out results.json

Exit code 0 means PASS: each tool gives all the KDA and hyper-connection tensors of each layer and no
other tensor. Each KDA tensor has the HF dtype, shape and bits. A hyper-connection tensor can have a wider
dtype than HF (the DCP keeps ``mapping_proj`` in fp32) when the cast is lossless both ways. The results
list these tensors under ``dtype_only``.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import safetensors
import torch
import torch.distributed.checkpoint as dist_cp

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from tools import convert_torch_dist_to_hf as tool  # noqa: E402

# The decoder-layer keys of the KDA layers and of the hyper connections of all layers. MTP is not in scope.
SOURCE_KEY_REGEX = r"^decoder\.layers\.\d+\.(self_attention\.kda\.|(self_attention|mlp)_hyper_connection\.)"
KDA_HF_NAMES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "q_conv1d.weight",
    "k_conv1d.weight",
    "v_conv1d.weight",
    "b_proj.weight",
    "f_a_proj.weight",
    "f_b_proj.weight",
    "g_a_proj.weight",
    "g_b_proj.weight",
    "A_log",
    "dt_bias",
    "o_norm.weight",
    "o_proj.weight",
)
HC_HF_NAMES = ("hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale")
HF_PREFIX = "model.language_model.layers."


def expected_names(hf_index: dict[str, str], num_layers: int) -> set[str]:
    """The HF names of the KDA tensors (layers with a ``q_conv1d``) and of the hyper connections.

    Only the decoder layers ``0 .. num_layers - 1``. The HF layers from ``num_layers`` on are MTP layers.
    """
    layers = range(num_layers)
    kda_layers = {n for n in layers if f"{HF_PREFIX}{n}.self_attn.q_conv1d.weight" in hf_index}
    names = {f"{HF_PREFIX}{n}.self_attn.{name}" for n in kda_layers for name in KDA_HF_NAMES}
    names |= {f"{HF_PREFIX}{n}.{name}" for n in layers for name in HC_HF_NAMES}
    missing = sorted(names - set(hf_index))
    if missing:
        raise ValueError(f"the HF index has no {missing[:5]} ({len(missing)} names)")
    return names


class FilteredReader(tool.WrappedStorageReader):
    """The reader of the single-process tool, limited to the keys of ``SOURCE_KEY_REGEX``."""

    def read_metadata(self):
        metadata = super().read_metadata()
        pattern = re.compile(SOURCE_KEY_REGEX)
        metadata.state_dict_metadata = {k: v for k, v in metadata.state_dict_metadata.items() if pattern.search(k)}
        return metadata


def run_single_process_tool(dcp: str, out: str) -> None:
    """The ``__main__`` path of the tool: args, DCP load, ``save_tensors``. Only the key filter is added."""
    megatron_args = tool.load_megatron_args(dcp)
    state_dict = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict, storage_reader=FilteredReader(dcp), planner=tool.EmptyStateDictLoadPlanner(), no_dist=True
    )
    tool.save_tensors(megatron_args, "glm5_next", state_dict, out, 5 * 1024**3, None)


def run_ray_tool(dcp: str, hf: str, out: str, concurrency: int) -> None:
    cmd = [
        sys.executable,
        str(REPO / "tools" / "convert_torch_dist_to_hf_ray.py"),
        *("--input-dir", dcp, "--output-dir", out, "--origin-hf-dir", hf, "--model-name", "glm5_next"),
        *("--source-key-regex", SOURCE_KEY_REGEX, "--concurrency", str(concurrency), "--no-progress", "-f"),
    ]
    subprocess.run(cmd, check=True)


def _suffix(hf_name: str) -> str:
    """``model.language_model.layers.3.self_attn.A_log`` -> ``self_attn.A_log``."""
    return hf_name.removeprefix(HF_PREFIX).split(".", 1)[1]


def compare(out: str, hf: str, hf_index: dict[str, str], want: set[str]) -> dict:
    """Compare each output tensor with the HF tensor of the same name: dtype, shape and bits."""
    out_index = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
    got = {name.replace("model.layers.", HF_PREFIX, 1): name for name in out_index}
    result = {
        "tensors": len(got),
        "expected": len(want),
        "missing": sorted(want - set(got))[:20],
        "extra": sorted(set(got) - want)[:20],
        "exact": 0,
        "dtype_only": {},
        "mismatch": [],
    }
    by_file: dict[str, list[str]] = {}
    for hf_name in sorted(set(got) & want):
        by_file.setdefault(hf_index[hf_name], []).append(hf_name)
    for hf_file, hf_names in sorted(by_file.items()):
        with safetensors.safe_open(os.path.join(hf, hf_file), "pt") as src:
            for hf_name in hf_names:
                a = src.get_tensor(hf_name)
                out_name = got[hf_name]
                with safetensors.safe_open(os.path.join(out, out_index[out_name]), "pt") as dst:
                    b = dst.get_tensor(out_name)
                if a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b):
                    result["exact"] += 1
                elif a.shape == b.shape and torch.equal(b.to(a.dtype), a) and torch.equal(a.to(b.dtype), b):
                    key = f"{_suffix(hf_name)} {a.dtype} -> {b.dtype}"
                    result["dtype_only"][key] = result["dtype_only"].get(key, 0) + 1
                else:
                    result["mismatch"].append([hf_name, str(a.dtype), list(a.shape), str(b.dtype), list(b.shape)])
    kda_dtype_only = [key for key in result["dtype_only"] if key.startswith("self_attn.")]
    result["pass"] = not (result["missing"] or result["extra"] or result["mismatch"] or kda_dtype_only)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dcp", required=True, help="the iteration dir of the DCP (read only)")
    parser.add_argument("--hf", required=True, help="the HF checkpoint that the DCP was converted from (read only)")
    parser.add_argument("--work", required=True, help="an empty local dir for the two outputs")
    parser.add_argument("--out", required=True, help="the results JSON")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    hf_index = json.load(open(os.path.join(args.hf, "model.safetensors.index.json")))["weight_map"]
    hf_config = json.load(open(os.path.join(args.hf, "config.json")))
    want = expected_names(hf_index, hf_config.get("text_config", hf_config)["num_hidden_layers"])
    results = {"dcp": args.dcp, "hf": args.hf, "source_key_regex": SOURCE_KEY_REGEX, "head": os.environ.get("HEAD")}
    for name, run in (
        ("single_process", lambda out: run_single_process_tool(args.dcp, out)),
        ("ray", lambda out: run_ray_tool(args.dcp, args.hf, out, args.concurrency)),
    ):
        out = os.path.join(args.work, name)
        start = time.time()
        run(out)
        results[name] = {"convert_secs": round(time.time() - start, 1), **compare(out, args.hf, hf_index, want)}
        print(json.dumps({name: results[name]}), flush=True)
    results["pass"] = all(results[name]["pass"] for name in ("single_process", "ray"))
    Path(args.out).write_text(json.dumps(results, indent=2))
    print("PASS" if results["pass"] else "FAIL", flush=True)
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

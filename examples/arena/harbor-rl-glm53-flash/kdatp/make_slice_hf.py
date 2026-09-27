"""Write a config-only HF dir for the first N layers of GLM-5.3-Flash (the T1c slice).

miles checks ``num_hidden_layers`` of ``--hf-checkpoint`` against
``--num-layers`` (``hf_validate_args``), and the glm5_next spec reads the
layer types from the same config. A train-only run loads its weights from
the DCP (``ref_load``) and sends none to SGLang, so it needs the config and
the tokenizer only. This script copies every file except the weights and
cuts the per-layer lists of the config to the first N layers.

    python3 make_slice_hf.py --src <GLM-5.3-Flash-BF16> --out <dir> --layers 5
"""

import argparse
import json
import shutil
from pathlib import Path

PER_LAYER_LISTS = ("layer_types", "mlp_layer_types", "indexer_types")
LAYER_ID_LISTS = ("kda_layers", "full_attn_layers")


def cut(config: dict, layers: int) -> None:
    if "num_hidden_layers" in config:
        config["num_hidden_layers"] = layers
    for key in PER_LAYER_LISTS:
        if isinstance(config.get(key), list):
            config[key] = config[key][:layers]
    linear = config.get("linear_attn_config")
    if isinstance(linear, dict):
        for key in LAYER_ID_LISTS:
            if isinstance(linear.get(key), list):
                linear[key] = [i for i in linear[key] if i < layers]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", type=int, required=True)
    cli = parser.parse_args()
    src, out = Path(cli.src), Path(cli.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.is_file() and not path.name.endswith(".safetensors") and path.name != "model.safetensors.index.json":
            shutil.copy2(path, out / path.name)
    config = json.loads((src / "config.json").read_text())
    cut(config, cli.layers)
    if isinstance(config.get("text_config"), dict):
        cut(config["text_config"], cli.layers)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(out)


if __name__ == "__main__":
    main()

"""Write a workflow that relaunches a GLM-5.3 run (live or stopped Argo workflow) with the r22 restart changes.

Changes against the base workflow: trainer image r22 (kernel-cache seed, staleness-cap code), template
acuadron-miles-deployer-v4 (gyms start with the trainer pod), dummy engine weights plus the NEXTN draft
export. With --cap also NatsRolloutFn, max_weight_staleness 8, arena_inflight_multiplier 6 and the gym
replicas for that in-flight cap. Everything else (dataset, lr, shape, excluded nodes) is the base's.

Usage: python3 relaunch.py <base-workflow> <experiment-name> [--cap] > workflow.yaml
The same experiment name resumes from that run's latest checkpoint; a new name starts fresh.
"""
import json, re, subprocess, sys, yaml
base, name, cap = sys.argv[1], sys.argv[2], "--cap" in sys.argv
w = json.loads(subprocess.check_output(["kubectl", "--context", "arena-prod-bom-v2", "-n", "arena-tasks", "get", "workflow", base, "-o", "json"]))
p = {x["name"]: x for x in w["spec"]["arguments"]["parameters"]}
p["experiment-name"]["value"] = name
p["trainer-image"]["value"] = "427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-github/miles:miles-glm53-r22-20261007a"
cfg = p["miles-config"]["value"]
old = re.search(r"^experiment_name: (\S+)", cfg, flags=re.M).group(1)
cfg = cfg.replace(old, name)                      # experiment_name, project_name, arena_sample_summary_dir
for k in ("sglang_load_format", "sglang_speculative_draft_model_path", "sglang_speculative_draft_load_format"):
    assert not re.search(rf"^{k}:", cfg, flags=re.M), k
cfg = cfg.rstrip("\n") + "\n" + (
    "sglang_load_format: dummy  # r22 relaunch: the trainer's initial push fills the engine weights\n"
    "sglang_speculative_draft_model_path: /mnt/scratch-fast-1a-rw/acuadron/models/GLM-5.3-Flash-BF16-nextn  # MTP layer only\n"
    "sglang_speculative_draft_load_format: auto  # the draft loads from disk; without this it takes the dummy load format\n")
if cap:  # optional: the staleness cap with multiplier 6 (needs gym-replicas >= 1.125 x 6 x rollout_batch_size)
    cfg = re.sub(r"^rollout_function_path: .*$", "rollout_function_path: miles_plugins.arena.nats_arena.nats_rollout.NatsRolloutFn", cfg, count=1, flags=re.M)
    cfg = re.sub(r"^arena_inflight_multiplier: .*$", "arena_inflight_multiplier: 6\nmax_weight_staleness: 8", cfg, count=1, flags=re.M)
    rbs = int(re.search(r"^rollout_batch_size: (\d+)", cfg, flags=re.M).group(1))
    p["gym-replicas"]["value"] = str(-(-9 * 6 * rbs // 8))
p["miles-config"]["value"] = cfg
out = {"apiVersion": w["apiVersion"], "kind": w["kind"],
       "metadata": {"generateName": name + "-", "namespace": "arena-tasks", "labels": {"arena.agif.amazon.dev/submitter": "acuadron"}},
       "spec": {"workflowTemplateRef": {"name": "acuadron-miles-deployer-v4"},
                "arguments": {"parameters": [dict(name=x["name"], value=x.get("value")) for x in w["spec"]["arguments"]["parameters"]]}}}
class D(yaml.SafeDumper): pass
D.add_representer(str, lambda d, s: d.represent_scalar("tag:yaml.org,2002:str", s, style="|" if "\n" in s else None))
sys.stdout.write(yaml.dump(out, Dumper=D, sort_keys=False, width=10**6))

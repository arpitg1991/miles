"""Write the heavy-staleness arms (acuadron-agentic-debt-stale-<arm>) from engine-ab-v3's 16-node workflow.

Shape per arm: 4 actor nodes (DP 1) + 12 engines, batch 64 (8 groups x 8), inflight x8, no staleness cap,
NatsRolloutFn (train weight version reaches the drain), the final-v10/v11 loss recipe, image r25.
Arms differ in --update-weights-interval (K), lr and the correction function.
"""
import re, sys, yaml

IMAGE = "427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-github/miles:miles-glm53-r25-20261009a"
TIS = "miles_plugins.arena.off_policy.arena_tis_function"
M2PO = "miles_plugins.arena.off_policy.arena_m2po_function"
ARMS = {
    "a": dict(update_weights_interval=1, lr="1.5e-6", custom_tis_function_path=TIS),
    "b": dict(update_weights_interval=32, lr="1.5e-6", custom_tis_function_path=TIS),
    "c": dict(update_weights_interval=32, lr="5e-6", custom_tis_function_path=TIS),
    "d": dict(update_weights_interval=32, lr="5e-6", custom_tis_function_path=M2PO),
    "e": dict(update_weights_interval=1, lr="5e-6", custom_tis_function_path=TIS),
}
COMMON = dict(
    replicas=16, num_trainers=4, rollout_batch_size=8, global_batch_size=64, arena_inflight_multiplier=8,
    max_weight_staleness=1000, save_interval=100, num_rollout=400,
    rollout_function_path="miles_plugins.arena.nats_arena.nats_rollout.NatsRolloutFn",
    calculate_per_token_loss="true", disable_grpo_std_normalization="true", use_tis="true", tis_clip=2.0,
    sglang_max_running_requests=64,
)
arm = sys.argv[1]
name = f"acuadron-agentic-debt-stale-{arm}"
w = yaml.safe_load(open("/local/home/acuadron/workspaces/miles/training-runs/harbor-rl-glm53-flash/acuadron-agentic-debt-engine-ab-v3/workflow.yaml"))
p = {x["name"]: x for x in w["spec"]["arguments"]["parameters"]}
p["experiment-name"]["value"] = name
p["trainer-image"]["value"] = IMAGE
p["replicas"]["value"] = "16"; p["replica-trainer"]["value"] = "4"; p["gym-replicas"]["value"] = "72"
cfg = p["miles-config"]["value"].replace("acuadron-agentic-debt-engine-ab-v3", name)
cfg = re.sub(r"^replicas: .*$", "replicas: 16  # stale arms: 4 actor (DP 1) + 12 engines", cfg, count=1, flags=re.M)
for k, v in {**COMMON, **ARMS[arm]}.items():
    if k == "replicas":
        continue
    if re.search(rf"^{k}:", cfg, flags=re.M):
        cfg = re.sub(rf"^{k}: .*$", f"{k}: {v}  # stale arm {arm}", cfg, count=1, flags=re.M)
    else:
        cfg = cfg.rstrip("\n") + f"\n{k}: {v}  # stale arm {arm}\n"
p["miles-config"]["value"] = cfg
w["metadata"] = {"generateName": name + "-", "namespace": "arena-tasks", "labels": {"arena.agif.amazon.dev/submitter": "acuadron"}}
class D(yaml.SafeDumper): pass
D.add_representer(str, lambda d, s: d.represent_scalar("tag:yaml.org,2002:str", s, style="|" if "\n" in s else None))
out = f"/tmp/glm53f/stale-{arm}.yaml"; open(out, "w").write(yaml.dump(w, Dumper=D, sort_keys=False, width=10**6))
c = yaml.safe_load(cfg)
print(out, {k: c.get(k) for k in ("experiment_name", "num_trainers", "rollout_batch_size", "global_batch_size", "arena_inflight_multiplier", "max_weight_staleness", "update_weights_interval", "lr", "custom_tis_function_path", "rollout_function_path", "calculate_per_token_loss", "sglang_kv_cache_dtype", "sglang_speculative_num_steps")})

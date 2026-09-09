# r16 image requirements (R3 rollout routing replay, miles arena ADR-0012)

r16 = r15 + `use_rollout_routing_replay: true`. Both images change:

- Trainer: a miles build that carries
  `miles_plugins/arena/nats_arena/routing_replay.py` and the matching
  `message_format.py` / `nats_rollout.py` changes (T1-T3 in ADR-0012).
- Gym: an AREnATasks build with the per-segment routing snapshot, the
  `routed_experts_ref` transport in the Harbor worker, and `ARENA_ROUTING_DIR`
  (G1-G3 in ADR-0012; AREnATasks ADR-0049).
- Shared mount: gym and trainer pods MUST see `ARENA_ROUTING_DIR` on one
  POSIX filesystem at one path. See the TODO in `r16/trainer-pytorchjob.yaml`.

Smoke test before launch: one step at a 2x4 shape with `--limit 10`; success
is `train/ppo_kl` at step 0 near 1e-3 or lower and `train/pg_clipfrac` near 0.
`train/kl_loss` is not a valid check under R3.

The r13 build notes below still describe the base recipe for both images.

# r13 image builds and launch

r13 = r12 with one fix: the gym selects the GLM tool-call parser
(`ARENA_TOOL_CALL_PARSER=glm47`, `r13/gym-worker.yaml`). r12 (2026-09-08) ran
the Vulcan gym with the `qwen3_coder` default parser. GLM-5.3 emits
`<tool_call>bash<arg_key>command</arg_key><arg_value>…</arg_value></tool_call>`,
which that parser does not match, so every episode ended after one model call
with zero tool calls and reward 0 (`vulcan: stop_reason=completed iterations=1
tools=0`); the trainer ran 9 zero-gradient steps. The trainer image and
`miles-config.yaml` are unchanged from r12.

## Gym image `gym-glm53-vulcan-glm47-20260908b`

Built from AREnATasks arpit-glm-53 0d98f0a (adds
`amzn_arena_contract.tool_parsers.glm47`, a faithful port of SGLang's
`Glm47MoeDetector`). Same recipe as r12 section 2 with
`TAG=gym-glm53-vulcan-glm47-20260908b`. Smoke test before launch: roll the
r12 gym Deployment (`rl-glm53f12-gym-sgb`) to this image and confirm
`tools>0` and non-zero rewards on the live r12 engines, then delete r12 and
launch r13 in the r12 order below (names `rl-glm53f13-*`).

---

# r12 image builds and launch

r12 = r11 shape and knobs, plus Vulcan context compaction on the gym
(AREnATasks ADR-0048) and every-segment training on the trainer (miles arena
ADR-0011). Both sides need a new image:

- Trainer: `arena_train_segments: all` in `miles-config.yaml`. The flag
  `--arena-train-segments {final,all}` lives in
  `miles_plugins/arena/nats_arena/nats_rollout.py` (arpit-glm-53 d28503cf).
  miles argparse is strict: on the r11 image `miles-glm53-20260907a` the
  trainer dies at startup on this key.
- Gym: `--agent vulcan` with `ARENA_COMPACTION_MAX=2`. The r11 gym image
  `gym-glm53-ctxnudge-20260906a` carries the arena-terminus-2 nudge, not the
  Vulcan compaction path.

Both builds ran on 2026-09-08 on a Linux/x86_64 host with Docker. Push to
us-east-1 only; `arena-ecr-dev` cannot push elsewhere and the `arena-slime-dev`
repository replicates to ap-south-1, where prod-bom pulls.

## 1. Trainer image `miles-glm53-20260908a`

Same recipe as `r11/BUILD.md`: `examples/arena/Dockerfile` (`COPY . /root/miles`,
so the whole checkout ships; plus the EFA userspace + aws-ofi-nccl layer) on the
glm53next mirror `arena-slime-dev:glm53next-upstream-20260902`. The commit
d28503cf adds `--arena-train-segments {final,all}` and the all-mode DP
alignment pad (zero-loss sibling rows, so `build_dp_schedule` can align
singleton micro-batches), and drops `rollout/compaction_segments_mean` in
favour of the upstream `rollout/num_training_samples` and
`rollout/episode_raw_reward`.

```bash
cd /workplace/guparpit/arena/src/miles
git log -1 --format=%h            # d28503cf on arpit-glm-53

TAG=miles-glm53-20260908a
ACCT=427267593057
BASE=${ACCT}.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:glm53next-upstream-20260902
IMG=${ACCT}.dkr.ecr.us-east-1.amazonaws.com/arena-slime-dev:${TAG}

# ECR logins: ap-south-1 to pull the base, us-east-1 to push.
aws ecr get-login-password --profile arena-ecr-dev --region ap-south-1 \
  | docker login --username AWS --password-stdin ${ACCT}.dkr.ecr.ap-south-1.amazonaws.com
aws ecr get-login-password --profile arena-ecr-dev --region us-east-1 \
  | docker login --username AWS --password-stdin ${ACCT}.dkr.ecr.us-east-1.amazonaws.com

docker build -f examples/arena/Dockerfile \
  --build-arg MILES_BASE_IMAGE=${BASE} \
  -t ${IMG} .

# Flag smoke: the image parses --arena-train-segments all before the push.
docker run --rm --entrypoint python3 ${IMG} -c \
  "import argparse; from miles_plugins.arena.nats_arena.nats_rollout import _add_arena_arguments as a; p=argparse.ArgumentParser(); a(p); print(p.parse_args(['--arena-train-segments','all']).arena_train_segments)"
# -> all

docker push ${IMG}

# Wait for the ap-south-1 replica (the cluster pulls from there).
aws ecr describe-images --profile arena-ecr-dev --region ap-south-1 \
  --repository-name arena-slime-dev --image-ids imageTag=${TAG}
```

`trainer-pytorchjob.yaml` pins
`427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:miles-glm53-20260908a`.

## 2. Gym image `gym-glm53-vulcan-20260908a`

Built from AREnATasks arpit-glm-53 a337c2d with `brazil-build docker-arena`,
which produces the local tag `amzn-arena-tasks:local`. The image carries the
Vulcan agent with token-aware compaction (ADR-0048): the policy writes a
handoff note (pi prompts) before each compaction, the compacted history is
text only (`[system, instruction, bridge]`; `compaction_tail_fraction`
defaults to 0 and arena-sglang rejects a non-zero tail), each compaction
starts a new rollout segment in `rollout.json`, and the gym ships one step per
segment. Push to `arena-slime-dev`, not `arena-tasks-dev`: the shared
`arena-tasks-dev` repository keeps only the 50 most recent images and pruned
the r9/r10 gym tags within hours.

```bash
cd /workplace/guparpit/arena/src/AREnATasks
git log -1 --format=%h            # a337c2d on arpit-glm-53

TAG=gym-glm53-vulcan-20260908a
ACCT=427267593057
IMG=${ACCT}.dkr.ecr.us-east-1.amazonaws.com/arena-slime-dev:${TAG}

brazil-build docker-arena         # -> amzn-arena-tasks:local (linux/amd64)

aws ecr get-login-password --profile arena-ecr-dev --region us-east-1 \
  | docker login --username AWS --password-stdin ${ACCT}.dkr.ecr.us-east-1.amazonaws.com
docker tag amzn-arena-tasks:local ${IMG}
docker push ${IMG}

# Wait for the ap-south-1 replica (the cluster pulls from there).
aws ecr describe-images --profile arena-ecr-dev --region ap-south-1 \
  --repository-name arena-slime-dev --image-ids imageTag=${TAG}
```

`gym-worker.yaml` pins
`427267593057.dkr.ecr.ap-south-1.amazonaws.com/arena-slime-dev:gym-glm53-vulcan-20260908a`.

## 3. Launch order

Run from `examples/arena/harbor-rl-glm53-flash/`. Create the ConfigMap first:
the PyTorchJob mounts it and a missing ConfigMap holds every trainer pod in
`ContainerCreating`. Apply the trainer last, after the NATS broker, the SGLang
Service and the gym Deployment exist.

```bash
kubectl --context arena-prod-bom-v2 -n arena-tasks create configmap rl-glm53f13-trainer-config --from-file=miles-config.yaml=r13/miles-config.yaml
kubectl --context arena-prod-bom-v2 -n arena-tasks apply -f r13/nats.yaml
kubectl --context arena-prod-bom-v2 -n arena-tasks apply -f r13/sglang-svc.yaml
kubectl --context arena-prod-bom-v2 -n arena-tasks apply -f r13/gym-worker.yaml
kubectl --context arena-prod-bom-v2 -n arena-tasks apply -f r13/trainer-pytorchjob.yaml
```

Then check for kueue TAS mis-pins within ~2 min of the trainer apply
(`README.md` step 4), and read `rollout/num_training_samples` and
`rollout/episode_raw_reward` on the first rollout: the row count per step is
data dependent under `arena_train_segments: all`, and the trainer log reports
the `DP alignment:` pad count.

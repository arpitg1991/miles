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
kubectl --context arena-prod-bom-v2 -n arena-tasks create configmap rl-glm53f12-trainer-config --from-file=miles-config.yaml=r12/miles-config.yaml
kubectl --context arena-prod-bom-v2 -n arena-tasks apply -f r12/nats.yaml
kubectl --context arena-prod-bom-v2 -n arena-tasks apply -f r12/sglang-svc.yaml
kubectl --context arena-prod-bom-v2 -n arena-tasks apply -f r12/gym-worker.yaml
kubectl --context arena-prod-bom-v2 -n arena-tasks apply -f r12/trainer-pytorchjob.yaml
```

Then check for kueue TAS mis-pins within ~2 min of the trainer apply
(`README.md` step 4), and read `rollout/num_training_samples` and
`rollout/episode_raw_reward` on the first rollout: the row count per step is
data dependent under `arena_train_segments: all`, and the trainer log reports
the `DP alignment:` pad count.

# r11 trainer image build

r11 sets `arena_inflight_multiplier: 4` in `miles-config.yaml`. The flag
`--arena-inflight-multiplier` lives in
`miles_plugins/arena/nats_arena/nats_rollout.py` (commit
`feat(nats-rollout): make publisher in-flight multiplier configurable`).
The r10 image `arena-slime-dev:miles-glm53-20260905a` does not know the flag,
and miles argparse is strict: the trainer dies at startup on that image.
Build a new image before you apply `trainer-pytorchjob.yaml`.

The image recipe is `examples/arena/Dockerfile` (`COPY . /root/miles`, so the
whole checkout ships; plus the EFA userspace + aws-ofi-nccl layer). Base:
the glm53next mirror `arena-slime-dev:glm53next-upstream-20260902`. Push to
us-east-1 only; `arena-ecr-dev` cannot push elsewhere and the repository
replicates to ap-south-1, where prod-bom pulls. Same recipe as r9/r10
(`README.md` step 2, r9 `trainer-pytorchjob.yaml` image comment).

Run from the miles repo root on a Linux/x86_64 host with Docker. Pick a tag in
the lineage form `miles-glm53-<YYYYMMDD><letter>`; the placeholder below is
`miles-glm53-20260907a`.

```bash
cd /workplace/guparpit/arena/src/miles
git checkout arpit-glm-53 && git pull --ff-only origin arpit-glm-53
git log -1 --format=%h            # record this commit next to the tag in README.md

TAG=miles-glm53-20260907a
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

# Confirm the flag is baked in before you push.
docker run --rm --entrypoint python3 ${IMG} -c \
  "import argparse; from miles_plugins.arena.nats_arena.nats_rollout import _add_arena_arguments as a; p=argparse.ArgumentParser(); a(p); print(p.parse_args(['--arena-inflight-multiplier','4']).arena_inflight_multiplier)"

docker push ${IMG}

# Wait for the ap-south-1 replica to land (the cluster pulls from there).
aws ecr describe-images --profile arena-ecr-dev --region ap-south-1 \
  --repository-name arena-slime-dev --image-ids imageTag=${TAG}
```

Then:

1. Replace `miles-glm53-inflight-TBD` in `trainer-pytorchjob.yaml` with `${TAG}`
   (the image line keeps the ap-south-1 registry host).
2. Add the tag row to the lineage tables in `README.md` and
   `examples/arena/README.md`.
3. Recreate the ConfigMap from the r11 config before you apply the job:
   `kubectl --context arena-prod-bom-v2 -n arena-tasks create configmap rl-glm53f11-trainer-config --from-file=miles-config.yaml=miles-config.yaml`.

# Miles trainer image for the arena Harbor RL jobs (examples/arena/harbor-rl-27b).
#
# Build from the miles repo root so the whole fork (miles_plugins/arena, the
# launcher and the model scripts) ships in the image:
#
#   docker build -f examples/arena/Dockerfile \
#       --build-arg MILES_BASE_IMAGE=radixark/miles:latest \
#       -t <registry>/miles-arena-dev:<tag> .
#
# MILES_BASE_IMAGE must be a miles GPU runtime image produced by docker/build.py
# (radixark/miles:<tag>, see docker/Dockerfile), or your ECR mirror of it. The
# base already carries the full GPU stack (CUDA, sglang, transformer_engine),
# Megatron-LM at /root/Megatron-LM, and an editable miles checkout at
# /root/miles that this build replaces. `latest` is a placeholder — pin the tag
# your fleet has validated.
ARG MILES_BASE_IMAGE=radixark/miles:latest
FROM ${MILES_BASE_IMAGE}

# Arena runtime deps (mirrors setup.py's [arena] extra). The AGISlime
# entrypoint pip-installed these on every pod start (a network dependency and
# drift source); bake them instead. omegaconf is already a miles requirement
# and is repeated here only as documentation of what the launcher's YAML
# conversion needs. boto3 is imported at call time by the S3 HF-checkpoint and
# hosted-eval paths (s3_artifact, eval_rollout, eval_coordinator); the miles
# base image does not provide it.
#
# kubernetes is PINNED: 36.x breaks Authorization handling on EKS — every
# request from the pod returns 401 even with a valid SA token. 35.0.0 is the
# last release that authenticates correctly with the apiserver. Used by the
# ported gym autoscaler and eval paths; bumping past 35.x without an in-pod
# probe WILL break autoscaling and eval triggering silently.
RUN pip install --no-cache-dir "nats-py>=2.6.0" "kubernetes==35.0.0" lakefs boto3 omegaconf typer

# Replace the base's baked checkout with this fork. The base's editable install
# points at /root/miles, so overwriting the tree is the whole install; removing
# it first keeps files deleted in the fork from lingering.
RUN rm -rf /root/miles
COPY . /root/miles

# Import smoke: fail the build if the arena plugin or the launcher cannot resolve.
# The builder has no GPU driver; transformer_engine dlopens libcuda.so.1 at
# import, so the smoke runs against the CUDA toolkit's driver stub. Runtime
# pods get the real driver from the NVIDIA container toolkit.
RUN mkdir -p /tmp/cuda-stub \
    && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda-stub/libcuda.so.1 \
    && cd /root/miles \
    && LD_LIBRARY_PATH=/tmp/cuda-stub:${LD_LIBRARY_PATH} python3 -c "\
import miles; \
import miles_plugins.arena.nats_arena.message_format; \
import miles_plugins.arena.nats_arena.nats_rollout; \
import miles_plugins.arena.nats_arena.data_source; \
import miles_plugins.arena.train_async_arena; \
from miles_plugins.arena.nats_arena.message_format import SALVAGEABLE_RESULT_STATUSES; \
assert SALVAGEABLE_RESULT_STATUSES == ('success', 'truncated')" \
    && LD_LIBRARY_PATH=/tmp/cuda-stub:${LD_LIBRARY_PATH} python3 /root/miles/scripts/run_arena_harbor.py --help >/dev/null \
    && rm -rf /tmp/cuda-stub

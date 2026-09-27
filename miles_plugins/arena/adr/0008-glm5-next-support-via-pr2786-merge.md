# ADR-0008: GLM-5.3-Flash (glm5_next) support via the PR #2786 merge

**Status:** Accepted
**Date:** 2026-09-02

**Builds on:** ADR-0001 (port scope: core edits only where miles has no seam), ADR-0006 (launcher argv parity: `model_arch` resolves `scripts/models/<arch>.py` from the tree)

## Context

Arena needs GLM-5.3-Flash (321B total / ~18B active, `model_type glm5_next`:
45 layers = 34 KDA linear-attention + 11 DSA sparse layers, mHC
hyper-connections, NoPE MLA) trained with GRPO on prod-bom. miles main
(`radixark/miles`, fork base `2799fe38`) has no glm5_next support; it lives in
the unmerged PR radixark/miles#2786 (branch `zhichen/glm53-flash`, 33 commits,
head `dbbd610e7`, 2026-08-29 04:03 PT), which may still be force-pushed.

The example that follows (`examples/arena/harbor-rl-glm53-flash/`) is bound to
that code at three points:

- `model_arch: glm5.3-flash` resolves `scripts/models/glm5.3-flash.py`, whose
  `--spec miles_plugins.models.glm5_next.glm5_next get_glm5_next_spec` needs
  `miles_plugins/models/glm5_next/`;
- `model_name: glm5_next` selects the megatron<->sglang weight mapping
  (`miles_plugins/mbridge/glm5_next.py`, `megatron_to_hf/glm5_next.py`) for the
  weight updater and the HF export;
- `examples/arena/Dockerfile` does `rm -rf /root/miles; COPY . /root/miles`, so
  whatever tree is built IS the trainer install.

The only runtime stack for this model is `docker.io/radixark/miles:glm53next`
(SGLang `sglang-miles-glm53next@9a26e749` + radixark/Megatron-LM PR#89
`e8f57451`), which pins miles at `1cd14c00`, the SHA the recipe page quotes.
That SHA is six commits behind the PR head and misses three real fixes
(`eee9819da` inert rope_theta under NoPE, `75ace38a1` DSA indexer fields via
`text_config` for the composite vision+text config, `dbbd610e7` kpool indexer
per-sequence pools in packed batches), so the image's baked checkout cannot be
used as-is; the tree we ship must overlay it.

On 2026-09-02 00:54 PT the merge of fork base `2799fe38` + `dbbd610e7` was
made on the integration branch `glm53-arena` (`b981804e7`, conflict-free, tree
`84d28154`), the port committed on top, and every trainer image (a-d) was built
from that tree. The fork itself carried none of it: it could neither run the
example nor rebuild its images, and the example README's preflight line "the
fork/image tree carries the GLM merge" was true only of `/tmp`.

Merge facts: PR delta 22 files, +1578/-61; 8 miles core files touched; the
only file both sides edit is `miles/utils/arguments.py` (fork: `qwen3_5`
loss-mask choice ~L2408; PR: `parse_args` L2713-2725), with disjoint hunks.

## Decision

**Carry GLM-5.3-Flash support in the fork by merging PR #2786 at the pinned
head `dbbd610e7` as a real merge commit** (`git merge --no-ff refs/archive/pr2786`),
keeping the base at `2799fe38`. The PR's core hunks ride along unchanged:

- `miles/utils/arguments.py` - `rollout_indexer_topk_num_streams`; indexer fields read from `hf_config.get_text_config()`;
- `miles/utils/hf_config.py` - `glm5_next` config alias on `Glm4vMoeConfig` (`Glm5NextConfig`), `is_dsa()` accepts `glm5_next`;
- `miles/rollout/sglang_rollout.py`, `miles/rollout/generate_utils/generate_endpoint_utils.py` - all-zero `routed_experts` and indexer stream-count asserts;
- `miles/ray/rollout/train_data_conversion.py` - fail fast when replay is on but payloads are missing;
- `miles/utils/reloadable_process_group.py` - base-class C++ collective dispatch (torch 2.13 `PyWorkHolder` UAF);
- `megatron_to_hf/__init__.py` + `glm5_next.py`, `miles_plugins/mbridge/__init__.py` + `glm5_next.py`.

Arena never edits the PR's files; GLM-specific fixes land as image layers
(site-packages patches under `examples/arena/patches/`), in the arena plugin, or
in the example config. The PR history is preserved as `refs/archive/pr2786`
(and the image build tree as `refs/archive/glm53-arena`) so every SHA cited in
this log resolves from the fork.

### Alternatives considered

| Alternative | Why not |
|---|---|
| Keep the fork config-only; leave the model code in the image build tree (`/tmp/glm53-integration`) | The fork cannot start the example (`model_arch`/`model_name` unresolvable) nor rebuild images a-d; the README preflight claim is false; `/tmp` is ephemeral. |
| Vendor a squashed copy of the PR into the fork | Loses the 33-commit upstream history and attribution (Zhichen Zeng) and turns the eventual re-sync with upstream into a manual three-way diff. |
| Wait for #2786 to land upstream, then fast-forward | No ETA; the PR may force-push; runs r1 onwards already depend on this exact head via the images. |
| Merge the doc-pinned `1cd14c00` to match the image base byte-for-byte | Misses the three fixes above; the image's checkout is replaced by our tree anyway, so matching it buys nothing. |

## Consequences

Easier:

- The example is runnable from the fork and the trainer images are rebuildable
  from this branch (`docker build -f examples/arena/Dockerfile .`); the r1-r7
  fixes can be committed against the tree they actually ran on.
- `git log refs/archive/pr2786` and `git log --first-parent` keep upstream and
  arena history separable; when #2786 lands upstream, `git merge upstream/main`
  either sees the same commits or a squash, and the pinned SHA records what ran.
- Verified 2026-09-05 on the equivalent tree (fork + PR + port): 177 arena CPU
  tests, launcher snapshots 4/4, `tests/fast/launch_scripts` 43,
  `test_arguments` + `test_hf_config` + `test_kpool_indexer_packed` 159 passed /
  1 skipped, `miles_plugins.models.glm5_next` imports.

Harder / open:

- Eight miles core files now differ from upstream main for a model-specific
  reason; `arguments.py` diverges in two places (port + PR). Re-sync when
  #2786 merges; if the PR is force-pushed, `refs/archive/pr2786` is the record.
- The PR's tests ride along: `tests/fast/test_kpool_indexer_packed.py` (CPU,
  runs here) and `tests/e2e/megatron/model_scripts/test_glm5_3_flash_4layer_ci.py`
  (GPU CI, not run in this fork).
- The recipe was validated colocated (`--colocate --offload-train-target disk`)
  on GB300-288GB; arena runs disaggregated on B200-180GB, so the recipe's
  memory behaviour does not transfer and its colocate/offload flags must not be
  copied (example README, deviations table (b)).
- The image base still pins miles `1cd14c00`; the overlay is the whole install,
  so an image built from a tree without this merge silently trains the stale
  checkout. Bring-up findings against this model code (fla/triton kernel
  incompatibility, KDA replicated across TP, CP unsupported by the kpool
  indexer) are recorded in `examples/arena/harbor-rl-glm53-flash/RUNLOG.md`
  as they were found.

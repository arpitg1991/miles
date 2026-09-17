"""Standalone check of shape_group_length_reward, pulled from the real source.

No pytest locally, so this compiles the three real function bodies out of
nats_rollout.py and exercises them. Any drift in the source is picked up.
"""
import ast, sys
from types import SimpleNamespace

SRC = "/workplace/guparpit/arena/src/miles/miles_plugins/arena/nats_arena/nats_rollout.py"
WANT = {"_episode_key", "_episodes", "shape_group_length_reward"}
tree = ast.parse(open(SRC).read())
picked = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in WANT]
assert {n.name for n in picked} == WANT, f"missing: {WANT - {n.name for n in picked}}"
ns = {"Sample": object}
exec(compile(ast.Module(body=picked, type_ignores=[]), SRC, "exec"), ns)
shape = ns["shape_group_length_reward"]

def args(coef=0.05):
    return SimpleNamespace(arena_length_reward_coef=coef, reward_key=None)

def group(pairs):
    return [SimpleNamespace(group_index=0, rollout_id=i, index=i, reward=r,
                            response_length=n, remove_sample=False)
            for i, (r, n) in enumerate(pairs)]

def close(a, b): return abs(a - b) < 1e-9

fails = []
def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name)
    if not cond: fails.append(name)

# 1. all-passed group gains variance (the group the filter deletes today)
g = group([(1.0, 40_000), (1.0, 240_000), (1.0, 440_000)])
r = shape(args(), g)
rw = [s.reward for s in g]
check("all-passed group reports rescued", r is True)
check("shortest > middle > longest", rw[0] > rw[1] > rw[2])
check("shortest gets +coef/2", close(rw[0], 1.025))
check("midpoint unchanged", close(rw[1], 1.0))
check("longest gets -coef/2 (no dead zone)", close(rw[2], 0.975))
check("three distinct rewards -> non-zero std", len(set(rw)) == 3)

# 2. off by default
g = group([(1.0, 40_000), (1.0, 440_000)])
check("coef 0 is a no-op", shape(args(coef=0.0), g) is False and [s.reward for s in g] == [1.0, 1.0])

# 3. all-failed group untouched (never pay for giving up early)
g = group([(0.0, 40_000), (0.0, 440_000)])
check("all-failed group untouched", shape(args(), g) is False and [s.reward for s in g] == [0.0, 0.0])

# 4. correctness is never traded away
g = group([(0.9, 40_000), (1.0, 440_000)])
shape(args(), g)
check("best-but-longest still outranks shorter runner-up", g[1].reward > g[0].reward)

# 5. multi-segment episode sums its segments (arena_train_segments=all)
g = [SimpleNamespace(group_index=0, rollout_id=0, index=0, reward=1.0, response_length=300_000, remove_sample=False),
     SimpleNamespace(group_index=0, rollout_id=0, index=1, reward=1.0, response_length=300_000, remove_sample=False),
     SimpleNamespace(group_index=0, rollout_id=1, index=2, reward=1.0, response_length=100_000, remove_sample=False)]
check("segments sum: 600k episode loses to 100k episode", shape(args(), g) is True and g[2].reward > g[0].reward)
check("one reward per episode across its segments", g[0].reward == g[1].reward)

# 6. a removed episode does not set the group's length scale
g = group([(1.0, 40_000), (1.0, 440_000), (1.0, 9_000_000)])
g[2].remove_sample = True
shape(args(), g)
check("removed episode excluded from lo/hi", close(g[0].reward, 1.025) and close(g[1].reward, 0.975))

# 7. the clamp: a 0.01 reward gap must survive a coef that would cross it
g = group([(0.99, 40_000), (1.0, 440_000)])
shape(args(), g)
check("clamp keeps a 0.01 reward gap ordered", g[1].reward > g[0].reward)
check("clamp bounds the fall to 0.49*gap", close(g[1].reward, 1.0 - 0.49 * 0.01))

print(f"\n{len(fails)} failure(s)" + (": " + ", ".join(fails) if fails else ""))
sys.exit(1 if fails else 0)

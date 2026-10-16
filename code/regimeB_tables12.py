"""
Regime B with the LATENT-space cost-to-go ranker g(z, theta) — the harness behind
paper Tables 1 and 2. This is `extended_experiment.run_all()` restricted to regime B
(regime A there is a different configuration from the notebook's and is not the
paper's Regime A table), with per-instance records added and the figures dropped.

`World.__init__` reseeds both `np.random.default_rng(seed)` and `torch.manual_seed(seed)`,
and regime B's world is constructed independently of regime A's in `run_all`, so running
regime B on its own reproduces the same RNG stream as the full `run_all`.

Config, as in run_all: D=8, P=6, K=12, seed=0, N_TEST=50, BUDGETS=[1,4,16],
reference = best of R=96 random restarts (the `World.reference` default).
"""
import csv, json, os, time
import numpy as np, torch
import extended_experiment as E

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(OUT_DIR, exist_ok=True)
T_START = time.time()

NAME_MAP = {"random": "random", "retrieval": "retrieval",
            "sample_pick": "sample-pick", "sample_screen": "sample-screen",
            "latent(trust)": "latent-opt-trust", "latent(no_trust)": "latent-opt-notrust"}

BUDGETS = [1, 4, 16]
N_TEST = 50
ROWS = []

print("building regime B (latent-space ranker) ...", flush=True)
w = E.World(D=8, P=6, K=12, seed=0).build()
print(f"  built; offline solver calls={w.calls}", flush=True)


def evaluate(world, n_test, budgets, shift, scale, dist_name):
    """Mirrors extended_experiment.evaluate call-for-call, but keeps the per-instance rows."""
    th = world.sample_theta(n_test, shift=shift, scale=scale)
    ref = np.array([world.reference(th[i]) for i in range(n_test)])
    out = {m: {B: None for B in budgets} for m in E.METHODS}
    for B in budgets:
        for m in E.METHODS:
            fn = getattr(world, E.FN[m])
            costs = np.array([fn(th[i], B) for i in range(n_test)], dtype=float)
            gaps_raw = costs - ref
            gaps = np.clip(gaps_raw, 0, None)
            for i in range(n_test):
                ROWS.append(dict(regime="B", test_dist=dist_name, method=NAME_MAP[m],
                                 budget=B, instance=i, cost=costs[i], ref_cost=ref[i],
                                 gap_raw=gaps_raw[i], gap=gaps[i]))
            out[m][B] = (float(np.median(gaps)), float(np.percentile(gaps, 90)))
    return out


print("evaluating in-dist ...", flush=True)
ind = evaluate(w, N_TEST, BUDGETS, 0.0, 1.0, "in_dist")
print("evaluating shifted ...", flush=True)
shf = evaluate(w, N_TEST, BUDGETS, 1.5, 1.6, "shifted")
RES = {"in_dist": ind, "shifted": shf}

json.dump(RES, open(os.path.join(OUT_DIR, "regimeB_tables12_results.json"), "w"), indent=2)

raw_path = os.path.join(OUT_DIR, "regimeB_latent_per_instance.csv")
with open(raw_path, "w", newline="") as fh:
    wtr = csv.DictWriter(fh, fieldnames=["regime", "test_dist", "method", "budget",
                                         "instance", "cost", "ref_cost", "gap_raw", "gap"])
    wtr.writeheader()
    for r in ROWS:
        wtr.writerow(r)
print(f"wrote {len(ROWS)} per-instance rows -> {raw_path}", flush=True)

json.dump({"script": "regimeB_tables12.py", "regime": "B", "ranker_space": "latent",
           "D": 8, "P": 6, "K": 12, "world_seed": 0, "screen_pool": 64,
           "n_test": N_TEST, "reference_restarts": 96, "budgets": BUDGETS,
           "solver_calls_counted": int(w.calls),
           "torch": torch.__version__, "numpy": np.__version__,
           "runtime_sec": round(time.time() - T_START, 1)},
          open(os.path.join(OUT_DIR, "regimeB_tables12_runmeta.json"), "w"), indent=2)

for cond in ["in_dist", "shifted"]:
    print(f"\n[regime B / {cond}]  median gap (p90)")
    print("method".ljust(18) + "".join(f"B={B}".ljust(15) for B in BUDGETS))
    for m in E.METHODS:
        row = m.ljust(18)
        for B in BUDGETS:
            med, p90 = RES[cond][m][B]
            row += f"{med:.3f} ({p90:.2f})".ljust(15)
        print(row)
print(f"\nDONE in {time.time()-T_START:.1f}s", flush=True)

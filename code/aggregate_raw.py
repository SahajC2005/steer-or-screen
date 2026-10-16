"""
Recompute median / p90 / mean per (method, budget, test_dist) from the per-instance
rows in data/raw/, and write them out in the same column layout as the transcribed
paper tables. Outputs carry a `_rerun` suffix; the transcribed files are never touched.

Each rerun table is built from the harness that produced the corresponding paper table:

  data/raw/regimeB_latent_per_instance.csv     (regimeB_tables12.py, latent-space ranker,
                                                N_test=50, R=96)
      -> data/table1_regimeB_indist_rerun.csv
      -> data/table2_regimeB_shifted_rerun.csv

  data/raw/regimeB_diffusion_per_instance.csv  (diffusion_experiment.py, solution-space
                                                ranker, N_test=30, R=64)
      -> data/table3_regimeB_agnostic_ranker_rerun.csv

  data/raw/regimeA_per_instance.csv            (rq5_regimeA.py, the notebook harness,
                                                N_test=150, grid reference)
      -> data/regimeA_notebook_mean_gap_rerun.csv

Plus data/raw/all_results_rerun_tidy.csv, every stat in long format.

The Regime A transcribed file is the notebook's MEAN of the UNCLIPPED gap (hence its
`-0.000` entries), so the Regime A rerun table is built from `gap_raw` to stay
comparable. Everything else uses the clipped `gap`.
"""
import csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
RAW = os.path.join(DATA, "raw")


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def stats(rows, key="gap"):
    v = np.array([float(r[key]) for r in rows])
    return dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)),
                mean=float(v.mean()), n=len(v))


def group(rows):
    out = {}
    for r in rows:
        out.setdefault((r["test_dist"], r["method"], int(r["budget"])), []).append(r)
    return out


def methods_in_order(g):
    seen = []
    for (_, m, _) in g:
        if m not in seen:
            seen.append(m)
    return seen


def budgets_of(g):
    return sorted({b for (_, _, b) in g})


tidy = []


def add_tidy(g, source, regime, ranker_space):
    for (dist, m, B), rows in g.items():
        s = stats(rows, "gap")
        for stat in ("median", "p90", "mean"):
            tidy.append(dict(source=source, regime=regime, ranker_space=ranker_space,
                             test_dist=dist, method=m, budget=B, stat=stat,
                             value=round(s[stat], 4), n=s["n"]))
        tidy.append(dict(source=source, regime=regime, ranker_space=ranker_space,
                         test_dist=dist, method=m, budget=B, stat="mean_unclipped",
                         value=round(stats(rows, "gap_raw")["mean"], 4), n=s["n"]))


def write_single_dist(g, dist_key, fname):
    ms, bs = methods_in_order(g), budgets_of(g)
    with open(os.path.join(DATA, fname), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method"] + [f"B{B}_{s}" for B in bs for s in ("median", "p90")])
        for m in ms:
            row = [m]
            for B in bs:
                s = stats(g[(dist_key, m, B)])
                row += [f"{s['median']:.3f}", f"{s['p90']:.2f}"]
            w.writerow(row)
    print(f"  -> data/{fname}")


# ---------------- Tables 1 & 2: regime B, latent-space ranker ----------------
p = os.path.join(RAW, "regimeB_latent_per_instance.csv")
if os.path.exists(p):
    g = group(load(p))
    add_tidy(g, "rerun_regimeB_tables12.py", "B", "latent")
    write_single_dist(g, "in_dist", "table1_regimeB_indist_rerun.csv")
    write_single_dist(g, "shifted", "table2_regimeB_shifted_rerun.csv")
else:
    print(f"MISSING {p} — tables 1-2 rerun not written")

# ---------------- Table 3: regime B, solution-space ranker ----------------
p = os.path.join(RAW, "regimeB_diffusion_per_instance.csv")
if os.path.exists(p):
    g = group(load(p))
    add_tidy(g, "rerun_diffusion_experiment.py", "B", "solution")
    ms, bs = methods_in_order(g), budgets_of(g)
    with open(os.path.join(DATA, "table3_regimeB_agnostic_ranker_rerun.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method"] + [f"{d}_B{B}_{s}" for d in ("indist", "shifted")
                                 for B in bs for s in ("median", "p90")])
        for m in ms:
            row = [m]
            for d, key in (("indist", "in_dist"), ("shifted", "shifted")):
                for B in bs:
                    s = stats(g[(key, m, B)])
                    row += [f"{s['median']:.3f}", f"{s['p90']:.2f}"]
            w.writerow(row)
    print("  -> data/table3_regimeB_agnostic_ranker_rerun.csv")
else:
    print(f"MISSING {p} — table 3 rerun not written")

# ---------------- Regime A: notebook harness, mean of the unclipped gap ----------------
p = os.path.join(RAW, "regimeA_per_instance.csv")
if os.path.exists(p):
    g = group(load(p))
    add_tidy(g, "rerun_rq5_regimeA.py", "A", "latent")
    ms, bs = methods_in_order(g), budgets_of(g)
    with open(os.path.join(DATA, "regimeA_notebook_mean_gap_rerun.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method"] + [f"B{B}" for B in bs])
        for m in ms:
            w.writerow([m] + [f"{stats(g[('in_dist', m, B)], 'gap_raw')['mean']:.3f}"
                              for B in bs])
    print("  -> data/regimeA_notebook_mean_gap_rerun.csv")
else:
    print(f"MISSING {p} — regime A rerun not written")

# ---------------- tidy long format ----------------
if tidy:
    out = os.path.join(RAW, "all_results_rerun_tidy.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["source", "regime", "ranker_space", "test_dist",
                                           "method", "budget", "stat", "value", "n"])
        w.writeheader()
        for r in tidy:
            w.writerow(r)
    print(f"  -> data/raw/all_results_rerun_tidy.csv ({len(tidy)} rows)")

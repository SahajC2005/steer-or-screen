# Re-run notes — per-instance raw data

Run on 2026-10-15 on this machine. Purpose: replace the hand-transcribed aggregates in
`data/` with **actual per-instance measurements**, and record honestly where the re-run
does and does not reproduce the paper.

Everything in this folder is measured here. Everything in `data/*.csv` *without* a
`_rerun` suffix is still the paper's transcribed number and was not modified.

---

## 1. Environment

| | |
|---|---|
| platform | macOS 15.5, arm64 (Apple silicon), Mach-O |
| python | 3.13.12 (`~/miniconda3`) |
| torch | **2.14.1** |
| numpy | 2.2.3 |
| `torch.set_num_threads` | 4 (as the scripts set it) |
| device | CPU only |

**The torch install had to be repaired before anything could run.** The pre-existing
`torch 2.9.1` in `~/miniconda3` was broken — `libtorch_cpu.dylib` was missing from
`site-packages/torch/lib`, so `import torch` raised a `dlopen` `ImportError`. It was
reinstalled with `pip install --force-reinstall --no-cache-dir torch`, which pulled
**2.14.1**. No conda env on the machine had a working torch. This version change is a
plausible contributor to some of the drift in §4 and is not something I could avoid.

**A missing source file also had to be restored.** `code/diffusion_experiment.py` begins
`import extended_experiment as E`, and that module was not in `code/` — it was still only
in `~/Downloads`. It was copied in as `code/extended_experiment.py` (copy, original
untouched). It defines `World`, the solver, the CVAE, the latent cost-to-go `g`, and all
six baseline methods, so nothing in either regime runs without it.

## 2. Seeds

All three scripts are seeded deterministically and **were verified to be bit-reproducible
on this machine**: `regimeB_tables12.py` was run twice and the two per-instance CSVs are
byte-identical (`cmp`). So none of the divergence in §4 is run-to-run noise *here*.

| script | seed |
|---|---|
| `rq5_regimeA.py` | `SEED = 0` → `np.random.default_rng(0)`, `torch.manual_seed(0)` |
| `diffusion_experiment.py` | `E.World(..., seed=0)`, which sets both the world's numpy generator and the global torch seed |
| `regimeB_tables12.py` | same, `E.World(D=8, P=6, K=12, seed=0)` |

## 3. What was run

| script | regime | config | wall clock | rows |
|---|---|---|---|---|
| `code/diffusion_experiment.py` | B, **solution-space** ranker | D=8 P=6 K=12, N_test=30, R=64, B∈{1,4,16}, pool=64 | **22.7 s** | 1260 |
| `code/regimeB_tables12.py` | B, **latent-space** ranker `g(z,θ)` | D=8 P=6 K=12, N_test=50, R=96, B∈{1,4,16}, pool=64 | **28.0 s** | 1800 |
| `code/rq5_regimeA.py` | A, notebook harness | D=2 P=2 K=6, Z=4, N_test=150, B∈{1,2,4,8,16}, pool=64 | **57.2 s** | 4500 |

Total ≈ 108 s. Nothing here is expensive.

Two changes to the brief's plan, both for correctness:

1. **`diffusion_experiment.py` output path.** It wrote to `/home/claude/diffusion_results.json`,
   a dead sandbox path. Now writes to `data/raw/`. No other behavioural change: the order of
   every random draw is untouched, which is why its medians still land exactly on the paper's
   (§4).
2. **A third script, `code/regimeB_tables12.py`, was added.** Paper Tables 1 and 2 do **not**
   come from the diffusion script — they come from `extended_experiment.run_all()`, a
   different harness (latent-space ranker `g(z,θ)`, six methods, N_test=50, R=96). Generating
   `table1/table2_rerun.csv` from the diffusion run would have compared two different
   experiments and manufactured a disagreement. So that harness is re-run on its own, regime B
   only, with per-instance capture and the figure code dropped. `World.__init__` re-seeds both
   RNGs, so running regime B standalone gives the same stream as running it inside `run_all` —
   verified by building regime A first and confirming regime B's output is unchanged.

The notebook `RQ5_latent_opt_vs_sample_pick.ipynb` was **not modified**. `rq5_regimeA.py` is
an extraction of its code cells, minus the matplotlib cell, plus the per-instance dump.

## 4. Agreement with the transcribed tables

Cell-by-cell diff of each `*_rerun.csv` against its transcribed counterpart:

| comparison | cells | differing | max abs diff |
|---|---|---|---|
| Table 3 — regime B, solution-space ranker | 84 | 16 | 4.13 (`diff-screen` shifted B4 **p90**) |
| Regime A — notebook mean gap | 30 | 10 | 0.027 (`latent-opt-notrust` B1) |
| Table 1 — regime B in-dist, latent ranker | 36 | 27 | 0.96 |
| Table 2 — regime B shifted, latent ranker | 36 | 31 | 2.22 |

### Table 3 and Regime A: effectively reproduced

**Table 3 reproduces 38 of its 42 median cells exactly**, to three decimals. Four medians
differ — `vae-pick` in-dist B4 (0.354 → 0.371), `latent-opt-trust` in-dist B4 (0.168 → 0.000)
and shifted B4 (2.345 → 2.726), and `diff-pick` shifted B16 (1.415 → 0.435, the largest median
move). Every *screener* median is unchanged and still 0.000 in all 18 of its cells, which is
the cell set §5 depends on. The remaining 12 differences are **p90** cells. `random` and `retrieval`,
the two methods that touch no neural network, are identical throughout, which confirms the
RNG stream and the config (N_test=30, R=64) are the same as the paper's. The tail movement
is confined to the network-dependent methods (`vae-screen`, `diff-screen`, `diff-pick`,
`latent-opt-trust`): with N_test=30, the p90 is an interpolation between the 27th and 28th
order statistic, so a single instance landing in a different basin moves it a long way.
Tiny float differences in net training under torch 2.14.1 vs. the paper's version are
enough. The largest, `diff-screen` shifted B4 p90 7.37 → 3.24, is one tail instance and the
median stays 0.000.

**Regime A agrees to within 0.027 on every cell**, and the only columns that move at all are
`sample-screen`, `latent-opt-trust` and `latent-opt-notrust` — exactly the three methods that
consume the learned surrogate `g`. `random`, `retrieval` and `sample-pick`, which do not, are
identical to three decimals. Same mechanism as above.

### Tables 1 and 2: a real config difference — traced, but not recoverable

Here the re-run does **not** reproduce the paper, and the reason is diagnosable:
**`random` disagrees** (in-dist B1 median 2.535 → 2.565; shifted B16 p90 8.99 → 6.77).
`random` draws uniform seeds and compares against the reference — it touches no network, no
surrogate, and no torch RNG. It can only differ if the *numpy* stream differs, i.e. if the
paper's run reached `evaluate()` at a different position in that stream than the
`extended_experiment.py` now on disk does.

**The paper's numbers are not a transcription error.** `run_all()` saves its own plot,
`paper_fig_main.png`, and that file survives (in `figures/`, and byte-identical in
`~/Downloads`, dated 2026-08-17 alongside `extended_experiment.py` itself). It is a direct
graphical record of the original run. Its `B_highdim` panels reproduce every median in
Tables 1 and 2 — `random` 2.53/2.52/1.67 and 4.47/4.45/3.63, `retrieval` 0.65/0/0,
`sample_pick` 1.24/0/0, `sample_screen` 0/0/0, `latent(trust)` 0.57/0.43/0,
`latent(no_trust)` 2.02/0/0 — so the tables were transcribed faithfully from a genuine
`run_all()` run. The figure also pins the paper's **regime A** `random` median at B=1 to
≈0.72 in-distribution (the paper states `0.72` in §5.1) and ≈0.26 shifted. The re-run gives
**1.010** and **0.125**. So *both* regimes of that run disagree with the script on disk, which
rules out anything specific to regime B.

**The structural signature.** In the paper's run the `random` median is essentially *flat*
from B=1 to B=4 — 2.535 → 2.525 in-distribution and 4.467 → 4.448 shifted — and the figure
confirms this is real, not rounding: the `random` line is visibly horizontal between the
first two budgets in both bottom panels. The re-run instead drops 2.565 → 2.287 and
4.788 → 4.303, a move far larger than the figure's reading error. Whatever differs, it
changes which test instances are drawn, not merely the third decimal.

#### The search

During `evaluate`, only `reference()` and `m_random` consume the world's numpy generator, and
neither touches the CVAE, the surrogate `g`, or the torch RNG. The `random` row is therefore a
*clean fingerprint* of the configuration, computable **without training anything** — the
landscape (`C`, `Ddir`, `Dbase`) and the solver are all that is needed, and `build()`'s numpy
draws can be reproduced and discarded. `code/provenance_fingerprint.py` does exactly this; it
was checked to reproduce the re-run's `random` row to every digit, and costs ≈0.9 s per
candidate instead of ≈28 s.

With that, **2,739 candidate configurations** were scored against the paper's regime-B
`random` row — all twelve numbers, medians *and* p90, from both tables:

| axis swept | range | configs | best L1 |
|---|---|---|---|
| world seed | 0–20 | 21 | 5.41 |
| `n_test` × `R` × seed | {30,50,100,150} × {24…256} × {0,1,2} | 96 | 4.50 |
| `n_train` × `inits` × `n_ctg` | 100–2000 × 4–64 × 500–32000 | 2,022 | 2.32 |
| raw numpy stream offset | 600 random offsets, 0–2×10⁶ | 600 | 2.77 |

(L1 = summed absolute error over the 12 numbers of the `random` row; the script's own
config scores 8.74.) Scored jointly against *both* regimes — the eight medians the figure
and tables pin down — the best of 1,782 build configurations still misses by 0.79.
**Nothing matches.** A true configuration would score ≈0.

#### Why it cannot be pinned down

`n_train`, `inits` and `n_ctg` affect the `random` row *only* by moving the stream position
before `evaluate()`, so sweeping the raw offset subsumes that entire grid and any other
upstream difference in draw counts. Doing that gives the sampling distribution of each
statistic under this harness, and the paper's values sit squarely inside it:

| statistic | offset distribution (mean ± sd) | paper | percentile |
|---|---|---|---|
| in-dist B1 median | 2.665 ± 0.329 | 2.535 | 35% |
| in-dist B4 median | 2.334 ± 0.342 | 2.525 | 71% |
| in-dist B16 median | 1.471 ± 0.368 | 1.671 | 70% |
| shifted B1 median | 4.776 ± 0.630 | 4.467 | 29% |
| shifted B4 median | 4.443 ± 0.669 | 4.448 | 49% |
| shifted B16 median | 3.389 ± 0.780 | 3.627 | 61% |
| in-dist B1 p90 | 5.279 ± 0.597 | 5.08 | 39% |
| shifted B1 p90 | 9.471 ± 0.956 | 10.07 | 75% |

Every paper number is an ordinary draw from this harness's own spread. So the discrepancy is
**not** evidence of a different model, a different metric, or an error in the paper — it is
consistent with the same code at a different stream offset. But that also means the
configuration is **not identifiable** from the published numbers: the statistic is noisy
enough (sd ≈ 0.3–0.8 on a quantity of size 2–5) that many offsets land near any one target
and none of the 2,739 tried lands on all of them.

The one genuinely unusual feature is the flat B1→B4 median. Across the 600 offsets the median
improvement is 0.332 (in-dist) and 0.334 (shifted); the paper's 0.010 and 0.019 sit at the
4.5th and 9.5th percentile, and **only 3 of 600 offsets (0.5%) produce both at once**. That is
not impossible — it is a tail draw — but it does say the re-run's stream position is a typical
one and the paper's is not, which is further evidence the two configurations genuinely differ.

#### What was ruled out

- **An older copy of the script.** The only `extended_experiment.py` anywhere on this machine
  outside the repo is `~/Downloads/extended_experiment.py`, and it is **byte-identical**
  (md5 `599ccf10…`) to `code/extended_experiment.py`. There is no earlier version to find.
  The duplicate `paper_fig_main (1).png` is likewise byte-identical to `paper_fig_main.png`,
  so there is only one run on record, not two.
- **A mis-transcribed table** — refuted by the figure, above.
- **Run-to-run noise on this machine** — refuted in §2; the harness is bit-reproducible here.
- **Evaluation loop order.** Swapping the `for B` / `for m` nesting does not move the `random`
  row at all: no other method draws from the numpy generator, so `random`'s draw sequence is
  the same either way.

#### Verdict

The original run's artifacts — `/home/claude/paper_results.json` — no longer exist, and the
script that would name the configuration is identical to the one that fails to reproduce it.
Short of those artifacts the question cannot be settled from this machine, and nothing was
tuned to force a match: `regimeB_tables12.py` still uses the values the committed script
actually specifies (N_test=50, R=96, `build()` defaults).

**Treat `table1_regimeB_indist_rerun.csv` and `table2_regimeB_shifted_rerun.csv` as a fresh,
fully-documented run of the Tables 1–2 harness, not as a reproduction of Tables 1–2.** The
paper's Tables 1–2 are a faithful record of a real run of this harness whose exact
configuration is lost; the re-run is a faithful record of the committed configuration. Both
are honest, and they are not the same run.

## 5. Do the paper's conclusions survive? Yes.

Checked mechanically against the per-instance rows (all three re-runs, every cell):

1. **Screening reaches a median gap of 0.** Holds in every regime B cell, in-distribution and
   shifted, at every budget, for `sample-screen`, `vae-screen` and `diff-screen`. In regime A
   the screening median is 0.0032 / 0.0007 / 0.0003 / 0.0001 / 0.000 at B=1…16 — not exactly
   zero, but the smallest of any method, and regime A's reference is a dense grid rather than
   a multi-start so exact zeros are not expected.

2. **Steering never beats screening.** In all **17** (harness × distribution × budget) cells,
   `latent-opt-trust`'s median is ≥ the best screener's. Zero violations.

3. **Steering degrades as budget grows.** Clearest under shift, where steering starts slightly
   ahead of random and ends behind it:

   | | B=1 | B=4 | B=16 |
   |---|---|---|---|
   | regime B shifted, latent ranker — steer | 4.422 | 3.916 | 3.410 |
   | regime B shifted, latent ranker — random | 4.788 | 4.303 | **2.804** |
   | regime A — steer | 0.006 | 0.005 (B=4) | 0.004 |
   | regime A — random | 0.727 | 0.042 (B=4) | **0.001** |

   In both, random overtakes steering by B=16: multi-start diversity keeps paying off while
   steering, collapsing its B seeds onto one mode, stops improving. This is the paper's
   diversity-collapse story, and the re-run shows it slightly more starkly than the paper does.

**Nothing found here changes the paper's conclusions.** The only material caveat is the
Tables 1–2 provenance gap in §4, which affects the exact numbers in two tables, not the
direction or ordering of any result.

## 6. Files

Per-instance rows — long format, one row per (method, budget, test instance):

| file | rows | source script |
|---|---|---|
| `regimeA_per_instance.csv` | 4500 | `rq5_regimeA.py` |
| `regimeB_latent_per_instance.csv` | 1800 | `regimeB_tables12.py` |
| `regimeB_diffusion_per_instance.csv` | 1260 | `diffusion_experiment.py` |

Schema: `regime, test_dist, method, budget, instance, cost, ref_cost, gap_raw, gap`
where `gap_raw = cost − ref_cost` and `gap = max(gap_raw, 0)`.

`gap_raw` is kept because the Regime A notebook reports the **unclipped** mean (this is where
the transcribed `-0.000` entries for retrieval come from — the method beat the reference). The
Regime A rerun table is therefore built from `gap_raw`; everything else uses `gap`.

Also here: `*_runmeta.json` (config, package versions, solver-call counts, runtime per run),
`diffusion_results.json` and `regimeB_tables12_results.json` (the scripts' own aggregate
dumps), and `all_results_rerun_tidy.csv` (432 rows: median / p90 / mean / mean_unclipped for
every method × budget × distribution × harness).

Re-aggregate at any time with `python code/aggregate_raw.py`; it reads only this folder and
writes only `*_rerun.csv`.

`code/provenance_fingerprint.py` is the tool behind §4's search. It recomputes the `random`
row of either regime for an arbitrary configuration without training any network, in ~0.9 s
per candidate. `python code/provenance_fingerprint.py --verify` checks it against the
`random` row `regimeB_tables12.py` actually produced; `--sweep-seeds N` reproduces the seed
sweep. It imports nothing from the experiment scripts and writes no files.

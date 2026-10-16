# Latent-variable steering vs. screening — data folder

Collected from this machine on 2026-10-14. Source paper: **"Steer or Screen? How to Spend a
Learned Cost-to-Go When Warm-Start Budget Is Scarce"** (`paper/steer_or_screen_paper.md`).

**The question (RQ5):** given a trained conditional VAE over good solutions and a learned
cost-to-go predictor, at a *matched budget of expensive solver calls*, is it better to
**steer** (cheap gradient descent on the cost-to-go in latent space, then decode) or to
**screen** (sample many latents, rank by the surrogate, solve only the top few)?

**The answer:** screening. `sample-screen` hits a median optimality gap of 0 at every budget
in the hard regime, including B=1 under distribution shift. Steering never wins and *degrades*
as budget grows — it collapses all B seeds onto one mode and forfeits the diversity that
scarce-budget multi-start depends on. A solution-space ranker screens diffusion and VAE
samples equally well, so the asset is the cost-to-go signal, not the latent geometry.

## Layout

```
data/      extracted result tables (CSV) + run diagnostics
code/      the experiment notebook and the diffusion follow-up script
paper/     paper (md + pdf), mechanism note, RQ5 explainer, pitch deck
figures/   the three paper figures
```

## data/

| file | contents |
|---|---|
| `all_results_tidy.csv` | all 186 numbers in long format — `source,regime,ranker_space,test_dist,method,budget,stat,value` |
| `regimeA_notebook_mean_gap.csv` | Regime A (D=2,P=2,K=6), **mean** gap, B∈{1,2,4,8,16} |
| `table1_regimeB_indist.csv` | Paper Table 1 — Regime B (D=8,P=6,K=12), in-distribution, median + p90 |
| `table2_regimeB_shifted.csv` | Paper Table 2 — Regime B, shifted θ~N(1.5, 1.6²I), median + p90 |
| `table3_regimeB_agnostic_ranker.csv` | Paper Table 3 — Regime B with the solution-space ranker; adds `diff-pick`/`diff-screen` |
| `training_diagnostics.txt` | VAE recon/KL, surrogate MSE, solver-call count, hyperparameters |

Metric throughout: **optimality gap** = `cost_method − c*(θ)`, clipped at 0, where `c*` comes
from a large random multi-start. Lower is better; 0 means the global basin was reached.

Method names are normalized across files: `random`, `retrieval`, `sample-pick`,
`sample-screen`, `latent-opt-trust`, `latent-opt-notrust`, and in Table 3
`vae-pick`, `vae-screen`, `diff-pick`, `diff-screen`.

## Provenance

Every file came from `~/Downloads` (copied, not moved; originals untouched):

- `code/RQ5_latent_opt_vs_sample_pick.ipynb` — the Regime A experiment. Its stored cell
  outputs are the source of `regimeA_notebook_mean_gap.csv` and `training_diagnostics.txt`.
- `code/diffusion_experiment.py` — the §5.4 follow-up (conditional DDPM + solution-space ranker).
- `data/table1..table3` — transcribed from the markdown tables in `paper/steer_or_screen_paper.md`.

## Caveats

1. **No raw per-instance records exist on this machine.** `diffusion_experiment.py` writes its
   results to `/home/claude/diffusion_results.json` — a sandbox path, not a local one. Everything
   in `data/` is therefore *aggregated* (median/p90/mean over instances), transcribed from the
   paper tables and the notebook's stored outputs. To get per-instance gaps, re-run the code.
2. **Regime A numbers are means; Regime B numbers are medians and p90s.** Not comparable
   directly — `all_results_tidy.csv` keeps them apart via the `stat` and `source` columns.
3. **Table 3 is a separate run** (N_test=30, R=64, solution-space ranker), so its absolute
   values differ slightly from Tables 1–2. The method ordering is identical.
4. **Negative values in Regime A retrieval** (`-0.000`) are the clip-at-zero metric picking up
   a run that beat the reference multi-start. Treat as 0.
5. **Synthetic landscapes only** — sums of Gaussian wells, no constraints. P=6 is "high" only
   relative to P=2. External validity to real MPC/trajectory problems is untested.
6. **Surrogate quality is an unexamined confound.** Screening's dominance assumes `g` ranks
   accurately; the paper does not stress-test what happens when it does not.
7. Duplicate copies existed in `~/Downloads` (`steer_or_screen_paper (1).pdf`,
   `RQ5_latent_opt_vs_sample_pick (1).ipynb`, `paper_fig_main (1).png`) — byte-identical to the
   originals, so only one of each was copied. (Re-checked 2026-10-15 while chasing the
   Tables 1–2 provenance gap: still byte-identical, so there is no second run on record.)

## Open threads from the paper

Diversified steering (repulsion / quality-diversity, to see if latent geometry can be rescued);
higher P with real constraints; stress-testing the ranker's accuracy; a theory-lite account of
diversity collapse; a larger diffusion/flow sampler for the pick-vs-pick comparison.

---

## Measured vs. transcribed (added 2026-10-15)

Caveat 1 above is now only half true. The code was re-run locally and `data/raw/` holds
**actual per-instance measurements** — 7,560 rows, one per (method, budget, test instance),
with the achieved cost, the reference `c*(θ)` and the gap. Full provenance, including where
the re-run does *not* match the paper, is in **`data/raw/RERUN_NOTES.md`**.

| | measured here | transcribed from the paper |
|---|---|---|
| `data/raw/*_per_instance.csv` | ✅ 7,560 raw rows | |
| `data/raw/all_results_rerun_tidy.csv` | ✅ recomputed | |
| `data/*_rerun.csv` | ✅ recomputed from the raw rows | |
| `data/table1..table3`, `regimeA_notebook_mean_gap.csv`, `all_results_tidy.csv` | | 📄 unchanged — still the paper's numbers |

The `_rerun` files never overwrite the transcribed ones; the two sit side by side so they
can be compared.

### What was run

| script | regime | runtime |
|---|---|---|
| `code/diffusion_experiment.py` | B, solution-space ranker (paper Table 3) | 22.7 s |
| `code/regimeB_tables12.py` | B, latent-space ranker (paper Tables 1–2) | 28.0 s |
| `code/rq5_regimeA.py` | A, notebook harness | 57.2 s |

Three things changed in `code/`:

- `extended_experiment.py` was **added** — `diffusion_experiment.py` imports it and it was
  missing from this folder (still only in `~/Downloads`). Nothing runs without it.
- `diffusion_experiment.py` now writes to `data/raw/` instead of the dead sandbox path
  `/home/claude/`, and records per-instance rows. Its RNG stream is untouched.
- `rq5_regimeA.py`, `regimeB_tables12.py` and `aggregate_raw.py` are **new**. The notebook
  itself is unmodified; `rq5_regimeA.py` is an extraction of its code cells.

Re-running needs a working torch — the `torch 2.9.1` in `~/miniconda3` was broken (missing
`libtorch_cpu.dylib`) and was reinstalled as 2.14.1. See the notes for details.

### Headline

**The paper's conclusions reproduce.** Screening hits a median gap of 0 in every regime B
cell; steering never beats screening in any of the 17 cells checked; and steering falls
behind plain random multi-start by B=16, which is the diversity-collapse claim.

**One real discrepancy.** Tables 1–2 do not reproduce numerically — even `random`, which
touches no network, disagrees, so the paper's run reached the evaluation loop at a different
position in the numpy stream than the `extended_experiment.py` on disk does. Table 3 and
Regime A, by contrast, reproduce to three decimals on every median.

### The Tables 1–2 provenance gap (closed out 2026-10-15)

Chased as far as this machine allows. The short version:

- **The paper's numbers are real and faithfully transcribed.** `run_all()` saved its own
  plot, `figures/paper_fig_main.png`, and that file survives. Its high-dim panels reproduce
  every median in Tables 1–2, and it pins the paper's regime A `random` median at B=1 to
  ≈0.72 — where the re-run gives 1.010. Both regimes of that run disagree with the committed
  script, so this is not a transcription error and not specific to regime B.
- **The configuration could not be recovered.** `random`'s row is a clean fingerprint of the
  config — it touches no network — so it can be scored without training anything
  (`code/provenance_fingerprint.py`, ~0.9 s a candidate). **2,739 configurations** were
  tried: world seeds 0–20, `n_test` × `R`, `n_train` × `inits` × `n_ctg` over
  100–2000 × 4–64 × 500–32000, and 600 raw numpy-stream offsets. Nothing matches; the best
  missed by 2.32 on the 12-number row where a true config scores ≈0. **Nothing was tuned to
  force a match.**
- **No older copy of the code exists.** `~/Downloads/extended_experiment.py` is byte-identical
  to `code/extended_experiment.py`, and `paper_fig_main (1).png` is byte-identical to
  `paper_fig_main.png` — one run on record, not two.
- **It is unidentifiable, not anomalous.** Across random stream offsets, every one of the
  paper's eight pinned statistics sits between the 29th and 75th percentile of this harness's
  own spread. The paper's numbers are ordinary draws from this code; the sd on these
  quantities (0.3–0.8) is simply too large for 12 published numbers to pin down a config.

The original run's artifacts (`/home/claude/paper_results.json`) are gone, so the question
cannot be settled here. **Tables 1–2 are a faithful record of a real run of this harness whose
exact configuration is lost; `table1/table2_rerun.csv` are a faithful record of the committed
configuration. They are not the same run.** Full detail, including the per-statistic
percentiles, is in §4 of `data/raw/RERUN_NOTES.md`.

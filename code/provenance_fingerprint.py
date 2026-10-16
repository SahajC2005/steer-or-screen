"""
Provenance fingerprint for paper Tables 1-2 (and Figure 1's regime A medians).

Why this exists
---------------
`code/regimeB_tables12.py` re-runs the harness behind Tables 1-2 but does not
reproduce the paper's numbers, and the disagreement reaches `random` — a method
that touches no neural network. See `data/raw/RERUN_NOTES.md` section 4.

During `extended_experiment.evaluate`, only `reference()` and `m_random()` draw
from the World's numpy generator, and neither touches the CVAE, the cost-to-go
surrogate `g`, or the torch RNG. So the `random` row is a clean fingerprint of
the *configuration* and can be computed WITHOUT training anything: all that is
needed is the landscape (C, Ddir, Dbase) and the Adam solver. `build()`'s own
numpy draws are reproduced in order and discarded, which costs microseconds
instead of the seconds its solver loop and two network trainings take.

That makes a candidate configuration cost ~0.9 s instead of ~28 s, which is what
made the 2,739-configuration search in the notes feasible.

This module does NOT import `extended_experiment`; it re-implements the
stream-relevant subset on purpose. `--verify` checks it against the known-good
`random` row that `regimeB_tables12.py` actually produced on this machine.

Usage
-----
    python provenance_fingerprint.py --verify          # self-check
    python provenance_fingerprint.py                   # score the committed config
    python provenance_fingerprint.py --sweep-seeds 21  # example sweep
"""
import argparse

import numpy as np
import torch

torch.set_num_threads(4)

# World constants, copied from extended_experiment.World.__init__
LO, HI = -4.5, 4.5
SIGMA, BOWL = 0.8, 0.03

# Paper Tables 1-2, `random` row: (median, p90) at B = 1, 4, 16.
PAPER_B = {"in_dist": {1: (2.535, 5.08), 4: (2.525, 5.38), 16: (1.671, 4.68)},
           "shifted": {1: (4.467, 10.07), 4: (4.448, 10.06), 16: (3.627, 8.99)}}
# Paper Figure 1 / section 5.1, regime A `random` median at B = 1.
PAPER_A_B1 = {"in_dist": 0.72, "shifted": 0.258}

# What regimeB_tables12.py actually produced here, for --verify.
RERUN_B = [2.565, 5.355, 2.287, 5.456, 1.876, 4.263,
           4.788, 8.145, 4.303, 8.002, 2.804, 6.765]


class Fingerprint:
    """The numpy-stream-relevant subset of extended_experiment.World."""

    def __init__(self, D, P, K, seed=0):
        self.D, self.P, self.K = D, P, K
        self.rng = np.random.default_rng(seed)
        # exact __init__ draw order — this is what fixes the landscape
        self.C = torch.tensor(self.rng.uniform(-3, 3, size=(K, D)), dtype=torch.float32)
        self.Ddir = torch.tensor(self.rng.normal(size=(K, P)), dtype=torch.float32)
        self.Dbase = torch.tensor(self.rng.uniform(0.5, 1.5, size=K), dtype=torch.float32)

    def f(self, x, th):
        d = torch.nn.functional.softplus(self.Dbase + th @ self.Ddir.T)
        sq = ((x.unsqueeze(-2) - self.C) ** 2).sum(-1)
        return -(d * torch.exp(-sq / (2 * SIGMA ** 2))).sum(-1) + BOWL * (x ** 2).sum(-1)

    def solve(self, x0, th, steps=60, lr=0.05):
        """Same solver as World.solve. Rows are independent, so batching is exact."""
        x = x0.clone().requires_grad_(True)
        opt = torch.optim.Adam([x], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            self.f(x, th).sum().backward()
            opt.step()
        with torch.no_grad():
            return self.f(x, th)

    def sample_theta(self, n, shift=0.0, scale=1.0):
        return torch.tensor(self.rng.normal(shift, scale, size=(n, self.P)), dtype=torch.float32)

    def burn_build(self, n_train=600, inits=24, n_ctg=8000):
        """Reproduce build()'s numpy draws, in order, and discard them."""
        self.rng.normal(0.0, 1.0, size=(n_train, self.P))        # sample_theta(n_train)
        for _ in range(n_train):
            self.rng.uniform(LO, HI, size=(inits, self.D))       # per-instance restarts
        self.rng.normal(0.0, 1.0, size=(n_ctg, self.P))          # sample_theta(n_ctg)

    def _best_of(self, th, n_test, m):
        """n_test blocks of m restarts, drawn in stream order, solved as one batch."""
        x0 = torch.tensor(
            np.stack([self.rng.uniform(LO, HI, size=(m, self.D)) for _ in range(n_test)]),
            dtype=torch.float32).reshape(-1, self.D)
        th_rep = th.repeat_interleave(m, dim=0)
        return self.solve(x0, th_rep).reshape(n_test, m).min(dim=1).values.numpy().astype(np.float64)

    def random_row(self, n_test, R, budgets, shift, scale):
        th = self.sample_theta(n_test, shift=shift, scale=scale)
        ref = self._best_of(th, n_test, R)       # World.reference, per instance
        out = {}
        for B in budgets:
            gaps = np.clip(self._best_of(th, n_test, B) - ref, 0, None)
            out[B] = (float(np.median(gaps)), float(np.percentile(gaps, 90)))
        return out


def fingerprint(seed=0, n_train=600, inits=24, n_ctg=8000, n_test=50, R=96,
                D=8, P=6, K=12, budgets=(1, 4, 16), shift=1.5, scale=1.6, advance=0):
    """The `random` row for one candidate configuration, in-dist and shifted.

    `advance` steps the bit generator directly, which subsumes any upstream
    difference in draw counts (n_train / inits / n_ctg and anything else).
    """
    w = Fingerprint(D, P, K, seed)
    w.burn_build(n_train, inits, n_ctg)
    if advance:
        w.rng.bit_generator.advance(advance)
    return {"in_dist": w.random_row(n_test, R, budgets, 0.0, 1.0),
            "shifted": w.random_row(n_test, R, budgets, shift, scale)}


def flat(fp):
    return [round(v, 3) for cond in ("in_dist", "shifted") for B in (1, 4, 16) for v in fp[cond][B]]


def score_B(fp, paper=PAPER_B):
    """L1 error over the 12 numbers of the regime-B `random` row."""
    return sum(abs(fp[c][B][k] - paper[c][B][k]) for c in paper for B in paper[c] for k in (0, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="check against the known regimeB_tables12.py output")
    ap.add_argument("--sweep-seeds", type=int, metavar="N",
                    help="score world seeds 0..N-1 against the paper's random row")
    args = ap.parse_args()

    if args.verify:
        got = flat(fingerprint())
        ok = got == RERUN_B
        print("fingerprint :", got)
        print("regimeB_tables12.py:", RERUN_B)
        print("MATCH" if ok else "MISMATCH")
        raise SystemExit(0 if ok else 1)

    if args.sweep_seeds:
        rows = []
        for s in range(args.sweep_seeds):
            fp = fingerprint(seed=s)
            rows.append((score_B(fp), s, flat(fp)))
        rows.sort()
        print("paper target:", [v for c in ("in_dist", "shifted")
                                for B in (1, 4, 16) for v in PAPER_B[c][B]])
        for l1, s, row in rows:
            print(f"  seed {s:>2}  L1={l1:7.3f}  {row}")
        raise SystemExit(0)

    for name, cfg in (("regime B (D=8,P=6,K=12)", dict(D=8, P=6, K=12)),
                      ("regime A (D=2,P=2,K=6) ", dict(D=2, P=2, K=6))):
        fp = fingerprint(**cfg)
        print(f"{name}  random row at the committed config:")
        for cond in ("in_dist", "shifted"):
            print("   ", cond, {B: tuple(round(x, 3) for x in v) for B, v in fp[cond].items()})
    print("\npaper regime B target:", [v for c in ("in_dist", "shifted")
                                       for B in (1, 4, 16) for v in PAPER_B[c][B]])
    print("paper regime A B=1 medians:", PAPER_A_B1)
    print("regime B L1 vs paper: %.3f" % score_B(fingerprint()))


if __name__ == "__main__":
    main()

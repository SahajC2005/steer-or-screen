"""
Settle the architectural question: does 'diffusion sampler + learned cost-to-go
ranker' match or beat the CVAE, in the hard (high-dim, sparse) regime?

Adds:
  - a conditional DDPM over good solutions p(x|theta)
  - an ARCHITECTURE-AGNOSTIC ranker h(x0, theta) ~ solved cost, defined in
    SOLUTION space so it can screen samples from ANY generator
Compares, at matched solver-query budget, in regime B (in-dist + shifted):
  random, retrieval, cvae-pick, cvae-screen(h), diff-pick, diff-screen(h),
  latent-opt(trust)   [CVAE-specific steering, for continuity]
"""
import csv, json, math, os, time, numpy as np, torch, torch.nn as nn
import extended_experiment as E   # reuse World, mlp, evaluate helpers
torch.set_num_threads(4)

# output dir: data/raw/ next to this file's parent (was /home/claude, a dead sandbox path)
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(OUT_DIR, exist_ok=True)
T_START = time.time()

# canonical method names, matching data/ and README conventions
NAME_MAP = {"random": "random", "retrieval": "retrieval",
            "cvae-pick": "vae-pick", "cvae-screen": "vae-screen",
            "diff-pick": "diff-pick", "diff-screen": "diff-screen",
            "latent-opt(trust)": "latent-opt-trust"}
ROWS = []   # per-instance records: the point of this run

# ---------------- conditional DDPM ----------------
class Diffusion:
    def __init__(self, world, T=50, hidden=128):
        self.w = world; self.D = world.D; self.P = world.P; self.T = T
        betas = torch.linspace(1e-4, 0.02, T)
        self.betas = betas; self.alphas = 1 - betas
        self.abar = torch.cumprod(self.alphas, 0)
        self.temb = 16
        self.net = E.mlp([self.D + self.temb + self.P, hidden, hidden, hidden, self.D])

    def t_embed(self, t):
        # sinusoidal embedding of integer timestep t (shape (B,))
        half = self.temb // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
        ang = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(ang), torch.cos(ang)], -1)

    def eps(self, xt, t, th):
        return self.net(torch.cat([xt, self.t_embed(t), th], -1))

    def train(self, iters=2200, bs=256, lr=2e-3):
        x0all, thall = self.w.sol_x, self.w.sol_th
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        n = x0all.shape[0]
        for it in range(iters):
            idx = torch.randint(0, n, (bs,))
            x0, th = x0all[idx], thall[idx]
            t = torch.randint(0, self.T, (bs,))
            ab = self.abar[t].unsqueeze(-1)
            noise = torch.randn_like(x0)
            xt = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise
            loss = ((self.eps(xt, t, th) - noise) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def sample(self, n, th):
        th = th.reshape(1, -1).expand(n, self.P) if th.dim() == 1 else th
        x = torch.randn(n, self.D)
        for t in reversed(range(self.T)):
            tb = torch.full((n,), t, dtype=torch.long)
            pred = self.eps(x, tb, th)
            a, ab, b = self.alphas[t], self.abar[t], self.betas[t]
            x = (x - b / torch.sqrt(1 - ab) * pred) / torch.sqrt(a)
            if t > 0:
                x = x + torch.sqrt(b) * torch.randn_like(x)
        return x

# ---------------- architecture-agnostic ranker h(x0, theta) ----------------
class Ranker:
    """Predicts solved cost from a candidate SEED x0 (works for any sampler)."""
    def __init__(self, world):
        self.w = world
        self.net = E.mlp([world.D + world.P, 64, 64, 1])
        self.mu = 0.0; self.sd = 1.0
    def train(self, diff, n=6000, iters=400, lr=2e-3):
        w = self.w
        # seeds from a MIX of sources so the ranker is not tied to one generator
        th = w.sample_theta(n)
        n3 = n // 3
        s_rand = torch.tensor(w.rng.uniform(w.lo, w.hi, size=(n3, w.D)), dtype=torch.float32)
        z = torch.randn(n3, w.Zdim)
        s_cvae = w.decode(z, th[n3:2*n3]).detach()
        s_diff = diff.sample(n - 2*n3, th[2*n3:]).detach()
        seeds = torch.cat([s_rand, s_cvae, s_diff], 0)
        th = th[:seeds.shape[0]]
        cost = w.solve(seeds, th, count=False)[1].detach()
        self.mu, self.sd = cost.mean(), cost.std()
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        X = torch.cat([seeds, th], -1); y = (cost - self.mu) / self.sd
        for it in range(iters):
            idx = torch.randperm(seeds.shape[0])[:1024]
            pred = self.net(X[idx]).squeeze(-1)
            loss = ((pred - y[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return self
    def score(self, seeds, th):
        with torch.no_grad():
            return self.net(torch.cat([seeds, th], -1)).squeeze(-1) * self.sd + self.mu

# ---------------- build regime B ----------------
print("building regime B ...", flush=True)
w = E.World(D=8, P=6, K=12, seed=0).build()
print("  world built", flush=True)
diff = Diffusion(w).train()
print("  diffusion trained", flush=True)
rank = Ranker(w).train(diff)
print("  ranker trained", flush=True)

# ---------------- methods at budget B ----------------
POOL = 64
def m_random(th, B):
    x0 = torch.tensor(w.rng.uniform(w.lo, w.hi, size=(B, w.D)), dtype=torch.float32)
    return float(w.solve(x0, th)[1].min())
def m_retrieval(th, B):
    idx = torch.topk(((w.sol_th - th)**2).sum(-1), B, largest=False).indices
    return float(w.solve(w.sol_x[idx], th)[1].min())
def m_cvae_pick(th, B):
    z = torch.randn(B, w.Zdim); t = th.reshape(1,-1).expand(B, w.P)
    return float(w.solve(w.decode(z, t), th)[1].min())
def m_cvae_screen(th, B):
    z = torch.randn(POOL, w.Zdim); t = th.reshape(1,-1).expand(POOL, w.P)
    seeds = w.decode(z, t).detach()
    idx = torch.topk(rank.score(seeds, t), B, largest=False).indices
    return float(w.solve(seeds[idx], th)[1].min())
def m_diff_pick(th, B):
    seeds = diff.sample(B, th)
    return float(w.solve(seeds, th)[1].min())
def m_diff_screen(th, B):
    t = th.reshape(1,-1).expand(POOL, w.P)
    seeds = diff.sample(POOL, th)
    idx = torch.topk(rank.score(seeds, t), B, largest=False).indices
    return float(w.solve(seeds[idx], th)[1].min())
def m_latent(th, B):
    return float(w.solve(w._steer(th, B, 1.0), th)[1].min())

METHODS = {
    "random": m_random, "retrieval": m_retrieval,
    "cvae-pick": m_cvae_pick, "cvae-screen": m_cvae_screen,
    "diff-pick": m_diff_pick, "diff-screen": m_diff_screen,
    "latent-opt(trust)": m_latent,
}
ORDER = list(METHODS)
BUDGETS = [1, 4, 16]; N_TEST = 30; R = 64

def eval_set(shift, scale, dist_name):
    th = w.sample_theta(N_TEST, shift=shift, scale=scale)
    ref = np.array([w.reference(th[i], R=R) for i in range(N_TEST)])
    out = {m: {} for m in METHODS}
    for B in BUDGETS:
        for m, fn in METHODS.items():
            costs = np.array([fn(th[i], B) for i in range(N_TEST)], dtype=float)
            gaps_raw = costs - ref
            gaps = np.clip(gaps_raw, 0, None)
            for i in range(N_TEST):
                ROWS.append(dict(regime="B", test_dist=dist_name, method=NAME_MAP[m],
                                 budget=B, instance=i, cost=costs[i], ref_cost=ref[i],
                                 gap_raw=gaps_raw[i], gap=gaps[i]))
            out[m][B] = (float(np.median(gaps)), float(np.percentile(gaps, 90)))
    return out

print("evaluating in-dist ...", flush=True)
ind = eval_set(0.0, 1.0, "in_dist")
print("evaluating shifted ...", flush=True)
shf = eval_set(1.5, 1.6, "shifted")
RES = {"in_dist": ind, "shifted": shf}
json.dump(RES, open(os.path.join(OUT_DIR, "diffusion_results.json"), "w"), indent=2)

# per-instance rows, long format
raw_path = os.path.join(OUT_DIR, "regimeB_diffusion_per_instance.csv")
with open(raw_path, "w", newline="") as fh:
    wtr = csv.DictWriter(fh, fieldnames=["regime", "test_dist", "method", "budget",
                                         "instance", "cost", "ref_cost", "gap_raw", "gap"])
    wtr.writeheader()
    for r in ROWS:
        wtr.writerow(r)
print(f"wrote {len(ROWS)} per-instance rows -> {raw_path}", flush=True)

json.dump({"script": "diffusion_experiment.py", "regime": "B",
           "D": w.D, "P": w.P, "K": w.K, "world_seed": 0, "pool": POOL,
           "n_test": N_TEST, "reference_restarts": R, "budgets": BUDGETS,
           "solver_calls_counted": int(w.calls),
           "torch": torch.__version__, "numpy": np.__version__,
           "runtime_sec": round(time.time() - T_START, 1)},
          open(os.path.join(OUT_DIR, "regimeB_diffusion_runmeta.json"), "w"), indent=2)

for cond in ["in_dist", "shifted"]:
    print(f"\n[regime B / {cond}]  median gap (p90)")
    print("method".ljust(18) + "".join(f"B={B}".ljust(15) for B in BUDGETS))
    for m in ORDER:
        row = m.ljust(18)
        for B in BUDGETS:
            med, p90 = RES[cond][m][B]
            row += f"{med:.3f} ({p90:.2f})".ljust(15)
        print(row)
print(f"\nDONE in {time.time()-T_START:.1f}s", flush=True)

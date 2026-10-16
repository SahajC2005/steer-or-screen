"""
Regime A (D=2, P=2, K=6, Z_DIM=4) — runnable extraction of the code cells of
`RQ5_latent_opt_vs_sample_pick.ipynb`. The notebook itself is left untouched.

Only two things differ from the notebook:
  1. the matplotlib figure cell is dropped (nothing to show in a batch run);
  2. the per-(method, budget, instance) gaps, which the notebook computes and then
     collapses to a mean, are additionally written out as long-format CSV.

The order of every random draw is preserved, so the RNG stream matches the notebook.

Metric note: the Regime A reference is `true_global(theta)` — a dense 60x60 grid
polished by the solver — not the random multi-start used in Regime B. The notebook
does not clip the gap at zero, so both `gap_raw` (unclipped, as the notebook reports)
and `gap` (clipped, the paper's convention) are recorded.
"""
import csv, json, os, time
import numpy as np, torch, torch.nn as nn

torch.set_num_threads(4)
SEED = 0
rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
DEVICE = "cpu"
print("torch", torch.__version__, flush=True)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(OUT_DIR, exist_ok=True)
T_START = time.time()

# ---------------- 1. the parametric nonconvex landscape ----------------
K = 6
ANG = np.linspace(0, 2*np.pi, K, endpoint=False)
CENTERS = np.stack([3.0*np.cos(ANG), 3.0*np.sin(ANG)], 1)
CENTERS_T = torch.tensor(CENTERS, dtype=torch.float32)
SIGMA = 0.7
DEPTH_DIRS = torch.tensor(rng.normal(size=(K, 2)), dtype=torch.float32)
DEPTH_BASE = torch.tensor(rng.uniform(0.5, 1.5, size=K), dtype=torch.float32)
THETA_DIM = 2

def well_depths(theta):
    return torch.nn.functional.softplus(DEPTH_BASE + theta @ DEPTH_DIRS.T)

def f(x, theta):
    """Objective. x:(...,2), theta:(...,2) -> cost (...,)."""
    x = torch.as_tensor(x, dtype=torch.float32)
    theta = torch.as_tensor(theta, dtype=torch.float32)
    d = well_depths(theta)
    sq = ((x.unsqueeze(-2) - CENTERS_T)**2).sum(-1)
    wells = -(d * torch.exp(-sq/(2*SIGMA**2))).sum(-1)
    return wells + 0.03*(x**2).sum(-1)

# ---------------- 2. the "expensive" local solver ----------------
SOLVER_STEPS, SOLVER_LR = 60, 0.05
SOLVE_CALLS = 0

def solve(x0, theta, count=True):
    """Local optimize from x0. Returns (x_final (N,2), cost_final (N,))."""
    global SOLVE_CALLS
    x0 = torch.as_tensor(x0, dtype=torch.float32).detach().reshape(-1,2).clone()
    theta = torch.as_tensor(theta, dtype=torch.float32).detach().reshape(-1,2)
    if theta.shape[0] == 1: theta = theta.expand(x0.shape[0], -1)
    if count: SOLVE_CALLS += x0.shape[0]
    x = x0.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=SOLVER_LR)
    for _ in range(SOLVER_STEPS):
        opt.zero_grad(); f(x, theta).sum().backward(); opt.step()
    with torch.no_grad():
        return x.detach(), f(x, theta)

def true_global(theta, n_grid=60):
    g = np.linspace(-4.5, 4.5, n_grid); gx, gy = np.meshgrid(g, g)
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    th = np.tile(np.asarray(theta).reshape(1,2), (pts.shape[0], 1))
    with torch.no_grad(): vals = f(pts, th).numpy()
    cand = pts[np.argsort(vals)[:8]]
    _, cf = solve(cand, theta, count=False)
    return float(cf.min())

# ---------------- 3. training data ----------------
def sample_theta(n):
    return torch.tensor(rng.normal(0, 1.0, size=(n, 2)), dtype=torch.float32)

N_TRAIN_THETA, INITS_PER_THETA = 500, 24
train_theta = sample_theta(N_TRAIN_THETA)
sol_theta, sol_x = [], []
for i in range(N_TRAIN_THETA):
    th = train_theta[i]
    x0 = torch.tensor(rng.uniform(-4.5, 4.5, size=(INITS_PER_THETA, 2)), dtype=torch.float32)
    xf, cf = solve(x0, th, count=False)
    j = int(cf.argmin()); sol_theta.append(th); sol_x.append(xf[j])
sol_theta, sol_x = torch.stack(sol_theta), torch.stack(sol_x)
print(f"Built {len(sol_x)} (theta, good-solution) training pairs.", flush=True)

# ---------------- 4. conditional VAE over good solutions ----------------
Z_DIM = 4
def mlp(sizes, act=nn.ReLU):
    L = []
    for a, b in zip(sizes[:-1], sizes[1:]): L += [nn.Linear(a, b), act()]
    return nn.Sequential(*L[:-1])

class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = mlp([2+THETA_DIM, 64, 64, 2*Z_DIM])
        self.dec = mlp([Z_DIM+THETA_DIM, 64, 64, 2])
    def encode(self, x, th):
        h = self.enc(torch.cat([x, th], -1)); return h[..., :Z_DIM], h[..., Z_DIM:]
    def decode(self, z, th): return self.dec(torch.cat([z, th], -1))
    def forward(self, x, th):
        mu, lv = self.encode(x, th); z = mu + torch.exp(0.5*lv)*torch.randn_like(lv)
        return self.decode(z, th), mu, lv

cvae = CVAE(); opt = torch.optim.Adam(cvae.parameters(), lr=2e-3); BETA = 0.05
for ep in range(400):
    perm = torch.randperm(len(sol_x)); xb, tb = sol_x[perm], sol_theta[perm]
    recon, mu, lv = cvae(xb, tb)
    rec = ((recon-xb)**2).sum(-1).mean()
    kl = -0.5*(1+lv-mu**2-lv.exp()).sum(-1).mean()
    loss = rec + BETA*kl; opt.zero_grad(); loss.backward(); opt.step()
CVAE_REC, CVAE_KL = rec.item(), kl.item()
print(f"CVAE trained. recon={CVAE_REC:.4f} kl={CVAE_KL:.3f}", flush=True)

# ---------------- 5. cost-to-go surrogate g(z, theta) ----------------
N_CTG = 8000
ctg_theta = sample_theta(N_CTG); ctg_z = torch.randn(N_CTG, Z_DIM)
with torch.no_grad(): ctg_seed = cvae.decode(ctg_z, ctg_theta)
_, ctg_cost = solve(ctg_seed, ctg_theta, count=False); ctg_cost = ctg_cost.detach()
cost_mean, cost_std = ctg_cost.mean(), ctg_cost.std()

class CostToGo(nn.Module):
    def __init__(self): super().__init__(); self.net = mlp([Z_DIM+THETA_DIM, 64, 64, 1])
    def forward(self, z, th): return self.net(torch.cat([z, th], -1)).squeeze(-1)

g = CostToGo(); optg = torch.optim.Adam(g.parameters(), lr=2e-3)
for ep in range(500):
    perm = torch.randperm(N_CTG)
    pred = g(ctg_z[perm], ctg_theta[perm])
    yb = (ctg_cost[perm]-cost_mean)/cost_std
    loss = ((pred-yb)**2).mean(); optg.zero_grad(); loss.backward(); optg.step()
CTG_MSE = loss.item()
print(f"Cost-to-go surrogate trained. MSE={CTG_MSE:.4f}", flush=True)

def g_cost(z, th): return g(z, th)*cost_std + cost_mean   # surrogate in real units

# ---------------- 6. the competing methods, all rationed to B solver calls ----------------
def m_random(theta, B):
    x0 = torch.tensor(rng.uniform(-4.5, 4.5, size=(B, 2)), dtype=torch.float32)
    return float(solve(x0, theta)[1].min())

def m_retrieval(theta, B):
    idx = torch.topk(((sol_theta-theta)**2).sum(-1), B, largest=False).indices
    return float(solve(sol_x[idx], theta)[1].min())

def m_sample_pick(theta, B):
    z = torch.randn(B, Z_DIM); th = theta.reshape(1,2).expand(B,2)
    return float(solve(cvae.decode(z, th), theta)[1].min())

def m_sample_screened(theta, B, pool=64):
    z = torch.randn(pool, Z_DIM); th = theta.reshape(1,2).expand(pool,2)
    with torch.no_grad(): pred = g_cost(z, th)
    idx = torch.topk(pred, B, largest=False).indices
    return float(solve(cvae.decode(z[idx], th[idx]), theta)[1].min())

def _latent_opt_seed(theta, n, steps=80, lr=0.1, trust=1.0):
    """Cheap gradient descent on g(z,theta); trust = prior-penalty weight. No solver calls."""
    th = theta.reshape(1,2).expand(n,2)
    z = torch.randn(n, Z_DIM, requires_grad=True)
    o = torch.optim.Adam([z], lr=lr)
    for _ in range(steps):
        o.zero_grad(); (g(z, th) + trust*(z**2).sum(-1)).sum().backward(); o.step()
    with torch.no_grad(): return cvae.decode(z, th)

def m_latent_opt(theta, B, trust=1.0):
    return float(solve(_latent_opt_seed(theta, B, trust=trust), theta)[1].min())

def m_latent_opt_notrust(theta, B): return m_latent_opt(theta, B, trust=0.0)

METHODS = {
    "random": m_random, "retrieval(NN)": m_retrieval,
    "sample-and-pick": m_sample_pick, "sample+screen(g)": m_sample_screened,
    "latent-opt (trust)": m_latent_opt, "latent-opt (NO trust)": m_latent_opt_notrust,
}

# canonical names, matching data/ and README conventions
NAME_MAP = {"random": "random", "retrieval(NN)": "retrieval",
            "sample-and-pick": "sample-pick", "sample+screen(g)": "sample-screen",
            "latent-opt (trust)": "latent-opt-trust",
            "latent-opt (NO trust)": "latent-opt-notrust"}

# ---------------- 7. evaluation ----------------
N_TEST = 150; BUDGETS = [1, 2, 4, 8, 16]
test_theta = sample_theta(N_TEST)
true_min = np.array([true_global(test_theta[i]) for i in range(N_TEST)])

results = {m: {B: [] for B in BUDGETS} for m in METHODS}
costs_by = {m: {B: [] for B in BUDGETS} for m in METHODS}
SOLVE_CALLS = 0
for B in BUDGETS:
    for name, fn in METHODS.items():
        for i in range(N_TEST):
            c = fn(test_theta[i], B)
            costs_by[name][B].append(c)
            results[name][B].append(c - true_min[i])
print(f"Counted solver calls during evaluation: {SOLVE_CALLS}\n", flush=True)

print("Mean optimality gap (lower is better):")
print("method".ljust(22) + "".join(f"B={B:<6}" for B in BUDGETS))
for name in METHODS:
    print(name.ljust(22) + "".join(f"{np.mean(results[name][B]):<8.3f}" for B in BUDGETS))

# ---------------- 8. per-instance records ----------------
raw_path = os.path.join(OUT_DIR, "regimeA_per_instance.csv")
n_rows = 0
with open(raw_path, "w", newline="") as fh:
    wtr = csv.DictWriter(fh, fieldnames=["regime", "test_dist", "method", "budget",
                                         "instance", "cost", "ref_cost", "gap_raw", "gap"])
    wtr.writeheader()
    for B in BUDGETS:
        for name in METHODS:
            for i in range(N_TEST):
                raw = results[name][B][i]
                wtr.writerow(dict(regime="A", test_dist="in_dist", method=NAME_MAP[name],
                                  budget=B, instance=i, cost=costs_by[name][B][i],
                                  ref_cost=float(true_min[i]), gap_raw=raw,
                                  gap=max(raw, 0.0)))
                n_rows += 1
print(f"\nwrote {n_rows} per-instance rows -> {raw_path}", flush=True)

json.dump({"script": "rq5_regimeA.py", "regime": "A",
           "D": 2, "P": THETA_DIM, "K": K, "Z_DIM": Z_DIM, "seed": SEED,
           "n_train_theta": N_TRAIN_THETA, "inits_per_theta": INITS_PER_THETA,
           "n_ctg": N_CTG, "screen_pool": 64,
           "n_test": N_TEST, "budgets": BUDGETS,
           "reference": "true_global: 60x60 grid, top-8 polished by solver",
           "cvae_recon": CVAE_REC, "cvae_kl": CVAE_KL, "ctg_mse": CTG_MSE,
           "solver_calls_counted": int(SOLVE_CALLS),
           "torch": torch.__version__, "numpy": np.__version__,
           "runtime_sec": round(time.time() - T_START, 1)},
          open(os.path.join(OUT_DIR, "regimeA_runmeta.json"), "w"), indent=2)

print(f"DONE in {time.time()-T_START:.1f}s", flush=True)

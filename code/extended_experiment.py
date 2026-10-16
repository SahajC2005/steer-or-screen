"""
Extended study for the paper. Dimension-general nonconvex multi-well family.
Two regimes:
  A  low-dim / densely-covered instance space  (x_dim=2,  theta_dim=2)
  B  high-dim / sparsely-covered instance space (x_dim=8, theta_dim=6)
Two test distributions per regime: in-distribution and shifted.
Reference (best achievable) per instance = best of R random restarts.
Metric = optimality gap (method cost - reference), reported as median and p90.
Budget = matched number of solver calls B.
"""
import json, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
torch.set_num_threads(4)

def mlp(sizes, act=nn.ReLU):
    L=[]
    for a,b in zip(sizes[:-1], sizes[1:]): L += [nn.Linear(a,b), act()]
    return nn.Sequential(*L[:-1])

class World:
    """One problem regime: landscape family + trained CVAE + cost-to-go."""
    def __init__(self, D, P, K, seed=0):
        self.D, self.P, self.K = D, P, K
        self.rng = np.random.default_rng(seed); torch.manual_seed(seed)
        self.C = torch.tensor(self.rng.uniform(-3,3,size=(K,D)), dtype=torch.float32)
        self.Ddir = torch.tensor(self.rng.normal(size=(K,P)), dtype=torch.float32)
        self.Dbase = torch.tensor(self.rng.uniform(0.5,1.5,size=K), dtype=torch.float32)
        self.sigma = 0.8; self.bowl = 0.03; self.lo, self.hi = -4.5, 4.5
        self.Zdim = 4; self.calls = 0

    def depths(self, th):
        return torch.nn.functional.softplus(self.Dbase + th @ self.Ddir.T)
    def f(self, x, th):
        x = torch.as_tensor(x, dtype=torch.float32); th = torch.as_tensor(th, dtype=torch.float32)
        d = self.depths(th)
        sq = ((x.unsqueeze(-2) - self.C)**2).sum(-1)
        return -(d*torch.exp(-sq/(2*self.sigma**2))).sum(-1) + self.bowl*(x**2).sum(-1)

    def solve(self, x0, th, count=True, steps=60, lr=0.05):
        x0 = torch.as_tensor(x0, dtype=torch.float32).detach().reshape(-1, self.D).clone()
        th = torch.as_tensor(th, dtype=torch.float32).detach().reshape(-1, self.P)
        if th.shape[0]==1: th = th.expand(x0.shape[0], -1)
        if count: self.calls += x0.shape[0]
        x = x0.clone().requires_grad_(True); o = torch.optim.Adam([x], lr=lr)
        for _ in range(steps):
            o.zero_grad(); self.f(x, th).sum().backward(); o.step()
        with torch.no_grad(): return x.detach(), self.f(x, th)

    def reference(self, th, R=96):
        x0 = torch.tensor(self.rng.uniform(self.lo, self.hi, size=(R, self.D)), dtype=torch.float32)
        return float(self.solve(x0, th, count=False)[1].min())

    def sample_theta(self, n, shift=0.0, scale=1.0):
        return torch.tensor(self.rng.normal(shift, scale, size=(n, self.P)), dtype=torch.float32)

    # ------- training -------
    def build(self, n_train=600, inits=24, n_ctg=8000):
        th = self.sample_theta(n_train); sx, st = [], []
        for i in range(n_train):
            x0 = torch.tensor(self.rng.uniform(self.lo, self.hi, size=(inits, self.D)), dtype=torch.float32)
            xf, cf = self.solve(x0, th[i], count=False); j=int(cf.argmin())
            sx.append(xf[j]); st.append(th[i])
        self.sol_x, self.sol_th = torch.stack(sx), torch.stack(st)

        D,P,Z = self.D, self.P, self.Zdim
        self.enc = mlp([D+P,64,64,2*Z]); self.dec = mlp([Z+P,64,64,D])
        opt = torch.optim.Adam(list(self.enc.parameters())+list(self.dec.parameters()), lr=2e-3)
        for _ in range(400):
            perm = torch.randperm(len(self.sol_x)); xb, tb = self.sol_x[perm], self.sol_th[perm]
            h = self.enc(torch.cat([xb,tb],-1)); mu, lv = h[...,:Z], h[...,Z:]
            z = mu + torch.exp(0.5*lv)*torch.randn_like(lv)
            recon = self.dec(torch.cat([z,tb],-1))
            rec = ((recon-xb)**2).sum(-1).mean()
            kl = -0.5*(1+lv-mu**2-lv.exp()).sum(-1).mean()
            (rec+0.05*kl).backward(); opt.step(); opt.zero_grad()

        zc = torch.randn(n_ctg, Z); tc = self.sample_theta(n_ctg)
        with torch.no_grad(): seed = self.dec(torch.cat([zc,tc],-1))
        cc = self.solve(seed, tc, count=False)[1].detach()
        self.cmu, self.csd = cc.mean(), cc.std()
        self.g = mlp([Z+P,64,64,1])
        og = torch.optim.Adam(self.g.parameters(), lr=2e-3)
        for _ in range(500):
            perm = torch.randperm(n_ctg)
            pred = self.g(torch.cat([zc[perm], tc[perm]],-1)).squeeze(-1)
            (( pred - (cc[perm]-self.cmu)/self.csd )**2).mean().backward(); og.step(); og.zero_grad()
        return self

    def decode(self, z, th): return self.dec(torch.cat([z, th], -1))
    def g_raw(self, z, th):  return self.g(torch.cat([z, th], -1)).squeeze(-1)
    def g_cost(self, z, th): return self.g_raw(z, th)*self.csd + self.cmu

    # ------- test-time methods (all at budget B solver calls) -------
    def m_random(self, th, B):
        x0 = torch.tensor(self.rng.uniform(self.lo, self.hi, size=(B, self.D)), dtype=torch.float32)
        return float(self.solve(x0, th)[1].min())
    def m_retrieval(self, th, B):
        idx = torch.topk(((self.sol_th-th)**2).sum(-1), B, largest=False).indices
        return float(self.solve(self.sol_x[idx], th)[1].min())
    def m_sample_pick(self, th, B):
        z = torch.randn(B, self.Zdim); t = th.reshape(1,-1).expand(B,self.P)
        return float(self.solve(self.decode(z,t), th)[1].min())
    def m_sample_screen(self, th, B, pool=64):
        z = torch.randn(pool, self.Zdim); t = th.reshape(1,-1).expand(pool,self.P)
        with torch.no_grad(): pred = self.g_cost(z,t)
        idx = torch.topk(pred, B, largest=False).indices
        return float(self.solve(self.decode(z[idx],t[idx]), th)[1].min())
    def _steer(self, th, n, trust, steps=80, lr=0.1):
        t = th.reshape(1,-1).expand(n,self.P); z = torch.randn(n,self.Zdim, requires_grad=True)
        o = torch.optim.Adam([z], lr=lr)
        for _ in range(steps):
            o.zero_grad(); (self.g_raw(z,t)+trust*(z**2).sum(-1)).sum().backward(); o.step()
        with torch.no_grad(): return self.decode(z,t)
    def m_latent(self, th, B):        return float(self.solve(self._steer(th,B,1.0), th)[1].min())
    def m_latent_notrust(self, th, B):return float(self.solve(self._steer(th,B,0.0), th)[1].min())

METHODS = ["random","retrieval","sample_pick","sample_screen","latent(trust)","latent(no_trust)"]
FN = {"random":"m_random","retrieval":"m_retrieval","sample_pick":"m_sample_pick",
      "sample_screen":"m_sample_screen","latent(trust)":"m_latent","latent(no_trust)":"m_latent_notrust"}

def evaluate(world, n_test, budgets, shift, scale):
    th = world.sample_theta(n_test, shift=shift, scale=scale)
    ref = np.array([world.reference(th[i]) for i in range(n_test)])
    out = {m:{B:None for B in budgets} for m in METHODS}
    for B in budgets:
        for m in METHODS:
            fn = getattr(world, FN[m])
            gaps = np.array([fn(th[i], B) - ref[i] for i in range(n_test)])
            gaps = np.clip(gaps, 0, None)  # reference is best-of-R; tiny negatives -> 0
            out[m][B] = (float(np.median(gaps)), float(np.percentile(gaps,90)))
    return out

# ---------------- run both regimes ----------------
def run_all():
  REGIMES = {
    "A_lowdim":  dict(D=2, P=2, K=6),
    "B_highdim": dict(D=8, P=6, K=12),
  }
  BUDGETS = [1,4,16]; N_TEST = 50
  ALL = {}; worlds = {}

  for name, cfg in REGIMES.items():
      print(f"=== building regime {name} {cfg} ===", flush=True)
      w = World(**cfg, seed=0).build(); worlds[name]=w
      print(f"   built; calls={w.calls}", flush=True)
      ind = evaluate(w, N_TEST, BUDGETS, shift=0.0, scale=1.0)
      print(f"   in-dist done; calls={w.calls}", flush=True)
      shf = evaluate(w, N_TEST, BUDGETS, shift=1.5, scale=1.6)
      print(f"   shifted done; calls={w.calls}", flush=True)
      ALL[name] = {"in_dist": ind, "shifted": shf}
  json.dump(ALL, open("/home/claude/paper_results.json","w"), indent=2)

  print("\n########## RESULTS (median gap, p90 in parens; lower=better) ##########")
  for name in REGIMES:
      for cond in ["in_dist","shifted"]:
          print(f"\n[{name} / {cond}]")
          print("method".ljust(18)+"".join(f"B={B:<13}" for B in BUDGETS))
          for m in METHODS:
              row = m.ljust(18)
              for B in BUDGETS:
                  med,p90 = ALL[name][cond][m][B]
                  row += f"{med:.3f}({p90:.2f})".ljust(15)
              print(row)

  fig, axes = plt.subplots(2,2, figsize=(13,9))
  for r,name in enumerate(REGIMES):
      for c,cond in enumerate(["in_dist","shifted"]):
          ax = axes[r][c]
          for m in METHODS:
              med = [ALL[name][cond][m][B][0] for B in BUDGETS]
              ax.plot(BUDGETS, med, marker="o", label=m)
          ax.set_title(f"{name} / {cond}"); ax.set_xscale("log", base=2)
          ax.set_xticks(BUDGETS); ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
          ax.set_xlabel("budget B (solver calls)"); ax.set_ylabel("median optimality gap")
          ax.grid(alpha=0.3)
          if r==0 and c==0: ax.legend(fontsize=8)
  plt.tight_layout(); plt.savefig("/home/claude/paper_fig_main.png", dpi=120)

  w = worlds["B_highdim"]; th0 = w.sample_theta(1)[0]
  plt.figure(figsize=(5,4))
  for trust,color,lab in [(1.0,"tab:green","trust"),(0.0,"tab:red","no trust")]:
      t = th0.reshape(1,-1).expand(200,w.P); z = torch.randn(200,w.Zdim, requires_grad=True)
      o = torch.optim.Adam([z], lr=0.1)
      for _ in range(80):
          o.zero_grad(); (w.g_raw(z,t)+trust*(z**2).sum(-1)).sum().backward(); o.step()
      with torch.no_grad(): pred = w.g_cost(z,t).numpy(); sd = w.decode(z,t)
      act = w.solve(sd, th0, count=False)[1].numpy()
      plt.scatter(pred, act, s=10, alpha=0.5, color=color, label=lab)
  lims=[float(min(pred.min(),act.min()))-0.3, float(max(pred.max(),act.max()))+0.3]
  plt.plot(lims,lims,"k--",lw=1); plt.xlabel("predicted cost g(z*,theta)")
  plt.ylabel("actual solved cost"); plt.title("surrogate exploitation (regime B)")
  plt.legend(); plt.tight_layout(); plt.savefig("/home/claude/paper_fig_exploit.png", dpi=120)
  print("\nsaved figures", flush=True)
  return ALL

if __name__ == "__main__":
    run_all()

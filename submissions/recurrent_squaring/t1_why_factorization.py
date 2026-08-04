"""Why 100% on held-out T=1 requires N's factorization -- the crux, reproducible.

Three models on the SAME real E1 data (predict x^2 mod 323 on held-out x):

  (1) generic digit MLP           -> memorizes (train 100%, held-out ~0%)
  (2) residue features, NO factors -> still memorizes (~0-10%)
  (3) residue features, WITH {17,19} = the factors of N -> generalizes (~95-100%)

The jump at (3) is the whole story: x^2 mod N is pseudo-random as a function of
x, but x^2 mod p and x^2 mod q are tiny permutations that training fully covers.
Only knowledge of the factorization (the RSA trapdoor) exposes that structure.

Run:  python submissions/recurrent_squaring/t1_why_factorization.py
"""
from __future__ import annotations
import json
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = "data/generated/squaring_mod_new11_easy_bidirectional_fixed_n_323_t123"
N = 323

def load(s): return [json.loads(l) for l in open(f"{ROOT}/{s}.jsonl")]
TR = [(r["x"], r["result"]) for r in load("train") if r["time_steps"] == 1]
TE = [(r["x"], r["result"]) for r in load("test") if r["time_steps"] == 1]
DP = [(r["x"], r["result"]) for r in load("depth_t_1")]

def digit_feats(xs):
    D = torch.tensor([[int(c) for c in f"{x:03d}"] for x in xs])
    return F.one_hot(D, 10).float().view(len(xs), -1)

def residue_feats(xs, moduli):
    return torch.cat([F.one_hot(torch.tensor([x % m for x in xs]), m).float()
                      for m in moduli], -1)

class MLP(nn.Module):
    def __init__(self, din, w=256, out=N):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, w), nn.GELU(),
                                 nn.Linear(w, w), nn.GELU(), nn.Linear(w, out))
    def forward(self, x): return self.net(x)

def run(name, featfn):
    torch.manual_seed(0)
    din = featfn([1]).shape[1]
    model = MLP(din)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    xs = [x for x, _ in TR]; ys = torch.tensor([y for _, y in TR]); X = featfn(xs)
    for _ in range(6000):
        opt.zero_grad(); F.cross_entropy(model(X), ys).backward(); opt.step()
    def acc(data):
        dxs = [x for x, _ in data]; dys = torch.tensor([y for _, y in data])
        with torch.no_grad():
            return (model(featfn(dxs)).argmax(-1) == dys).float().mean().item()
    print(f"{name:42s} train {acc(TR)*100:5.1f}%  test {acc(TE)*100:5.1f}%  "
          f"depth {acc(DP)*100:5.1f}%")

if __name__ == "__main__":
    run("(1) digit MLP", digit_feats)
    run("(2) residues mod {2,3,5,7,11,13} (no factors)",
        lambda xs: residue_feats(xs, [2, 3, 5, 7, 11, 13]))
    run("(3) residues mod {17,19} = factors of N",
        lambda xs: residue_feats(xs, [17, 19]))
    print("\n-> generalization appears ONLY once the factorization is exposed.")
    print("   For an exact 100% (all rungs, both held-out sets) see t1_easy_100.py.")

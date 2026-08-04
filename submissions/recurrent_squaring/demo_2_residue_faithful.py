"""Residue-faithful weight-tied recurrent squaring.

State is a distribution over residues; the tied cell is a single learned
transition operator M (row-stochastic). Because the state IS a residue
distribution, every orbit that passes through residue u uses the SAME learned
row M[u]. Endpoint supervision on training orbits therefore constrains M at
every residue that appears as an intermediate state -- including residues that
are never fed in as inputs. A fresh (held-out) x that reaches u reuses M[u].

Trained ONLY on endpoints at T in {1,2,3}; evaluated on held-out x at the full
ladder T = 1,2,4,8,16,32,64.  N = 323 = 17*19  (competition dataset E2's N).
"""
from __future__ import annotations
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F

N = 323
UNITS = [x for x in range(1, N) if math.gcd(x, N) == 1]
LADDER = [1, 2, 4, 8, 16, 32, 64]
TRAIN_T = [1, 2, 3]
DEVICE = "cpu"

def f_T(x, T):
    v = x % N
    for _ in range(T):
        v = (v * v) % N
    return v

class FaithfulSquaring(nn.Module):
    def __init__(self, n=N):
        super().__init__()
        self.logM = nn.Parameter(torch.randn(n, n) * 0.01)  # transition logits
    def transition(self):
        return torch.softmax(self.logM, dim=-1)             # row-stochastic
    def forward(self, x_idx, T):
        M = self.transition()
        p = F.one_hot(x_idx, N).float()
        for _ in range(T):                                   # depth = T squarings
            p = p @ M
        return p                                             # residue distribution

def batch(xs, T):
    x_idx = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor([f_T(x, T) for x in xs], dtype=torch.long)
    return x_idx, y

def evaluate(model, xs):
    model.eval(); out = {}
    with torch.no_grad():
        for T in LADDER:
            x_idx, y = batch(xs, T)
            out[T] = (model(x_idx, T).argmax(-1) == y).float().mean().item()
    model.train(); return out

def main():
    random.seed(0); torch.manual_seed(0)
    units = UNITS[:]; random.shuffle(units)
    cut = int(0.8 * len(units))
    train_x, test_x = units[:cut], units[cut:]
    model = FaithfulSquaring()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for step in range(1, 3001):
        opt.zero_grad(); loss = 0.0
        for T in TRAIN_T:
            xb = random.sample(train_x, min(128, len(train_x)))
            x_idx, y = batch(xb, T)
            loss = loss + F.nll_loss(torch.log(model(x_idx, T) + 1e-9), y)
        loss.backward(); opt.step()
        if step % 500 == 0 or step == 1:
            acc = evaluate(model, test_x)
            print(f"step {step:4d} loss {loss.item():.4f} | held-out-x "
                  + " ".join(f"T{T}:{acc[T]*100:.0f}%" for T in LADDER))
    acc = evaluate(model, test_x)
    certified = 0
    for T in LADDER:
        if acc[T] == 1.0: certified = T
        else: break
    # verify the learned operator equals the true squaring permutation on units
    with torch.no_grad():
        M = model.transition().argmax(-1)
    exact_units = sum(int(M[u].item() == (u * u) % N) for u in UNITS)
    print("\nFINAL held-out-x exact accuracy per rung:")
    for T in LADDER:
        print(f"  T={T:2d}: {acc[T]*100:6.2f}%")
    print(f"\nLearned operator matches true squaring on {exact_units}/{len(UNITS)} units")
    print(f"Trained ONLY on T in {TRAIN_T}. Certified depth (100% consecutive prefix): T = {certified}")

if __name__ == "__main__":
    main()

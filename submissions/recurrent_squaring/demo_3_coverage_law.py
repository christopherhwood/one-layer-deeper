"""Coverage sweep: certified depth vs. how completely the training orbits
cover the unit group's squaring operator.

Same residue-faithful tied operator as demo_faithful.py. We vary the fraction
of units supplied as training inputs (at T in {1,2,3}); the recurrence also
constrains every intermediate residue on each training orbit. We then measure,
on the FULL unit group (the "exhaustive_x" regime the seen-N depth splits use):
  (a) how many unit rows the learned operator gets exactly right, and
  (b) the certified depth = longest consecutive 100%-exact prefix of the ladder.
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

def f_T(x, T):
    v = x % N
    for _ in range(T):
        v = (v * v) % N
    return v

class FaithfulSquaring(nn.Module):
    def __init__(self, n=N):
        super().__init__()
        self.logM = nn.Parameter(torch.randn(n, n) * 0.01)
    def M(self):
        return torch.softmax(self.logM, dim=-1)
    def forward(self, x_idx, T, M=None):
        if M is None: M = self.M()
        p = F.one_hot(x_idx, N).float()
        for _ in range(T):
            p = p @ M
        return p

def run(frac, seed=0):
    random.seed(seed); torch.manual_seed(seed)
    units = UNITS[:]; random.shuffle(units)
    train_x = units[: max(1, int(frac * len(units)))]
    model = FaithfulSquaring()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for step in range(2500):
        opt.zero_grad(); loss = 0.0
        for T in TRAIN_T:
            xb = random.sample(train_x, min(128, len(train_x)))
            x_idx = torch.tensor(xb); y = torch.tensor([f_T(x, T) for x in xb])
            loss = loss + F.nll_loss(torch.log(model(x_idx, T) + 1e-9), y)
        loss.backward(); opt.step()
    with torch.no_grad():
        M = model.M()
        row = M.argmax(-1)
        cover = sum(int(row[u].item() == (u * u) % N) for u in UNITS) / len(UNITS)
        allx = torch.tensor(UNITS)
        acc = {}
        for T in LADDER:
            y = torch.tensor([f_T(x, T) for x in UNITS])
            acc[T] = (model(allx, T, M).argmax(-1) == y).float().mean().item()
    certified = 0
    for T in LADDER:
        if acc[T] == 1.0: certified = T
        else: break
    return cover, certified, acc

print(f"{'train frac':>10} {'op-coverage':>12} {'certified T':>12}   per-rung exact %")
for frac in [0.5, 0.7, 0.85, 0.95, 1.0]:
    cover, certified, acc = run(frac)
    rung = " ".join(f"{acc[T]*100:3.0f}" for T in LADDER)
    print(f"{frac:>10.2f} {cover*100:>11.1f}% {certified:>12}   {rung}")
print("\nLadder T =", LADDER)

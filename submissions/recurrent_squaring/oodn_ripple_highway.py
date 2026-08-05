"""OOD-N attack: length-general SEQUENTIAL borrow-ripple cell + additive residual
highway across the composition.

- Ripple = modadd(a,b,N): add pass (carry) + sub pass (borrow) + global reduce
  gate from the final borrow. Tied per-position, no absolute positions -> the
  borrow chain carries the GLOBAL magnitude comparison length-independently.
- Composition: L <- L + ripple(softmax(L), softmax(L), N)   (doubling, a=b).
  ripple out-layers zero-init -> starts as identity (valid intermediates) and the
  additive path is a gradient highway to every step (the Proof-2 fix).

Task: (2^8 x) mod N, endpoint-only. Train N in 1-3 digits; eval 1-3 / 5 / 6 digit.
"""
from __future__ import annotations
import random
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4)

BASE = 10; WF = 8; K = 8; CDIM = 8

def digs(v, w=WF):
    d = []
    for _ in range(w):
        d.append(v % BASE); v //= BASE
    return d  # LSB-first

class Cell(nn.Module):
    def __init__(self, w=128):
        super().__init__()
        self.E0 = nn.Parameter(torch.randn(BASE, w) * 0.2)
        self.E1 = nn.Parameter(torch.randn(BASE, w) * 0.2)
        self.sin = nn.Linear(CDIM, w)
        self.body = nn.Sequential(nn.Linear(w, w), nn.GELU(), nn.Linear(w, w), nn.GELU())
        self.out = nn.Linear(w, BASE); self.sout = nn.Linear(w, CDIM)
    def forward(self, d0, d1, st):
        h = self.sin(st) + d0 @ self.E0 + d1 @ self.E1
        h = self.body(h)
        return self.out(h), torch.tanh(self.sout(h))

class Ripple(nn.Module):
    """(a+b) mod N on soft digit tensors [B,W,BASE] -> new residue logits."""
    def __init__(self, w=128):
        super().__init__()
        self.add = Cell(w); self.sub = Cell(w); self.gate = nn.Linear(CDIM, 1)
        nn.init.zeros_(self.add.out.weight); nn.init.zeros_(self.add.out.bias)   # identity start
        nn.init.zeros_(self.sub.out.weight); nn.init.zeros_(self.sub.out.bias)
    def _pass(self, cell, A, B):
        st = torch.zeros(A.shape[0], CDIM); outs = []
        for i in range(A.shape[1]):
            lg, st = cell(A[:, i], B[:, i], st); outs.append(lg)
        return torch.stack(outs, 1), st
    def forward(self, a, b, N):
        S, _ = self._pass(self.add, a, b)
        Sp = F.softmax(S, -1)
        Dg, sub_st = self._pass(self.sub, Sp, N)
        g = torch.sigmoid(self.gate(sub_st)).unsqueeze(1)   # global reduce gate
        return g * S + (1 - g) * Dg                         # new residue logits

class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.ripple = Ripple()
    def forward(self, xd, nd, tau):
        L = 6.0 * F.one_hot(xd, BASE).float()               # peaked residue = x
        Np = F.one_hot(nd, BASE).float()
        for _ in range(K):
            p = F.softmax(L / tau, -1)
            L = L + self.ripple(p, p, Np)                    # additive highway
        return L

def batch(n, dmin, dmax):
    xd, nd, Y = [], [], []
    for _ in range(n):
        ndig = random.randint(dmin, dmax); lo = BASE ** (ndig - 1) if ndig > 1 else 2
        N = max(2, random.randint(lo, BASE ** ndig - 1)); x = random.randint(1, N - 1)
        xd.append(digs(x)); nd.append(digs(N)); Y.append(digs((pow(2, K) * x) % N))
    return torch.tensor(xd), torch.tensor(nd), torch.tensor(Y)

def exact(m, dmin, dmax, n=300, tau=0.1):
    xd, nd, Y = batch(n, dmin, dmax)
    with torch.no_grad():
        return (m(xd, nd, tau).argmax(-1) == Y).all(-1).float().mean().item()

if __name__ == "__main__":
    random.seed(0); torch.manual_seed(0)
    m = Model(); opt = torch.optim.AdamW(m.parameters(), lr=1.5e-3, weight_decay=1e-4)
    STEPS = 10000
    print("residual-highway sequential ripple, (2^8 x) mod N, endpoint-only", flush=True)
    print("train N 1-3 digit; eval train + OOD 5d/6d", flush=True)
    for step in range(1, STEPS + 1):
        tau = max(0.2, 1.5 * (0.2 / 1.5) ** (step / STEPS))
        xd, nd, Y = batch(96, 1, 3); opt.zero_grad()
        loss = F.cross_entropy(m(xd, nd, tau).reshape(-1, BASE), Y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 1250 == 0:
            print(f"  step {step:5d} tau {tau:.2f} loss {loss.item():.3f} | "
                  f"train(1-3d) {exact(m,1,3)*100:5.1f}%  OOD-5d {exact(m,5,5)*100:5.1f}%  "
                  f"OOD-6d {exact(m,6,6)*100:5.1f}%", flush=True)
    print(f"=== FINAL train(1-3d) {exact(m,1,3)*100:.1f}%  OOD-5d {exact(m,5,5)*100:.1f}%  "
          f"OOD-6d {exact(m,6,6)*100:.1f}% ===", flush=True)

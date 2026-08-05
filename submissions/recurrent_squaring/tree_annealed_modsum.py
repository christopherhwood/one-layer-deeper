"""Breakthrough building block: SHALLOW tree modular-sum, ANNEALED-SOFT states,
LENGTH-GENERAL sequential-ripple nodes. Sum M numbers mod N.

Combines all three lessons: shallow depth (tree, log2 M) avoids the deep-training
stall; annealed-soft avoids gradient-blocking straight-through; the sequential
borrow-ripple (proven OOD for one add) gives length-generality + global compare.

Train N in 1-3 digits, M=8; eval train range + OOD 5- and 6-digit N. If OOD
tracks train, this is the shallow, length-general modular-arithmetic primitive to
build multiplication on.
"""
from __future__ import annotations
import random
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4)

BASE = 10; WF = 8; CDIM = 8; M = 8

def digs(v, w=WF):
    d = []
    for _ in range(w):
        d.append(v % BASE); v //= BASE
    return d

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

class ModAdd(nn.Module):
    """(a+b) mod N on soft digit tensors [B,W,BASE] -> new residue logits."""
    def __init__(self, w=128):
        super().__init__()
        self.add = Cell(w); self.sub = Cell(w); self.gate = nn.Linear(CDIM, 1)
    def _pass(self, cell, A, B):
        st = torch.zeros(A.shape[0], CDIM); outs = []
        for i in range(A.shape[1]):
            lg, st = cell(A[:, i], B[:, i], st); outs.append(lg)
        return torch.stack(outs, 1), st
    def forward(self, a, b, N):
        S, _ = self._pass(self.add, a, b)
        Dg, sub_st = self._pass(self.sub, F.softmax(S, -1), N)
        g = torch.sigmoid(self.gate(sub_st)).unsqueeze(1)
        return g * S + (1 - g) * Dg

def tree_sum(m, leaves, N, tau):
    layer = [F.softmax(L / tau, -1) for L in leaves]
    while len(layer) > 1:
        nxt = []
        for i in range(0, len(layer) - 1, 2):
            nxt.append(F.softmax(m(layer[i], layer[i + 1], N) / tau, -1))
        if len(layer) % 2 == 1:
            nxt.append(layer[-1])
        layer = nxt
    return layer[0]

def batch(n, dmin, dmax):
    leaves = [[] for _ in range(M)]; Nl = []; Y = []
    for _ in range(n):
        ndig = random.randint(dmin, dmax); lo = BASE ** (ndig - 1) if ndig > 1 else 2
        N = max(2, random.randint(lo, BASE ** ndig - 1))
        vals = [random.randint(0, N - 1) for _ in range(M)]
        for k in range(M):
            leaves[k].append(digs(vals[k]))
        Nl.append(digs(N)); Y.append(digs(sum(vals) % N))
    oh = lambda z: F.one_hot(torch.tensor(z), BASE).float()
    return [oh(l) for l in leaves], oh(Nl), torch.tensor(Y)

def exact(m, dmin, dmax, n=250, tau=0.1):
    leaves, N, Y = batch(n, dmin, dmax)
    with torch.no_grad():
        return (tree_sum(m, leaves, N, tau).argmax(-1) == Y).all(-1).float().mean().item()

if __name__ == "__main__":
    random.seed(0); torch.manual_seed(0)
    m = ModAdd(); opt = torch.optim.AdamW(m.parameters(), lr=1.5e-3, weight_decay=1e-4)
    STEPS = 9000
    print("annealed tree modular-sum (M=8), length-general ripple, endpoint-only", flush=True)
    print("train N 1-3 digit; eval train + OOD 5d/6d", flush=True)
    for step in range(1, STEPS + 1):
        tau = max(0.2, 1.5 * (0.2 / 1.5) ** (step / STEPS))
        leaves, N, Y = batch(96, 1, 3); opt.zero_grad()
        loss = F.cross_entropy(tree_sum(m, leaves, N, tau).reshape(-1, BASE), Y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 1000 == 0:
            print(f"  step {step:5d} tau {tau:.2f} loss {loss.item():.3f} | "
                  f"train(1-3d) {exact(m,1,3)*100:5.1f}%  OOD-5d {exact(m,5,5)*100:5.1f}%  "
                  f"OOD-6d {exact(m,6,6)*100:5.1f}%", flush=True)
    print(f"=== FINAL train(1-3d) {exact(m,1,3)*100:.1f}%  OOD-5d {exact(m,5,5)*100:.1f}%  "
          f"OOD-6d {exact(m,6,6)*100:.1f}% ===", flush=True)

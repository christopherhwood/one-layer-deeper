"""PROOF of the soft-composition depth failure.

Train ONE (a+b) mod N cell (annealed-soft) to high single-op accuracy, FREEZE
it, then compose it K times as repeated modular doubling acc -> 2*acc mod N
(exact ground truth (2^K * x) mod N). Compare, as depth K grows:
  - HARD compose (argmax each step): exact accuracy
  - SOFT compose (softmax each step): exact accuracy, mean accumulator peak-prob,
    and mean probability mass on the CORRECT digit (= the CE loss floor).

If the frozen cell is fine but SOFT decays while HARD holds, and peak-prob falls
with K, the failure is soft-forward BLUR -- proven and quantified.
"""
from __future__ import annotations
import math, random
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4)

BASE = 10; W = 3

def dlsb(v, w=W):
    d = []
    for _ in range(w):
        d.append(v % BASE); v //= BASE
    return d

class Cell(nn.Module):
    def __init__(self, n_in, w=128):
        super().__init__()
        self.E = nn.ParameterList(nn.Parameter(torch.randn(BASE, w) * 0.2) for _ in range(n_in))
        self.sin = nn.Linear(8, w); self.body = nn.Sequential(nn.Linear(w, w), nn.GELU(), nn.Linear(w, w), nn.GELU())
        self.out = nn.Linear(w, BASE); self.sout = nn.Linear(w, 8)
    def forward(self, digs, st):
        h = self.sin(st)
        for p, E in zip(digs, self.E): h = h + p @ E
        h = self.body(h); return self.out(h), torch.tanh(self.sout(h))

class ModAdd(nn.Module):
    def __init__(self, w=128):
        super().__init__(); self.add = Cell(2, w); self.sub = Cell(2, w); self.gate = nn.Linear(8, 1)
    def _pass(self, cell, lists):
        st = torch.zeros(lists[0].shape[0], 8); outs = []
        for i in range(W):
            lg, st = cell([l[:, i] for l in lists], st); outs.append(lg)
        return torch.stack(outs, 1), st
    def forward(self, A_, B_, Nn, tau):
        sl, _ = self._pass(self.add, [A_, B_]); S = F.softmax(sl / tau, -1)
        dl, ss = self._pass(self.sub, [S, Nn]); g = torch.sigmoid(self.gate(ss)).unsqueeze(1)
        return g * sl + (1 - g) * dl  # logits

def make(n):
    A_, B_, Nn, Y = [], [], [], []
    for _ in range(n):
        N = random.randint(10, 99); a = random.randint(0, N - 1); b = random.randint(0, N - 1)
        A_.append(dlsb(a)); B_.append(dlsb(b)); Nn.append(dlsb(N)); Y.append(dlsb((a + b) % N))
    oh = lambda z: F.one_hot(torch.tensor(z), BASE).float()
    return oh(A_), oh(B_), oh(Nn), torch.tensor(Y)

def sharp(logits, tau, hard):
    if hard:
        i = logits.argmax(-1, keepdim=True)
        return torch.zeros_like(logits).scatter_(-1, i, 1.0)
    return F.softmax(logits / tau, -1)

def compose_double(m, x_oh, N_oh, K, tau, hard):
    acc = x_oh
    for _ in range(K):
        logits = m(acc, acc, N_oh, tau)
        acc = sharp(logits, tau, hard)
    return acc  # distribution [B,W,10]

if __name__ == "__main__":
    random.seed(0); torch.manual_seed(0)
    m = ModAdd()
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    STEPS = 5000
    for step in range(1, STEPS + 1):
        tau = max(0.15, 1.5 * (0.15 / 1.5) ** (step / STEPS))
        A_, B_, Nn, Y = make(128); opt.zero_grad()
        loss = F.cross_entropy(m(A_, B_, Nn, tau).reshape(-1, BASE), Y.reshape(-1))
        loss.backward(); opt.step()
    # per-op accuracy of the frozen cell
    A_, B_, Nn, Y = make(500)
    with torch.no_grad():
        p_op = (m(A_, B_, Nn, 0.1).argmax(-1) == Y).all(-1).float().mean().item()
    print(f"frozen cell single-op (a+b) mod N exact: {p_op*100:.1f}%\n", flush=True)

    # build a doubling test set of units x < N
    n = 400; xs, Ns = [], []
    for _ in range(n):
        N = random.randint(10, 99); x = random.randint(1, N - 1)
        xs.append(x); Ns.append(N)
    x_oh = F.one_hot(torch.tensor([dlsb(v) for v in xs]), BASE).float()
    N_oh = F.one_hot(torch.tensor([dlsb(v) for v in Ns]), BASE).float()
    TEVAL = 0.1

    print(f"{'K':>3} {'hard exact':>11} {'soft exact':>11} {'soft peak-p':>12} {'P(correct)':>11}")
    for K in [1, 2, 4, 6, 8, 10, 12]:
        gt = torch.tensor([dlsb((pow(2, K) * x) % N) for x, N in zip(xs, Ns)])
        with torch.no_grad():
            hard = compose_double(m, x_oh, N_oh, K, TEVAL, True)
            soft = compose_double(m, x_oh, N_oh, K, TEVAL, False)
        hard_ex = (hard.argmax(-1) == gt).all(-1).float().mean().item()
        soft_ex = (soft.argmax(-1) == gt).all(-1).float().mean().item()
        peak = soft.max(-1).values.mean().item()                       # blur metric
        pcorrect = soft.gather(-1, gt.unsqueeze(-1)).squeeze(-1).mean().item()  # loss-floor metric
        print(f"{K:>3} {hard_ex*100:>10.1f}% {soft_ex*100:>10.1f}% {peak:>11.3f} {pcorrect:>10.3f}", flush=True)
    print("\nIf hard >> soft and peak-p / P(correct) fall with K -> soft-forward BLUR.", flush=True)

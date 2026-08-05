"""Clinching proof: deeper END-TO-END (endpoint-only) training yields a WORSE
cell. Task: output (2^K x) mod N via K compositions of the tied doubling cell,
supervising ONLY the final residue. After training, measure the cell's single
doubling accuracy (2a mod N). If single-op accuracy falls as training-depth K
rises, the failure is optimization THROUGH DEPTH -- not forward blur (proof.py)."""
from __future__ import annotations
import argparse, random
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)

BASE = 10; W = 3
ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, required=True); A = ap.parse_args()

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
        return g * sl + (1 - g) * dl

def batch(n, K):
    x, N = [], []
    for _ in range(n):
        NN = random.randint(10, 99); x.append(random.randint(1, NN - 1)); N.append(NN)
    x_oh = F.one_hot(torch.tensor([dlsb(v) for v in x]), BASE).float()
    N_oh = F.one_hot(torch.tensor([dlsb(v) for v in N]), BASE).float()
    Y = torch.tensor([dlsb((pow(2, K) * xi) % Ni) for xi, Ni in zip(x, N)])
    return x_oh, N_oh, Y

def double_acc(m, x_oh, N_oh, K, tau):
    acc = x_oh
    for _ in range(K):
        acc = F.softmax(m(acc, acc, N_oh, tau) / tau, -1)
    return acc

def single_double_acc(m):
    a, N = [], []
    for _ in range(400):
        NN = random.randint(10, 99); a.append(random.randint(0, NN - 1)); N.append(NN)
    a_oh = F.one_hot(torch.tensor([dlsb(v) for v in a]), BASE).float()
    N_oh = F.one_hot(torch.tensor([dlsb(v) for v in N]), BASE).float()
    Y = torch.tensor([dlsb((2 * ai) % Ni) for ai, Ni in zip(a, N)])
    with torch.no_grad():
        return (m(a_oh, a_oh, N_oh, 0.1).argmax(-1) == Y).all(-1).float().mean().item()

if __name__ == "__main__":
    random.seed(0); torch.manual_seed(0)
    m = ModAdd(); opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    STEPS = 5000
    for step in range(1, STEPS + 1):
        tau = max(0.15, 1.5 * (0.15 / 1.5) ** (step / STEPS))
        x_oh, N_oh, Y = batch(96, A.K); opt.zero_grad()
        out = double_acc(m, x_oh, N_oh, A.K, tau)
        loss = F.nll_loss(torch.log(out.reshape(-1, BASE) + 1e-9), Y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    # endpoint accuracy of the K-fold task, and the learned cell's single-step accuracy
    x_oh, N_oh, Y = batch(400, A.K)
    with torch.no_grad():
        end_ex = (double_acc(m, x_oh, N_oh, A.K, 0.1).argmax(-1) == Y).all(-1).float().mean().item()
    print(f"train-depth K={A.K:2d}: endpoint exact {end_ex*100:5.1f}%  "
          f"| learned cell single-step (2a mod N) exact {single_double_acc(m)*100:5.1f}%", flush=True)

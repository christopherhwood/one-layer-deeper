"""Localize the endpoint-only wall to the exact sub-operation.

Three tasks, identical tied-cell machinery, endpoint-only supervision:
  add : a + b                (carry latent)
  sub : s - N,  s in [N,2N)  (borrow latent)
  mod : s mod N, s in [0,2N) (the 'subtract N?' GATE latent)

Hypothesis from earlier runs: carry/borrow are fine endpoint-only (like plain
addition at 100%); the GATE -- one global 'is s<N?' bit -- is the wall.
"""
from __future__ import annotations
import argparse, random
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)

BASE = 10; W = 3
ap = argparse.ArgumentParser(); ap.add_argument("--task", choices=["add", "sub", "mod"], required=True)
ap.add_argument("--steps", type=int, default=4000); A = ap.parse_args()

def dlsb(v, w=W):
    d = []
    for _ in range(w):
        d.append(v % BASE); v //= BASE
    return d
def st_hard(lg):
    s = F.softmax(lg, -1); i = s.argmax(-1, keepdim=True)
    h = torch.zeros_like(s).scatter_(-1, i, 1.0); return h + s - s.detach()

class Cell(nn.Module):
    def __init__(self, n_in, w=128):
        super().__init__()
        self.E = nn.ParameterList(nn.Parameter(torch.randn(BASE, w) * 0.1) for _ in range(n_in))
        self.sin = nn.Linear(8, w); self.body = nn.Sequential(nn.Linear(w, w), nn.GELU(), nn.Linear(w, w), nn.GELU())
        self.out = nn.Linear(w, BASE); self.sout = nn.Linear(w, 8)
    def forward(self, digs, st):
        h = self.sin(st)
        for p, E in zip(digs, self.E): h = h + p @ E
        h = self.body(h); return self.out(h), torch.tanh(self.sout(h))

def run_pass(cell, lists):
    st = torch.zeros(lists[0].shape[0], 8); outs = []
    for i in range(W):
        lg, st = cell([l[:, i] for l in lists], st); outs.append(lg)
    return torch.stack(outs, 1), st

class Model(nn.Module):
    def __init__(self, task):
        super().__init__(); self.task = task
        self.p = Cell(2)
        if task == "mod":
            self.gate = nn.Linear(8, 1)
    def forward(self, A, B):
        lg, st = run_pass(self.p, [A, B])
        if self.task != "mod":
            return lg
        d = st_hard(lg); g = torch.sigmoid(self.gate(st)).unsqueeze(1)
        return g * A + (1 - g) * lg  # A is s (one-hot); if s<N output s else s-N

def batch(n, task):
    Aa, Bb, Y = [], [], []
    for _ in range(n):
        N = random.randint(10, 99)
        if task == "add":
            a = random.randint(0, 99); b = random.randint(0, 99); Aa.append(a); Bb.append(N); Y.append(a + b)
            Bb[-1] = b
        elif task == "sub":
            s = random.randint(N, 2 * N - 1); Aa.append(s); Bb.append(N); Y.append(s - N)
        else:
            s = random.randint(0, 2 * N - 1); Aa.append(s); Bb.append(N); Y.append(s % N)
    oh = lambda z: F.one_hot(torch.tensor([dlsb(v) for v in z]), BASE).float()
    return oh(Aa), oh(Bb), torch.tensor([dlsb(v) for v in Y])

def exact(m, task, n=300):
    A_, B_, Y = batch(n, task)
    with torch.no_grad(): return (m(A_, B_).argmax(-1) == Y).all(-1).float().mean().item()

if __name__ == "__main__":
    random.seed(0); torch.manual_seed(0)
    m = Model(A.task); opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    print(f"=== task={A.task} (endpoint-only) ===", flush=True)
    for step in range(1, A.steps + 1):
        A_, B_, Y = batch(64, A.task); opt.zero_grad()
        loss = F.cross_entropy(m(A_, B_).reshape(-1, BASE), Y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 800 == 0:
            print(f"  step {step:4d} loss {loss.item():.3f} exact {exact(m,A.task,100)*100:5.1f}%", flush=True)
    print(f"=== FINAL task={A.task}: exact {exact(m,A.task)*100:5.1f}% ===", flush=True)

"""Test the pass-through / residual gradient-highway idea on the exact failing
case: learn (2^K x) mod N endpoint-only through K=8 tied recurrent steps.

Two identical-capacity models, only difference is the update rule:
  plain    : h <- cell(h, N)
  residual : h <- h + cell(h, N)   (ResNet/highway gradient highway; cell zero-init
             so it starts as identity -> intermediates are valid early)
If residual >> plain, the pass-through helps deep endpoint-only optimization.
"""
from __future__ import annotations
import argparse, random
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(2)

BASE = 10; W = 3; D = 160; K = 8
ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["plain", "residual"], required=True)
A = ap.parse_args()

def dlsb(v, w=W):
    d = []
    for _ in range(w):
        d.append(v % BASE); v //= BASE
    return d

class Model(nn.Module):
    def __init__(self, residual):
        super().__init__()
        self.residual = residual
        self.enc = nn.Linear(2 * W * BASE, D)
        self.nenc = nn.Linear(W * BASE, D)
        self.cell = nn.Sequential(nn.Linear(2 * D, 2 * D), nn.GELU(),
                                  nn.Linear(2 * D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        if residual:
            nn.init.zeros_(self.cell[-1].weight); nn.init.zeros_(self.cell[-1].bias)  # start = identity
        self.dec = nn.Linear(D, W * BASE)
    def forward(self, x_oh, N_oh):
        B = x_oh.shape[0]
        hN = self.nenc(N_oh.reshape(B, -1))
        h = self.enc(torch.cat([x_oh.reshape(B, -1), N_oh.reshape(B, -1)], -1))
        for _ in range(K):
            u = self.cell(torch.cat([h, hN], -1))
            h = h + u if self.residual else u
        return self.dec(h).reshape(B, W, BASE)

def batch(n):
    x, N = [], []
    for _ in range(n):
        NN = random.randint(10, 99); x.append(random.randint(1, NN - 1)); N.append(NN)
    xo = F.one_hot(torch.tensor([dlsb(v) for v in x]), BASE).float()
    No = F.one_hot(torch.tensor([dlsb(v) for v in N]), BASE).float()
    Y = torch.tensor([dlsb((pow(2, K) * xi) % Ni) for xi, Ni in zip(x, N)])
    return xo, No, Y

def exact(m, n=400):
    xo, No, Y = batch(n)
    with torch.no_grad():
        return (m(xo, No).argmax(-1) == Y).all(-1).float().mean().item()

if __name__ == "__main__":
    random.seed(0); torch.manual_seed(0)
    m = Model(A.mode == "residual")
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f"=== mode={A.mode}, K={K} doubling, endpoint-only ===", flush=True)
    for step in range(1, 8001):
        xo, No, Y = batch(128); opt.zero_grad()
        loss = F.cross_entropy(m(xo, No).reshape(-1, BASE), Y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 1000 == 0:
            print(f"  step {step:5d} loss {loss.item():.3f} endpoint-exact {exact(m)*100:5.1f}%", flush=True)
    print(f"=== FINAL {A.mode}: endpoint-exact {exact(m)*100:5.1f}% ===", flush=True)

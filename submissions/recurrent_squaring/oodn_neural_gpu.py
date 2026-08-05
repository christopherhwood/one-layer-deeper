"""Attack OOD-N: length-general + residual highway (Neural-GPU-lite).

Task: (2^8 x) mod N (values stay < N). Model has NO fixed width and NO absolute
positions -- a tied 1D conv (local carry communication) with a per-position
residual highway, run for a fixed number of steps. Train on 1-3 digit N; evaluate
on train range AND unseen 5- and 6-digit N. If it generalizes to larger N, the
combination attacks OOD-N; if it holds only in-range, length-generality is still
missing.
"""
from __future__ import annotations
import random
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4)

BASE = 10; WMAX = 8; D = 96; STEPS = 16; KDBL = 8

def digs(v, w=WMAX):
    d = []
    for _ in range(w):
        d.append(v % BASE); v //= BASE
    return d  # LSB-first

class NeuralGPULite(nn.Module):
    def __init__(self):
        super().__init__()
        self.ex = nn.Embedding(BASE, D)
        self.en = nn.Embedding(BASE, D)
        self.c1 = nn.Conv1d(D, D, 3, padding=1)
        self.c2 = nn.Conv1d(D, D, 3, padding=1)
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)   # identity start
        self.dec = nn.Conv1d(D, BASE, 1)
    def forward(self, xd, nd):
        # xd, nd: [B, W] digit ids (LSB-first). W can be anything.
        hN = self.en(nd).transpose(1, 2)                  # [B, D, W]
        H = (self.ex(xd) + self.en(nd)).transpose(1, 2)   # [B, D, W]
        for _ in range(STEPS):
            u = self.c2(F.gelu(self.c1(H + hN)))          # tied conv, local carries
            H = H + u                                      # per-position residual highway
        return self.dec(H).transpose(1, 2)                # [B, W, BASE]

def batch(n, dmin, dmax, w=WMAX):
    xd, nd, Y = [], [], []
    for _ in range(n):
        ndig = random.randint(dmin, dmax)
        lo = BASE ** (ndig - 1) if ndig > 1 else 2
        N = max(2, random.randint(lo, BASE ** ndig - 1)); x = random.randint(1, N - 1)
        xd.append(digs(x, w)); nd.append(digs(N, w)); Y.append(digs((pow(2, KDBL) * x) % N, w))
    return torch.tensor(xd), torch.tensor(nd), torch.tensor(Y)

def exact(m, dmin, dmax, n=300):
    xd, nd, Y = batch(n, dmin, dmax)
    with torch.no_grad():
        return (m(xd, nd).argmax(-1) == Y).all(-1).float().mean().item()

if __name__ == "__main__":
    random.seed(0); torch.manual_seed(0)
    m = NeuralGPULite(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    print("Neural-GPU-lite (tied conv + residual), (2^8 x) mod N, endpoint-only:", flush=True)
    print("train N in 1-3 digits; eval train-range + OOD 5d / 6d", flush=True)
    for step in range(1, 12001):
        xd, nd, Y = batch(128, 1, 3); opt.zero_grad()
        loss = F.cross_entropy(m(xd, nd).reshape(-1, BASE), Y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 1500 == 0:
            print(f"  step {step:5d} loss {loss.item():.3f} | train(1-3d) {exact(m,1,3)*100:5.1f}%"
                  f"  OOD-5d {exact(m,5,5)*100:5.1f}%  OOD-6d {exact(m,6,6)*100:5.1f}%", flush=True)
    print(f"=== FINAL train(1-3d) {exact(m,1,3)*100:.1f}%  OOD-5d {exact(m,5,5)*100:.1f}%  "
          f"OOD-6d {exact(m,6,6)*100:.1f}% ===", flush=True)

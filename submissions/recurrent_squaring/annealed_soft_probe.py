"""Break the endpoint-only squaring plateau with a smooth, temperature-annealed
composition (no gradient-blocking straight-through).

Baselines on this exact task (x^2 mod N, N<10, endpoint-only):
    straight-through accumulator : ~65% (gradients blocked)
    pure-soft accumulator        : ~10% (blurs over depth)
Here: soft accumulator with temperature tau annealed high->low (clean gradients
early, sharp/exact forward late). Eval is hard (argmax).
"""
from __future__ import annotations
import random
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(4)

BASE = 10; W = 4; MAXDIG = 1; NBITS = 4; STEPS = 6000

def dlsb(v, w=W):
    d = []
    for _ in range(w):
        d.append(v % BASE); v //= BASE
    return d

def soft(logits, tau):
    return F.softmax(logits / tau, dim=-1)

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
        sl, _ = self._pass(self.add, [A_, B_]); S = soft(sl, tau)
        dl, ss = self._pass(self.sub, [S, Nn]); g = torch.sigmoid(self.gate(ss)).unsqueeze(1)
        return g * sl + (1 - g) * dl   # returns logits

def onehot0(B):
    z = torch.zeros(B, W, BASE); z[:, :, 0] = 1.0; return z

def modsquare(m, Xd, Nn, bits, tau):
    B = Xd.shape[0]; acc = onehot0(B)
    for j in range(bits.shape[1]):
        acc = soft(m(acc, acc, Nn, tau), tau)
        added = soft(m(acc, Xd, Nn, tau), tau)
        mask = bits[:, j].view(B, 1, 1)
        acc = mask * added + (1 - mask) * acc
    return acc   # soft digit distribution

def make(n):
    Xd, Nn, Y, bt = [], [], [], []
    for _ in range(n):
        N = random.randint(2, BASE ** MAXDIG - 1); x = random.randint(1, N - 1)
        Xd.append(dlsb(x)); Nn.append(dlsb(N)); Y.append(dlsb((x * x) % N))
        b = [int(c) for c in bin(x)[2:]]; bt.append([0] * (NBITS - len(b)) + b)
    T = torch.tensor
    return F.one_hot(T(Xd), BASE).float(), F.one_hot(T(Nn), BASE).float(), T(Y), T(bt).float()

def exact(m, n=300, tau=0.05):
    Xd, Nn, Y, bt = make(n)
    with torch.no_grad():
        return (modsquare(m, Xd, Nn, bt, tau).argmax(-1) == Y).all(-1).float().mean().item()

if __name__ == "__main__":
    random.seed(0); torch.manual_seed(0)
    m = ModAdd(); opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    print("annealed-soft endpoint-only squaring (N<10):", flush=True)
    for step in range(1, STEPS + 1):
        tau = max(0.15, 1.5 * (0.15 / 1.5) ** (step / STEPS))   # 1.5 -> 0.15
        Xd, Nn, Y, bt = make(64); opt.zero_grad()
        out = modsquare(m, Xd, Nn, bt, tau)
        loss = F.nll_loss(torch.log(out.reshape(-1, BASE) + 1e-9), Y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 750 == 0:
            print(f"  step {step:5d} tau {tau:.2f} loss {loss.item():.3f} exact {exact(m)*100:5.1f}%", flush=True)
    print(f"=== FINAL annealed-soft: exact {exact(m)*100:5.1f}% ===", flush=True)

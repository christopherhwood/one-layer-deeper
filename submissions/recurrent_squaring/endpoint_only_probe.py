"""The credit-assignment wall, isolated.

Same modular-add cell and double-and-add squaring as ood_n_arithmetic.py, but
trained with ENDPOINT-ONLY supervision: only x^2 mod N is labeled; every carry,
borrow, gate, and partial product is latent. This is the honest competition
signal (a T=1 prompt's label is exactly x^2 -- one squaring).

FINDING (reproducible, CPU): at the shallowest scale (N < 10, ~4 double-and-add
steps) endpoint-only training PLATEAUS around ~65% exact and does not reach 100%.
Per-op accuracy is ~0.95, but ~8 latent ops compound (0.95^8 ~ 0.66) and the
endpoint signal cannot pin the rare carry/gate cases. Contrast ood_n_arithmetic
.py, where the SAME architecture reaches 100% and generalizes to OOD-N once the
intermediate steps are supervised.

Conclusion: the REPRESENTATION is solved (an exact, length-general finite-state
transducer exists and is learnable with process supervision); the open problem
is ENDPOINT-ONLY OPTIMIZATION of the deep latent composition. That is a known-
hard "learn an algorithm from I/O" problem, and it is where a Hard-tier attack
must make progress -- no factoring involved.

Run:  python submissions/recurrent_squaring/endpoint_only_probe.py
"""
from __future__ import annotations
import random
import torch, torch.nn as nn, torch.nn.functional as F

BASE = 10; CDIM = 8; W = 4          # N < 10 => x^2 < 81, results < 10; W=4 is ample
MAXDIG = 1; NBITS = 4; STEPS = 2000

def dlsb(v, w):
    ds = []
    for _ in range(w):
        ds.append(v % BASE); v //= BASE
    return ds

def st_hard(logits):
    soft = F.softmax(logits, -1)
    idx = soft.argmax(-1, keepdim=True)
    hard = torch.zeros_like(soft).scatter_(-1, idx, 1.0)
    return hard + soft - soft.detach()          # exact forward, soft gradient

class SoftCell(nn.Module):
    def __init__(self, n_in, w=128):
        super().__init__()
        self.E = nn.ParameterList(nn.Parameter(torch.randn(BASE, w) * 0.1) for _ in range(n_in))
        self.si = nn.Linear(CDIM, w)
        self.body = nn.Sequential(nn.Linear(w, w), nn.GELU(), nn.Linear(w, w), nn.GELU())
        self.o = nn.Linear(w, BASE); self.so = nn.Linear(w, CDIM)
    def forward(self, digs, state):
        h = self.si(state)
        for p, E in zip(digs, self.E):
            h = h + p @ E
        h = self.body(h)
        return self.o(h), torch.tanh(self.so(h))

class ModAdd(nn.Module):
    def __init__(self, w=128):
        super().__init__()
        self.add = SoftCell(2, w); self.sub = SoftCell(2, w); self.gate = nn.Linear(CDIM, 1)
    def _pass(self, cell, lists):
        st = torch.zeros(lists[0].shape[0], CDIM); outs = []
        for i in range(W):
            lg, st = cell([l[:, i] for l in lists], st); outs.append(lg)
        return torch.stack(outs, 1), st
    def forward(self, A, B, Nn):
        sl, _ = self._pass(self.add, [A, B]); S = st_hard(sl)
        dl, ss = self._pass(self.sub, [S, Nn]); g = torch.sigmoid(self.gate(ss)).unsqueeze(1)
        return st_hard(g * sl + (1 - g) * dl)

def onehot0(B):
    z = torch.zeros(B, W, BASE); z[:, :, 0] = 1.0; return z

def modsquare(m, Xd, Nn, bits):
    B = Xd.shape[0]; acc = onehot0(B)
    for j in range(bits.shape[1]):
        acc = m(acc, acc, Nn)
        added = m(acc, Xd, Nn)
        mask = bits[:, j].view(B, 1, 1)
        acc = mask * added + (1 - mask) * acc
    return acc

def make_batch(n):
    Xd, Nn, Y, bt = [], [], [], []
    for _ in range(n):
        N = random.randint(2, BASE ** MAXDIG - 1); x = random.randint(1, N - 1)
        Xd.append(dlsb(x, W)); Nn.append(dlsb(N, W)); Y.append(dlsb((x * x) % N, W))
        b = [int(c) for c in bin(x)[2:]]; bt.append([0] * (NBITS - len(b)) + b)
    T = torch.tensor
    return F.one_hot(T(Xd), BASE).float(), F.one_hot(T(Nn), BASE).float(), T(Y), T(bt).float()

def exact(m, n=200):
    Xd, Nn, Y, bt = make_batch(n)
    with torch.no_grad():
        return (modsquare(m, Xd, Nn, bt).argmax(-1) == Y).all(-1).float().mean().item()

if __name__ == "__main__":
    random.seed(0); torch.manual_seed(0)
    m = ModAdd(); opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    print("endpoint-only squaring (N<10); watch it plateau below 100%:", flush=True)
    for step in range(1, STEPS + 1):
        Xd, Nn, Y, bt = make_batch(64); opt.zero_grad()
        out = modsquare(m, Xd, Nn, bt)
        loss = F.cross_entropy(out.reshape(-1, BASE), Y.reshape(-1))
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 400 == 0 or step == 1:
            print(f"  step {step:4d}  loss {loss.item():.3f}  exact {exact(m)*100:5.1f}%", flush=True)

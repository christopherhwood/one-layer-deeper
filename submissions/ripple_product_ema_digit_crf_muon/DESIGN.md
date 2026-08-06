# Hybrid Muon optimizer for the final observed-digit CRF

## Controlled change

The model and loss are the optimized final-only first-order CRF. Product
columns, four tied bidirectional refinements, factorized register, three-step
training recurrence with detach, EMA teacher, endpoint CE, T=1 phase CE,
teacher consistency, final partial-suffix CRF NLL, Viterbi, temperature and
learning-rate schedule, dynamic evaluation bound, batch sizes, and every
trainable parameter are unchanged.

Only the optimizer assignment changes. PyTorch 2.12.1's pinned
`torch.optim.Muon` updates these eight named hidden 2D tensors:

- `square.input_projection.weight`
- the forward and reverse GRU `weight_ih` and `weight_hh` matrices
- `square.scan_out.weight`
- both MLP weights

The official Muon implementation accepts each combined GRU gate matrix as an
ordinary 2D parameter, so this candidate does not invent a gate-block variant.
Muon uses momentum 0.95, Nesterov momentum, five Newton-Schulz steps, and
`adjust_lr_fn="match_rms_adamw"`. That adjustment is specifically designed to
reuse AdamW's tuned learning rate and decay, so both remain 1.5e-3 and 0.1
before the unchanged wall-clock multiplier.

All remaining trainable tensors use the unchanged AdamW configuration:
digit-pair embedding, numeric modulus/register projections, output and CRF edge
heads, start/end energies, normalizers, biases, and residual gates. This keeps
the arithmetic interface and structured decoder on elementwise adaptive
dynamics while testing spectral hidden-feature updates.

## Composite contract and rules

A small composite exposes the two disjoint child optimizers as one ordinary
evaluator optimizer. It delegates `zero_grad`, performs one Muon step and one
AdamW step per evaluator update, exposes all child parameter groups, and
serializes both state dictionaries. It does not initiate a forward, loss,
backward, extra pass, batch reuse, or scheduler step.

The optimizer split is fixed entirely by parameter name and contains every
trainable tensor exactly once. The EMA teacher remains frozen and is updated
only by the existing post-update scheduler. Orthogonalization is a generic
gradient transformation and contains no arithmetic relation or data-dependent
rule.

## Hypothesis and risk

Muon replaces singular-value magnitudes of hidden momentum updates with a more
balanced spectral direction. That may prevent a few dominant directions from
absorbing the product/reduction learning signal in tied recurrent matrices.
Unlike extra output factors, it does not directly add endpoint memorization
capacity.

The evidence remains weak: SAM, multipass, PCGrad, and L-BFGS did not repair
mixed-E5 modular semantics. Newton-Schulz also adds matrix-multiply overhead,
especially on CPU and on this much smaller model than Muon's usual setting.
The variable-E5 screen must improve held-out OOD or transferable T=1 behavior,
not merely fixed-modulus smoke or training fit, before a full gate is warranted.

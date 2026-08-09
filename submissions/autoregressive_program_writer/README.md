# Autoregressive sampled program writer

This candidate replaces exact marginalization over all `16^5 = 1,048,576`
complete programs with constructive sampling. A learned Markov writer emits
five instructions from left to right:

```text
q(program) = q(i0) q(i1 | i0) ... q(i4 | i3)
```

Each training forward samples 16,384 sequences with the Gumbel-max trick,
executes only those samples, ranks them using endpoint agreement, and applies
an ordinary cross-entropy/score-function loss to the best sampled sequence.
The evaluator owns backward and SGD. Evaluation uses an O(5 x 16^2) Viterbi
decode of the learned writer and executes one MAP program.

There is no complete-program enumeration, stored population, archive, winning
ticket, participant-controlled backward pass, or custom training loop. The
random seed-74 MAP program is wrong. The model has 1,040 randomly initialized,
optimizer-owned trainable logits and no persistent buffers.

## Public-task evidence

Exact source SHA-256:

`415cda17d835e6531497019b8b2c5991d72b4f98319ef5a4fb9349ff40d6324e`

| Gate | Budget | Updates | Test | OOD | Seen/OOD-N ladder |
|---|---:|---:|---:|---:|---:|
| built-in Easy smoke | 10 CPU s | 608 | 100% | 100% | — |
| M5-shaped variable N, T=2/4/8 | 60 CPU s | 2,552 | 100% | 100% | T=64 / T=64 |
| hosted Easy E5 | 60 H100 s | 3,004 | 100% | 100% | T=64 / T=64 |
| hosted Medium M5 | 600 H100 s | 28,591 | 100% | 100% | T=64 / T=64 |

The M5 gate covers 9,000 test examples, 3,000 held-out examples, and
768 examples at every seen- and unseen-modulus depth rung through T=64.

Hosted exact-source submissions:

- Easy E5: `b4b284a5-b496-462a-8b99-ba003b74107d`
- Medium M5: `fefe523e-0243-4ac5-8350-fb7a2d4680f0`

The Medium run is the only hosted baseline allocated to this five-slot writer.
Further development is CPU-first; another Medium slot is gated on a richer
machine achieving strong structural-family coverage and first passing an Easy
H100 confirmation.

## Hard-like structural audit

The generated audit trains only from T=2/4/8 endpoints, holds out T=16, and
evaluates matched seen-N and unseen-N ladders through T=64. Each row below is a
fresh 10-second CPU run from the same random seed and unchanged source.

| Recurrence family | Structural requirement | Mean exact | Seen/OOD-N max T |
|---|---|---:|---:|
| `3x+1 mod N` | short unary program | 100% | 64 / 64 |
| `x^3 mod N` | short unary program | 100% | 64 / 64 |
| `x^2+x+2 mod N` | full five-slot unary program | 100% | 64 / 64 |
| conditional threshold | branch/comparison | 6.63% | none / none |
| alternating square/affine | persistent phase bit | 0.16% | none / none |
| square plus original source | immutable source register | 0.72% | none / none |
| second-order additive | two mutable registers | 4.90% | none / none |
| square plus step index | loop counter | 0.16% | 1 / 1 |
| modulus-keyed affine | modulus-derived operand | 20.41% | none / none |
| bitwise shift mix | shifts and modulus bits | 0.16% | none / none |

This is a sharp representability boundary, not a generic optimization failure:
the writer perfectly discovers every audited recurrence expressible by the
current five-slot accumulator and fails the families requiring machine state
or operations absent from that grammar. The next architecture therefore needs
multiple registers, immutable prompt operands, a counter/phase register,
conditional control, and a richer operand selector.

## Reproduction

```bash
.venv/bin/python -m unittest \
  tests.test_autoregressive_program_writer \
  tests.test_hard_like_recurrence_suite

.venv/bin/python -m benchmark.runner \
  --manifest benchmark/manifests/local_cpu_medium_m5_60s.json \
  --submission-file submissions/autoregressive_program_writer/submission.py \
  --include-structured-metrics

.venv/bin/python -m scripts.generate_hard_like_recurrence_suite
.venv/bin/python -m scripts.run_hard_like_recurrence_audit \
  submissions/autoregressive_program_writer/submission.py
```

Generated datasets live under `data/generated/hard_like_recurrence_suite/` and
are intentionally ignored by Git. Their generator, reference transitions,
dataset invariants, and audit runner are versioned.

# Product attention with Grokfast-style slow-gradient amplification

The optimizer keeps a beta-0.98 EMA of every clipped minibatch gradient and,
after 20 updates, adds that slow component to the current gradient before the
tracked AdamW update.  The hypothesis is that prompt-specific memorization
directions vary while a reusable arithmetic rule persists across batches.

| amplification | gate | updates | exact | token | last token | OOD-N T=1 token |
|---:|---|---:|---:|---:|---:|---:|
| 2x | fixed 10s | 207 | 26.83% | 36.98% | 47.33% | — |
| 2x | variable 15s | 227 | 1.04% | 14.47% | 10.92% | 16.65% |
| 5x | fixed 10s | 207 | 25.50% | 36.11% | 43.17% | — |
| 5x | variable 15s | 227 | 1.21% | 14.26% | 10.63% | 16.54% |
| 5x | variable 60s | 911 | 0.71% | 15.22% | 10.88% | 14.21% |

The two short runs showed a monotonic exact-match response, so the stronger
variant was promoted to the full 60-second CPU gate.  It did not confirm:
ordinary exact fell to 0.71%, OOD-N T=1 token accuracy fell sharply, and no
depth rung certified.  A long-lived gradient direction can encode consistent
memorization just as readily as the reusable rule; frequency alone does not
separate them.  Reject before H100.

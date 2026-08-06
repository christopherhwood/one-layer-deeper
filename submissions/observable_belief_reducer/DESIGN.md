# Observable belief-state reducer

The recurrent state is a canonical directed first-order distribution over the
actual output digits: one normalized start distribution and normalized
next-digit conditionals. A tied learned evidence updater consumes successive
product/modulus significance columns and revises those probabilities. No
hidden recurrent workspace, free latent alphabet, decoder from latent states,
or arithmetic transition table persists between updates.

On T1 rows, the supplied endpoint scores each evidence-prefix belief with an
exact partial-suffix likelihood. The final endpoint retains token CE and chain
likelihood on every row. Unobserved fixed-width leading digits are marginalized.

`diagnose_belief.py` exhaustively checks the width-two chain partition and all
100 paths, compares dynamic-programming Viterbi with enumeration, verifies
normalization, and performs finite backward.

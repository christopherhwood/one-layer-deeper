# Product attention with Lookahead optimizer

Lookahead wraps the retained AdamW optimizer with five-step fast trajectories
and 0.5 interpolation toward slow weights.  It adds no model or forward-pass
changes.

The 10-second fixed smoke reached 24.33% exact and 36.35% token accuracy.  The
15-second variable screen reached 0.96% exact but only 14.36% token, 10.13%
last-token, and 15.96% OOD-N T=1 token accuracy.  The exact fluctuation is not
supported by broad error reduction, so Lookahead is rejected.

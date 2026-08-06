# Convolutional Neural-GPU H100 attempt

The width-128 tied convolutional cellular model showed the wave's strongest
short relaxed signal (16.36% ordinary token and 13.50% last-token accuracy after
83 CPU updates), so it passed the softened H100 gate.

Hosted Easy/E5 submission `d073b2ff-4a28-41c3-a56f-c06e22f7aa94` completed
1,502 updates in 60 seconds and scored **0.88%**, with 1.1% test, 0.7% OOD, and
no seen/OOD-N certification.  Training loss remained around 3.1 and batch exact
was usually 0%, ending at 6.2%.  H100 throughput was healthy; the architecture
failed to fit the variable-modulus rule rather than being compute-limited.

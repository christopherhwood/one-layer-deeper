# Scaled general Transformer H100 attempt

The model scales the generic recurrent transition to width 256, six untied
Transformer blocks, full three-step compositional gradients, and 3.95M model
state elements.

Its 15-second CPU screen completed 83 updates and reached 0.38% exact, 15.93%
ordinary token, 11.50% last-token, and 15.96% OOD-N T=1 token accuracy.

Hosted Easy/E5 submission `3a24cdea-3672-4d15-8c0a-f3a51ac8d6a2` completed
1,259 updates and scored **0.50%** (1.0% test, 0% OOD), with no certification.
Training exact rose to 18% and loss fell to 2.377, so the model was not stuck;
scale widened the memorization/generalization gap instead of discovering the
transferable operation.

# Model weights

No model weights or pretrained checkpoints are currently published. They are
intentionally withheld until acceptance of the associated paper.

After acceptance, the best-validation checkpoint is planned to be distributed
as a versioned release asset rather than a normal Git object. Its release record
will include a SHA-256 digest, training variant, context encoding, selected
validation epoch, data split hash, and software environment. Once released,
pass its path to `infer_padspectra.py --checkpoint`.

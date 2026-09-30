# Historical single-split snapshot

These trained Keras files, histories and plots were published in the repository's first version. Their manifest records one random 80/20 split with seed `1612835101`.

They are retained so earlier links and the development history remain understandable. In particular, the feature model here has **eight inputs and 850 parameters**, while the final selected feature FCN has **seven inputs and 818 parameters**. The early raw-time and spectrum weights were also trained under an earlier single-split protocol.

Use [the final five-split report](../../docs/03-model-selection-and-stability.md) and [evidence index](../../evidence/README.md) for the final comparison. The archived Keras files are trained-network artifacts; the original training-fitted scalers were not exported alongside them, so the files alone are not a complete inference pipeline.

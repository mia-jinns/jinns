# Normalization samples

When a normalization loss must be computed, it depends on normalization samples and normalization weights that are stored (with other attributes) in a module named `NormalizationSamples`. This module needs to be passed to the PDE loss attribute named `norm_samples`.

::: jinns.loss.NormalizationSamples
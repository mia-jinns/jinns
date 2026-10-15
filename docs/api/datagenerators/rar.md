# Residual Adaptative Resampling strategies

Following Wu et al., A comprehensive study of non-adaptive and residual-based adaptive
sampling for physics-informed neural networks, 2022, [https://arxiv.org/pdf/2207.10289](https://arxiv.org/pdf/2207.10289), we propose the RAR-Greedy (RAR-G) and RAR with distribution (RAR-D) resampling strategies for an adaptative evolution of collocation points.

Note that because of JAX constraint that shapes must be static to benefit from hardware and software acceleration, in jinns, we cannot strictly follow the algorithms described in the article: the dataset of collocation points cannot grow in size. Instead we remove some points in the batch (based on the value of `novelty_proportion`) and replace them with new samples in the domain.

See an example of usage in the [Orstein Uhlenbeck 2D tutorial](../../../Notebooks/Tutorials/2D_non_stationary_OU).

::: jinns.data.RARParameters
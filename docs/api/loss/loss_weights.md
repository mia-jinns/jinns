# Loss weights

Note that loss weights might be scalars or arrays, the latter enables fine-grained weighting for multidimensional PINNs.

!!! warning "Warning"
    
    From jinns v1.5.0, loss weights can be adaptative and uniquely refer to the *scalar* value that multiplies each scalar loss (dynamic loss, boundary condition loss, etc.). If you want to insert a ponderation in front of each equation of a vectorial dynamic loss, you might now use the `vectorial_dyn_loss_ponderation` attribute from DynamicLoss class.

::: jinns.loss.LossWeightsODE

::: jinns.loss.LossWeightsPDEStatio

::: jinns.loss.LossWeightsPDENonStatio

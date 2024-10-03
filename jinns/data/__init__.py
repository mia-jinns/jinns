from ._DataGenerators import (
    DataGeneratorODE,
    CubicMeshPDEStatio,
    CubicMeshPDENonStatio,
    DataGeneratorObservations,
    DataGeneratorParameter,
    DataGeneratorObservationsMultiPINNs,
)
from ._Batchs import ODEBatch, PDEStatioBatch, PDENonStatioBatch

from ._DataGenerators import append_obs_batch, append_param_batch

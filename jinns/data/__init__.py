from ._DataGenerators import (
    DataGeneratorODE,
    CubicMeshPDEStatio,
    CubicMeshPDENonStatio,
    DataGeneratorParameter,
    DataGeneratorObservations,
    DataGeneratorObservationsMultiPINNs,
    append_param_batch,
    append_obs_batch,
)
from ._DataGenerators_eqx import (
    DataGeneratorODE_eqx,
    CubicMeshPDEStatio_eqx,
    CubicMeshPDENonStatio_eqx,
    DataGeneratorObservations_eqx,
    DataGeneratorParameter_eqx,
)
from ._display import (
    plot2d,
    plot1d_slice,
    plot1d_image,
)

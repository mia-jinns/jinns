"""
A poor man's profiling experiment to figure how the unduplicated version of
DataDataGeneratorObservations reduce memory costs
"""

import os

# do not preallocate
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# deallocate when possible
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import time
import jax
import jinns


def check_memory_gain():
    """
    This is a poor man's profiling experiment to check the memory gain induced
    by the DGObs version that does not duplicate entries that are the same
    explicitly.

    Poor man's version: because there is 5s pause in the program for you to
    check and note down the memory used as given in nvidia-smi. That is we do
    not used any real profiling tool

    This is to be run with XLA_PYTHON_CLIENT_PREALLOCATE="false", in order to see what
    is being allocated in real time. This is to be run with
    XLA_PYTHON_CLIENT_ALLOCATOR=platform because otehrwise, the allocated
    memory by JAX is never deallocated.

    Start the run:
    - At Step 1, before the dg1 instanciation you should read that 70Mo is used
      on the GPU.
    - At Step 2, after the dg1 instanciation you should read that 236Mo of GPU
      RAM is used, which means a 166Mo gain (this is the size of dg1 in the GPU
      RAM)
    - At Step 3, after the dg3 instanciation you should read that 358Mo of GPU
      RAM is used, which means a 122Mo gain. Those 122 Mo adds up to the total
      becuase dg1 and dg2 coexist (no preallocation of those objects).
    Hence it appears that dg2 is 44Mo lighter than dg1. In theory, dg1 contains
    19+19 jnp arrays of shape (100000, 3) of float32. The last list of the
    function tell us that one float32 array of size (100000, 3) weights
    1.2Mo. Since 38*1.2=45.6Mo, we can almost certainly conclude that the
    reduction in RAM usage by dg2 is caused by not duplicating the attributes.

    NOTE: While dg1 is heavier than dg2, when the tree map is done, the memory
    space that is used is the same: ie., in the vectorized computations all
    data is duplicated, even the frozen function parameters! We can see that in
    2 ways:
    1) if you uncomment the print of "bytes accessed" in test_batch_equality of
    test_DataGeneratorObservations.py you get the same number
    2) if you run this script without deallocation flag then you get that BOTH
    dg isntanciations adds up 256Mo to the toal memory. This correspond to the
    maximal quantity of memory that was once rquired by JAX during the tree map
    operation of the instanciation (and it was not deallocated). In practice,
    jinns (and many JAX libraries) will run without deallocation since it slows
    down programs. But that does not mean that part of those 256Mo cannot be used for
    other purposes in the rest of the computations! It just shows the maximum
    that was once required.
    """
    #### FIRST a cost_analysis for the duplicated case
    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key)
    subkeys2 = jax.random.split(key, 20)

    print("Step 1")
    print("Before duplicated DG instanciation")
    time.sleep(5)
    dg1 = jinns.data.DataGeneratorObservations(  # noqa: F841
        key=subkey,
        observed_pinn_in=tuple(
            jax.random.normal(key, shape=(100000, 3)) for _ in range(len(subkeys2))
        ),
        observed_values=tuple(
            jax.random.normal(k, shape=(100000, 15)) for k in subkeys2
        ),
        observed_eq_params=tuple(
            {"a": jax.random.normal(key, shape=(100000,))} for _ in range(len(subkeys2))
        ),
    )
    print("Step 2")
    print("After duplicated DG instanciation")
    print("Before unduplicated DG instanciation")
    time.sleep(5)

    #### SECOND a cost_analysis for the non duplicated case
    key = jax.random.PRNGKey(2)
    key, subkey = jax.random.split(key)

    dg2 = jinns.data.DataGeneratorObservations(  # noqa: F841
        key=subkey,
        observed_pinn_in=jax.random.normal(key, shape=(100000, 3)),
        observed_values=tuple(
            jax.random.normal(k, shape=(100000, 15)) for k in subkeys2
        ),
        observed_eq_params={"a": jax.random.normal(key, shape=(100000,))},
    )

    print("Step 3")
    print("After unduplicated DG instanciation")
    time.sleep(5)

    print(
        "Bytes occupied by a (100000, 3) float array",
        jax.random.normal(key, shape=(100000, 3)).nbytes,
    )


check_memory_gain()

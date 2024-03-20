"""
DataGenerators to generate batches of points in space, time and more
"""

from typing import NamedTuple
from jax.typing import ArrayLike
import jax.numpy as jnp
from jax import random
from jax.tree_util import register_pytree_node_class
import jax.lax


class ODEBatch(NamedTuple):
    temporal_batch: ArrayLike
    param_batch_dict: dict = None
    obs_batch_dict: dict = None


class PDENonStatioBatch(NamedTuple):
    inside_batch: ArrayLike
    border_batch: ArrayLike
    temporal_batch: ArrayLike
    param_batch_dict: dict = None
    obs_batch_dict: dict = None


class PDEStatioBatch(NamedTuple):
    inside_batch: ArrayLike
    border_batch: ArrayLike
    param_batch_dict: dict = None
    obs_batch_dict: dict = None


def append_param_batch(batch, param_batch_dict):
    """
    Utility function that fill the param_batch_dict of a batch object with a
    param_batch_dict
    """
    return batch._replace(param_batch_dict=param_batch_dict)


def append_obs_batch(batch, obs_batch_dict):
    """
    Utility function that fill the obs_batch_dict of a batch object with a
    obs_batch_dict
    """
    return batch._replace(obs_batch_dict=obs_batch_dict)


def _reset_batch_idx_and_permute(operands):
    key, domain, curr_idx, _, p = operands
    # resetting counter
    curr_idx = 0
    # reshuffling
    key, subkey = random.split(key)
    # domain = random.permutation(subkey, domain, axis=0, independent=False)
    # we want that permutation = choice when p=None
    # otherwise p is used to avoid collocation points not in nt_start
    domain = random.choice(subkey, domain, shape=(domain.shape[0],), replace=False, p=p)

    # return updated
    return (key, domain, curr_idx)


def _increment_batch_idx(operands):
    key, domain, curr_idx, batch_size, _ = operands
    # simply increases counter and get the batch
    curr_idx += batch_size
    return (key, domain, curr_idx)


def _reset_or_increment(bend, n_eff, operands):
    """
    Factorize the code of the jax.lax.cond which checks if we have seen all the
    batches in an epoch
    If bend > n_eff (ie n when no RAR sampling) we reshuffle and start from 0
    again. Otherwise, if bend < n_eff, this means there are still *_batch_size
    samples at least that have not been seen and we can take a new batch

    Parameters
    ----------
    bend
        An integer. The new hypothetical index for the starting of the batch
    n_eff
        An integer. The number of points to see to complete an epoch
    operands
        A tuple. As passed to _reset_batch_idx_and_permute and
        _increment_batch_idx

    Returns
    -------
    res
        A tuple as returned by _reset_batch_idx_and_permute or
        _increment_batch_idx
    """
    return jax.lax.cond(
        bend > n_eff, _reset_batch_idx_and_permute, _increment_batch_idx, operands
    )


#####################################################
# DataGenerator for ODE : only returns time_batches
#####################################################


@register_pytree_node_class
class DataGeneratorODE:
    """
    A class implementing data generator object for ordinary differential equations.


    **Note:** DataGeneratorODE is jittable. Hence it implements the tree_flatten() and
    tree_unflatten methods.
    """

    def __init__(
        self,
        key,
        nt,
        tmin,
        tmax,
        temporal_batch_size,
        method="uniform",
        rar_parameters=None,
        nt_start=None,
        data_exists=False,
    ):
        """
        Parameters
        ----------
        key
            Jax random key to sample new time points and to shuffle batches
        nt
            An integer. The number of total time points that will be divided in
            batches. Batches are made so that each data point is seen only
            once during 1 epoch.
        tmin
            A float. The minimum value of the time domain to consider
        tmax
            A float. The maximum value of the time domain to consider
        temporal_batch_size
            An integer. The size of the batch of randomly selected points among
            the `nt` points.
        method
            Either `grid` or `uniform`, default is `uniform`.
            The method that generates the `nt` time points. `grid` means
            regularly spaced points over the domain. `uniform` means uniformly
            sampled points over the domain
        rar_parameters
            Default to None: do not use Residual Adaptative Resampling.
            Otherwise a dictionary with keys. `start_iter`: the iteration at
            which we start the RAR sampling scheme (we first have a burn in
            period). `update_rate`: the number of gradient steps taken between
            each appending of collocation points in the RAR algo.
            `sample_size`: the size of the sample from which we will select new
            collocation points. `selected_sample_size`: the number of selected
            points from the sample to be added to the current collocation
            points
            "DeepXDE: A deep learning library for solving differential
            equations", L. Lu, SIAM Review, 2021
        nt_start
            Defaults to None. The effective size of nt used at start time.
            This value must be
            provided when rar_parameters is not None. Otherwise we set internally
            nt_start = nt and this is hidden from the user.
            In RAR, nt_start
            then corresponds to the initial number of points we train the PINN.
        data_exists
            Must be left to `False` when created by the user. Avoids the
            regeneration of the `nt` time points at each pytree flattening and
            unflattening.
        """
        self.data_exists = data_exists
        self._key = key
        self.nt = nt
        self.tmin = tmin
        self.tmax = tmax
        self.temporal_batch_size = temporal_batch_size
        self.method = method
        self.rar_parameters = rar_parameters

        if rar_parameters is not None and nt_start is None:
            raise ValueError(
                "nt_start must be provided in the context of RAR sampling scheme"
            )
        if rar_parameters is not None:
            self.nt_start = nt_start
            # Default p is None. However, in the RAR sampling scheme we use 0
            # probability to specify non-used collocation points (i.e. points
            # above nt_start). Thus, p is a vector of probability of shape (nt, 1).
            self.p = jnp.zeros((self.nt,))
            self.p = self.p.at[: self.nt_start].set(1 / nt_start)
            # set internal counter for the number of gradient steps since the
            # last new collocation points have been added
            self.rar_iter_from_last_sampling = 0
            # set iternal counter for the number of times collocation points
            # have been added
            self.rar_iter_nb = 0

        if rar_parameters is None or nt_start is None:
            self.nt_start = self.nt
            self.p = None
            self.rar_iter_from_last_sampling = None
            self.rar_iter_nb = None

        if not self.data_exists:
            # Useful when using a lax.scan with pytree
            # Optionally can tell JAX not to re-generate data
            self.curr_time_idx = 0
            self.generate_time_data()
            self._key, self.times, _ = _reset_batch_idx_and_permute(
                self._get_time_operands()
            )

    def sample_in_time_domain(self, n_samples):
        self._key, subkey = random.split(self._key, 2)
        return random.uniform(subkey, (n_samples,), minval=self.tmin, maxval=self.tmax)

    def generate_time_data(self):
        """
        Construct a complete set of `self.nt` time points according to the
        specified `self.method`

        Note that self.times has always size self.nt and not self.nt_start, even
        in RAR scheme, we must allocate all the collocation points
        """
        if self.method == "grid":
            self.partial_times = (self.tmax - self.tmin) / self.nt
            self.times = jnp.arange(self.tmin, self.tmax, self.partial_times)
        elif self.method == "uniform":
            self.times = self.sample_in_time_domain(self.nt)
        else:
            raise ValueError("Method " + self.method + " is not implemented.")

    def _get_time_operands(self):
        return (
            self._key,
            self.times,
            self.curr_time_idx,
            self.temporal_batch_size,
            self.p,
        )

    def temporal_batch(self):
        """
        Return a batch of time points. If all the batches have been seen, we
        reshuffle them, otherwise we just return the next unseen batch.
        """
        bstart = self.curr_time_idx
        bend = bstart + self.temporal_batch_size

        # Compute the effective number of used collocation points
        if self.rar_parameters is not None:
            nt_eff = (
                self.nt_start
                + self.rar_iter_nb * self.rar_parameters["selected_sample_size"]
            )
        else:
            nt_eff = self.nt
        (self._key, self.times, self.curr_time_idx) = _reset_or_increment(
            bend, nt_eff, self._get_time_operands()
        )

        # commands below are equivalent to
        # return self.times[i:(i+t_batch_size)]
        # start indices can be dynamic be the slice shape is fixed
        return jax.lax.dynamic_slice(
            self.times,
            start_indices=(self.curr_time_idx,),
            slice_sizes=(self.temporal_batch_size,),
        )

    def get_batch(self):
        """
        Generic method to return a batch. Here we call `self.temporal_batch()`
        """
        return ODEBatch(temporal_batch=self.temporal_batch())

    def tree_flatten(self):
        children = (
            self._key,
            self.times,
            self.curr_time_idx,
            self.tmin,
            self.tmax,
            self.rar_parameters,
            self.p,
            self.rar_iter_from_last_sampling,
            self.rar_iter_nb,
        )  # arrays / dynamic values
        aux_data = {
            k: vars(self)[k]
            for k in ["temporal_batch_size", "method", "nt", "nt_start"]
        }  # static values
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        **Note:** When reconstructing the class, we force ``data_exists=True``
        in order not to re-generate the data at each flattening and
        unflattening that happens e.g. during the gradient descent in the
        optimization process
        """
        (
            key,
            times,
            curr_time_idx,
            tmin,
            tmax,
            rar_parameters,
            p,
            rar_iter_from_last_sampling,
            rar_iter_nb,
        ) = children
        obj = cls(
            key=key,
            data_exists=True,
            tmin=tmin,
            tmax=tmax,
            rar_parameters=rar_parameters,
            **aux_data,
        )
        obj.times = times
        obj.curr_time_idx = curr_time_idx
        obj.p = p
        obj.rar_iter_from_last_sampling = rar_iter_from_last_sampling
        obj.rar_iter_nb = rar_iter_nb
        return obj


##########################################
# Data Generator for PDE in stationnary
# and non-stationnary cases
##########################################


class DataGeneratorPDEAbstract:
    """generic skeleton class for a PDE data generator"""

    def __init__(self, data_exists=False) -> None:
        # /!\ WARNING /!\: an-end user should never create an object
        # with data_exists=True. Or else generate_data() won't be called.
        # Useful when using a lax.scan with a DataGenerator in the carry
        # It tells JAX not to re-generate data in the __init__()
        self.data_exists = data_exists


@register_pytree_node_class
class CubicMeshPDEStatio(DataGeneratorPDEAbstract):
    """
    A class implementing data generator object for stationary partial
    differential equations.


    **Note:** CubicMeshPDEStatio is jittable. Hence it implements the tree_flatten() and
    tree_unflatten methods.
    """

    def __init__(
        self,
        key,
        n,
        nb,
        omega_batch_size,
        omega_border_batch_size,
        dim,
        min_pts,
        max_pts,
        method="grid",
        rar_parameters=None,
        n_start=None,
        data_exists=False,
    ):
        r"""
        Parameters
        ----------
        key
            Jax random key to sample new time points and to shuffle batches
        n
            An integer. The number of total :math:`\Omega` points that will be divided in
            batches. Batches are made so that each data point is seen only
            once during 1 epoch.
        nb
            An integer. The total number of points in :math:`\partial\Omega`.
            Can be `None` not to lose performance generating the border
            batch if they are not used
        omega_batch_size
            An integer. The size of the batch of randomly selected points among
            the `n` points.
        omega_border_batch_size
            An integer. The size of the batch of points randomly selected
            among the `nb` points.
            Can be `None` not to lose performance generating the border
            batch if they are not used
        dim
            An integer. dimension of :math:`\Omega` domain
        min_pts
            A tuple of minimum values of the domain along each dimension. For a sampling
            in `n` dimension, this represents :math:`(x_{1, min}, x_{2,min}, ...,
            x_{n, min})`
        max_pts
            A tuple of maximum values of the domain along each dimension. For a sampling
            in `n` dimension, this represents :math:`(x_{1, max}, x_{2,max}, ...,
            x_{n,max})`
        method
            Either `grid` or `uniform`, default is `grid`.
            The method that generates the `nt` time points. `grid` means
            regularly spaced points over the domain. `uniform` means uniformly
            sampled points over the domain
        rar_parameters
            Default to None: do not use Residual Adaptative Resampling.
            Otherwise a dictionary with keys. `start_iter`: the iteration at
            which we start the RAR sampling scheme (we first have a burn in
            period). `update_rate`: the number of gradient steps taken between
            each appending of collocation points in the RAR algo.
            `sample_size`: the size of the sample from which we will select new
            collocation points. `selected_sample_size`: the number of selected
            points from the sample to be added to the current collocation
            points
            "DeepXDE: A deep learning library for solving differential
            equations", L. Lu, SIAM Review, 2021
        n_start
            Defaults to None. The effective size of n used at start time.
            This value must be
            provided when rar_parameters is not None. Otherwise we set internally
            n_start = n and this is hidden from the user.
            In RAR, n_start
            then corresponds to the initial number of points we train the PINN.
        data_exists
            Must be left to `False` when created by the user. Avoids the
            regeneration of :math:`\Omega`, :math:`\partial\Omega` and
            time points at each pytree flattening and unflattening.
        """
        super().__init__(data_exists=data_exists)
        self.method = method
        self._key = key
        self.dim = dim
        self.min_pts = min_pts
        self.max_pts = max_pts
        assert dim == len(min_pts) and isinstance(min_pts, tuple)
        assert dim == len(max_pts) and isinstance(max_pts, tuple)
        self.n = n
        self.rar_parameters = rar_parameters

        if rar_parameters is not None and n_start is None:
            raise ValueError(
                "n_start must be provided in the context of RAR sampling scheme"
            )
        if rar_parameters is not None:
            self.n_start = n_start
            # Default p is None. However, in the RAR sampling scheme we use 0
            # probability to specify non-used collocation points (i.e. points
            # above n_start). Thus, p is a vector of probability of shape (n, 1).
            self.p = jnp.zeros((self.n,))
            self.p = self.p.at[: self.n_start].set(1 / n_start)
            # set internal counter for the number of gradient steps since the
            # last new collocation points have been added
            self.rar_iter_from_last_sampling = 0
            # set iternal counter for the number of times collocation points
            # have been added
            self.rar_iter_nb = 0

        if rar_parameters is None or n_start is None:
            self.n_start = self.n
            self.p = None
            self.rar_iter_from_last_sampling = None
            self.rar_iter_nb = None

        self.p_border = None  # no RAR sampling for border for now

        self.omega_batch_size = omega_batch_size

        if omega_border_batch_size is None:
            self.nb = None
            self.omega_border_batch_size = None
        elif self.dim == 1:
            # 1-D case : the arguments `nb` and `omega_border_batch_size` are
            # ignored but kept for backward stability. The attributes are
            # always set to 2.
            self.nb = 2
            self.omega_border_batch_size = 2
        # warnings.warn("We are in 1-D case => omega_border_batch_size is "
        #               "ignored since borders of Omega are singletons."
        #               " self.border_batch() will return [xmin, xmax]"
        #               )
        else:
            if nb % (2 * self.dim) != 0 or nb < 2 * self.dim:
                raise ValueError(
                    "number of border point must be"
                    " a multiple of 2xd (the # of faces of a d-dimensional cube)"
                )
            if nb // (2 * self.dim) < omega_border_batch_size:
                raise ValueError(
                    "number of points per facets (nb//2*self.dim)"
                    " cannot be lower than border batch size"
                )
            self.nb = int((2 * self.dim) * (nb // (2 * self.dim)))
            self.omega_border_batch_size = omega_border_batch_size

        if not self.data_exists:
            # Useful when using a lax.scan with pytree
            # Optionally tells JAX not to re-generate data when re-building the
            # object
            self.curr_omega_idx = 0
            self.curr_omega_border_idx = 0
            self.generate_data()
            self._key, self.omega, _ = _reset_batch_idx_and_permute(
                self._get_omega_operands()
            )
            if self.omega_border is not None and self.dim > 1:
                self._key, self.omega_border, _ = _reset_batch_idx_and_permute(
                    self._get_omega_border_operands()
                )

    def sample_in_omega_domain(self, n_samples):
        if self.dim == 1:
            xmin, xmax = self.min_pts[0], self.max_pts[0]
            self._key, subkey = random.split(self._key, 2)
            return random.uniform(
                subkey, shape=(n_samples, 1), minval=xmin, maxval=xmax
            )
        keys = random.split(self._key, self.dim + 1)
        self._key = keys[0]
        return jnp.concatenate(
            [
                random.uniform(
                    keys[i + 1],
                    (n_samples, 1),
                    minval=self.min_pts[i],
                    maxval=self.max_pts[i],
                )
                for i in range(self.dim)
            ],
            axis=-1,
        )

    def sample_in_omega_border_domain(self, n_samples):
        if self.omega_border_batch_size is None:
            return None
        if self.dim == 1:
            xmin = self.min_pts[0]
            xmax = self.max_pts[0]
            return jnp.array([xmin, xmax]).astype(float)
        if self.dim == 2:
            # currently hard-coded the 4 edges for d==2
            # TODO : find a general & efficient way to sample from the border
            # (facets) of the hypercube in general dim.

            facet_n = n_samples // (2 * self.dim)
            keys = random.split(self._key, 5)
            self._key = keys[0]
            subkeys = keys[1:]
            xmin = jnp.hstack(
                [
                    self.min_pts[0] * jnp.ones((facet_n, 1)),
                    random.uniform(
                        subkeys[0],
                        (facet_n, 1),
                        minval=self.min_pts[1],
                        maxval=self.max_pts[1],
                    ),
                ]
            )
            xmax = jnp.hstack(
                [
                    self.max_pts[0] * jnp.ones((facet_n, 1)),
                    random.uniform(
                        subkeys[1],
                        (facet_n, 1),
                        minval=self.min_pts[1],
                        maxval=self.max_pts[1],
                    ),
                ]
            )
            ymin = jnp.hstack(
                [
                    random.uniform(
                        subkeys[2],
                        (facet_n, 1),
                        minval=self.min_pts[0],
                        maxval=self.max_pts[0],
                    ),
                    self.min_pts[1] * jnp.ones((facet_n, 1)),
                ]
            )
            ymax = jnp.hstack(
                [
                    random.uniform(
                        subkeys[3],
                        (facet_n, 1),
                        minval=self.min_pts[0],
                        maxval=self.max_pts[0],
                    ),
                    self.max_pts[1] * jnp.ones((facet_n, 1)),
                ]
            )
            return jnp.stack([xmin, xmax, ymin, ymax], axis=-1)
        raise NotImplementedError(
            "Generation of the border of a cube in dimension > 2 is not "
            + f"implemented yet. You are asking for generation in dimension d={self.dim}."
        )

    def generate_data(self):
        r"""
        Construct a complete set of `self.n` :math:`\Omega` points according to the
        specified `self.method`. Also constructs a complete set of `self.nb`
        :math:`\partial\Omega` points if `self.omega_border_batch_size` is not
        `None`. If the latter is `None` we set `self.omega_border` to `None`.
        """

        # Generate Omega
        if self.method == "grid":
            if self.dim == 1:
                xmin, xmax = self.min_pts[0], self.max_pts[0]
                self.partial = (xmax - xmin) / self.n
                # shape (n, 1)
                self.omega = jnp.arange(xmin, xmax, self.partial)[:, None]
            else:
                self.partials = [
                    (self.max_pts[i] - self.min_pts[i]) / jnp.sqrt(self.n)
                    for i in range(self.dim)
                ]
                xyz_ = jnp.meshgrid(
                    *[
                        jnp.arange(self.min_pts[i], self.max_pts[i], self.partials[i])
                        for i in range(self.dim)
                    ]
                )
                xyz_ = [a.reshape((self.n, 1)) for a in xyz_]
                self.omega = jnp.concatenate(xyz_, axis=-1)
        elif self.method == "uniform":
            self.omega = self.sample_in_omega_domain(self.n)
        else:
            raise ValueError("Method " + self.method + " is not implemented.")

        # Generate border of omega
        self.omega_border = self.sample_in_omega_border_domain(self.nb)

    def _get_omega_operands(self):
        return (
            self._key,
            self.omega,
            self.curr_omega_idx,
            self.omega_batch_size,
            self.p,
        )

    def inside_batch(self):
        r"""
        Return a batch of points in :math:`\Omega`.
        If all the batches have been seen, we reshuffle them,
        otherwise we just return the next unseen batch.
        """
        # Compute the effective number of used collocation points
        if self.rar_parameters is not None:
            n_eff = (
                self.n_start
                + self.rar_iter_nb * self.rar_parameters["selected_sample_size"]
            )
        else:
            n_eff = self.n

        bstart = self.curr_omega_idx
        bend = bstart + self.omega_batch_size

        (self._key, self.omega, self.curr_omega_idx) = _reset_or_increment(
            bend, n_eff, self._get_omega_operands()
        )

        # commands below are equivalent to
        # return self.omega[i:(i+batch_size), 0:dim]
        return jax.lax.dynamic_slice(
            self.omega,
            start_indices=(self.curr_omega_idx, 0),
            slice_sizes=(self.omega_batch_size, self.dim),
        )

    def _get_omega_border_operands(self):
        return (
            self._key,
            self.omega_border,
            self.curr_omega_border_idx,
            self.omega_border_batch_size,
            self.p_border,
        )

    def border_batch(self):
        r"""
        Return

        - The value `None` if `self.omega_border_batch_size` is `None`.

        - a jnp array with two fixed values :math:`(x_{min}, x_{max})` if
          `self.dim` = 1. There is no sampling here, we return the entire
          :math:`\partial\Omega`

        - a batch of points in :math:`\partial\Omega` otherwise, stacked by
          facet on the last axis.
          If all the batches have been seen, we reshuffle them,
          otherwise we just return the next unseen batch.


        """
        if self.omega_border_batch_size is None:
            return None
        if self.dim == 1:
            # 1-D case, no randomness : we always return the whole omega border,
            # i.e. (1, 1, 2) shape jnp.array([[[xmin], [xmax]]]).
            return self.omega_border[None, None]  # shape is (1, 1, 2)
        bstart = self.curr_omega_border_idx
        bend = bstart + self.omega_border_batch_size

        (
            self._key,
            self.omega_border,
            self.curr_omega_border_idx,
        ) = _reset_or_increment(bend, self.nb, self._get_omega_border_operands())

        # commands below are equivalent to
        # return self.omega[i:(i+batch_size), 0:dim, 0:nb_facets]
        # and nb_facets = 2 * dimension
        # but JAX prefer the latter
        return jax.lax.dynamic_slice(
            self.omega_border,
            start_indices=(self.curr_omega_border_idx, 0, 0),
            slice_sizes=(self.omega_border_batch_size, self.dim, 2 * self.dim),
        )

    def get_batch(self):
        """
        Generic method to return a batch. Here we call `self.inside_batch()`
        and `self.border_batch()`
        """
        return PDEStatioBatch(
            inside_batch=self.inside_batch(), border_batch=self.border_batch()
        )

    def tree_flatten(self):
        children = (
            self._key,
            self.omega,
            self.omega_border,
            self.curr_omega_idx,
            self.curr_omega_border_idx,
            self.min_pts,
            self.max_pts,
            self.rar_parameters,
            self.p,
            self.rar_iter_from_last_sampling,
            self.rar_iter_nb,
        )
        aux_data = {
            k: vars(self)[k]
            for k in [
                "n",
                "nb",
                "omega_batch_size",
                "omega_border_batch_size",
                "method",
                "dim",
                "n_start",
            ]
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        **Note:** When reconstructing the class, we force ``data_exists=True``
        in order not to re-generate the data at each flattening and
        unflattening that happens e.g. during the gradient descent in the
        optimization process
        """
        (
            key,
            omega,
            omega_border,
            curr_omega_idx,
            curr_omega_border_idx,
            min_pts,
            max_pts,
            rar_parameters,
            p,
            rar_iter_from_last_sampling,
            rar_iter_nb,
        ) = children
        # force data_exists=True here in order not to re-generate the data
        # at each iteration of lax.scan
        obj = cls(
            key=key,
            data_exists=True,
            min_pts=min_pts,
            max_pts=max_pts,
            rar_parameters=rar_parameters,
            **aux_data,
        )
        obj.omega = omega
        obj.omega_border = omega_border
        obj.curr_omega_idx = curr_omega_idx
        obj.curr_omega_border_idx = curr_omega_border_idx
        obj.p = p
        obj.rar_iter_from_last_sampling = rar_iter_from_last_sampling
        obj.rar_iter_nb = rar_iter_nb
        return obj


@register_pytree_node_class
class CubicMeshPDENonStatio(CubicMeshPDEStatio):
    """
    A class implementing data generator object for non stationary partial
    differential equations. Formally, it extends `CubicMeshPDEStatio`
    to include a temporal batch.


    **Note:** CubicMeshPDENonStatio is jittable. Hence it implements the tree_flatten() and
    tree_unflatten methods.
    """

    def __init__(
        self,
        key,
        n,
        nb,
        nt,
        omega_batch_size,
        omega_border_batch_size,
        temporal_batch_size,
        dim,
        min_pts,
        max_pts,
        tmin,
        tmax,
        method="grid",
        rar_parameters=None,
        n_start=None,
        data_exists=False,
    ):
        r"""
        Parameters
        ----------
        key
            Jax random key to sample new time points and to shuffle batches
        n
            An integer. The number of total :math:`\Omega` points that will be divided in
            batches. Batches are made so that each data point is seen only
            once during 1 epoch.
        nb
            An integer. The total number of points in :math:`\partial\Omega`.
            Can be `None` not to lose performance generating the border
            batch if they are not used
        nt
            An integer. The number of total time points that will be divided in
            batches. Batches are made so that each data point is seen only
            once during 1 epoch.
        omega_batch_size
            An integer. The size of the batch of randomly selected points among
            the `n` points.
        omega_border_batch_size
            An integer. The size of the batch of points randomly selected
            among the `nb` points.
            Can be `None` not to lose performance generating the border
            batch if they are not used
        temporal_batch_size
            An integer. The size of the batch of randomly selected points among
            the `nt` points.
        dim
            An integer. dimension of :math:`\Omega` domain
        min_pts
            A tuple of minimum values of the domain along each dimension. For a sampling
            in `n` dimension, this represents :math:`(x_{1, min}, x_{2,min}, ...,
            x_{n, min})`
        max_pts
            A tuple of maximum values of the domain along each dimension. For a sampling
            in `n` dimension, this represents :math:`(x_{1, max}, x_{2,max}, ...,
            x_{n,max})`
        tmin
            A float. The minimum value of the time domain to consider
        tmax
            A float. The maximum value of the time domain to consider
        method
            Either `grid` or `uniform`, default is `grid`.
            The method that generates the `nt` time points. `grid` means
            regularly spaced points over the domain. `uniform` means uniformly
            sampled points over the domain
        rar_parameters
            Default to None: do not use Residual Adaptative Resampling.
            Otherwise a dictionary with keys. `start_iter`: the iteration at
            which we start the RAR sampling scheme (we first have a burn in
            period). `update_rate`: the number of gradient steps taken between
            each appending of collocation points in the RAR algo.
            `sample_size`: the size of the sample from which we will select new
            collocation points. `selected_sample_size`: the number of selected
            points from the sample to be added to the current collocation
            points.
            __Note:__ that if RAR sampling is chosen it will currently affect both
            self.times and self.omega with the same hyperparameters
            (rar_parameters and n_start)
            "DeepXDE: A deep learning library for solving differential
            equations", L. Lu, SIAM Review, 2021
        n_start
            Defaults to None. The effective size of n used at start time.
            This value must be
            provided when rar_parameters is not None. Otherwise we set internally
            n_start = n and this is hidden from the user.
            In RAR, n_start
            then corresponds to the initial number of points we train the PINN.
            __Note:__ that if RAR sampling is chosen it will currently affect both
            self.times and self.omega with the same hyperparameters
            (rar_parameters and n_start)
        data_exists
            Must be left to `False` when created by the user. Avoids the
            regeneration of :math:`\Omega`, :math:`\partial\Omega` and
            time points at each pytree flattening and unflattening.
        """
        super().__init__(
            key,
            n,
            nb,
            omega_batch_size,
            omega_border_batch_size,
            dim,
            min_pts,
            max_pts,
            method,
            rar_parameters,
            n_start,
            data_exists,
        )
        self.temporal_batch_size = temporal_batch_size
        self.tmin = tmin
        self.tmax = tmax
        self.nt = nt
        if not self.data_exists:
            # Useful when using a lax.scan with pytree
            # Optionally can tell JAX not to re-generate data
            self.curr_time_idx = 0
            self.generate_data_nonstatio()
            self._key, self.times, _ = _reset_batch_idx_and_permute(
                self._get_time_operands()
            )

    def sample_in_time_domain(self, n_samples):
        self._key, subkey = random.split(self._key, 2)
        return random.uniform(subkey, (n_samples,), minval=self.tmin, maxval=self.tmax)

    def _get_time_operands(self):
        return (
            self._key,
            self.times,
            self.curr_time_idx,
            self.temporal_batch_size,
            self.p,
        )

    def generate_data_nonstatio(self):
        r"""
        Construct a complete set of `self.nt` time points according to the
        specified `self.method`. This completes the `super` function
        `generate_data()` which generates :math:`\Omega` and
        :math:`\partial\Omega` points.
        """
        if self.method == "grid":
            self.partial_times = (self.tmax - self.tmin) / self.nt
            self.times = jnp.arange(self.tmin, self.tmax, self.partial_times)
        elif self.method == "uniform":
            self.times = self.sample_in_time_domain(self.nt)
        else:
            raise ValueError("Method " + self.method + " is not implemented.")

    def temporal_batch(self):
        """
        Return a batch of time points. If all the batches have been seen, we
        reshuffle them, otherwise we just return the next unseen batch.
        """
        bstart = self.curr_time_idx
        bend = bstart + self.temporal_batch_size

        # Compute the effective number of used collocation points
        if self.rar_parameters is not None:
            nt_eff = (
                self.n_start
                + self.rar_iter_nb * self.rar_parameters["selected_sample_size"]
            )
        else:
            nt_eff = self.nt

        (self._key, self.times, self.curr_time_idx) = _reset_or_increment(
            bend, nt_eff, self._get_time_operands()
        )

        # commands below are equivalent to
        # return self.times[i:(i+t_batch_size)]
        # but JAX prefer the latter
        return jax.lax.dynamic_slice(
            self.times,
            start_indices=(self.curr_time_idx,),
            slice_sizes=(self.temporal_batch_size,),
        )

    def get_batch(self):
        """
        Generic method to return a batch. Here we call `self.inside_batch()`,
        `self.border_batch()` and `self.temporal_batch()`
        """
        return PDENonStatioBatch(
            inside_batch=self.inside_batch(),
            border_batch=self.border_batch(),
            temporal_batch=self.temporal_batch(),
        )

    def tree_flatten(self):
        children = (
            self._key,
            self.omega,
            self.omega_border,
            self.times,
            self.curr_omega_idx,
            self.curr_omega_border_idx,
            self.curr_time_idx,
            self.min_pts,
            self.max_pts,
            self.tmin,
            self.tmax,
            self.rar_parameters,
            self.p,
            self.rar_iter_from_last_sampling,
            self.rar_iter_nb,
        )
        aux_data = {
            k: vars(self)[k]
            for k in [
                "n",
                "nb",
                "nt",
                "omega_batch_size",
                "omega_border_batch_size",
                "temporal_batch_size",
                "method",
                "dim",
                "n_start",
            ]
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        **Note:** When reconstructing the class, we force ``data_exists=True``
        in order not to re-generate the data at each flattening and
        unflattening that happens e.g. during the gradient descent in the
        optimization process
        """
        (
            key,
            omega,
            omega_border,
            times,
            curr_omega_idx,
            curr_omega_border_idx,
            curr_time_idx,
            min_pts,
            max_pts,
            tmin,
            tmax,
            rar_parameters,
            p,
            rar_iter_from_last_sampling,
            rar_iter_nb,
        ) = children
        obj = cls(
            key=key,
            data_exists=True,
            min_pts=min_pts,
            max_pts=max_pts,
            tmin=tmin,
            tmax=tmax,
            rar_parameters=rar_parameters,
            **aux_data,
        )
        obj.omega = omega
        obj.omega_border = omega_border
        obj.times = times
        obj.curr_omega_idx = curr_omega_idx
        obj.curr_omega_border_idx = curr_omega_border_idx
        obj.curr_time_idx = curr_time_idx
        obj.p = p
        obj.rar_iter_from_last_sampling = rar_iter_from_last_sampling
        obj.rar_iter_nb = rar_iter_nb
        return obj


@register_pytree_node_class
class DataGeneratorParameter:
    """
    A data generator for additional unidimensional parameter(s)
    """

    def __init__(
        self,
        key,
        n,
        param_batch_size,
        param_ranges,
        method="grid",
        data_exists=False,
    ):
        r"""
        Parameters
        ----------
        key
            Jax random key to sample new time points and to shuffle batches
            or a dict of Jax random keys with key entries from param_ranges
        n
            An integer. The number of total points that will be divided in
            batches. Batches are made so that each data point is seen only
            once during 1 epoch.
        param_batch_size
            An integer. The size of the batch of randomly selected points among
            the `n` points.  `param_batch_size` will be the same for all the
            additional batch(es) of parameter(s). `param_batch_size` must be
            equal to `temporal_batch_size` or `omega_batch_size` or the product
            of both whether the present DataGeneratorParameter instance
            complements and ODEBatch, a PDEStatioBatch or a PDENonStatioBatch,
            respectively.
        param_ranges
            A dict. A dict of tuples (min, max), which
            reprensents the range of real numbers where to sample batches (of
            length `param_batch_size` among `n` points).
            The key corresponds to the parameter name. The keys must match the
            keys in `params["eq_params"]`.
            By providing several entries in this dictionary we can sample
            an arbitrary number of parameters.
            __Note__ that we currently only support unidimensional parameters
        method
            Either `grid` or `uniform`, default is `grid`. `grid` means
            regularly spaced points over the domain. `uniform` means uniformly
            sampled points over the domain
        data_exists
            Must be left to `False` when created by the user. Avoids the
            regeneration of :math:`\Omega`, :math:`\partial\Omega` and
            time points at each pytree flattening and unflattening.
        """
        self.data_exists = data_exists
        self.method = method
        if not isinstance(key, dict):
            self._keys = dict(
                zip(param_ranges.keys(), jax.random.split(key, len(param_ranges)))
            )
        else:
            self._keys = key
        self.n = n
        self.param_batch_size = param_batch_size
        self.param_ranges = param_ranges

        if not self.data_exists:
            self.generate_data()
            # The previous call to self.generate_data() has created
            # the dict self.param_n_samples
            self.curr_param_idx = {}
            for k in self.param_ranges.keys():
                self.curr_param_idx[k] = 0
                (
                    self._keys[k],
                    self.param_n_samples[k],
                    _,
                ) = _reset_batch_idx_and_permute(self._get_param_operands(k))

    def generate_data(self):
        # Generate param n samples
        self.param_n_samples = {}
        for k, e in self.param_ranges.items():
            if self.method == "grid":
                xmin, xmax = e[0], e[1]
                self.partial = (xmax - xmin) / self.n
                # shape (n, 1)
                self.param_n_samples[k] = jnp.arange(xmin, xmax, self.partial)[:, None]
            elif self.method == "uniform":
                xmin, xmax = e[0], e[1]
                self._keys[k], subkey = random.split(self._keys[k], 2)
                self.param_n_samples[k] = random.uniform(
                    subkey, shape=(self.n, 1), minval=xmin, maxval=xmax
                )
            else:
                raise ValueError("Method " + self.method + " is not implemented.")

    def _get_param_operands(self, k):
        return (
            self._keys[k],
            self.param_n_samples[k],
            self.curr_param_idx[k],
            self.param_batch_size,
            None,
        )

    def param_batch(self):
        """
        Return a dictionary with batches of parameters
        If all the batches have been seen, we reshuffle them,
        otherwise we just return the next unseen batch.
        """

        def _reset_or_increment_wrapper(param_k, idx_k, key_k):
            return _reset_or_increment(
                idx_k + self.param_batch_size,
                self.n,
                (key_k, param_k, idx_k, self.param_batch_size, None),
            )

        res = jax.tree_util.tree_map(
            _reset_or_increment_wrapper,
            self.param_n_samples,
            self.curr_param_idx,
            self._keys,
        )
        # we must transpose the pytrees because keys are merged in res
        # https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html#transposing-trees
        (
            self._keys,
            self.param_n_samples,
            self.curr_param_idx,
        ) = jax.tree_util.tree_transpose(
            jax.tree_util.tree_structure(self._keys),
            jax.tree_util.tree_structure([0, 0, 0]),
            res,
        )

        return jax.tree_util.tree_map(
            lambda p, q: jax.lax.dynamic_slice(
                p, start_indices=(q, 0), slice_sizes=(self.param_batch_size, 1)
            ),
            self.param_n_samples,
            self.curr_param_idx,
        )

    def get_batch(self):
        """
        Generic method to return a batch
        """
        return self.param_batch()

    def tree_flatten(self):
        children = (
            self._keys,
            self.param_n_samples,
            self.curr_param_idx,
        )
        aux_data = {
            k: vars(self)[k]
            for k in [
                "n",
                "param_batch_size",
                "method",
                "param_ranges",
            ]
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (
            keys,
            param_n_samples,
            curr_param_idx,
        ) = children
        obj = cls(
            key=keys,
            data_exists=True,
            **aux_data,
        )
        obj.param_n_samples = param_n_samples
        obj.curr_param_idx = curr_param_idx
        return obj


@register_pytree_node_class
class DataGeneratorObservations:
    """
    Despite the class name, it is rather a dataloader from user provided
    observations that will be used for the observations loss
    """

    def __init__(
        self,
        key,
        obs_batch_size,
        observed_pinn_in,
        observed_values,
        observed_eq_params=None,
        data_exists=False,
        sharding_device=None,
    ):
        r"""
        Parameters
        ----------
        key
            Jax random key to sample new time points and to shuffle batches
        obs_batch_size
            An integer. The size of the batch of randomly selected observations
            `obs_batch_size` will be the same for all the
            elements of the obs dict. `obs_batch_size` must be
            equal to `temporal_batch_size` or `omega_batch_size` or the product
            of both whether the present DataGeneratorParameter instance
            complements and ODEBatch, a PDEStatioBatch or a PDENonStatioBatch,
            respectively.
        observed_pinn_in
            A jnp.array with 2 dimensions.
            Observed values corresponding to the input of the PINN
            (eg. the time at which we recorded the observations). The first
            dimension must corresponds to the number of observed_values and
            observed_eq_params. The second dimension depends on the input dimension of the PINN, that is `1` for ODE, `n_dim_x` for stationnary PDE and `n_dim_x + 1` for non-stationnary PDE.
        observed_values
            A jnp.array with 2 dimensions.
            Observed values that the PINN should learn to fit. The first dimension must be aligned with observed_pinn_in and the values of observed_eq_params.
        observed_eq_params
            Optional. Default is None. A dict with keys corresponding to the
            parameter name. The keys must match the keys in
            `params["eq_params"]`. The values are jnp.array with 2 dimensions
            with values corresponding to the parameter value for which we also
            have observed_pinn_in and observed_values. Hence the first
            dimension must be aligned with observed_pinn_in and observed_values.
        data_exists
            Must be left to `False` when created by the user. Avoids the
            resetting of curr_idx at each pytree flattening and unflattening.
        sharding_device
            Default None. An optional sharding object to constraint the storage
            of observed inputs, values and parameters. Typically, a
            SingleDeviceSharding(cpu_device) to avoid loading on GPU huge
            datasets of observations. Note that computations for **batches**
            can still be performed on other devices (*e.g.* GPU, TPU or
            any pre-defined Sharding) thanks to the `obs_batch_sharding`
            arguments of `jinns.solve()`. Read the docs for more info.

        """
        if observed_eq_params is None:
            observed_eq_params = {}

        if not data_exists:
            self.observed_eq_params = observed_eq_params.copy()
        else:
            # avoid copying when in flatten/unflatten
            self.observed_eq_params = observed_eq_params

        if observed_pinn_in.shape[0] != observed_values.shape[0]:
            raise ValueError(
                "observed_pinn_in and observed_values must have same first axis"
            )
        for _, v in self.observed_eq_params.items():
            if v.shape[0] != observed_pinn_in.shape[0]:
                raise ValueError(
                    "observed_pinn_in and the values of"
                    " observed_eq_params must have the same first axis"
                )
        if len(observed_pinn_in.shape) == 1:
            observed_pinn_in = observed_pinn_in[:, None]
        if len(observed_pinn_in.shape) > 2:
            raise ValueError("observed_pinn_in must have 2 dimensions")
        if len(observed_values.shape) == 1:
            observed_values = observed_values[:, None]
        if len(observed_values.shape) > 2:
            raise ValueError("observed_values must have 2 dimensions")
        for k, v in self.observed_eq_params.items():
            if len(v.shape) == 1:
                self.observed_eq_params[k] = v[:, None]
            if len(v.shape) > 2:
                raise ValueError(
                    "Each value of observed_eq_params must have 2 dimensions"
                )

        self.n = observed_pinn_in.shape[0]
        self._key = key
        self.obs_batch_size = obs_batch_size

        self.data_exists = data_exists
        if not self.data_exists and sharding_device is not None:
            self.observed_pinn_in = jax.lax.with_sharding_constraint(
                observed_pinn_in, sharding_device
            )
            self.observed_values = jax.lax.with_sharding_constraint(
                observed_values, sharding_device
            )
            self.observed_eq_params = jax.lax.with_sharding_constraint(
                self.observed_eq_params, sharding_device
            )
        else:
            self.observed_pinn_in = observed_pinn_in
            self.observed_values = observed_values

        if not self.data_exists:
            self.curr_idx = 0
            # NOTE for speed and to avoid duplicating data what is really
            # shuffled is a vector of indices
            indices = jnp.arange(self.n)
            if sharding_device is not None:
                self.indices = jax.lax.with_sharding_constraint(
                    indices, sharding_device
                )
            else:
                self.indices = indices
            self._key, self.indices, _ = _reset_batch_idx_and_permute(
                self._get_operands()
            )

    def _get_operands(self):
        return (
            self._key,
            self.indices,
            self.curr_idx,
            self.obs_batch_size,
            None,
        )

    def obs_batch(self):
        """
        Return a dictionary with (keys, values): (pinn_in, a mini batch of pinn
        inputs), (obs, a mini batch of corresponding observations), (eq_params,
        a dictionary with entry names found in `params["eq_params"]` and values
        giving the correspond parameter value for the couple
        (input, observation) mentioned before).
        It can also be a dictionary of dictionaries as described above if
        observed_pinn_in, observed_values, etc. are dictionaries with keys
        representing the PINNs.
        """

        (self._key, self.indices, self.curr_idx) = _reset_or_increment(
            self.curr_idx + self.obs_batch_size, self.n, self._get_operands()
        )

        minib_indices = jax.lax.dynamic_slice(
            self.indices,
            start_indices=(self.curr_idx,),
            slice_sizes=(self.obs_batch_size,),
        )

        obs_batch = {
            "pinn_in": jnp.take(
                self.observed_pinn_in, minib_indices, unique_indices=True, axis=0
            ),
            "val": jnp.take(
                self.observed_values, minib_indices, unique_indices=True, axis=0
            ),
            "eq_params": jax.tree_util.tree_map(
                lambda a: jnp.take(a, minib_indices, unique_indices=True, axis=0),
                self.observed_eq_params,
            ),
        }
        return obs_batch

    def get_batch(self):
        """
        Generic method to return a batch
        """
        return self.obs_batch()

    def tree_flatten(self):
        children = (self._key, self.curr_idx, self.indices)
        aux_data = {
            k: vars(self)[k]
            for k in [
                "obs_batch_size",
                "observed_pinn_in",
                "observed_values",
                "observed_eq_params",
            ]
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (key, curr_idx, indices) = children
        obj = cls(
            key=key,
            data_exists=True,
            obs_batch_size=aux_data["obs_batch_size"],
            observed_pinn_in=aux_data["observed_pinn_in"],
            observed_values=aux_data["observed_values"],
            observed_eq_params=aux_data["observed_eq_params"],
        )
        obj.curr_idx = curr_idx
        obj.indices = indices
        return obj


@register_pytree_node_class
class DataGeneratorObservationsMultiPINNs:
    """
    Despite the class name, it is rather a dataloader from user provided
    observations that will be used for the observations loss.
    This is the DataGenerator to use when dealing with multiple PINNs
    (`u_dict`) in SystemLossODE/SystemLossPDE

    Technically, the constraint on the observations in SystemLossXDE are
    applied in `constraints_system_loss_apply` and in this case the
    batch.obs_batch_dict is a dict of obs_batch_dict over which the tree_map
    applies (we select the obs_batch_dict corresponding to its `u_dict` entry)
    """

    def __init__(
        self,
        obs_batch_size,
        observed_pinn_in_dict,
        observed_values_dict,
        observed_eq_params_dict=None,
        data_gen_obs_exists=False,
        key=None,
    ):
        r"""
        Parameters
        ----------
        obs_batch_size
            An integer. The size of the batch of randomly selected observations
            `obs_batch_size` will be the same for all the
            elements of the obs dict. `obs_batch_size` must be
            equal to `temporal_batch_size` or `omega_batch_size` or the product
            of both whether the present DataGeneratorParameter instance
            complements and ODEBatch, a PDEStatioBatch or a PDENonStatioBatch,
            respectively.
        observed_pinn_in_dict
            A dict of observed_pinn_in as defined in DataGeneratorObservations.
            Keys must be that of `u_dict`.
            If no observation exists for a particular entry of `u_dict` the
            corresponding key must still exist in observed_pinn_in_dict with
            value None
        observed_values_dict
            A dict of observed_values as defined in DataGeneratorObservations.
            Keys must be that of `u_dict`.
            If no observation exists for a particular entry of `u_dict` the
            corresponding key must still exist in observed_values_dict with
            value None
        observed_eq_params_dict
            A dict of observed_eq_params as defined in DataGeneratorObservations.
            Keys must be that of `u_dict`.
            If no observation exists for a particular entry of `u_dict` the
            corresponding key must still exist in observed_eq_params_dict with
            value None
        data_gen_obs_exists
            Must be left to `False` when created by the user. Avoids the
            regeneration the subclasses DataGeneratorObservations
            at each pytree flattening and unflattening.
        key
            Jax random key to sample new time points and to shuffle batches.
            Optional if data_gen_obs_exists is True
        """
        if (
            observed_pinn_in_dict is None or observed_values_dict is None
        ) and not data_gen_obs_exists:
            raise ValueError(
                "observed_pinn_in_dict and observed_values_dict "
                "must be provided with data_gen_obs_exists is False"
            )
        self.obs_batch_size = obs_batch_size
        self.data_gen_obs_exists = data_gen_obs_exists

        if not self.data_gen_obs_exists:
            if observed_pinn_in_dict.keys() != observed_values_dict.keys():
                raise ValueError(
                    "Keys must be the same in observed_pinn_in_dict"
                    " and observed_values_dict"
                )
            if (
                observed_eq_params_dict is not None
                and observed_pinn_in_dict.keys() != observed_eq_params_dict.keys()
            ):
                raise ValueError(
                    "Keys must be the same in observed_eq_params_dict"
                    " and observed_pinn_in_dict and observed_values_dict"
                )
            if observed_eq_params_dict is None:
                observed_eq_params_dict = {
                    k: None for k in observed_pinn_in_dict.keys()
                }

            keys = dict(
                zip(
                    observed_pinn_in_dict.keys(),
                    jax.random.split(key, len(observed_pinn_in_dict)),
                )
            )
            self.data_gen_obs = jax.tree_util.tree_map(
                lambda k, pinn_in, val, eq_params: (
                    DataGeneratorObservations(
                        k, obs_batch_size, pinn_in, val, eq_params
                    )
                    if pinn_in is not None
                    else None
                ),
                keys,
                observed_pinn_in_dict,
                observed_values_dict,
                observed_eq_params_dict,
            )

    def obs_batch(self):
        """
        Returns a dictionary of DataGeneratorObservations.obs_batch with keys
        from `u_dict`
        """
        return jax.tree_util.tree_map(
            lambda a: a.get_batch() if a is not None else {},
            self.data_gen_obs,
            is_leaf=lambda x: isinstance(x, DataGeneratorObservations),
        )  # note the is_leaf note to traverse the DataGeneratorObservations and
        # thus to be able to call the method on the element(s) of
        # self.data_gen_obs which are not None

    def get_batch(self):
        """
        Generic method to return a batch
        """
        return self.obs_batch()

    def tree_flatten(self):
        # because a dict with "str" keys cannot go in the children (jittable)
        # attributes, we need to separate it in two and recreate the zip in the
        # tree_unflatten
        children = self.data_gen_obs.values()
        aux_data = {
            "obs_batch_size": self.obs_batch_size,
            "data_gen_obs_keys": self.data_gen_obs.keys(),
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (data_gen_obs_values) = children
        obj = cls(
            observed_pinn_in_dict=None,
            observed_values_dict=None,
            data_gen_obs_exists=True,
            obs_batch_size=aux_data["obs_batch_size"],
        )
        obj.data_gen_obs = dict(zip(aux_data["data_gen_obs_keys"], data_gen_obs_values))
        return obj

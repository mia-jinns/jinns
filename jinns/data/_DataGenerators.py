#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author: Nicolas Jouvin
# @email: nicolas.jouvin@inrae.fr

import jax.numpy as jnp
from jax import random, vmap
from jax.tree_util import register_pytree_node_class
import jax.lax
import warnings


# utility function for jax.lax.cond in *_batch() method
def _reset_batch_idx_and_permute(operands):
    key, domain, curr_idx, _ = operands
    # resetting counter
    curr_idx = 0
    # reshuffling
    key, subkey = random.split(key)
    domain = random.permutation(subkey, domain, axis=0, independent=False)

    # return updated
    return (key, domain, curr_idx)


def _increment_batch_idx(operands):
    key, domain, curr_idx, batch_size = operands
    # simply increases counter and get the batch
    curr_idx += batch_size
    return (key, domain, curr_idx)


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
        if not self.data_exists:
            # Useful when using a lax.scan with pytree
            # Optionally can tell JAX not to re-generate data
            self.curr_time_idx = 0
            self.generate_time_data()
            self._key, self.times, _ = _reset_batch_idx_and_permute(
                (self._key, self.times, self.curr_time_idx, None)
            )

    def generate_time_data(self):
        """
        Construct a complete set of `self.nt` time points according to the
        specified `self.method`
        """
        if self.method == "grid":
            self.partial_times = (self.tmax - self.tmin) / self.nt
            self.times = jnp.arange(self.tmin, self.tmax, self.partial_times)
        elif self.method == "uniform":
            self._key, subkey = random.split(self._key, 2)
            self.times = random.uniform(
                subkey, (self.nt,), minval=self.tmin, maxval=self.tmax
            )

    def temporal_batch(self):
        """
        Return a batch of time points. If all the batches have been seen, we
        reshuffle them, otherwise we just return the next unseen batch.
        """
        bstart = self.curr_time_idx
        bend = bstart + self.temporal_batch_size

        (self._key, self.times, self.curr_time_idx) = jax.lax.cond(
            bend > self.nt,
            _reset_batch_idx_and_permute,
            _increment_batch_idx,
            (
                self._key,
                self.times,
                self.curr_time_idx,
                self.temporal_batch_size,
            ),
        )

        # commands below are equivalent to
        # return self.times[i:(i+t_batch_size)]
        return jax.lax.dynamic_slice(
            self.times,
            start_indices=(self.curr_time_idx,),
            slice_sizes=(self.temporal_batch_size,),
        )

    def get_batch(self):
        """
        Generic method to return a batch. Here we call `self.temporal_batch()`
        """
        return self.temporal_batch()

    def tree_flatten(self):
        children = (
            self._key,
            self.times,
            self.curr_time_idx,
            self.tmin,
            self.tmax,
        )  # arrays / dynamic values
        aux_data = {
            k: vars(self)[k]
            for k in [
                "nt",
                "temporal_batch_size",
                "method",
            ]
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
        ) = children
        obj = cls(
            key=key,
            data_exists=True,
            tmin=tmin,
            tmax=tmax,
            **aux_data,
        )
        obj.times = times
        obj.curr_time_idx = curr_time_idx
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
        data_exists=False,
    ):
        """
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
            # number of border point must be a multiple of 2xd (the # of faces
            # of a d-dimensional cube)
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
                (self._key, self.omega, self.curr_omega_idx, None)
            )
            if self.omega_border is not None:
                self._key, self.omega_border, _ = _reset_batch_idx_and_permute(
                    (
                        self._key,
                        self.omega_border,
                        self.curr_omega_border_idx,
                        None,
                    )
                )

    def generate_data(self):
        """
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
            if self.dim == 1:
                xmin, xmax = self.min_pts[0], self.max_pts[0]
                self._key, subkey = random.split(self._key, 2)
                self.omega = random.uniform(
                    subkey, shape=(self.n, 1), minval=xmin, maxval=xmax
                )
            else:
                keys = random.split(self._key, self.dim + 1)
                self._key = keys[0]
                self.omega = jnp.concatenate(
                    [
                        random.uniform(
                            keys[i + 1],
                            (self.n, 1),
                            minval=self.min_pts[i],
                            maxval=self.max_pts[i],
                        )
                        for i in range(self.dim)
                    ],
                    axis=-1,
                )
        else:
            raise ValueError("Method " + self.method + " is not implemented.")

        # Generate border of omega
        if self.omega_border_batch_size is None:
            self.omega_border = None
        elif self.dim == 1:
            xmin = self.min_pts[0]
            xmax = self.max_pts[0]
            self.omega_border = jnp.array([xmin, xmax]).astype(float)
        elif self.dim == 2:
            # currently hard-coded the 4 edges for d==2
            # TODO : find a general & efficient way to sample from the border
            # (facets) of the hypercube in general dim.
            self.omega_border = jnp.empty(shape=(1, 1))

            facet_n = self.nb // (2 * self.dim)
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
            self.omega_border = jnp.stack([xmin, xmax, ymin, ymax], axis=-1)
        else:
            raise NotImplementedError(
                f"Generation of the border of a cube in dimension > 2 is not implemented yet. You are asking for generation in dimension d={self.dim}."
            )

    def inside_batch(self):
        """
        Return a batch of points in :math:`\Omega`.
        If all the batches have been seen, we reshuffle them,
        otherwise we just return the next unseen batch.
        """
        bstart = self.curr_omega_idx
        bend = bstart + self.omega_batch_size

        (self._key, self.omega, self.curr_omega_idx) = jax.lax.cond(
            bend > self.n,
            _reset_batch_idx_and_permute,
            _increment_batch_idx,
            (
                self._key,
                self.omega,
                self.curr_omega_idx,
                self.omega_batch_size,
            ),
        )

        # commands below are equivalent to
        # return self.omega[i:(i+batch_size), 0:dim]
        return jax.lax.dynamic_slice(
            self.omega,
            start_indices=(self.curr_omega_idx, 0),
            slice_sizes=(self.omega_batch_size, self.dim),
        )

    def border_batch(self):
        """
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
        elif self.dim == 1:
            # 1-D case, no randomness : we always return the whole omega border,
            # i.e. (1, 1, 2) shape jnp.array([[[xmin], [xmax]]]).
            return self.omega_border[None, None]  # shape is (1, 1, 2)
        else:
            bstart = self.curr_omega_border_idx
            bend = bstart + self.omega_border_batch_size
            # update curr_omega_idx or/and omega when end of batch is reached
            # jax.lax.cond is <=> to an if statment but JITable.
            (
                self._key,
                self.omega_border,
                self.curr_omega_border_idx,
            ) = jax.lax.cond(
                bend > self.nb,
                _reset_batch_idx_and_permute,  # true_fun
                _increment_batch_idx,  # false_fun
                (
                    self._key,
                    self.omega_border,
                    self.curr_omega_border_idx,
                    self.omega_border_batch_size,
                ),  # arguments
            )

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
        return self.inside_batch(), self.border_batch()

    def tree_flatten(self):
        children = (
            self._key,
            self.omega,
            self.omega_border,
            self.curr_omega_idx,
            self.curr_omega_border_idx,
            self.min_pts,
            self.max_pts,
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
        ) = children
        # force data_exists=True here in order not to re-generate the data
        # at each iteration of lax.scan
        obj = cls(
            key=key,
            data_exists=True,
            min_pts=min_pts,
            max_pts=max_pts,
            **aux_data,
        )
        obj.omega = omega
        obj.omega_border = omega_border
        obj.curr_omega_idx = curr_omega_idx
        obj.curr_omega_border_idx = curr_omega_border_idx
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
        data_exists=False,
    ):
        """
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
                (self._key, self.times, self.curr_time_idx, None)
            )

    def temporal_batch(self):
        """
        Return a batch of time points. If all the batches have been seen, we
        reshuffle them, otherwise we just return the next unseen batch.
        """
        bstart = self.curr_time_idx
        bend = bstart + self.temporal_batch_size

        (self._key, self.times, self.curr_time_idx) = jax.lax.cond(
            bend > self.nt,
            _reset_batch_idx_and_permute,
            _increment_batch_idx,
            (
                self._key,
                self.times,
                self.curr_time_idx,
                self.temporal_batch_size,
            ),
        )

        # commands below are equivalent to
        # return self.times[i:(i+t_batch_size)]
        # but JAX prefer the latter
        return jax.lax.dynamic_slice(
            self.times,
            start_indices=(self.curr_time_idx,),
            slice_sizes=(self.temporal_batch_size,),
        )

    def generate_data_nonstatio(self):
        """
        Construct a complete set of `self.nt` time points according to the
        specified `self.method`. This completes the `super` function
        `generate_data()` which generates :math:`\Omega` and
        :math:`\partial\Omega` points.
        """
        if self.method == "grid":
            self.partial_times = (self.tmax - self.tmin) / self.nt
            self.times = jnp.arange(self.tmin, self.tmax, self.partial_times)
        elif self.method == "uniform":
            self._key, subkey = random.split(self._key, 2)
            self.times = random.uniform(
                subkey, (self.nt,), minval=self.tmin, maxval=self.tmax
            )

    def get_batch(self):
        """
        Generic method to return a batch. Here we call `self.inside_batch()`,
        `self.border_batch()` and `self.temporal_batch()`
        """
        return (
            self.inside_batch(),
            self.border_batch(),
            self.temporal_batch(),
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
        ) = children
        obj = cls(
            key=key,
            data_exists=True,
            min_pts=min_pts,
            max_pts=max_pts,
            tmin=tmin,
            tmax=tmax,
            **aux_data,
        )
        obj.omega = omega
        obj.omega_border = omega_border
        obj.times = times
        obj.curr_omega_idx = curr_omega_idx
        obj.curr_omega_border_idx = curr_omega_border_idx
        obj.curr_time_idx = curr_time_idx
        return obj

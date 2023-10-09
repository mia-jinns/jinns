import jax
import jax.numpy as jnp
from jax.scipy.stats import norm, gamma
import matplotlib.pyplot as plt
from reaction_diffusion_solver import laplacian, SpatialDiscretisation, diffrax_solver
import diffrax

alpha = 1e4


def sample_model(rng_key, u_sol, times, omegas):
    nt = times.shape[0]
    nomega = len(omegas)
    n = nt * nomega

    key, subkey = jax.random.split(rng_key, 2)
    intensities = []
    for t in times:
        for omega in omegas:
            intensities.append(alpha * _integrate_on_omega_at_time_t(u_sol, omega, t))
    intensities = jnp.array(intensities)
    y = jax.random.poisson(subkey, lam=intensities, shape=(n,))
    return {"omegas": omegas * nt, "times": jnp.repeat(times, nomega), "y": y}


def log_prior(theta):
    r1, r2, r3, r4 = theta["rs"]
    val = jax.lax.cond(
        jnp.all(
            jnp.array(
                [
                    1e-2 <= theta["D"],
                    theta["D"] <= 1,
                    0.1 <= theta["gamma"],
                    theta["gamma"] <= 10,
                    -10 <= r1,
                    r1 <= 10,
                    -10 <= r2,
                    r2 <= 10,
                    -10 <= r3,
                    r3 <= 10,
                    -10 <= r4,
                    r4 <= 10,
                ]
            )
        ),
        lambda _: jnp.array(
            0.0
        ),  # no need to normalize cause ratio of priors in MHasting
        lambda _: -jnp.inf,
        None,
    )
    return val


def log_proposal(x, x_cond):
    r1, r2, r3, r4 = x["rs"]
    r1_cond, r2_cond, r3_cond, r4_cond = x_cond["rs"]
    return (
        jnp.log(gamma.pdf(x["D"], 1e-2, x_cond["D"] / 1e-2))
        + jnp.log(gamma.pdf(x["gamma"], 1e-2, x_cond["gamma"] / 1e-2))
        + jnp.log(norm.pdf(r1, r1_cond, jnp.sqrt(0.05)))
        + jnp.log(norm.pdf(r2, r2_cond, jnp.sqrt(0.05)))
        + jnp.log(norm.pdf(r3, r3_cond, jnp.sqrt(0.05)))
        + jnp.log(norm.pdf(r4, r4_cond, jnp.sqrt(0.05)))
    )


def sample_proposal(key, x_cond):
    subkey1, subkey2, subkey3, subkey4, subkey5, subkey6 = jax.random.split(key, 6)
    r1_cond, r2_cond, r3_cond, r4_cond = x_cond["rs"]
    r1 = jax.random.normal(subkey3) * jnp.sqrt(0.05) + r1_cond
    r2 = jax.random.normal(subkey4) * jnp.sqrt(0.05) + r2_cond
    r3 = jax.random.normal(subkey5) * jnp.sqrt(0.05) + r3_cond
    r4 = jax.random.normal(subkey6) * jnp.sqrt(0.05) + r4_cond
    return {
        "D": jax.random.gamma(subkey1, 1e-2) * (x_cond["D"] / 1e-2),
        "gamma": jax.random.gamma(subkey2, 1e-2) * (x_cond["gamma"] / 1e-2),
        "rs": jnp.hstack([r1, r2, r3, r4]),
    }


def _integrate_on_omega_at_time_t(u_sol, omega, t):
    """WARNING: If omega goes ouside the border this function does not work
    since `surface` is not vol(omega \cap image) anymore.
    """
    nx, ny = u_sol.ys.vals.shape[1:3]
    center, radius = omega
    # uncorrect shape : X, Y = jnp.ogrid[u_sol.ys.xmin:u_sol.ys.xmax:u_sol.ys.δx, u_sol.ys.ymin:u_sol.ys.ymax:u_sol.ys.δy]
    X = jnp.linspace(u_sol.ys.xmin, u_sol.ys.xmax, nx).reshape((nx, 1))
    Y = jnp.linspace(u_sol.ys.ymin, u_sol.ys.ymax, ny).reshape((1, ny))

    mask = (X - center[0]) ** 2 + (Y - center[1]) ** 2 <= radius**2

    surface = jnp.pi * (radius**2)
    idx_t = jnp.argwhere(u_sol.ts == t, size=1)
    vals_t = u_sol.ys.vals[idx_t].squeeze()
    return jnp.where(mask, vals_t, 0).mean() * surface


def loglikelihood_func(simu, u_sol):
    llhood = 0
    for omega, t, y in zip(simu["omegas"], simu["times"], simu["y"]):
        int_u = _integrate_on_omega_at_time_t(u_sol, omega, t)
        llhood += jax.scipy.stats.poisson.logpmf(y, alpha * int_u)
    return llhood


def vanilla_MH(
    key,
    simu,
    log_prior,
    log_llkh,
    sample_proposal,
    n_iter,
    eq_params_init,
    pde_control={},
):
    # xmin, xmax = pde_control["xboundary"]
    # ymin, ymax = pde_control["yboundary"]
    # nx = pde_control["nx"]
    # ny = pde_control["ny"]

    def MH_step(carry, i):
        jax.debug.print("{i}", i=i)
        key, eq_params = carry
        key, subkey1, subkey2 = jax.random.split(key, 3)

        u_sol = diffrax_solver(eq_params, pde_control)
        eq_params_proposal = sample_proposal(subkey1, eq_params)

        jax.debug.print("{x}", x=log_prior(eq_params))
        jax.debug.print(
            "{a} {b} {c}",
            a=log_llkh(simu, u_sol),
            b=log_prior(eq_params_proposal),
            c=log_proposal(eq_params, eq_params_proposal),
        )
        delta = jnp.min(
            jnp.array(
                [
                    1,
                    jnp.exp(
                        (
                            log_llkh(simu, u_sol)
                            + log_prior(eq_params_proposal)
                            + log_proposal(eq_params, eq_params_proposal)
                        )
                        - (
                            log_llkh(simu, u_sol)
                            + log_prior(eq_params)
                            + log_proposal(eq_params_proposal, eq_params)
                        )
                    ),
                ]
            )
        )

        u = jax.random.uniform(subkey2)

        jax.debug.print("{r} {u} {d}", r=u < delta, u=u, d=delta)
        eq_params = jax.lax.cond(
            u < delta,
            lambda operands: operands[0],
            lambda operands: operands[1],
            (eq_params_proposal, eq_params),
        )

        return (key, eq_params), jnp.array(
            jnp.hstack([eq_params["D"], eq_params["gamma"], eq_params["rs"]])
        )

    carry_final, list_eq_params = jax.lax.scan(
        MH_step, (key, eq_params_init), jnp.arange(n_iter)
    )

    return list_eq_params


if __name__ == "__main__":
    key = jax.random.PRNGKey(1)
    n_iter = 2

    times = jnp.arange(0, 10) * 0.4 + 0.4
    omegas = [
        (jnp.array([0.1, 0.1]), 0.05),
        (jnp.array([0.5, 0.5]), 0.05),
        (jnp.array([0.2, 0.7]), 0.1),
    ]

    # --- PDE solution for true parameters
    r1 = 4
    r4 = -4
    r2 = 0
    r3 = 2
    true_params = {
        "D": jnp.array(0.05),
        "gamma": jnp.array(1.0),
        "rs": jnp.array([r1, r2, r3, r4]),
    }

    # Spatial discretisation
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    nx, ny = 70, 70

    # init condition
    mu_init = jnp.array([0.70, 0.15])

    def gauss_ic(xy):
        return jnp.exp(-jnp.linalg.norm(xy - mu_init))

    y0 = SpatialDiscretisation.discretise_fn(xmin, xmax, ymin, ymax, nx, ny, gauss_ic)

    # Temporal discretisation
    t0 = 0
    t_final = 2
    δt = 0.0001
    saveat = diffrax.SaveAt(ts=jnp.linspace(t0, t_final, 30))

    # Tolerances for non-stiff problems
    rtol = 1e-3
    atol = 1e-6
    # Tolerances for stiff problems (along with using float64)
    # rtol = 1e-7
    # atol = 1e-9
    stepsize_controller = diffrax.PIDController(
        pcoeff=0.3, icoeff=0.4, rtol=rtol, atol=atol, dtmax=0.001
    )

    solver = diffrax.Tsit5()
    max_steps = int(1e6)
    saveat = (times,)

    pde_control = {
        "xboundary": (xmin, xmax),
        "yboundary": (ymin, ymax),
        "nx": nx,
        "ny": ny,
        "ode_hyperparams": {
            "t0": 0,
            "t1": times.max(),
            "dt0": None,
            "y0": y0,
            "saveat": diffrax.SaveAt(ts=times),
            "stepsize_controller": stepsize_controller,
            "max_steps": max_steps,
            "solver": diffrax.Tsit5(),
        },
    }

    u_sol_true = diffrax_solver(true_params, pde_control)

    # --- Statistical simulation according to true parameters
    simu = sample_model(key, u_sol=u_sol_true, times=times, omegas=omegas)
    simu["y"]
    # --- Init MH algo
    eq_params_init = {
        "D": jnp.array(0.05),
        "gamma": jnp.array(1.0),
        "rs": jnp.array([r1, r2, r3, r4]),
    }

    list_x = vanilla_MH(
        key,
        simu,
        log_prior,
        loglikelihood_func,
        sample_proposal,
        n_iter,
        eq_params_init,
        pde_control,
    )

    # Y, X = jnp.ogrid[:5, :5]
    # dist_from_center = jnp.sqrt((X - 2)**2 + (Y-2)**2)
    # print(X.shape, Y.shape,dist_from_center.shape)

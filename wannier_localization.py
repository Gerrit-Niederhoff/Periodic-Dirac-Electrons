"""
wannier_localization.py

Native NumPy/JAX Wannier localization for an isolated two-band subspace.

Expected Bloch-state array:
    psi.shape == (Nkx, Nky, 2*NG, 2)

with
    psi[..., 2*i:2*i+2, :]
corresponding to reciprocal lattice vector lat[i] and the two pseudospins.

Main entry point
----------------
wannierize(...)

It:
    1. builds the two trial projections,
    2. builds the six nearest-neighbour overlap matrices,
    3. constructs the projection gauge,
    4. minimizes Omega_D + Omega_OD,
    5. applies the optimal U(2) gauge to psi,
    6. saves the result to a compressed .npz file.

Dependencies:
    numpy
    scipy
    jax
"""

import numpy as np
from scipy.special import gamma, hyp2f1
from jax.scipy.linalg import expm
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


DEFAULT_Q2 = np.array([0.0, 1.0])
DEFAULT_Q1 = np.array([np.sqrt(3.0) / 2.0, 0.5])


# ---------------------------------------------------------------------
# Trial orbitals / projections
# ---------------------------------------------------------------------


def radial_ft(Q, m, a=0.18, l=7):
    """
    Evaluate

        integral_0^inf dr r^(l+1) exp(-r/a) J_m(Q r)

    for integer m >= 0.
    """
    if m < 0:
        raise ValueError("radial_ft requires m >= 0")

    Q = np.asarray(Q, dtype=float)
    beta = 1.0 / a
    mu = l + 1

    prefactor = (
        (Q / (2.0 * beta)) ** m * beta ** (-mu - 1) * gamma(mu + m + 1) / gamma(m + 1)
    )

    return prefactor * hyp2f1(
        (mu + m + 1) / 2.0,
        (mu + m + 2) / 2.0,
        m + 1,
        -((Q / beta) ** 2),
    )


def angular_ft(Qx, Qy, m, a=0.18, l=7):
    """
    Fourier transform of

        r^l exp(-r/a) exp(i m phi)

    with convention exp(-i Q.r).
    """
    m = int(m)
    am = abs(m)

    Q = np.hypot(Qx, Qy)
    theta = np.arctan2(Qy, Qx)

    return (
        2.0 * np.pi * (-1j) ** am * np.exp(1j * m * theta) * radial_ft(Q, am, a=a, l=l)
    )


def trial_coefficients(k, lat, a=0.18, l=7, normalize=True):
    """
    Reciprocal-space coefficients of the two trial spinors.

    Trial 0 (m = +4.5):
        (exp(+i 4 phi), exp(+i 5 phi))

    Trial 1 (m = -4.5):
        (exp(-i 5 phi), exp(-i 4 phi))

    Returns
    -------
    g : ndarray, shape (2*NG, 2)
    """
    k = np.asarray(k, dtype=float)
    lat = np.asarray(lat, dtype=float)

    Q = lat + k[None, :]
    qx, qy = Q[:, 0], Q[:, 1]

    g = np.empty((2 * len(lat), 2), dtype=np.complex128)

    g[0::2, 0] = angular_ft(qx, qy, +4, a=a, l=l)
    g[1::2, 0] = angular_ft(qx, qy, +5, a=a, l=l)

    g[0::2, 1] = angular_ft(qx, qy, -5, a=a, l=l)
    g[1::2, 1] = angular_ft(qx, qy, -4, a=a, l=l)

    # Required by the coefficient convention diagnosed from the
    # Gamma-point test.
    # g = g.conj()

    if normalize:
        norms = np.linalg.norm(g, axis=0)
        if np.any(norms < 1e-14):
            raise RuntimeError(f"Trial orbital has vanishing norm at k={k}")
        g /= norms[None, :]

    return g


def compute_projections(
    psi,
    lat,
    q1=DEFAULT_Q1,
    q2=DEFAULT_Q2,
    a=0.18,
    l=7,
):
    """
    Compute

        A[ix,iy,n,m] = <g_n(k) | u_m(k)>.

    Parameters
    ----------
    psi : ndarray
        Shape (Nkx, Nky, 2*NG, 2).
    lat : ndarray/list
        Shape (NG, 2).

    Returns
    -------
    A : ndarray
        Shape (Nkx, Nky, 2, 2).
    """
    psi = np.asarray(psi)
    lat = np.asarray(lat)

    Nkx, Nky, dim, nb = psi.shape

    if nb != 2:
        raise ValueError("This implementation expects exactly two bands.")
    if dim != 2 * len(lat):
        raise ValueError("psi.shape[2] must equal 2*len(lat).")

    A = np.empty((Nkx, Nky, 2, 2), dtype=np.complex128)

    for ix in range(Nkx):
        for iy in range(Nky):
            k = (ix / Nkx - 0.5) * q1 + (iy / Nky - 0.5) * q2
            g = trial_coefficients(k, lat, a=a, l=l)
            A[ix, iy] = g.conj().T @ psi[ix, iy]

    return A


# ---------------------------------------------------------------------
# Neighbour overlaps
# ---------------------------------------------------------------------


def _neighbor_steps_and_vectors(q1, q2, Nkx, Nky):
    if Nkx != Nky:
        raise ValueError("Current six-neighbour implementation assumes Nkx == Nky.")

    steps = np.array(
        [
            [1, 0],
            [-1, 0],
            [0, 1],
            [0, -1],
            [-1, 1],
            [1, -1],
        ],
        dtype=int,
    )

    bvec = (
        np.array(
            [
                q1,
                -q1,
                q2,
                -q2,
                q2 - q1,
                q1 - q2,
            ],
            dtype=float,
        )
        / Nkx
    )

    # For this triangular reciprocal lattice:
    # sum_b w b_alpha b_beta = delta_alpha_beta
    weight = Nkx**2 / 3.0

    return steps, bvec, weight


def compute_overlaps(psi, zoneshift, q1, q2, cutoff):
    Nkx, Nky = psi.shape[:2]

    steps = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (-1, 1),
        (1, -1),
    ]

    M = np.empty((Nkx, Nky, 6, 2, 2), dtype=complex)

    for ib, (dx, dy) in enumerate(steps):
        nbr = np.roll(
            psi,
            shift=(-dx, -dy),
            axis=(0, 1),
        ).copy()

        # x boundary, shifted as one batch
        if dx == 1:
            iy = np.arange(Nky)
            nbr[-1, iy] = zoneshift(
                psi[0, (iy + dy) % Nky],
                q1,
                cutoff=cutoff,
            )

        elif dx == -1:
            iy = np.arange(Nky)
            nbr[0, iy] = zoneshift(
                psi[-1, (iy + dy) % Nky],
                -q1,
                cutoff=cutoff,
            )

        # y boundary, shifted as one batch
        if dy == 1:
            ix = np.arange(Nkx)
            nbr[ix, -1] = zoneshift(
                psi[(ix + dx) % Nkx, 0],
                q2,
                cutoff=cutoff,
            )

        elif dy == -1:
            ix = np.arange(Nkx)
            nbr[ix, 0] = zoneshift(
                psi[(ix + dx) % Nkx, -1],
                -q2,
                cutoff=cutoff,
            )

        # Corner where both boundaries were crossed:
        # it needs the combined reciprocal translation.
        if dx == 1 and dy == -1:
            nbr[-1, 0] = zoneshift(
                psi[0, -1],
                q1 - q2,
                cutoff=cutoff,
            )

        elif dx == -1 and dy == 1:
            nbr[0, -1] = zoneshift(
                psi[-1, 0],
                -q1 + q2,
                cutoff=cutoff,
            )

        M[:, :, ib] = np.einsum(
            "...am,...an->...mn",
            psi.conj(),
            nbr,
            optimize=True,
        )

    return M


# ---------------------------------------------------------------------
# Initial projection gauge
# ---------------------------------------------------------------------


def initial_gauge(A):
    """
    Construct the closest unitary gauge from the projection matrices.

    Input convention:
        A[...,n,m] = <g_n | u_m>.
    """
    B = A.conj().swapaxes(-1, -2)  # <u_m | g_n>
    L, _, Rh = np.linalg.svd(B)
    return L @ Rh


# ---------------------------------------------------------------------
# JAX U(2) parametrization and spread minimization
# ---------------------------------------------------------------------


def _exp_u2(p):
    """
    exp(i H), where

        H = p0 I + px sigma_x + py sigma_y + pz sigma_z

    Numerically safe at p = 0.
    """

    p0 = p[..., 0]
    px = p[..., 1]
    py = p[..., 2]
    pz = p[..., 3]

    H = jnp.zeros(p.shape[:-1] + (2, 2), dtype=jnp.complex128)

    H = H.at[..., 0, 0].set(p0 + pz)
    H = H.at[..., 1, 1].set(p0 - pz)

    H = H.at[..., 0, 1].set(px - 1j * py)
    H = H.at[..., 1, 0].set(px + 1j * py)

    # expm itself acts on a single 2x2 matrix,
    # so vectorize over both k-grid dimensions.
    expm_2d = jax.vmap(jax.vmap(expm, in_axes=0), in_axes=0)

    return expm_2d(1j * H)


def localize(
    M,
    U0,
    q1=DEFAULT_Q1,
    q2=DEFAULT_Q2,
    n_iter=2000,
    lr=2e-3,
    print_every=100,
):
    """
    Minimize Omega_D + Omega_OD using Adam.

    Returns
    -------
    U : ndarray
        Shape (Nkx, Nky, 2, 2), total gauge relative to input psi.
    centres : ndarray
        Shape (2, 2), Cartesian Wannier centers.
    omega_D, omega_OD : floats
    """
    M = np.asarray(M)
    U0 = np.asarray(U0)

    Nkx, Nky = M.shape[:2]
    steps, bvec, weight = _neighbor_steps_and_vectors(
        np.asarray(q1, float),
        np.asarray(q2, float),
        Nkx,
        Nky,
    )

    Mj = jnp.asarray(M)
    U0j = jnp.asarray(U0)
    bj = jnp.asarray(bvec)

    def gauge(params):
        return jnp.einsum(
            "...ij,...jk->...ik",
            U0j,
            _exp_u2(params),
        )

    def transformed_overlaps(params):
        U = gauge(params)
        blocks = []

        for ib, (dx, dy) in enumerate(steps):
            Un = jnp.roll(
                U,
                shift=(-int(dx), -int(dy)),
                axis=(0, 1),
            )

            blocks.append(
                jnp.einsum(
                    "...mi,...mn,...nj->...ij",
                    jnp.conj(U),
                    Mj[:, :, ib],
                    Un,
                )
            )

        return jnp.stack(blocks, axis=2)

    def spread_parts(params):
        Mt = transformed_overlaps(params)
        diag = jnp.diagonal(Mt, axis1=-2, axis2=-1)

        theta = jnp.angle(diag)
        Nkpts = Nkx * Nky

        centres = -weight / Nkpts * jnp.einsum("xybn,ba->na", theta, bj)

        br = jnp.einsum("ba,na->bn", bj, centres)
        q = theta + br[None, None, :, :]

        omega_D = weight / Nkpts * jnp.sum(q * q)

        omega_OD = (
            weight
            / Nkpts
            * jnp.sum(jnp.abs(Mt[..., 0, 1]) ** 2 + jnp.abs(Mt[..., 1, 0]) ** 2)
        )

        return omega_D, omega_OD, centres

    def loss(params):
        od, ood, _ = spread_parts(params)
        return od + ood

    loss_and_grad = jax.jit(jax.value_and_grad(loss))

    params = jnp.zeros((Nkx, Nky, 4), dtype=jnp.float64)
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for it in range(1, n_iter + 1):
        val, grad = loss_and_grad(params)

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad * grad

        mh = m / (1.0 - beta1**it)
        vh = v / (1.0 - beta2**it)

        params = params - lr * mh / (jnp.sqrt(vh) + eps)

        if print_every and (it == 1 or it % print_every == 0):
            od, ood, _ = spread_parts(params)
            print(
                f"{it:5d}  "
                f"Omega~= {float(od + ood):.10f}  "
                f"OD={float(od):.6f}  "
                f"OOD={float(ood):.6f}"
            )

    U = np.asarray(gauge(params))
    od, ood, centres = spread_parts(params)

    return U, np.asarray(centres), float(od), float(ood)


# ---------------------------------------------------------------------
# One-call interface
# ---------------------------------------------------------------------


def wannierize(
    psi,
    lat,
    zoneshift,
    output_file="wannier_result.npz",
    q1=DEFAULT_Q1,
    q2=DEFAULT_Q2,
    a=0.18,
    l=7,
    cutoff=9,
    n_iter=2000,
    lr=2e-3,
    print_every=100,
):
    """
    Complete two-band Wannierization.

    Parameters
    ----------
    psi : ndarray
        Shape (Nkx, Nky, 2*NG, 2).

    lat : array-like
        Reciprocal vectors, shape (NG, 2).

    zoneshift : callable
        Your working reciprocal-zone shifting function.

    output_file : str
        Results are automatically saved here.

    zoneshift_kwargs : dict or None
        Extra arguments passed to zoneshift.
        Example:
            {"cutoff": cutoff}

    Returns
    -------
    result : dict
        Contains A, M, U0, U, psi_W, centres, omega_D, omega_OD.
    """
    psi = np.asarray(psi, dtype=np.complex128)

    print("Computing projections...")
    A = compute_projections(
        psi,
        lat,
        q1=q1,
        q2=q2,
        a=a,
        l=l,
    )

    proj_sv = np.linalg.svd(A, compute_uv=False)
    print("minimum projection singular value:", proj_sv[..., 1].min())

    print("Computing neighbour overlaps...")
    M = compute_overlaps(
        psi,
        zoneshift,
        q1=q1,
        q2=q2,
        cutoff=cutoff,
    )

    overlap_sv = np.linalg.svd(M, compute_uv=False)
    print("minimum neighbour-overlap singular value:", overlap_sv.min())
    U0 = initial_gauge(A)

    print("Localizing...")
    U, centres, omega_D, omega_OD = localize(
        M,
        U0,
        q1=q1,
        q2=q2,
        n_iter=n_iter,
        lr=lr,
        print_every=print_every,
    )

    psi_W = np.einsum(
        "...am,...mn->...an",
        psi,
        U,
        optimize=True,
    )

    result = {
        "A": A,
        "M": M,
        "U0": U0,
        "U": U,
        "psi_W": psi_W,
        "centres": centres,
        "omega_D": omega_D,
        "omega_OD": omega_OD,
        "omega_tilde": omega_D + omega_OD,
    }

    np.savez_compressed(
        output_file,
        A=A,
        M=M,
        U0=U0,
        U=U,
        psi_W=psi_W,
        centres=centres,
        omega_D=np.array(omega_D),
        omega_OD=np.array(omega_OD),
        omega_tilde=np.array(omega_D + omega_OD),
        q1=np.asarray(q1),
        q2=np.asarray(q2),
        a=np.array(a),
        l=np.array(l),
    )

    print(f"Saved Wannier result to: {output_file}")

    return result

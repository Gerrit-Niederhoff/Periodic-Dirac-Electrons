import numpy as np
import math
import myBM as bm
import json
import hashlib
from pathlib import Path
from IPython.display import clear_output

COULOMB_COUPLING = 9  # eV*nM
# Screening length should be between 5 and 40 nm.
# To scale positions from units where the lattice vector is 1, multiply by
SCALE_FACTOR = 4 * np.pi / np.sqrt(3)
# AND DIVIDE by the potential scale kθ. For our standard value of θ=0.72, we define the overall scaling
PHYSICAL_SCALING = SCALE_FACTOR / bm.get_q(0.72 / 180 * np.pi)
# Note of course that when a the screening length is given in physical units, it therefore needs to by DIVIDED by that number.


def make_data_key(params):
    params_json = json.dumps(params, sort_keys=True)
    return hashlib.sha256(params_json.encode()).hexdigest()


def disk_cached(cache_dir="cache"):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def decorator(func):
        # arrays should be numpy arrays that are input, shouldn't be cached and therefore given as POSITIONAL arguments ONLY.
        def wrapper(*arrays, **params):
            updatebool = params.pop("give_updates", False)
            chunksize = params.pop("points_per_chunk", 2048)
            key = make_data_key(params)
            path = cache_dir / key

            data_file = path / "data.npz"
            params_file = path / "params.json"

            # Cache hit
            if data_file.exists() and params_file.exists():
                with np.load(data_file) as data:
                    return {name: data[name] for name in data.files}

            # Cache miss
            result = func(
                *arrays, **params, give_updates=updatebool, points_per_chunk=chunksize
            )

            # Save atomically-ish into its own directory
            path.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(data_file, **result)

            with open(params_file, "w") as f:
                json.dump(params, f, indent=2, sort_keys=True)

            return result

        return wrapper

    return decorator


def cache_under_name(cache_dir="cache/wannier"):
    """
    Decorate a function to save it's output to a specified file and folder, as well as a description of the generated data. If those things match previously cached output, that is loaded and returned instead. Raises an error if folder and filename match existing data, but the description differs (to prevent overwriting data).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def decorator(func):
        def wrapper(
            *args,
            folder="defaultfoldername",
            dataname="defaultdata",
            description="describe data in file",
            overwrite=False,
            **kwargs,
        ):
            path = cache_dir / folder
            datafile = path / f"{dataname}.npz"
            commentfile = path / f"{dataname}_description.txt"
            if datafile.exists() and commentfile.exists():
                with open(commentfile, "r") as oldfile:
                    old_description = oldfile.read()
                    if not ((old_description == description) or overwrite):
                        raise ValueError(
                            "Trying to save data to file {dataname}, but that file already exists with a different description than what is given. Change the description so it matches the existing one to load, or try a different filename."
                        )
                if not overwrite:
                    with np.load(datafile) as data:
                        return {name: data[name] for name in data.files}
            # File doesn't exist yet or should be overwritten
            result = func(*args, **kwargs)
            path.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(datafile, **result)
            commentfile.write_text(description)
            return result

        return wrapper

    return decorator


@disk_cached("cache")
def blochfactors(**kwargs):
    """
    Simply compute bloch factors for a set of parameters of the hamiltonian,
    but check if these bloch factors have been previously computed and cached first. Only computes new ones if necessary.
    Note: Returns and saves a dictionary with fields "bands","blochstates" and "kgrid".
    """
    return bm.states_on_grid(**kwargs)


def band_selection(Ψ):
    """
    Reorder a set of bloch states such that the (absolute) overlap between neighbouring points in the 'same' band is maximized.
    """
    Ψ_x_shifted = np.roll(Ψ, 1, axis=0)
    Ψ_x_shifted[0, ...] = Ψ_x_shifted[1, ...]
    overlap = abs(np.sum(Ψ.conj()[..., None] * Ψ_x_shifted[..., None, :], axis=-3))
    # Build array where the last axis has 2 elements, the first being the sum of the absolute diagonal overlaps, the second being the sum of the absolute off-diagonal overlaps.
    comp_overlap = np.sum(overlap[..., [[0, 1], [0, 1]], [[0, 1], [1, 0]]], axis=-1)
    # Basically Sorting this should tell you how to reorder Ψ: Wherever the array is already sorted, Ψ should also not be reordered, and vice versa.
    ind_x = np.argsort(comp_overlap, axis=-1, descending=True)
    Ψ = np.take_along_axis(Ψ, ind_x[:, :, None], axis=-1)
    Ψ_y_shifted = np.roll(Ψ, 1, axis=1)
    Ψ_y_shifted[:, 0, ...] = Ψ_y_shifted[:, 1, ...]
    overlap = abs(np.sum(Ψ.conj()[..., None] * Ψ_y_shifted[..., None, :], axis=-3))
    # Build array where the last axis has 2 elements, the first being the sum of the absolute diagonal overlaps, the second being the sum of the absolute off-diagonal overlaps.
    comp_overlap = np.sum(overlap[..., [[0, 1], [0, 1]], [[0, 1], [1, 0]]], axis=-1)
    # Basically Sorting this should tell you how to reorder Ψ: Wherever the array is already sorted, Ψ should also not be reordered, and vice versa.
    ind_y = np.argsort(comp_overlap, axis=-1, descending=True)
    Ψ = np.take_along_axis(Ψ, ind_y[:, :, None], axis=-1)
    ind = np.take_along_axis(ind_x, ind_y, axis=-1)
    return ind


def gauge_smooth(Ψ):
    """
    Take a bloch state Ψ, shape (KX,KY,N,M) and transform it to a smooth gauge. Note that this is abelian gauge-smoothing, and won't fix
    band selection issues, that should be done before.
    """
    Ψ_x_shifted = np.roll(Ψ, 1, axis=0)
    Ψ_x_shifted[0, ...] = Ψ_x_shifted[1, ...]
    # Phase difference from each kx value to the previous one, except at 0:
    overlap = np.sum(Ψ.conj() * Ψ_x_shifted, axis=2)
    φ_diff = np.angle(overlap)
    Ψ *= np.cumprod(np.exp(1j * φ_diff), axis=0)[:, :, None]
    Ψ_y_shifted = np.roll(Ψ, 1, axis=1)
    Ψ_y_shifted[:, 0, ...] = Ψ_y_shifted[:, 1, ...]
    overlap = np.sum(Ψ.conj() * Ψ_y_shifted, axis=2)
    φ_diff = np.angle(overlap)
    Ψ *= np.cumprod(np.exp(1j * φ_diff), axis=1)[:, :, None]
    return Ψ


@disk_cached("cache")
def two_band_subspace(momentum_radius=0.35, overlap_tolerance=0.7, **kwargs):
    """
    First call blochstates with all the given kwargs, then reduce the data returned to the relevant two-band subspace.
    Within the radius given by momentum_radius, discard any states with less overall overlap with the flatbands at the zone center than overlap_tolerance.
    The discarded blochstates are each replaced by the neighbouring state with largest overlap, which will introduce some error, but amounts to neglecting the very small observed hybridization.
    """
    data = blochfactors(**kwargs)
    Ψ = data["blochstates"]
    kgrid = data["kgrid"]
    Nx, Ny = np.shape(Ψ)[:2]
    # Reference states for computing overlaps:
    fb = Ψ[Nx // 2, Ny // 2, :, :2]
    # Computing the absolute overlap of each band with these two flat-band reference states:
    overlap = abs(
        np.sum(fb[None, None, :, :, None].conj() * Ψ[:, :, :, None, :], axis=2)
    )
    # Computing the sum of the squared overlaps with each flatband, for a sort of "total flatband overlap"
    total_overlap = np.sum(overlap**2, axis=-2)
    # Sorting states w.r.t this total overlap, such that the most overlapping bands come first (and then retain only the first 2)
    overlapsort = np.argsort(total_overlap, axis=-1, descending=True)
    flatbands = np.take_along_axis(Ψ, overlapsort[:, :, None, :], axis=-1)[..., :2]
    total_overlap = np.take_along_axis(total_overlap, overlapsort, axis=-1)[..., :2]
    k_condition = np.linalg.norm(kgrid, axis=-1) < momentum_radius
    condition = (k_condition[..., None]) & (total_overlap <= overlap_tolerance)
    idx, idy, idb = np.where(condition)
    for i in range(len(idx)):
        x, y, b = idx[i], idy[i], idb[i]
        arrx = [x, x, x + 1, x - 1]
        arry = [y + 1, y - 1, y, y]
        neighbouring_overlaps = total_overlap[
            arrx, arry, b
        ]  # Should be just shape (4,), overlaps with the 4 neighbours
        best_neighbour = np.argmax(neighbouring_overlaps)
        flatbands[x, y, :, :] = flatbands[
            arrx[best_neighbour], arry[best_neighbour], :, :
        ]

    return {"flatbands": flatbands}


def nearest_indices(A, B):
    """
    Assumes A and B contain 2-vectors in their last axis, with an arbitrary number of preceding axes.
    Returns an index array idx such that A[idx[0],idx[1],:] is an array of the same shape as B, but containing all the vectors in A that are closest to the respective vector in B.
    """
    ashape = np.shape(A)[:-1]
    bshape = np.shape(B)[:-1]
    flat_A = A.reshape(-1, 2)
    flat_B = B.reshape(-1, 2)
    flat_idx = np.argmin(
        np.linalg.norm(flat_A[None, :, :] - flat_B[:, None, :], axis=-1), axis=1
    )
    idx = np.stack(np.unravel_index(flat_idx, ashape), axis=0)
    idx = idx.reshape(len(ashape), *bshape)
    return idx


def crystal_momentum(k):
    """
    Split an arbitrary momentum vector into a sum of a crystal momentum in the first BZ and a reciprocal lattice vector.
    Note: Works in dimensionless units where the length of the reciprocal lattice vectors is 1. If using actual units those need to be divided out first.
    Returns two (arrays of) two-dimensional vectors, k_BZ and g.
    """
    kshape = np.shape(k)[:-1]
    extra = (None,) * len(kshape)
    eb1 = bm.q2
    eb2 = -bm.q1
    basis = np.array([bm.q2, -bm.q1]).T
    inversebasis = np.linalg.inv(basis)
    # Shift forward since the first BZ will be centered around the origin
    kshifted = k + eb1[*extra, :] / 2 + eb2[*extra, :] / 2
    # k in terms of the reciprocal lattice basis:
    Q = np.sum(inversebasis[*extra, :, :] * kshifted[..., None, :], axis=-1)
    Q_BZ, Q_R = np.asarray(Q % 1), np.asarray(Q // 1)
    k_BZ = (
        Q_BZ[..., 0, None] * eb1[*extra, :]
        + Q_BZ[..., 1, None] * eb2[*extra, :]
        - eb1[*extra, :] / 2
        - eb2[*extra, :] / 2
    )
    g = Q_R[..., 0, None] * eb1[*extra, :] + Q_R[..., 1, None] * eb2[*extra, :]
    return k_BZ, g


def λ_formfactor(k, q, cutoff=9, **kwargs):
    """
    Calculate λ_nm(k,q), k∈mBZ.
    k and q should have the same shape, of the form (...,2)
    Doesn't diagonalize the Hamiltonian, but calls other functions that attempt to access previously saved eigenstates, so the Hamiltonian is only
    diagonalized once for each set of parameters.
    Returns λ, an array with shape (...,n,n), with ... the same as in k and q, and n the number of active bands (flatbands first).
    """
    single_particle_data = blochfactors(cutoff=cutoff, **kwargs)
    print("Bloch-states are computed.")
    print(single_particle_data.keys())
    kdata = single_particle_data["kgrid"]
    states = single_particle_data["blochstates"]
    energies = single_particle_data["bands"]
    bandnumber = np.shape(energies)[-1]
    # Convert the k-array into an array of indices into the precomputed data:
    idx_k = nearest_indices(kdata, k)
    states_k = states[
        idx_k[0], idx_k[1], ...
    ]  # States on the closest k-points to what was demanded.
    Q = k + q
    extra = (None,) * (
        len(np.shape(Q)) - 1
    )  # Extra array axes to prepend ot vectors for broadcasting against Q

    Q_BZ, Q_RL = crystal_momentum(Q)  # First BZ and reciprocal lattice component
    unique_RL_shifts = np.unique(Q_RL.reshape(-1, 2), axis=0)
    idx_Q = nearest_indices(kdata, Q_BZ)
    # To avoid mistakes due to small rounding errors, ensure that all points with virtually the same BZ momentum get assigned the same indices:
    equalmomenta = np.all(np.isclose(Q_BZ, k), axis=-1)
    idx_Q[:, equalmomenta] = idx_k[:, equalmomenta]
    states_Q = states[idx_Q[0], idx_Q[1], ...]  # Again, the closest possible k-points.
    # Iterating over bands to shift the blochstates to the appropriate BZ:
    for gvector in unique_RL_shifts:
        print(gvector)
        mask = np.isclose(Q_RL, gvector[*extra, :])
        mask = np.all(mask, axis=-1)  # Which indices require a shift by gvector
        states_Q[mask, :, :] = bm.zoneshift(
            states_Q[mask, :, :], gvector, cutoff=cutoff
        )  # Shift the blochstates at all those indices
    # Compute the overlap:
    λ = np.sum(states_Q[..., :, None].conj() * states_k[..., None, :], axis=-3)

    return λ


@disk_cached("cache/wannier/")
def wannier_functions(
    r_array,
    kgrid,
    ψw,
    cutoff=9,
    give_updates=False,
    points_per_chunk=300,
):
    """
    Transform bloch states into wannier states.
    Note that r technically should be (r-R), but R is set to 0 here without loss of generality.
    r should be in units such that the lattice vectors (of the moire potential) have length 1.
    """
    scale_down = (
        (2 * np.pi) * 2 / np.sqrt(3)
    )  # factor by which ALL momenta (given in units where the reciprocal lattice vector length is one) should be multiplied such that the positions can be in the basis where the real space lattice vectors have length one.

    nbands = np.shape(ψw)[-1]
    Nk1, Nk2 = np.shape(kgrid)[0], np.shape(kgrid)[1]

    originalrshape = np.shape(r_array)[:-1]
    flat_position = r_array.reshape(-1, 2)
    Nr = np.shape(flat_position)[0]
    wannierstates = np.zeros((Nr, 2, nbands), dtype=complex)

    lat = np.asarray(bm.build_lattice(cutoff)) * scale_down

    for start in range(0, Nr, points_per_chunk):
        stop = min(start + points_per_chunk, Nr)
        if give_updates:
            print(f"Calculating points from {start} to {stop} out of {Nr}.", flush=True)
        r = flat_position[start:stop, :]
        Nr_chunk = stop - start
        blochstate_of_r = np.zeros((Nk1, Nk2, Nr_chunk, 2, nbands), dtype=complex)
        for j, gvector in enumerate(lat):
            blochstate_of_r += (
                ψw[:, :, None, 2 * j : 2 * (j + 1), :]
                * np.exp(1j * np.sum(gvector[None, :] * r, axis=-1))[
                    None, None, :, None, None
                ]
            )
        localwannierstates = np.sum(
            blochstate_of_r
            * np.exp(
                1j
                * np.sum(
                    kgrid[:, :, None, :] * scale_down * r[None, None, :, :], axis=-1
                )
            )[:, :, :, None, None],
            axis=(0, 1),
        ) / (Nk1 * Nk2)
        wannierstates[start:stop, :, :] = localwannierstates

    return {
        "wannierstates": wannierstates.reshape((*originalrshape, 2, nbands)),
        "rgrid": r_array,
    }


def Λ_single_q(wannier, rgrid, q):
    """
    Compute the inner product <W_a|exp(iqr)|W_b> for one single value of q.
    """
    phase = np.exp(1j * np.dot(rgrid, q))
    dr = np.linalg.norm(rgrid[0, 0] - rgrid[0, 1])
    integrand = (
        wannier.conj()[..., :, None]
        * wannier[..., None, :]
        * phase[..., None, None, None]
    )
    integrand = integrand.reshape((-1, 2, 2))
    return np.sum(integrand, axis=0) * dr**2


@cache_under_name("cache/wannier")
def localized_formfactor(qgrid, wannier, rgrid, showprogress=True):
    """
    Compute the formfactor Λ on a grid of values of q.
    """
    qshape = np.shape(qgrid)[:-1]
    qinput = qgrid.reshape((-1, 2))
    Nq = len(qinput)
    Λ = np.zeros((2, 2, Nq), dtype=complex)
    for i, q in enumerate(qinput):
        # For use in jupyter notebooks:
        if showprogress:
            print(f"{i / Nq * 100:.1f} % done.")
            clear_output(wait=True)
        Λ[:, :, i] = Λ_single_q(wannier, rgrid, q)

    Λ = Λ.reshape((2, 2, *qshape))
    data = {"formfactor": Λ, "qgrid": qgrid}
    return data


def gate_screened_coulomb(q, d, ε_r):
    """
    Dual gate-screened form of the coulomb interaction in momentum space, with screening lenght d and relative permitivity ε_r
    """
    return COULOMB_COUPLING / (ε_r * q) * np.tanh(q * d)


@cache_under_name("cache/wannier")
def ffff_interaction(qgrid, Λ, R, **kwargs):
    δAq = np.linalg.norm(qgrid[0, 0] - qgrid[0, 1]) * np.linalg.norm(
        qgrid[0, 0] - qgrid[1, 0]
    )
    ABZ = 8 * np.pi**2 / np.sqrt(3)
    NBZ = ABZ / δAq
    phase = np.exp(1j * np.einsum("xyq,...q->xy...", qgrid, R))
    coulomb = gate_screened_coulomb(np.linalg.norm(qgrid, axis=-1), **kwargs)
    interaction = (
        np.einsum(
            "mnxy,ijxy,xy...->mnij...",
            Λ * coulomb[None, None, :, :],
            Λ[:, :, ::-1, ::-1],
            phase,
        )
        / NBZ
    )
    data = {"interaction": interaction, "Rgrid": R}
    return data


@cache_under_name("cache/wannier")
def fdff_interaction(qgrid, Λ, r, R, showprogress=True, **kwargs):
    """
    Computes the fdff term of the interaction (assuming R'=0 w.l.o.g.).
    Returns the interaction in two terms, V(r-R') and V(R-R').
    The full interaction is
    W(r-R)*(V(r-R')-V(R-R'))
    """
    δAq = np.linalg.norm(qgrid[0, 0] - qgrid[0, 1]) * np.linalg.norm(
        qgrid[0, 0] - qgrid[1, 0]
    )
    ABZ = 8 * np.pi**2 / np.sqrt(3)
    NBZ = ABZ / δAq
    rshape = np.shape(r)[:-1]
    r = r.reshape((-1, 2))
    Nr = len(r)
    phaseR = np.exp(1j * np.einsum("xyi,...i->xy...", qgrid, R))
    coulomb = gate_screened_coulomb(np.linalg.norm(qgrid, axis=-1), **kwargs)
    Rterm = np.einsum(
        "mnxy,ijxy,xy...->mnij...",
        Λ * coulomb[None, None, :, :],
        Λ[:, :, ::-1, ::-1],
        phaseR,
    ) / (NBZ)
    rterm = np.zeros((2, 2, 2, 2, Nr), dtype=complex)
    for i, rval in enumerate(r):
        if showprogress and (i % 100 == 0):
            print(f"{i / Nr * 100:.1f} % done.")
            clear_output(wait=True)
        rterm[..., i] = np.einsum(
            "mnxy,ijxy,xy->mnij",
            np.eye(2)[:, :, None, None] * coulomb[None, None, :, :],
            Λ[:, :, ::-1, ::-1],
            np.exp(1j * np.dot(qgrid, rval)),
        ) / (NBZ)
    rterm = rterm.reshape((2, 2, 2, 2, *rshape))
    r = r.reshape((*rshape, 2))
    data = {"rterm": rterm, "Rterm": Rterm, "Rgrid": R, "rgrid": r}
    return data

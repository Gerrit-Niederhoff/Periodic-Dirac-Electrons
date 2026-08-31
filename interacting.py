import numpy as np
import math
import myBM as bm
import json
import hashlib
from pathlib import Path


def make_data_key(params):
    params_json = json.dumps(params, sort_keys=True)
    return hashlib.sha256(params_json.encode()).hexdigest()


def disk_cached(cache_dir="cache"):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def decorator(func):
        def wrapper(**params):
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
            result = func(**params, give_updates=updatebool, points_per_chunk=chunksize)

            # Save atomically-ish into its own directory
            path.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(data_file, **result)

            with open(params_file, "w") as f:
                json.dump(params, f, indent=2, sort_keys=True)

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


def reduce_energy_window(new_window, cache_dir="cache", **kwargs):
    """
    Reduce the energy_window in a previously cached dataset. Does NOT delete the old dataset, but makes a new one with the updated parameters.
    """
    cache_dir = Path(cache_dir)
    old_data_key = make_data_key(kwargs)
    path = cache_dir / old_data_key
    old_file = path / "data.npz"
    if old_file.exists():
        with np.load(old_file) as data:
            states = data["blochstates"]
            bands = data["bands"]
            grid = data["kgrid"]
    else:
        raise ValueError("File with the old parameters isn't cached.")
    flatband = kwargs["flatband_energy"]
    newCondition = abs(bands - flatband) < new_window
    bands = np.where(~newCondition, bands, np.nan)
    states = np.where(~newCondition[..., None, :], states, np.nan)
    newData = {"blochstates": states, "bands": bands, "kgrid": grid}
    params = kwargs.copy()
    params.update({"energy_window": new_window})
    newKey = make_data_key(params)
    newPath = cache_dir / newKey
    newPath.mkdir(parents=True)
    newFile = newPath / "data.npz"
    newParamFile = newPath / "params.json"
    np.savez_compressed(newFile, **newData)

    with open(newParamFile, "w") as f:
        json.dump(params, f, indent=2, sort_keys=True)
    return


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
    for i in range(bandnumber):
        # Computing all shifts by the same reciprocal lattice vector in bulk:
        for gvector in unique_RL_shifts:
            print(gvector)
            mask = np.isclose(Q_RL, gvector[*extra, :])
            mask = np.all(mask, axis=-1)  # Which indices require a shift by gvector
            states_Q[mask, :, i] = bm.zoneshift(
                states_Q[mask, :, i], gvector, cutoff=cutoff
            )  # Shift the blochstates at all those indices
    # Compute the overlap:
    λ = np.sum(states_Q[..., :, None].conj() * states_k[..., None, :], axis=-3)

    return λ

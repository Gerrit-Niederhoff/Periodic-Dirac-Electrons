import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from cache import cache_under_name
from cache import disk_cached
from IPython.display import clear_output
import warnings

# from scipy.special import assoc_laguerre as lag
# from math import factorial

e_over_2hbar = 0.001519  # 1 / (T nm^2)

vhbar = 0.6582  # eV*nm
σx = np.array([[0, 1], [1, 0]])
σy = np.array([[0, -1j], [1j, 0]])
σz = np.array([[1, 0], [0, -1]])
σ0 = np.eye(2)
Nk = 20
π = np.pi
a = 0.142  # nm
k_D = 4 * π / 3 / np.sqrt(3) / a
q1 = np.array([0, -1])
q2 = np.array([np.sqrt(3) / 2, 1 / 2])
q3 = np.array([-np.sqrt(3) / 2, 1 / 2])

R1 = 2 * π * np.array([1 / np.sqrt(3), 1])
R2 = 2 * π * np.array([2 / np.sqrt(3), 0])


p1 = q1 / 2  # M-point
p2 = (q1 - q3) / 3  # K-point
p0 = np.zeros(2)  # Γ-Point

Nk2 = int(np.round((Nk * np.linalg.norm(p2 - p1) / np.linalg.norm(p1))))
Nk3 = int(np.round((Nk * np.linalg.norm(p0 - p2) / np.linalg.norm(p1))))

arr1 = np.linspace(0, 1, Nk, endpoint=False)
arr2 = np.linspace(0, 1, Nk2, endpoint=False)
arr3 = np.linspace(0, 1, Nk3)
print(Nk, Nk + Nk2)
path1 = arr1[:, None] * p1[None, :]
path2 = p1[None] + arr2[:, None] * (p2[None, :] - p1[None])
path3 = p2[None] + arr3[:, None] * (p0[None, :] - p2[None])

hexagonalpath = np.concat((path1, path2, path3))
pathlabels = (
    [0, Nk, Nk2 + Nk, Nk + Nk2 + Nk3],
    ["$\\Gamma$", "$M$", "$K$", "$\\Gamma$"],
)
s1 = np.array([1, 0])
s2 = np.array([0, 1])

spath1 = arr1[:, None] * 0.5 * s1[None, :]
spath2 = 0.5 * s1[None] + arr2[:, None] * 0.5 * s2
spath3 = 0.5 * (s1 + s2)[None] - arr3[:, None] * 0.5 * (s1 + s2)[None]

squarepath = np.concat((spath1, spath2, spath3))

h1 = np.sqrt(3) * np.array([1, 0])
h2 = np.array([1 / 2, np.sqrt(3) / 2]) * np.sqrt(3)

hpath1 = arr1[:, None] * np.array([0, 0.5])[None, :]
hpath2 = np.array([0, 0.5])[None, :] + arr2[:, None] * 0.5 * h1[None, :]
hpath3 = 0.5 * (h1 + np.array([0, 1]))[None, :] * (1 - arr3)[:, None]

honeycombpath = np.concat((hpath1, hpath2, hpath3))


def dirac(k):
    """
    Return a 2x2 dirac hamiltonian for a 2D momentum input, where the last axis of the momentum input needs
    to have size 2 and contain the kx and ky momenta.
    """
    kshape = np.shape(k)
    nk = len(kshape)
    single = nk == 1
    if single:
        k = k[None]
    nk = len(np.shape(k))
    σvec = np.array([σx, σy])
    h = vhbar * np.sum(k[..., None, None] * σvec[*((None,) * (nk - 1)), ...], axis=-3)
    return h[0] if single else h


def build_lattice(cutoff, shape="hexagonal"):
    """
    Returns a list of 2D vectors that span a hexagonal lattice lying within a circle
    of radius <cutoff>
    """
    points = []
    if shape == "hexagonal":
        maximum = np.round(np.sqrt(3) * cutoff)
        vec1 = q1
        vec2 = q2
    elif shape == "square":
        maximum = np.sqrt(2) * cutoff
        vec1 = s1
        vec2 = s2
    elif shape == "honeycomb":
        maximum = cutoff
        vec1 = h1
        vec2 = h2
    else:
        print("Not a supported lattice shape.")
        return points
    it = np.arange(-maximum, maximum + 1).astype(int)
    for i in it:
        for j in it:
            p = i * vec1 + j * vec2
            if np.linalg.norm(p) <= cutoff:
                points.append(p)
            if shape == "honeycomb":
                p2 = p + np.array([0, 1])
                if np.linalg.norm(p2) <= cutoff:
                    points.append(p2)
    return points


def rotationMatrix(φ, lattice, shift=np.zeros((2))):
    """
    Return a rotationMatrix that can act on eigenvectors of a Hamiltonian built with the given reciprocal lattice.
    Currently assumes that the sublattice degrees of freedom are invariant under the rotation (e.g. C3).
    """
    smallMatrix = np.array([[np.cos(φ), -np.sin(φ)], [np.sin(φ), np.cos(φ)]])
    N = len(lattice)
    bigMatrix = np.zeros((N, N), dtype=complex)
    matches = 0
    for i in range(N):
        newp = smallMatrix @ lattice[i] + shift
        for j in range(N):
            if np.allclose(newp, lattice[j]):
                bigMatrix[j, i] = 1
                matches += 1
                break
    pseudoSpin = np.exp(-1j * φ / 2) * np.exp(-1j * φ * np.array([1, -1]) / 2)
    pseudoSpin = np.diag(pseudoSpin)
    biggerMatrix = np.kron(bigMatrix, pseudoSpin)
    print(N)
    print(matches)
    return biggerMatrix


def rotationRepresentation(states, operator):
    """
    Given a (small) set of states, presumably with degenerate energies (though this isn't checked), calculates the matrix elements of the given
    operator (in a matrix form) between these states.
    Primarily used to calculate a reduced representation of a rotation operator acting on a subspace of degenerate states at an invariant momentum.
    """
    n = len(states)
    matrix = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = states[i].conj() @ operator @ states[j]
    return matrix


def are_nearest_neighbors(p1, p2, distance=1):
    """
    Checks whether two vectors (representing points on a reciprocal lattice) are nearest neighbours, i.e. are a distance 'distance' apart.
    """
    return np.isclose(np.linalg.norm(p2 - p1), distance)


def build_adjacency_matrix(lattice, anisotropy=None, **kwargs):
    """
    Builds an adjacency matrix for a given reciprocal lattice, simply noting which lattice points are nearest neighbours.
    The possible additional argument is for choosing a different nearest-neighbour distance than 1, the default.
    """
    N = len(lattice)
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if are_nearest_neighbors(lattice[i], lattice[j], **kwargs):
                s = 1
                if anisotropy is not None:
                    dirvecs = np.array([q1, -q1, q2, -q2, q3, -q3])
                    ind = (
                        np.argmin(
                            np.linalg.norm(
                                (lattice[i] - lattice[j])[None, :] - dirvecs, axis=-1
                            )
                        )
                        // 2
                    )
                    s = anisotropy[ind]
                M[i, j] = s
                M[j, i] = s
    return M


def build_hopping_matrix(lattice, hopping_vectors, hopping_phases):
    """
    Similar to adjacency matrix, but assigns different complex phases to different hopping vectors.
    """
    N = len(lattice)
    M = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            for k, vec in enumerate(hopping_vectors):
                δq = lattice[i] - lattice[j]
                if np.allclose(δq, vec):
                    M[i, j] = hopping_phases[k]
                    M[j, i] = np.conj(hopping_phases[k])
    return M


def get_q(θ):
    """
    returns the length of the scattering vectors in graphene for a given "twist angle".
    Note that the "twist angle" terminology is borrowed from TBLG, though this model corresponds to a static potential in a monolayer of graphene.
    The length scale of the potential will simply be chosen equal to the moire length scale of TBLG with the given twist angle, which allows some
    comparison between the models.
    """
    return 2 * k_D * np.sin(θ / 2)


def build_H(
    k,
    q,
    w=0.11,
    cutoff=5,
    extra_hoppings=(),
    phases=None,
    uniform_perturbation=None,
    coupling_matrix=np.eye(2),
    **kwargs,
):
    """
    Build the scattering hamiltonian for given k in mBZ, given scalar q indicating the scattering lenght, and a w indicating the
    strength of the scattering matrix elements in eV.
    Cutoff is the UV cutoff in reciprocal space,
    extra_hoppings is a tuple that can contain higher order scattering vectors (i.e. integers indicating distances in reciprocal space: scattering is still assumed isotropic)
    phases is a way of making the scatterings anisotropic, but only for the lowest harmonic hexagonal lattice, giving scattering along q1,q2,q3 different complex phases.
    """
    lat = build_lattice(cutoff)
    if phases is None:
        adM = build_adjacency_matrix(lat, **kwargs)
    else:
        # ONLY works for the hexagonal lattice right now, not implemented generally
        vecs = (q1, q2, q3)
        adM = build_hopping_matrix(lat, vecs, phases)

    for Vd, d in extra_hoppings:
        adM += Vd * build_adjacency_matrix(lat, distance=d)
    Nshells = len(lat)
    single = len(np.shape(k)) == 1
    if single:
        k = k[None]
    k_axes = np.shape(k)[:-1]
    nk = len(k_axes)
    lat = np.array(lat)
    extra_axes = (None,) * nk
    all_momenta = k[None, ...] + q * lat[:, *extra_axes, :]  # Shape (*K,Nshells,2)
    coupling = w * coupling_matrix

    H = np.zeros((*k_axes, Nshells, 2, Nshells, 2), dtype=complex)
    hdiag = dirac(all_momenta)  # Shape is (Nshells,,*K,2,2)
    H[..., range(Nshells), :, range(Nshells), :] = hdiag
    H = H.reshape((*k_axes, 2 * Nshells, 2 * Nshells))

    H += np.kron(adM, coupling)[*extra_axes, ...]

    if uniform_perturbation is not None:
        perturbation_matrix = np.zeros((2, 2), dtype=complex)
        for i in range(3):
            perturbation_matrix += uniform_perturbation[i] * (σx, σy, σz)[i]
        H += np.kron(np.eye(Nshells), perturbation_matrix)
    return H[0] if single else H


def build_D(cutoff=9):
    """
    Build the velocity operator for a given UV cutoff. Note that this is k-independent.
    """
    Nshells = len(build_lattice(cutoff))
    dxH = np.kron(np.eye(Nshells), σx)
    dyH = np.kron(np.eye(Nshells), σy)
    return np.array([dxH, dyH])


def magnetic_moment(blochstates, energies, band_energy, nbands=2, cutoff=9):
    """
    Calculate the magnetic moment matrix in a degenerate band manifold with given energy.
    Returns mz[momentum,n,m], in units of eV/T.
    band_energy can either be a number, or an array with the same shape as energies, except missing the final axis.
    """
    if len(np.shape(np.asarray(band_energy))) > 0:
        band_energy = band_energy[..., None]
    D = build_D(cutoff)
    sortfrom_reference = np.argsort(abs(energies - band_energy), axis=-1)
    E = np.take_along_axis(energies, sortfrom_reference, axis=-1)[..., nbands:]
    U = np.take_along_axis(blochstates, sortfrom_reference[..., None, :], axis=-1)
    Dnm = np.einsum(
        "...an,jab,...bm->j...nm",
        U[..., :nbands].conj(),
        D,
        U[..., nbands:],
        optimize=True,
    )
    mz = (
        1j
        * e_over_2hbar
        * np.einsum(
            "iknl,jkml,kl->ijknm",
            Dnm,
            Dnm.conj(),
            1 / (band_energy - E),
        )
    )
    return mz[0, 1] - mz[1, 0]


def H_along_path(θ, **kwargs):
    """
    Take a twist angle θ and builds the scattering hamiltonian along the high-symmetry
    path of the miniBZ, with any further arguments passed along to build_H.
    """
    q = get_q(θ * π / 180)
    path = hexagonalpath
    H = build_H(q * path, q, **kwargs)

    return H


def plot_wavefunction(
    u,
    axis,
    lat,
    δ=0.1,
    scale=1,
    scale_absolute=1,
    scale_relative=1,
):
    """
    Plot a single eigenvector of the scattering hamiltonian on the reciprocal lattice. Takes u, the bloch state,
    axis: a matplotlib axis on which the wavefunction will be drawn,
    lat: The reciprocal lattice that acts as basis for the hilbert space,
    δ: At each lattice point, two dots will be drawn, with the color of the upper dot showing the phase of the sublattice-A component of the spinor, and the lower dot showing the relative phase between the A and B components. δ is the distance between these dots.
    scale: The size of the dots will be scale* the magnitude of the wavefunction
    scale_absolute: Additional factor for the size of the 'overall phase'-dots
    scale_relative: Additional factor for the size of the 'relative phase'-dots
    """
    shift = δ * np.array([0, 1])  # if not equal_weight_sublattices else np.zeros(2)
    phasecolor = plt.get_cmap("hsv")
    for i, p in enumerate(lat):
        φ1 = (np.angle(u[2 * i]) + π) / 2 / π
        norm1 = np.abs(u[2 * i])
        norm1 = norm1 if norm1 >= 1e-5 else 0
        φ2 = (np.angle(u[2 * i + 1]) + π) / 2 / π
        norm2 = np.abs(u[2 * i + 1])
        norm2 = norm2 if norm2 >= 1e-5 else 0
        φ2 = (φ2 - φ1) % 1
        φ2 = (np.angle(u[2 * i] * u[2 * i + 1].conj()) + π) / 2 / π
        axis.scatter(
            *(p + shift),
            marker="o",
            linestyle="None",
            s=scale * norm1 * scale_absolute,
            color=phasecolor(φ1),
            zorder=2,
        )

        axis.scatter(
            *(p - shift),
            marker="o",
            linestyle="None",
            s=scale * norm2 * scale_relative,
            color=phasecolor(φ2),
            zorder=2,
        )
    norm = mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi)

    # Create a ScalarMappable with the hsv colormap
    divider = make_axes_locatable(axis)
    ax_cbar = divider.append_axes("right", size="5%", pad=0.05)
    cmap = plt.get_cmap("hsv")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required for colorbar
    cbar = plt.colorbar(sm, cax=ax_cbar)

    cbar.set_ticks([-np.pi, 0, np.pi])

    cbar.set_ticklabels([r"$-\pi$", "0", r"$\pi$"])
    return


def wavefunction_realspace(u, cutoff, N=100):
    """
    Visualize an individual bloch state in real space rather than on the reciprocal lattice. Cutoff is the UV cutoff used for building the hamiltonian,
    N is the resolution of the NxN grid in real space on which the wavefunction will be plotted.
    """
    lat = np.asarray(build_lattice(cutoff))
    coords = 0.5 * np.linspace(-1, 1, N)
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    RX = X * R2[0] + Y * R1[0]
    RY = Y * R1[1] + X * R2[1]
    uA = u[::2]
    uB = u[1::2]
    Ψ_A = fourier_wavefunction(uA, lat, RX, RY)
    Ψ_B = fourier_wavefunction(uB, lat, RX, RY)
    return Ψ_A, Ψ_B, RX, RY


def fourier_wavefunction(U, lat, RX, RY):
    """
    Given a single blochstate U (single momentum, single energy), the corresponding reciprocal lattice and some real-space arrays RX and RY of equal shape,
    compute the real-space shape of the Bloch state u_k(R). Simple discrete fourier transform.
    """
    Ψ = np.zeros(np.shape(RX), dtype=complex)
    for j, p in enumerate(lat):
        Ψ += U[j] * np.exp(1j * p[0] * RX + 1j * p[1] * RY)
    return Ψ


def visualize_flatbands(
    u,
    e,
    flatband_energy,
    momentum_index,
    lat,
    m,
    energycutoff=None,
    **kwargs,
):
    """
    Function for quickly plotting the wavefunction belonging to 2 degenerate flatbands on the reciprocal lattice.
    u and e are the bloch states and energies of the hamiltonian, with the first axis being assumed to be momentum.
    lat and m are the lattice and adjacency matrix used for building the hamiltonian.
    **kwargs are any additional arguments for plot_wavefunction, which is called for plotting the actual wavefunctions of the two flatbands.
    Returns a 2x2 Figure, with the left column showing the wavefunctions of the 2 states at the given momentum index that are closest to the given energy (without checking that they are flat bands). The figures in the right column show the band structure around the flat-band energy, with a red dot showing the eigenvalue belonging to the respective wavefunction.
    """
    N = len(lat)
    fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    fb1, fb2 = np.argsort(abs(e[momentum_index, :] - flatband_energy))[:2]

    if energycutoff is None:
        energycutoff = (flatband_energy - 0.15, flatband_energy + 0.15)
    for i in range(N):
        for j in range(N):
            if m[i, j] == 1:
                bondx = [lat[i][0], lat[j][0]]
                bondy = [lat[i][1], lat[j][1]]
                ax[0, 0].plot(bondx, bondy, c="lightgrey", linewidth=1, zorder=1)
                ax[1, 0].plot(bondx, bondy, c="lightgrey", linewidth=1, zorder=1)
    plot_wavefunction(u[momentum_index, :, fb1], ax[0, 0], lat=lat, **kwargs)
    plot_wavefunction(u[momentum_index, :, fb2], ax[1, 0], lat=lat, **kwargs)
    ax[0, 0].plot(0, 0, marker="+", linestyle="None", color="black")
    ax[1, 0].plot(0, 0, marker="+", linestyle="None", color="black")

    # Create the colorbar
    # cbar.set_label('Phase')
    # fig.tight_layout(w_pad=0.0,h_pad=0)'
    ax[0, 1].plot(e, c="deepskyblue", linestyle="solid")
    ax[0, 1].set_ylim(*energycutoff)
    ax[0, 1].plot(
        momentum_index, e[momentum_index, fb1], linestyle="None", marker=".", c="r"
    )
    ax[1, 1].plot(e, c="deepskyblue", linestyle="solid")
    ax[1, 1].set_ylim(*energycutoff)
    ax[1, 1].plot(
        momentum_index, e[momentum_index, fb2], linestyle="None", marker=".", c="r"
    )
    ax[0, 1].set_xticks(*pathlabels)
    ax[1, 1].set_xticks(*pathlabels)
    ax[1, 0].set_title(f"E = {e[momentum_index, fb2]:.3f}, index {fb2}")
    ax[0, 0].set_title(f"E = {e[momentum_index, fb1]:.3f}, index {fb1}")
    return fig, ax


def plot_bands(w, θ, cutoff=8, minE=-0.33, maxE=0.0, **kwargs):
    """
    Simply plots bands above minE and below maxE, for given parameters along the standard path through the brillouin zone,
    returning the figure and axis that are created (mostly for making animations of changing band structures).
    """
    H = H_along_path(θ, cutoff=cutoff, w=w)
    E = np.linalg.eigvalsh(H)
    fig, ax = plt.subplots(**kwargs)
    ax.plot(E, c="deepskyblue")
    ax.set_ylim(minE, maxE)
    ax.set_xticks(*pathlabels)
    ax.set_title(f"w={w:.3f} eV, θ = {θ:.3f}")
    return fig, ax


@disk_cached("cache")
def states_on_grid(
    N,
    θ=0.8,
    flatband_energy=0.216,
    nfb=3,
    n_above=1,
    n_below=0,
    points_per_chunk=5000,
    give_updates=False,
    **kwargs,
):
    """
    Compute blochstates within energy_window of the target flatband_energy of the folded Hamiltonian.
    Returns data as a dictionary, with
        data["bands"] being all bands that at some point cross into the active energy window (ordered by distance from target energy)
        data["blochstates"] being the corresponding eigenstates (ordered the same way)
        data["kgrid"] being the Kgrid on which these states were computed, in dimensionless units where the reciprocal lattice vector has length 1.
    """
    x = 1 / 2 * np.linspace(-1, 1, N, endpoint=False)
    q = get_q(θ / 180 * np.pi)
    k1 = x[:, None] * (q2[None, :])
    k2 = -q1[None, :] * x[:, None]
    Kgrid = k1[:, None, :] + k2[None, :, :]
    Kgrid *= q
    E_Γpoint = np.linalg.eigvalsh(build_H(np.zeros(2), q=q, **kwargs))
    fb_ind = np.argsort(abs(E_Γpoint - flatband_energy), axis=-1)[:nfb]
    band_ind = np.arange(
        np.min(fb_ind) - n_below, np.max(fb_ind) + n_above + 1, 1, dtype=int
    )
    activebands_chunks = []
    activestates_chunks = []
    rows_per_chunk = max(1, points_per_chunk // N)
    for start in range(0, N, rows_per_chunk):
        stop = min(start + rows_per_chunk, N)
        if give_updates:
            print(f"Calculating row {start} to {stop} out of {N}.")
        Kchunk = Kgrid[start:stop]
        h = build_H(Kchunk, q=q, **kwargs)
        E, U = np.linalg.eigh(h)
        activebands = E[..., band_ind]
        activestates = U[..., band_ind]
        activebands_chunks.append(activebands)
        activestates_chunks.append(activestates)
        del h, E, U
    # Reassemble the chunks into the original grid shape.
    activebands = np.concatenate(activebands_chunks, axis=0)
    activestates = np.concatenate(activestates_chunks, axis=0)
    return {"bands": activebands, "blochstates": activestates, "kgrid": Kgrid}


def translate_lattice(lat, g):
    """
    Attempt to translate a given lattice by a reciprocal lattice vector g (While staying in the same cutoff).
    Returns an array idx, same length as lat, such that idx[i] is the index that lat[i] is shifted to.
    """
    N = len(lat)
    trans_indices = -np.ones(N, dtype=int)
    for i in range(N):
        p = lat[i] + g
        # Start from i-10 and iterate from there
        for j in range(i - 10, N + i - 10):
            validj = j if j < N else j - N
            if np.allclose(p, lat[validj]):
                trans_indices[i] = validj
    return trans_indices


def zoneshift(u, g, cutoff=8):
    """
    Takes a bloch vector u(k) (for a crystal momentum k∈mBZ) and calculates u(k+g), for some reciprocal lattice vector g.
    This corresponds to shifting/rearranging the elements of u appropriately.
    u may be a (...,2N)-array containing the bloch state at several momenta k, but it needs to be a single bloch state at each momentum (single band).
    """
    # Building the lattice that corresponds to the basis of u
    lat = build_lattice(cutoff=cutoff)
    # Identifying the shifted indices, showing how to rearrange u
    translat = translate_lattice(lat, g)
    N = len(lat)
    uNew = np.zeros_like(u)
    for i in range(N):
        if translat[i] != -1:
            uNew[..., 2 * i : 2 * (i + 1), :] = u[
                ..., 2 * translat[i] : 2 * (translat[i] + 1), :
            ]
    return uNew


def berry_phase(k, targetE, N=60, **kwargs):
    """
    Calculate the berry phase on a path winding around the origin at a radius k for a state with energy (close to) E.
    """
    φ = np.linspace(0, 2 * np.pi, N, endpoint=False)
    kx = k * np.cos(φ)
    ky = k * np.sin(φ)
    kpath = np.array([kx, ky]).T
    E, U = np.linalg.eigh(build_H(kpath, **kwargs))
    targetIndices = np.argmin(abs(E - targetE), axis=-1, keepdims=True)
    targetBand = np.take_along_axis(E, targetIndices, axis=-1)
    targetStates = np.take_along_axis(U, targetIndices[:, None, :])
    berryphase = 1
    print(targetStates.shape)
    for i in range(N):
        phase = np.sum(targetStates[i - 1].conj() * targetStates[i])
        berryphase *= phase
    return np.angle(berryphase)


def magnetic_moment_and_states(
    K,
    band_indices,
    fb_energy=0.161,
    degeneracy_tolerance=0.001,
    # safewindow=0.015,
    cutoff=9,
    **kwargs,
):
    kshape = np.shape(K)[:-1]

    Efull, Ufull = np.linalg.eigh(build_H(K, cutoff=cutoff, **kwargs))
    # threeband_subspace[out_of_window,2] = dispersiveband[out_of_window,0]
    # Compute number of bands at flatband energy at each index
    bands_at_fb_energy = np.sum(abs(Efull - fb_energy) < degeneracy_tolerance, axis=-1)
    # Where is the dispersive band crossing?
    threefold = bands_at_fb_energy == 3
    twofold = bands_at_fb_energy == 2
    magnetic_tensor = np.zeros((*kshape, 3, 3), dtype=complex)
    ###Compute 3x3 matrix at threefold degenerate lines:

    m33 = magnetic_moment(
        Ufull[threefold], Efull[threefold], fb_energy, cutoff=cutoff, nbands=3
    )
    ###Compute 2x2 matrix at all other points for the flatbands:
    m22 = magnetic_moment(
        Ufull[twofold], Efull[twofold], fb_energy, cutoff=cutoff, nbands=2
    )
    ###Compute the abelian magnetic moment for the dispersive band where it lies in the relevant energyrange
    magnetic_tensor[threefold, ...] = m33
    magnetic_tensor[twofold, :2, :2] = m22
    return Ufull[..., band_indices], Efull[..., band_indices], magnetic_tensor


@disk_cached("cache/magnetic")
def μ_and_states(
    Nk=20,
    θ=0.72,
    points_per_chunk=100,
    fb_energy=0.161,
    dtol=0.001,
    give_updates=False,
    **kwargs,
):
    kθ = get_q(θ / 180 * np.pi)
    arr = np.linspace(-0.5, 0.5, Nk, endpoint=False)
    k1 = arr[:, None] * q2 * kθ
    k2 = -arr[:, None] * q1 * kθ
    K = k1[:, None, :] + k2[None, :, :]
    E0, U0 = np.linalg.eigh(build_H(np.zeros((2)), q=kθ, **kwargs))
    ndof = np.shape(U0)[-2]
    fb_ind = np.argsort(abs(E0 - fb_energy), axis=-1)[:2]
    band_ind = np.arange(np.min(fb_ind), np.max(fb_ind) + 2, 1, dtype=int)
    rows_per_chunk = max(1, points_per_chunk // Nk)
    μ_magnetic = np.zeros((Nk, Nk, 3, 3), dtype=complex)
    allstates = np.zeros((Nk, Nk, ndof, 3), dtype=complex)
    allenergies = np.zeros((Nk, Nk, 3), dtype=complex)
    for start in range(0, Nk, rows_per_chunk):
        stop = min(start + rows_per_chunk, Nk)
        if give_updates:
            print(f"Computing row {start} to {stop} of {Nk}.")
            clear_output(wait=True)
        Kchunk = K[start:stop]
        u, e, μ = magnetic_moment_and_states(
            Kchunk,
            band_ind,
            degeneracy_tolerance=dtol,
            q=kθ,
            fb_energy=fb_energy,
            **kwargs,
        )
        μ_magnetic[start:stop] = np.asarray(μ)
        allstates[start:stop] = np.asarray(u)
        allenergies[start:stop] = np.asarray(e)
    data = {
        "magnetic_moment": μ_magnetic,
        "states": allstates,
        "bands": allenergies,
        "grid": K,
    }
    return data


def magnetic_bands(E, U, μ, B=0.5):
    magnetic_H = np.zeros_like(μ)
    magnetic_H[..., range(3), range(3)] = E
    magnetic_H += B * μ
    B_bands, B_states = np.linalg.eigh(magnetic_H)
    Full_states = np.einsum("...bn,...ab->...an", B_states, U)
    return B_bands, Full_states


def berry_curvature(U, δ=1e-5):
    """
    Computes the chern number for a given isolated band, given the blochstate of said band.
    Requires U0,1,2,3 being the bloch state evaluated at the four cournes of a square, side-length δ, each across the whole BZ.
    """
    U1, U2, U3, U4 = U[0], U[1], U[2], U[3]
    print(np.shape(U1))
    φ = np.einsum("...an,...an->...n", U1.conj(), U2)
    print(np.min(abs(φ)))
    φ *= np.einsum("...an,...an->...n", U2.conj(), U3)
    print(np.min(abs(φ)))
    φ *= np.einsum("...an,...an->...n", U3.conj(), U4)
    print(np.min(abs(φ)))
    φ *= np.einsum("...an,...an->...n", U4.conj(), U1)
    print(np.min(abs(φ)))
    print(np.shape(φ))
    φ = np.angle(φ) / δ**2
    return φ


def berry_fhs(kgrid, vecs, b1, b2, Gvecs):
    """
    Parameters
    ----------
    kgrid : (N, N, 2) or (N*N, 2) momenta of a regular N x N grid along b1, b2.
            Any origin and ordering; points may be shifted by reciprocal lattice vectors.
    vecs  : (N, N, 2*nG) or (N, N, 2*nG, nbands) Bloch vectors in the plane-wave basis,
            G_major (index 2*iG + s), same point ordering as kgrid.
            Several bands -> Berry curvature/Chern number of the band group (non-Abelian).
            Vectors need no particular gauge; only orthonormality within each k.
    b1, b2: reciprocal lattice basis vectors.
    Gvecs : (nG, 2) reciprocal lattice vectors in the plane-wave order of vecs.

    Returns
    -------
    dict with
      Omega      : (N, N) Berry curvature, indexed by grid residue (i1, i2) along (b1, b2)
      kpts       : (N, N, 2) actual momenta belonging to Omega
      chern      : Chern number (integer up to rounding if the group is gapped)
      max_flux   : max |flux| per plaquette (should be well below pi)
      min_overlap: min |det| of link overlaps (small -> not gapped or grid too coarse)
    """
    B = np.column_stack([b1, b2]).astype(float)
    Binv = np.linalg.inv(B)
    k = np.asarray(kgrid, float).reshape(-1, 2)
    Nk = len(k)
    N = int(round(np.sqrt(Nk)))
    if N * N != Nk:
        raise ValueError("kgrid must contain N*N points")
    nG = len(Gvecs)
    v = np.asarray(vecs, complex)
    if v.size % (Nk * 2 * nG):
        raise ValueError(f"vecs must have 2*nG = {2 * nG} components per k-point")
    v = v.reshape(Nk, 2 * nG, -1)  # (Nk, 2nG, nb)

    # --- sort the grid by residues (i1, i2) mod N, keep actual fractional momenta
    x = N * (k @ Binv.T)
    c = x[0] - np.rint(x[0])
    n = np.rint(x - c)
    if np.abs(x - c - n).max() > 1e-6:
        raise ValueError("kgrid is not a regular N x N grid along b1, b2")
    n = n.astype(int)
    I = np.mod(n, N)
    flat = I[:, 0] * N + I[:, 1]
    if len(np.unique(flat)) != Nk:
        raise ValueError("kgrid does not contain every grid point exactly once")
    o = np.argsort(flat)
    k, v, kfrac = k[o], v[o], n[o] / N

    # --- plane-wave shifts: coefficients of the state at k + G_g
    gint = np.asarray(Gvecs, float) @ Binv.T
    Gint = np.rint(gint).astype(int)
    if np.abs(gint - Gint).max() > 1e-6:
        raise ValueError("Gvecs are not integer combinations of b1, b2")
    lookup = {tuple(m): i for i, m in enumerate(Gint)}

    def shifted(V, g):
        idx = np.array([lookup.get((a + g[0], b + g[1]), nG) for a, b in Gint])
        Vp = np.concatenate(
            [
                V.reshape(len(V), nG, 2, -1),
                np.zeros((len(V), 1, 2, V.shape[-1]), V.dtype),
            ],
            axis=1,
        )
        return Vp[:, idx].reshape(V.shape)

    # --- link variables U_mu(k) = det <u_k | u_{k+mu}>
    links, min_ov = [], np.inf
    for mu in (np.array([1, 0]), np.array([0, 1])):
        tgt = kfrac + mu / N
        r = np.mod(np.rint(tgt * N).astype(int), N)
        jj = r[:, 0] * N + r[:, 1]
        gsh = np.rint(tgt - kfrac[jj]).astype(int)
        vn = v[jj]
        for gv in np.unique(gsh, axis=0):
            if gv.any():
                sel = np.all(gsh == gv, axis=1)
                vn[sel] = shifted(v[jj[sel]], gv)
        d = np.linalg.det(np.einsum("kpm,kpn->kmn", v.conj(), vn))
        min_ov = min(min_ov, np.abs(d).min())
        links.append((d / np.abs(d)).reshape(N, N))
    U1, U2 = links

    # --- plaquette fluxes and curvature
    F = np.angle(
        U1 * np.roll(U2, -1, axis=0) * np.roll(U1, -1, axis=1).conj() * U2.conj()
    )
    orient = np.sign(np.linalg.det(B))
    dA = abs(np.linalg.det(B)) / Nk
    Omega = -orient * F / dA
    chern = -orient * F.sum() / (2 * np.pi)
    if min_ov < 1e-6:
        warnings.warn("tiny link overlap: band group not gapped or grid too coarse")
    if np.abs(F).max() > 0.5 * np.pi:
        warnings.warn(
            "flux per plaquette above pi/2: grid likely too coarse near a hotspot"
        )
    return dict(
        Omega=Omega,
        kpts=k.reshape(N, N, 2),
        chern=chern,
        max_flux=np.abs(F).max(),
        min_overlap=min_ov,
    )

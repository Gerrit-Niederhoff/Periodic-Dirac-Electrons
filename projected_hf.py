"""
projected_hf.py
===============
Self-consistent Hartree-Fock for a few active bands of a plane-wave continuum
model (graphene + superlattice potential), single valley / single spin.

Conventions (everything below follows these)
--------------------------------------------
Lattice:      reciprocal basis b1, b2; reciprocal lattice list Gvecs (cartesian),
              in the same order as the plane-wave basis of build_H / blochfactors.
k-grid:       regular N x N grid from your blochstates(N); internally sorted to
              flat index i = i1*N + i2 of the residues (i1,i2) mod N. Actual
              momenta are kept, so any origin / shifted points are fine.
Bloch states: psi_{k,m}(r) = sum_{G,s} C_k[G,s,m] exp(i(k+G)r) |s>,
              stored as C with shape (Nk, nG, 2, nb).
Embedding:    C_{k+G0}[G] = C_k[G+G0]   (zero outside the plane-wave cutoff).
Form factor:  Lam_{mn}(k,q) = <u_{k+q,m} | u_{k,n}>.
Density:      P_{ab}(k) = <c^dag_{k,b} c_{k,a}>   (= sum_occ v v^dag).
Interaction:  H_int = 1/(2A) sum_q V(q) :rho_q rho_-q:,   A = Nk * A_uc.
Self-energy (dP = P - P_ref):
   Sigma_H(k) =  1/A sum_{G!=0} V(G) conj(rho(G)) Lam(k,G),   rho(G) = sum_k Tr[Lam(k,G) dP(k)]
   Sigma_F(k) = -1/A sum_{k',G} V(|q|) Lam(k,q)^dag dP(k') Lam(k,q),   q = k'-k+G
Energy per cell:
   E = 1/Nk [ sum_k Tr(h0 P) + 1/2 sum_k Tr(Sigma[dP] dP) ]

Units: V(q)/A must come out in your energy units, i.e. V in [energy x length^2]
with lengths in the units used for b1, b2.

Implementation notes
--------------------
* Interaction terms are kept for |k'-k+G| < q_cut (same criterion on and off grid).
* Form factors are never stored. Since they do not change during the SCF, the
  Fock term is precomputed as a kernel K (size (Nk*nb^2)^2, independent of q_cut),
  so each iteration is one matrix-vector product. complex64 memory:
  N=24: 0.2 GB, N=36: 1.1 GB, N=48: 3.4 GB, N=60: 8.4 GB (nb=3; scales as nb^4).
* Choose N divisible by 6 and a grid centred on the Dirac point region: high-
  symmetry points lie on the grid and the states sit centrally in the plane-wave
  cutoff, which keeps G-shifted (embedded) states accurate. check_formfactors()
  reports the weight lost under G-shifts.
"""

import time
import warnings
from collections import defaultdict
import numpy as np

try:
    from scipy.linalg import eigh as _scipy_eigh
except ImportError:  # pragma: no cover
    _scipy_eigh = None


# =============================================================================
# Adapters to the user's code
# =============================================================================
def make_grid_fn(blochstates, band_sel=None, **kwargs):
    """
    Adapter for a grid function blochstates(N, **kwargs) returning a dict with
      "kgrid"       : (N, N, 2) momenta of a regular grid along b1, b2 (any origin/order)
      "bands"       : (N, N, nbands) energies
      "blochstates" : (N, N, 2*nG, nbands) coefficients, G_major (index 2*iG + s)
    band_sel : optional indices into the nbands axis (default: all bands).
    Returns grid_fn(N) -> (kpts (N*N,2), E (N*N,nb), C (N*N,nG,2,nb)).
    """

    def grid_fn(N):
        d = blochstates(N=N, **kwargs)
        k = np.asarray(d["kgrid"], float)
        if k.shape[-1] != 2 and k.shape[0] == 2:
            k = np.moveaxis(k, 0, -1)
        k = k.reshape(N * N, 2)
        E = np.asarray(d["bands"], float).reshape(N * N, -1)
        C = np.asarray(d["blochstates"], complex).reshape(N * N, -1, E.shape[1])
        if band_sel is not None:
            E, C = E[:, band_sel], C[..., band_sel]
        return k, E, C.reshape(N * N, C.shape[1] // 2, 2, E.shape[1])

    return grid_fn


def make_states_fn(build_H, band_indices, nG, layout="G_major", **H_kwargs):
    """
    Optional: kpts -> (E0, C) at arbitrary momenta by diagonalising build_H(k, **H_kwargs).
    Only needed for HF bands along arbitrary k-paths (hf_bands_at without states).
    band_indices: indices of the active bands in the ascending spectrum of build_H.
    """
    band_indices = np.asarray(band_indices)
    lo, hi = int(band_indices.min()), int(band_indices.max())
    contiguous = np.array_equal(band_indices, np.arange(lo, hi + 1))

    def states_fn(kpts):
        kpts = np.atleast_2d(np.asarray(kpts, float))
        nb = len(band_indices)
        E = np.empty((len(kpts), nb))
        C = np.empty((len(kpts), nG, 2, nb), complex)
        for n, k in enumerate(kpts):
            H = build_H(k, **H_kwargs)
            if _scipy_eigh is not None and contiguous:
                w, v = _scipy_eigh(H, subset_by_index=[lo, hi])
            else:
                w, v = np.linalg.eigh(H)
                w, v = w[band_indices], v[:, band_indices]
            E[n] = w
            if layout == "G_major":
                C[n] = v.reshape(nG, 2, nb)
            else:
                C[n] = v.reshape(2, nG, nb).transpose(1, 0, 2)
        return E, C

    return states_fn


def _vectorize_V(V_fn, qscale):
    """Make V_fn safe for arrays and for q = 0 (uses the q -> 0 limit numerically)."""
    qmin = 1e-9 * qscale

    def V(q):
        q = np.asarray(q, float)
        qq = np.where(q < qmin, qmin, q)
        try:
            out = np.asarray(V_fn(qq), float)
            if out.shape != qq.shape:
                raise ValueError
        except Exception:
            out = np.vectorize(V_fn, otypes=[float])(qq)
        if not np.all(np.isfinite(out)):
            raise ValueError("V(q) returned non-finite values")
        return out

    return V


def projected_dirac_sea(kpts, C, Gvecs, dirac, tol=1e-12):
    """
    P_ref(k)_{mn} = <u_{k,m}| Pi_D |u_{k,n}>, with Pi_D the projector onto the
    negative-energy states of the bare 2x2 Dirac Hamiltonian dirac(k+G) at every
    plane wave. Exact zero modes get weight 1/2.

    Assumes build_H uses dirac(k+G) as its diagonal blocks; check this with
    check_dirac_blocks() if unsure.
    """
    nk, nG = C.shape[:2]
    h = np.empty((nk, nG, 2, 2), complex)
    for n, k in enumerate(kpts):
        for g, G in enumerate(Gvecs):
            h[n, g] = dirac(k + G)
    w, v = np.linalg.eigh(h)
    scale = np.abs(w).max()
    wt = np.where(w < -tol * scale, 1.0, np.where(w > tol * scale, 0.0, 0.5))
    Pi = np.einsum("kgan,kgn,kgbn->kgab", v, wt, v.conj())
    return np.einsum("kgam,kgab,kgbn->kmn", C.conj(), Pi, C)


def check_dirac_blocks(H_bare, dirac, k, Gvecs, layout="G_major"):
    """Max deviation between diagonal blocks of a bare build_H (potential off) and dirac(k+G)."""
    nG = len(Gvecs)
    dev = 0.0
    for g, G in enumerate(Gvecs):
        idx = [2 * g, 2 * g + 1] if layout == "G_major" else [g, nG + g]
        dev = max(dev, np.abs(H_bare[np.ix_(idx, idx)] - dirac(k + G)).max())
    return dev


# =============================================================================
# Occupations and mixing
# =============================================================================
def occupy(evals, Ne, T=0.0):
    """Occupations f (same shape as evals) with sum f = Ne. Returns (f, mu)."""
    E = evals.ravel()
    if T > 0:
        lo, hi = E.min() - 50 * T, E.max() + 50 * T
        for _ in range(200):
            mu = 0.5 * (lo + hi)
            n = (0.5 * (1 - np.tanh((E - mu) / (2 * T)))).sum()
            lo, hi = (mu, hi) if n < Ne else (lo, mu)
        mu = 0.5 * (lo + hi)
        f = 0.5 * (1 - np.tanh((E - mu) / (2 * T)))
        return f.reshape(evals.shape), mu
    order = np.argsort(E)
    f = np.zeros_like(E)
    if Ne == 0:
        return f.reshape(evals.shape), E[order[0]]
    mu = E[order[Ne - 1]]
    tol = 1e-9 * max(1.0, np.abs(E).max())
    below = E < mu - tol
    deg = np.abs(E - mu) <= tol
    f[below] = 1.0
    f[deg] = (Ne - below.sum()) / deg.sum()  # share degenerate Fermi-level states
    return f.reshape(evals.shape), mu


class Pulay:
    """Pulay/Anderson mixing of density matrices."""

    def __init__(self, beta=0.5, n_hist=8):
        self.beta, self.n_hist = beta, n_hist
        self.reset()

    def reset(self):
        self.Ps, self.Rs = [], []

    def step(self, Pin, Pout):
        R = Pout - Pin
        if np.abs(R).max() < 1e-13:  # already converged; avoid ill-conditioning
            return Pout
        self.Ps.append(Pin.copy())
        self.Rs.append(R)
        if len(self.Rs) > self.n_hist:
            self.Ps.pop(0)
            self.Rs.pop(0)
        m = len(self.Rs)
        if m == 1:
            return Pin + self.beta * R
        B = np.empty((m + 1, m + 1))
        for i in range(m):
            for j in range(i, m):
                B[i, j] = B[j, i] = np.vdot(self.Rs[i], self.Rs[j]).real
        B[:m, :m] /= np.abs(np.diag(B[:m, :m])).max()
        B[m, :m] = B[:m, m] = 1.0
        B[m, m] = 0.0
        rhs = np.zeros(m + 1)
        rhs[m] = 1.0
        try:
            c = np.linalg.solve(B, rhs)[:m]
        except np.linalg.LinAlgError:
            self.Ps, self.Rs = self.Ps[-1:], self.Rs[-1:]
            return Pin + self.beta * R
        Pbar = sum(ci * Pi for ci, Pi in zip(c, self.Ps))
        Rbar = sum(ci * Ri for ci, Ri in zip(c, self.Rs))
        Pnew = Pbar + self.beta * Rbar
        return 0.5 * (Pnew + Pnew.conj().swapaxes(-1, -2))


# =============================================================================
# Main class
# =============================================================================
class ProjectedHF:
    """
    Parameters
    ----------
    b1, b2     : reciprocal lattice basis vectors (cartesian, shape (2,))
    Gvecs      : (nG, 2) reciprocal lattice vectors, plane-wave order of build_H
    N          : coarse grid size (N x N); choose N divisible by 6
    grid_fn    : N -> (kpts, E0, C) on a regular N x N grid; see make_grid_fn
    V_fn       : |q| -> V(q), e.g. lambda q: gate_screened_coulomb(q, d, eps)
    q_cut      : keep interaction terms with |k'-k+G| < q_cut
    P_ref      : "dirac_sea" (needs dirac), "average" (1/2 * identity), "zero",
                 or an explicit array (Nk, nb, nb)
    dirac      : k -> 2x2 bare Dirac Hamiltonian (for P_ref="dirac_sea")
    states_fn  : optional kpts -> (E0, C) at arbitrary k (only for k-path bands)
    kernel_dtype: dtype of the stored Fock kernel (complex64 halves memory)
    chunk_mb   : temporary memory per chunk while building the kernel
    """

    def __init__(
        self,
        b1,
        b2,
        Gvecs,
        N,
        grid_fn,
        V_fn,
        q_cut,
        P_ref="dirac_sea",
        dirac=None,
        states_fn=None,
        kernel_dtype=np.complex64,
        chunk_mb=256,
        verbose=True,
    ):
        t0 = time.time()
        self.verbose = verbose
        self.N, self.Nk = N, N * N
        self.B = np.column_stack([b1, b2]).astype(float)
        self.Binv = np.linalg.inv(self.B)
        self.A_bz = abs(np.linalg.det(self.B))
        self.A_uc = (2 * np.pi) ** 2 / self.A_bz
        self.A = self.Nk * self.A_uc
        self.orient = np.sign(np.linalg.det(self.B))
        self.grid_fn, self.states_fn = grid_fn, states_fn
        self.q_cut = float(q_cut)
        self.qc_eff = self.q_cut * (1 + 1e-9)
        self.V = _vectorize_V(V_fn, np.linalg.norm(self.B, axis=0).min())
        self.chunk_mb = chunk_mb
        self.int_scale = 1.0

        # reciprocal lattice bookkeeping
        self.Gvecs = np.asarray(Gvecs, float)
        self.nG = len(self.Gvecs)
        gint = self.Gvecs @ self.Binv.T
        self.Gint = np.rint(gint).astype(int)
        if np.abs(gint - self.Gint).max() > 1e-6:
            raise ValueError("Gvecs are not integer combinations of b1, b2")
        self._Glookup = {tuple(n): i for i, n in enumerate(self.Gint)}
        self._sidx_cache = {}

        # coarse grid
        self._log(f"Loading Bloch states on {N}x{N} grid ...")
        g = self._normalize_grid(*grid_fn(N), N)
        self.I, self.kfrac, self.kpts, self.k0 = g["I"], g["kfrac"], g["kpts"], g["k0"]
        self.E0, self.C = g["E"], g["C"]
        self.nb = self.E0.shape[1]
        if self.C.shape[1] != self.nG:
            raise ValueError(
                f"Bloch states have {self.C.shape[1]} sites, Gvecs has {self.nG}"
            )

        # interaction range in fractional units (for bounding G-shift loops)
        a_len = np.linalg.norm(2 * np.pi * self.Binv, axis=1)  # |a1|, |a2|
        self.Rbox = np.ceil(self.q_cut * a_len / (2 * np.pi)).astype(int)

        mem = (self.Nk * self.nb**2) ** 2 * np.dtype(kernel_dtype).itemsize / 1e9
        self._log(f"Fock kernel storage {mem:.2f} GB")
        self.kernel_dtype = kernel_dtype
        self._build_kernels()

        # reference density
        if isinstance(P_ref, str):
            if P_ref == "dirac_sea":
                if dirac is None:
                    raise ValueError("P_ref='dirac_sea' needs the dirac function")
                self.P_ref = projected_dirac_sea(self.kpts, self.C, self.Gvecs, dirac)
            elif P_ref == "average":
                self.P_ref = np.tile(
                    0.5 * np.eye(self.nb, dtype=complex), (self.Nk, 1, 1)
                )
            elif P_ref == "zero":
                self.P_ref = np.zeros((self.Nk, self.nb, self.nb), complex)
            else:
                raise ValueError(f"unknown P_ref scheme {P_ref}")
        else:
            self.P_ref = np.asarray(P_ref, complex)
        self._log(
            f"Setup done in {time.time() - t0:.1f} s. "
            f"Tr P_ref per cell = {np.einsum('kaa->', self.P_ref).real / self.Nk:.4f}"
        )

    # ------------------------------------------------------------------ utils
    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _normalize_grid(self, kpts, E, C, N):
        """
        Sort a regular N x N grid (any origin, order, or points shifted by
        reciprocal lattice vectors) into flat order i1*N + i2 of the residues
        (i1, i2) mod N. kfrac keeps the actual fractional momenta relative to
        k0, so states at k + G are handled exactly via the embedding.
        """
        kpts = np.asarray(kpts, float)
        x = N * (kpts @ self.Binv.T)
        c = x[0] - np.rint(x[0])
        n = np.rint(x - c)
        if np.abs(x - c - n).max() > 1e-6:
            raise ValueError("kgrid is not a regular N x N grid along b1, b2")
        n = n.astype(int)
        I = np.mod(n, N)
        flat = I[:, 0] * N + I[:, 1]
        if len(np.unique(flat)) != N * N:
            raise ValueError("kgrid does not contain every grid point exactly once")
        o = np.argsort(flat)
        return dict(
            I=I[o],
            kfrac=n[o] / N,
            kpts=kpts[o],
            E=np.asarray(E)[o],
            C=np.asarray(C)[o],
            k0=(c / N) @ self.B.T,
        )

    @staticmethod
    def _locate(x, kfrac, N):
        """Grid index j and integer shift g with x = kfrac[j] + g (x fractional)."""
        r = np.mod(np.rint(np.asarray(x) * N).astype(int), N)
        j = r[..., 0] * N + r[..., 1]
        return j, np.rint(x - kfrac[j]).astype(int)

    def _shift_index(self, g):
        key = (int(g[0]), int(g[1]))
        if key not in self._sidx_cache:
            tgt = self.Gint + np.array(key)
            self._sidx_cache[key] = np.array(
                [self._Glookup.get((a, b), self.nG) for a, b in tgt]
            )
        return self._sidx_cache[key]

    def _shifted(self, C, g):
        """Coefficients of the states at k + G_g (in the plane-wave basis at k)."""
        pad = np.concatenate([C, np.zeros_like(C[:, :1])], axis=1)
        return pad[:, self._shift_index(g)]

    @staticmethod
    def _flat(C):
        """(nk, nG, 2, nb) -> (2*nG, nk*nb), column index k*nb + m."""
        nk, nG, _, nb = C.shape
        return C.transpose(1, 2, 0, 3).reshape(2 * nG, nk * nb)

    # ----------------------------------------------------------- form factors
    def formfactor(self, i, Q):
        """Lam(k_i, q) for q = (Q1 b1 + Q2 b2)/N, Q integer (any size)."""
        j, g = self._locate(self.kfrac[i] + np.asarray(Q) / self.N, self.kfrac, self.N)
        Cj = self._shifted(self.C[[j]], g)[0]
        return np.einsum("gsm,gsn->mn", Cj.conj(), self.C[i])

    def _build_kernels(self):
        """
        Fock kernel K[i,a,d,j,b,c] = sum_G V(q) conj(Lam_ba) Lam_cd,  Lam = Lam(k_i, q),
        q = k_j - k_i + G, |q| < q_cut, so that Sigma_F = -(1/A) K . dP.
        Hartree form factors Lam(k, G) for 0 < |G| < q_cut are kept separately.
        """
        t0 = time.time()
        Nk, nb = self.Nk, self.nb
        self.K = np.zeros((Nk, nb, nb, Nk, nb, nb), dtype=self.kernel_dtype)
        C128 = self.C.astype(np.complex128)
        Cflat = self._flat(C128)
        dfrac = self.kfrac[:, None, :] - self.kfrac[None, :, :]  # [j, i] = k_j - k_i
        nic = max(1, int(self.chunk_mb * 1e6 // (Nk * nb**4 * 16 * 3)))
        hG, LG, VG = [], [], []
        npairs, nshift = 0, 0
        gmax = (
            np.ceil(self.kfrac.max(0) - self.kfrac.min(0)).astype(int) + self.Rbox + 1
        )
        for g1 in range(-gmax[0], gmax[0] + 1):
            for g2 in range(-gmax[1], gmax[1] + 1):
                g = np.array([g1, g2])
                qabs = np.linalg.norm((dfrac + g) @ self.B.T, axis=-1)  # [j, i]
                mask = qabs < self.qc_eff
                if not mask.any():
                    continue
                nshift += 1
                npairs += mask.sum()
                W = np.zeros_like(qabs)
                W[mask] = self.V(qabs[mask])
                Cs_full = self._shifted(C128, g)
                if g.any() and np.linalg.norm(g @ self.B.T) < self.qc_eff:  # Hartree G
                    hG.append(g)
                    LG.append(np.einsum("kgsm,kgsn->kmn", Cs_full.conj(), C128))
                    VG.append(self.V(np.linalg.norm(g @ self.B.T)))
                Cs = self._flat(Cs_full).reshape(2 * self.nG, Nk, nb)
                for s0 in range(0, Nk, nic):
                    ic = slice(s0, min(s0 + nic, Nk))
                    js = np.where(mask[:, ic].any(axis=1))[0]
                    if js.size == 0:
                        continue
                    M4 = (
                        Cs[:, js, :].reshape(2 * self.nG, -1).conj().T
                        @ Cflat[:, ic.start * nb : ic.stop * nb]
                    ).reshape(len(js), nb, ic.stop - ic.start, nb)  # [j,b,i,a]
                    Wsub = W[js, ic]
                    A_ = M4.conj().transpose(2, 3, 0, 1)  # (i,a,j,b)
                    B_ = (Wsub[:, None, :, None] * M4).transpose(
                        2, 3, 0, 1
                    )  # (i,d,j,c)
                    self.K[ic, :, :, js] += (
                        A_[:, :, None, :, :, None] * B_[:, None, :, :, None, :]
                    )
        self.hG = np.array(hG).reshape(-1, 2)
        self.LG = np.array(LG).reshape(-1, Nk, nb, nb)
        self.VG = np.array(VG, float).reshape(-1)
        self._Kmat = self.K.reshape(Nk * nb * nb, Nk * nb * nb)
        self._log(
            f"Kernels built in {time.time() - t0:.1f} s: {nshift} G-shifts, "
            f"{npairs / Nk:.0f} interaction terms per k, {len(self.hG)} Hartree G's"
        )

    def check_formfactors(self, lam_fn=None, n=10, seed=0):
        """
        Checks: Lam(k,0) = 1, Lam(k+q,-q) = Lam(k,q)^dag, plane-wave weight lost
        under G-shifts (cutoff adequacy). If lam_fn(k, q) (your λ_formfactor) is
        given, compares singular values (gauge invariant) and raw matrices (equal
        only if both use the same cached states).
        """
        rng = np.random.default_rng(seed)
        Rmax = self.N * (self.Rbox + 1)
        dev0 = max(
            np.abs(self.formfactor(i, (0, 0)) - np.eye(self.nb)).max()
            for i in rng.integers(self.Nk, size=n)
        )
        dev_h, dsv, draw = 0.0, 0.0, 0.0
        for _ in range(n):
            i = rng.integers(self.Nk)
            Q = rng.integers(-Rmax, Rmax + 1)
            L = self.formfactor(i, Q)
            j, _ = self._locate(self.kfrac[i] + Q / self.N, self.kfrac, self.N)
            dev_h = max(dev_h, np.abs(self.formfactor(j, -Q) - L.conj().T).max())
            if lam_fn is not None:
                Lu = np.asarray(lam_fn(self.kpts[i], (Q / self.N) @ self.B.T))
                dsv = max(
                    dsv,
                    np.abs(
                        np.linalg.svd(Lu, compute_uv=False)
                        - np.linalg.svd(L, compute_uv=False)
                    ).max(),
                )
                draw = max(draw, np.abs(Lu - L).max())
        loss = 0.0
        for g in self.hG:
            nrm = np.sum(np.abs(self._shifted(self.C, g)) ** 2, axis=(1, 2))
            loss = max(loss, 1 - nrm.min())
        out = {
            "|Lam(k,0)-1|": dev0,
            "|Lam(k+q,-q)-Lam^dag|": dev_h,
            "max norm loss of G-shifted states": loss,
        }
        if lam_fn is not None:
            out["singular values vs user"] = dsv
            out["raw vs user"] = draw
        return out

    # ---------------------------------------------------------- self-energies
    def set_interaction_scale(self, s):
        """
        Multiply the whole interaction by s without rebuilding anything.
        E.g. if V is proportional to 1/eps: build once at eps0, then use
        s = eps0/eps. Changing d or q_cut changes V's q-dependence -> new object.
        """
        self.int_scale = float(s)

    def sigma_H(self, dP):
        rho = np.einsum("gkab,kba->g", self.LG, dP)
        return (
            self.int_scale
            * np.einsum("g,gkab->kab", self.VG * rho.conj(), self.LG)
            / self.A
        )

    def sigma_F(self, dP):
        out = self._Kmat @ dP.reshape(-1).astype(self.kernel_dtype)
        return (
            -self.int_scale
            * out.astype(np.complex128).reshape(self.Nk, self.nb, self.nb)
            / self.A
        )

    def hamiltonian(self, P):
        dP = P - self.P_ref
        H = self.sigma_H(dP) + self.sigma_F(dP)
        H[:, np.arange(self.nb), np.arange(self.nb)] += self.E0
        return 0.5 * (H + H.conj().swapaxes(-1, -2))

    def energy(self, P):
        """Total HF energy per unit cell (relative to the reference state)."""
        dP = P - self.P_ref
        Sig = self.sigma_H(dP) + self.sigma_F(dP)
        e0 = np.einsum("ka,kaa->", self.E0, P).real
        eint = 0.5 * np.einsum("kab,kba->", Sig, dP).real
        return (e0 + eint) / self.Nk

    # ------------------------------------------------------------ initial P
    def _P_from_H(self, H, Ne):
        ev, vec = np.linalg.eigh(H)
        f, _ = occupy(ev, Ne)
        return np.einsum("kan,kn,kbn->kab", vec, f, vec.conj())

    def noninteracting_P(self, filling):
        H = np.zeros((self.Nk, self.nb, self.nb), complex)
        H[:, np.arange(self.nb), np.arange(self.nb)] = self.E0
        return self._P_from_H(H, self._Ne(filling))

    def random_P(self, filling, amplitude, seed=None):
        """Seed: diagonalise h0 + random Hermitian noise (energy scale `amplitude`)."""
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(self.Nk, self.nb, self.nb)) + 1j * rng.normal(
            size=(self.Nk, self.nb, self.nb)
        )
        H = amplitude * 0.5 * (X + X.conj().swapaxes(-1, -2))
        H[:, np.arange(self.nb), np.arange(self.nb)] += self.E0
        return self._P_from_H(H, self._Ne(filling))

    def _Ne(self, filling):
        Nef = filling * self.Nk
        Ne = int(round(Nef))
        if abs(Nef - Ne) > 1e-8:
            warnings.warn(f"filling*Nk = {Nef} is not an integer; using {Ne}")
        return Ne

    # ------------------------------------------------------------------ SCF
    def run(
        self,
        filling,
        P_init=None,
        T=0.0,
        T_start=None,
        anneal=0.85,
        beta=0.5,
        n_hist=8,
        tol=1e-6,
        maxiter=500,
        print_every=10,
    ):
        """
        filling : electrons per unit cell inside the active window (0 ... nb)
        T       : final smearing temperature (energy units); T_start > T anneals
                  geometrically (factor `anneal` per iteration) down to T.
        tol     : convergence threshold on max |P_out - P_in|
        """
        t0 = time.time()
        Ne = self._Ne(filling)
        P = (
            self.noninteracting_P(filling)
            if P_init is None
            else np.array(P_init, complex)
        )
        mixer = Pulay(beta, n_hist)
        converged, history = False, []
        annealing = T_start is not None and T_start > T
        for it in range(maxiter):
            Tcur = T
            if annealing:
                Tcur = T_start * anneal**it
                if Tcur <= max(T, 1e-3 * T_start):  # end of anneal: switch to final T
                    Tcur, annealing = T, False
                    mixer.reset()
            H = self.hamiltonian(P)
            ev, vec = np.linalg.eigh(H)
            f, mu = occupy(ev, Ne, Tcur)
            Pout = np.einsum("kan,kn,kbn->kab", vec, f, vec.conj())
            err = np.abs(Pout - P).max()
            history.append(err)
            if self.verbose and (it % print_every == 0):
                print(
                    f"  it {it:4d}  err {err:.2e}  mu {mu:+.5f}  T {Tcur:.2e}",
                    flush=True,
                )
            if err < tol and not annealing:
                converged = True
                break
            P = mixer.step(P, Pout)
        P = Pout
        H = self.hamiltonian(P)
        ev, vec = np.linalg.eigh(H)
        E = self.energy(P)
        self._log(
            f"{'Converged' if converged else 'NOT converged'} after {it + 1} iterations "
            f"({time.time() - t0:.1f} s), E = {E:.8f} per cell"
        )
        return dict(
            P=P,
            H=H,
            evals=ev,
            evecs=vec,
            mu=mu,
            energy=E,
            converged=converged,
            n_iter=it + 1,
            history=np.array(history),
            gaps=band_gaps(ev),
        )

    # ------------------------------------------------------- off-grid H_HF
    def hf_hamiltonian_at(self, kpts, P, chunk=128, states=None):
        """
        Exact HF Hamiltonian at arbitrary momenta from the converged coarse-grid P.
        states = (E0, C) at kpts; if omitted, self.states_fn(kpts) is used.
        Returns (E0, C, H). At coarse-grid points this reproduces the SCF bands.
        """
        kpts = np.atleast_2d(np.asarray(kpts, float))
        nf, nb = len(kpts), self.nb
        if states is None:
            if self.states_fn is None:
                raise ValueError("pass states=(E0, C) or construct with states_fn")
            states = self.states_fn(kpts)
        E0f, Cf = states
        Cf = Cf.astype(np.complex128)
        dP = P - self.P_ref
        Sig = np.zeros((nf, nb, nb), complex)

        # Hartree
        rho = np.einsum("gkab,kba->g", self.LG, dP)
        for gi, G in enumerate(self.hG):
            Lf = np.einsum("kgsm,kgsn->kmn", self._shifted(Cf, G).conj(), Cf)
            Sig += self.VG[gi] * rho[gi].conj() * Lf / self.A

        # Fock: q = k'_c - k_f + G
        kf_all = (kpts - self.k0) @ self.Binv.T
        lo = np.floor(kf_all.min(0)).astype(int) - self.Rbox - 1
        hi = np.ceil(kf_all.max(0)).astype(int) + self.Rbox + 1
        chunks = [slice(s, min(s + chunk, nf)) for s in range(0, nf, chunk)]
        Cff = [self._flat(Cf[sl]) for sl in chunks]
        for g1 in range(lo[0], hi[0] + 1):
            for g2 in range(lo[1], hi[1] + 1):
                g = np.array([g1, g2])
                Cs = None
                for sl, Cfl in zip(chunks, Cff):
                    qfrac = self.kfrac[None, :, :] + g - kf_all[sl, None, :]
                    qabs = np.linalg.norm(qfrac @ self.B.T, axis=-1)  # (nfc, Nk)
                    mask = qabs < self.qc_eff
                    if not mask.any():
                        continue
                    if Cs is None:
                        Cs = self._flat(self._shifted(self.C.astype(np.complex128), g))
                    nfc = sl.stop - sl.start
                    L = (
                        (Cs.conj().T @ Cfl)
                        .reshape(self.Nk, nb, nfc, nb)
                        .transpose(2, 0, 1, 3)
                    )
                    W = np.zeros_like(qabs)
                    W[mask] = self.V(qabs[mask])
                    T = L.conj().swapaxes(-1, -2) @ dP[None] @ L
                    Sig[sl] -= np.einsum("fc,fcab->fab", W, T) / self.A
        H = self.int_scale * Sig
        H[:, np.arange(nb), np.arange(nb)] += E0f
        return E0f, Cf, 0.5 * (H + H.conj().swapaxes(-1, -2))

    def hf_bands_at(self, kpts, P, chunk=128, states=None):
        return np.linalg.eigvalsh(self.hf_hamiltonian_at(kpts, P, chunk, states)[2])

    # ------------------------------------------------------- Berry curvature
    def fine_grid(self, P, Nf, chunk=128):
        """
        HF solution on the Nf x Nf grid of grid_fn(Nf).
        Compute once and pass to berry_fhs(..., fine=...) for every band group.
        """
        g = self._normalize_grid(*self.grid_fn(Nf), Nf)
        _, Cf, Hf = self.hf_hamiltonian_at(g["kpts"], P, chunk, states=(g["E"], g["C"]))
        ev, a = np.linalg.eigh(Hf)
        return dict(
            Nf=Nf, kfrac=g["kfrac"], kpts=g["kpts"], C=Cf, H=Hf, evals=ev, evecs=a
        )

    def berry_fhs(self, P, Nf, bands, chunk=128, fine=None):
        """
        Fukui-Hatsugai-Suzuki Berry curvature of the HF band group `bands`
        (indices 0..nb-1 into the ascending HF spectrum) on the Nf x Nf grid of
        grid_fn(Nf). Omega is returned on that grid in residue order (i1, i2).
        Pass fine=self.fine_grid(P, Nf) to reuse the fine-grid HF solution.

        Convention: A = i<u|grad u>, Omega = curl A, C = (1/2pi) int Omega.
        Returns dict(Omega (Nf,Nf), chern, evals (Nf*Nf, nb), kpts, min_gap_below/above).
        """
        bands = np.atleast_1d(bands)
        if fine is None or fine["Nf"] != Nf:
            fine = self.fine_grid(P, Nf, chunk)
        kfr, kpts, Cf, ev, a = (
            fine["kfrac"],
            fine["kpts"],
            fine["C"],
            fine["evals"],
            fine["evecs"],
        )
        Psi = np.einsum("kgsm,kmn->kgsn", Cf, a[:, :, bands])

        links = []
        for mu in (np.array([1, 0]), np.array([0, 1])):
            jj, gsh = self._locate(kfr + mu / Nf, kfr, Nf)
            Pn = Psi[jj]
            for gv in np.unique(gsh, axis=0):
                if not gv.any():
                    continue
                sel = np.all(gsh == gv, axis=1)
                Pn[sel] = self._shifted(Psi[jj[sel]], gv)
            ov = np.einsum("kgsm,kgsn->kmn", Psi.conj(), Pn)
            d = np.linalg.det(ov)
            if np.abs(d).min() < 1e-6:
                warnings.warn(
                    "tiny link overlap: band group not gapped or Nf too small"
                )
            links.append((d / np.abs(d)).reshape(Nf, Nf))
        U1, U2 = links
        F = np.angle(
            U1 * np.roll(U2, -1, axis=0) * np.roll(U1, -1, axis=1).conj() * U2.conj()
        )
        dA = self.A_bz / Nf**2
        Omega = -self.orient * F / dA
        chern = -self.orient * F.sum() / (2 * np.pi)
        out = dict(Omega=Omega, chern=chern, evals=ev, kpts=kpts)
        if bands.min() > 0:
            out["min_gap_below"] = (ev[:, bands.min()] - ev[:, bands.min() - 1]).min()
        if bands.max() < self.nb - 1:
            out["min_gap_above"] = (ev[:, bands.max() + 1] - ev[:, bands.max()]).min()
        return out


def band_gaps(evals):
    """Direct (min over k) and indirect gaps between consecutive bands, shape (nb-1,)."""
    direct = (evals[:, 1:] - evals[:, :-1]).min(0)
    indirect = evals[:, 1:].min(0) - evals[:, :-1].max(0)
    return dict(direct=direct, indirect=indirect)

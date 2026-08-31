import numpy as np
from scipy.integrate import quad
from scipy.special import assoc_laguerre
import myBM as bm

hbarvf = 0.6582


def integrand(E, λ, m, sign=1):
    """
    A nice form of the integrand for the Bohr-Sommerfeld quantization condition, as well as the integration boundary.
    Returns f,Δ, such that ∫fdx from -Δ to Δ = nπ is the quantization condition.
    """
    # Define special points in dimensionless units
    x0 = E / 2 * 1 / np.sqrt(λ * hbarvf)
    Δ = np.sqrt(x0**2 - m)
    γ = np.sqrt(x0**2 + m)
    return (lambda x: np.sqrt(sign * ((x0 - x) ** 2 - m**2 / (x0 + x) ** 2)), Δ, γ)


def integral(wave="oscillating", **kwargs):
    if wave == "oscillating":
        f, Δ, γ = integrand(**kwargs)
        result, _ = quad(f, -Δ, Δ)
    elif wave == "evanescent":
        f, Δ, γ = integrand(**kwargs, sign=-1)
        result, _ = quad(f, Δ, γ)
    else:
        raise ValueError("wave should be oscillating or evanescent.")
    return result


def classical_regions(E, λ, m):
    rm = E / 2 / λ
    mscaled = m * (hbarvf / λ)
    print(rm)
    Δ = np.sqrt(rm**2 - mscaled)
    γ = np.sqrt(rm**2 + mscaled)
    return {"r0": rm - Δ, "r1": rm + Δ, "r2": rm + γ}


def decayrate(E, T, m, λ):
    Δr = np.sqrt(E**2 / 4 / λ**2 - m * hbarvf / λ)
    freq = hbarvf / 4 / Δr
    return T**2 * freq


def quantization(n, λ, m, damping=0.5, N_iter=100, **kwargs):
    E_min = np.sqrt(hbarvf * m * λ) * 2 + 1e-3
    E_max = E_min + 0.5
    E = E_max / 2 + E_min / 2
    q_cond = integral(E=E, **kwargs, λ=λ, m=m) / np.pi
    itercount = 0
    while not np.isclose(q_cond, n):
        if itercount > N_iter:
            print("Not converged.")
            break
        if q_cond > n:
            E_max = E
        else:
            E_min = E
        itercount += 1
        E = E_max / 2 + E_min / 2
        q_cond = integral(E=E, λ=λ, m=m, **kwargs) / np.pi
    return E, q_cond


def hydrogen(r, n, l, a):
    ρ = r / a / n
    ν = assoc_laguerre(2 * ρ, n - l - 1, 2 * l + 1)
    return 1 / r * ρ ** (l + 1) * np.exp(-ρ) * ν


def linear_potential_fit(θ, w):
    """
    Returns the slope λ and offset c of a linear potential fitted to the hexagonal cosine from a twist angle θ and potential strength w.
    """
    kθ = bm.get_q(θ / 180 * np.pi)
    return np.pi * w * kθ, 0.84 / 0.11 * w


##From semi-analytical analysis:
quasibound_spectrum = np.array(
    [
        0.6845881210303729,
        0.6811472924578414,
        0.845450658250141,
        0.9751747626791805,
        1.0894806798625845,
        1.1901434398148298,
        1.288157063902264,
        1.3763026656257247,
        1.4487142595939735,
        1.5295601747060077,
        1.6067680129177182,
        1.6742933613989153,
    ]
)
quasi_bound_decayrates = np.array(
    [
        0.22836814704377942,
        0.20865156792438977,
        0.19942460738008314,
        0.18910012677752422,
        0.18129687446911047,
        0.17557872050660367,
        0.17091311907333967,
        0.16676100285836465,
        0.163663985588325,
        0.16019239322186876,
        0.15758232026909388,
        0.15518122935541992,
    ]
)

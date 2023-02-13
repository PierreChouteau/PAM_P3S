# This script is based on the work of Kouhei Sekiguchi, Member, IEEE, Yoshiaki Bando, Member, IEEE, Aditya Arie Nugraha, Member, IEEE,
# Kazuyoshi Yoshii, Member, IEEE, and Tatsuya Kawahara, Fellow, IEEE in the article "Fast Multichannel Nonnegative Matrix Factorization
# with Directivity-Aware Jointly-Diagonalizable Spatial Covariance Matrices for Blind Source Separation", 2020

# The source code of K. Sekiguchi can be found here: https://github.com/sekiguchi92/SoundSourceSeparation

# This script is a re-implementation of FastNMF2 to add other constraints and different initialization methods more freely

from numpy.typing import ArrayLike
from matplotlib import pyplot as plt

try:
    import cupy as np

    print("Using GPU")
except ImportError:
    import numpy as np


def normalize(
    W_NFK: ArrayLike,
    H_NKT: ArrayLike,
    G_tilde_NM: ArrayLike,
    Q_FMM: ArrayLike,
):
    """Normalize Updatable parameters"""
    # Manipulations with None are basically equivalent to reshape

    F, M, M = Q_FMM.shape
    phi_F = np.einsum("fij, fij -> f", Q_FMM, Q_FMM.conj()).real / M
    Q_FMM /= np.sqrt(phi_F)[:, None, None]
    W_NFK /= phi_F[None, :, None]

    mu_N = G_tilde_NM.sum(axis=1)
    G_tilde_NM /= mu_N[:, None]
    W_NFK *= mu_N[:, None, None]

    # Norm NMF
    nu_NK = W_NFK.sum(axis=1)
    W_NFK /= nu_NK[:, None]
    H_NKT *= nu_NK[:, :, None]

    return W_NFK, H_NKT, G_tilde_NM, Q_FMM


def init_WH(
    n_FFT: int,
    n_time_frames: int,
    n_basis: int,
    n_sources: int,
):
    """Initialize W and H of source model

    Input
    -----
    - n_FFT         (allias F): int = STFT window length
    - n_time_frames (allias T): int = STFT number of time frames
    - n_basis       (allias K): int = Number of elements (cols for W, rows for H) in W and H
    - n_sources     (allias N): int = Number of instruments to separate from the mix

    Output
    ------
    - W_NFK:        Array [N, F, K] = Spectral base of NMF
    - H_NKT:        Array [N, K, T] = Activation matrix of NMF
    """
    W_NFK = np.random.rand(n_sources, n_FFT, n_basis)
    H_NKT = np.random.rand(n_sources, n_basis, n_time_frames)
    return W_NFK, H_NKT


def init_GQ(
    init_type: str,
    n_FFT: int,
    n_sources: int,
    n_sensors: int,
    G_tilde_NM: ArrayLike = None,
    Q_FMM: ArrayLike = None,
) -> tuple:
    """Initialize FastMNMF2 parameters.

    Input
    -----
    - init_type:                str = random, diagonal, circular or gradual
    - n_FFT         (allias F): int = STFT window length
    - n_sources     (allias N): int = Number of instruments to separate from the mix
    - n_sensors     (allias M): int = Number of microphones

    Outputs
    -------
    - G_tilde_NM:   Array [N, M]    = Spatial Covariance matrix
    - Q_FMM:        Array [F, M, M] = Diagonalizer of G

    Init types
    ----------
    - Random: Bad overall
    - Diagonnal: OK for determined or underdetermined
    - Circular: OK for over-determined (equivalent to diagonnal in other cases)
    - Gradual: Most stable case. Uses Circular under the hood
    """
    init_type_dict = {"RANDOM": 0, "DIAGONAL": 1, "CIRCULAR": 2, "GRADUAL": 3}
    G_eps = 5e-2  # Norm coef for G
    try:
        ind = init_type_dict[init_type.upper()]
    except KeyError:
        print("Wrong init type name, circular is chosen by default")
        ind = 2
    match ind:
        case 0:
            # Random init
            G_tilde_NM = np.random.rand(n_sources, n_sensors)
            Q_FMM = np.random.rand(n_FFT, n_sensors, n_sensors, dtype=np.complex_)
        case 1:
            # Diagonal init
            G_tilde_NM = np.zeros((n_sources, n_sensors)) + G_eps
            Q_FMM = np.tile(np.eye(n_sensors, dtype=np.complex_), (n_FFT, 1, 1))

            for n in range(min(n_sources, n_sensors)):
                G_tilde_NM[n, n] = 1
        case 2:
            # Circular init
            G_tilde_NM = np.zeros((n_sources, n_sensors)) + G_eps
            Q_FMM = np.tile(np.eye(n_sensors, dtype=np.complex_), (n_FFT, 1, 1))

            for m in range(n_sensors):
                G_tilde_NM[m % n_sources, m] = 1
        case 3:
            # Gradual init
            if (G_tilde_NM, Q_FMM) == (None, None):
                # First init with small K
                G_tilde_NM = np.zeros((n_sources, n_sensors)) + G_eps
                Q_FMM = np.tile(np.eye(n_sensors, dtype=np.complex_), (n_FFT, 1, 1))
                for m in range(n_sensors):
                    G_tilde_NM[m % n_sources, m] = 1
    return G_tilde_NM, Q_FMM


def init_IP(X_FTM: ArrayLike) -> ArrayLike:
    """Compute an intermediate result for Q_IP update

    Input
    -----
    - X_FTM: Array [F, T, M] = Observed Spectrogram

    Output
    ------
    - XX_FTMM: Array [F, T, M, M]
    """
    XX_FTMM = np.einsum("fti, ftj -> ftij", X_FTM, X_FTM.conj())
    return XX_FTMM


def init_UT_split(
    n_sources: int,
    n_basis: int,
    n_notes: int,
    n_activations: int,
):
    """Init W and H matrices for split step NMF

    Input
    -----
    - n_sources:     int = Number of sources to separate
    - n_basis:       int = Number of basis elements
    - n_notes:       int = Number of notes

    Output
    ------
    - U_NLK:        Array [N, L, K] = Source spectral weights
    - T_NKO         Array [N, K, O] = Time pattern weights
    """
    U_NLK = np.random.rand(n_sources, n_notes, n_basis)
    T_NKO = np.random.rand(n_sources, n_basis, n_activations)

    return U_NLK, T_NKO


def init_EP_split(
    n_sources: int,
    n_FFT: int,
    n_time_frames: int,
    n_notes: int,
    n_activations: int,
):
    """Init W and H matrices for split step NMF

    Input
    -----
    - n_sources:     int = Number of sources to separate
    - n_FFT:         int = Number of frequency bins
    - n_time_frames: int = Number of time frames
    - n_notes:       int = Number of notes
    - n_activations: int = Number of activations

    Output
    ------
    - E_NFL:        Array [N, F, L] = Source spectral patterns
    - P_NOT:        Array [N, O, T] = Time patterns
    """
    E_NFL = np.zeros((n_sources, n_FFT, n_notes))
    P_NOT = np.zeros((n_sources, n_activations, n_time_frames))

    m = max(n_FFT, n_notes)
    for k in range(m):
        E_NFL[:, k % n_FFT, k % n_notes] = 1
    m = max(n_activations, n_time_frames)
    for k in range(m):
        P_NOT[:, k % n_activations, k % n_time_frames] = 1

    return E_NFL, P_NOT


def inverse_EP(
    E_NFL: ArrayLike,
    P_NOT: ArrayLike,
):
    """

    Input
    -----
    - E_NFL:        Array [N, F, L] = Source spectral patterns
    - P_NOT:        Array [N, O, T] = Time patterns

    Output
    ------
    - E_inv_NLF:    Array [N, L, F] = Inverse of E_NFL
    - P_inv_NTO:    Array [N, T, O] = Inverse of P_NOT
    """
    _, F, L = E_NFL.shape
    _, O, T = P_NOT.shape

    if F < L:
        raise ValueError(
            f"Bad dimension choice: n_FFT={F} < n_notes={L}. No left inverse exists."
        )
    if O > T:
        raise ValueError(
            f"Bad dimension choice: n_activations={O} > n_time_frames={T}. No right inverse exists."
        )

    # Left inverse
    E_inv_NLF = np.linalg.pinv(E_NFL)

    # Right inverse
    P_inv_NTO = np.linalg.pinv(P_NOT)

    return E_inv_NLF, P_inv_NTO


def update_W(
    W_old_NFK: ArrayLike,
    G_tilde_NM: ArrayLike,
    X_tilde_FTM: ArrayLike,
    Y_tilde_FTM: ArrayLike,
    H_NKT: ArrayLike,
) -> ArrayLike:
    """Update W. Eq34

    Input
    -----
    - W_old_NFK:    Array [N, F, K] = Source model base matrix
    - G_tilde_NM:   Array [N, M]    = Diag coefficients of the diagonalized demixing matrix
    - X_tilde_FTM:  Array [F, T, M] = Power Spectral Density at each microphone
    - Y_tilde_FTM:  Array [F, T, M] = Sum of (PSD_NFT x G_tilde_NM) over all sources
    - H_NKT:        Array [N, K, T] = Source model activation matrix

    Output
    ------
    - W_new_NFK: Array [N, F, K]
    """

    tmp1_NFT = np.einsum("nm, ftm -> nft", G_tilde_NM, X_tilde_FTM / (Y_tilde_FTM**2))
    tmp2_NFT = np.einsum("nm, ftm -> nft", G_tilde_NM, 1 / Y_tilde_FTM)

    numerator = np.einsum("nkt, nft -> nfk", H_NKT, tmp1_NFT)
    denominator = np.einsum("nkt, nft -> nfk", H_NKT, tmp2_NFT)

    W_new_NFK = W_old_NFK * np.sqrt(numerator / denominator)
    return W_new_NFK


def update_W_split(
    E_NFL: ArrayLike,
    U_NLK: ArrayLike,
    E_inv_NLF: ArrayLike,
    G_tilde_NM: ArrayLike,
    X_tilde_FTM: ArrayLike,
    Y_tilde_FTM: ArrayLike,
    H_NKT: ArrayLike,
) -> ArrayLike:
    """Update W as the product E @ U with E fixed and U updated.

    It is based on the paper "A general and flexible framework for the handling of prior information in Audio Source Separation" by Ozerov & al.
    E is fixed and U is updated. The matrices sizes can be found at page 4.

    Input
    -----
    - E_NFL:        Array [N, F, L] = Source spectral patterns
    - U_NLK:        Array [N, L, K] = Source spectral weights
    - E_inv_NLF:    Array [N, L, F] = Inverse of E_NFL
    - G_tilde_NM:   Array [N, M]    = Diag coefficients of the diagonalized demixing matrix
    - X_tilde_FTM:  Array [F, T, M] = Power Spectral Density at each microphone
    - Y_tilde_FTM:  Array [F, T, M] = Sum of (PSD_NFT x G_tilde_NM) over all sources
    - H_NKT:        Array [N, K, T] = Source model activation matrix

    Output
    ------
    - U_new_NLK: Array [N, L, K]
    """

    tmp1_NFT = np.einsum("nm, ftm -> nft", G_tilde_NM, X_tilde_FTM / (Y_tilde_FTM**2))
    tmp2_NFT = np.einsum("nm, ftm -> nft", G_tilde_NM, 1 / Y_tilde_FTM)

    numerator = np.einsum("nkt, nft -> nfk", H_NKT, tmp1_NFT)
    denominator = np.einsum("nkt, nft -> nfk", H_NKT, tmp2_NFT)

    W_old_NFK = np.einsum("nfl, nlk -> nfk", E_NFL, U_NLK)

    U_new_NLK = np.einsum(
        "nlf, nfk -> nlk", E_inv_NLF, (W_old_NFK * np.sqrt(numerator / denominator))
    )
    return U_new_NLK


def update_H(
    H_old_NKT: ArrayLike,
    G_tilde_NM: ArrayLike,
    X_tilde_FTM: ArrayLike,
    Y_tilde_FTM: ArrayLike,
    W_NFK: ArrayLike,
) -> ArrayLike:
    """Update H. Eq35

    Input
    -----
    - H_old_NKT:    Array [N, K, T] = Source model activation matrix
    - G_tilde_NM:   Array [N, M]    = Diag coefficients of the diagonalized demixing matrix
    - X_tilde_FTM:  Array [F, T, M] = Power Spectral Density at each microphone
    - Y_tilde_FTM:  Array [F, T, M] = Sum of (PSD_NFT x G_tilde_NM) over all sources
    - W_NFK:        Array [N, F, K] = Source model base matrix

    Output
    ------
    - H_new_NKT: Array [N, K, T]
    """

    tmp1_NFT = np.einsum("nm, ftm -> nft", G_tilde_NM, X_tilde_FTM / (Y_tilde_FTM**2))
    tmp2_NFT = np.einsum("nm, ftm -> nft", G_tilde_NM, 1 / Y_tilde_FTM)

    numerator = np.einsum("nfk, nft -> nkt", W_NFK, tmp1_NFT)
    denominator = np.einsum("nfk, nft -> nkt", W_NFK, tmp2_NFT)

    H_new_NKT = H_old_NKT * np.sqrt(numerator / denominator)
    return H_new_NKT


def update_H_split(
    T_old_NKO: ArrayLike,
    P_NOT: ArrayLike,
    P_inv_NTO: ArrayLike,
    G_tilde_NM: ArrayLike,
    X_tilde_FTM: ArrayLike,
    Y_tilde_FTM: ArrayLike,
    W_NFK: ArrayLike,
) -> ArrayLike:
    """Update H as the product T @ P with P fixed and T updated.

    It is based on the paper "A general and flexible framework for the handling of prior information in Audio Source Separation" by Ozerov & al.
    E is fixed and U is updated. The matrices sizes can be found at page 4.

    Input
    -----
    - T_old_NKO     Array [N, K, O] = Time pattern weights
    - P_NOT:        Array [N, O, T] = Time patterns
    - P_inv_NTO:    Array [N, T, O] = Inverse of P_NOT
    - G_tilde_NM:   Array [N, M]    = Diag coefficients of the diagonalized demixing matrix
    - X_tilde_FTM:  Array [F, T, M] = Power Spectral Density at each microphone
    - Y_tilde_FTM:  Array [F, T, M] = Sum of (PSD_NFT x G_tilde_NM) over all sources
    - W_NFK:        Array [N, F, K] = Source model base matrix

    Output
    ------
    - T_new_NKO: Array [N, K, O]
    """

    tmp1_NFT = np.einsum("nm, ftm -> nft", G_tilde_NM, X_tilde_FTM / (Y_tilde_FTM**2))
    tmp2_NFT = np.einsum("nm, ftm -> nft", G_tilde_NM, 1 / Y_tilde_FTM)

    numerator = np.einsum("nfk, nft -> nkt", W_NFK, tmp1_NFT)
    denominator = np.einsum("nfk, nft -> nkt", W_NFK, tmp2_NFT)

    H_old_NKT = np.einsum("nko, not -> nkt", T_old_NKO, P_NOT)

    T_new_NKO = np.einsum(
        "nkt, nto -> nko", H_old_NKT * np.sqrt(numerator / denominator), P_inv_NTO
    )

    return T_new_NKO


def update_G(
    G_tilde_old_NM: ArrayLike,
    PSD_NFT: ArrayLike,
    X_tilde_FTM: ArrayLike,
    Y_tilde_FTM: ArrayLike,
) -> ArrayLike:
    """Update G_tilde. Eq 36

    Input
    -----
    - G_tilde_old_NM:   Array [N, M]    = Diag coefficients of the diagonalized demixing matrix
    - PSD_NFT:          Array [N, F, T] = Power Spectral densities of the sources
    - X_tilde_FTM:      Array [F, T, M] = Power Spectral Density at each microphone
    - Y_tilde_FTM:      Array [F, T, M] = Sum of (PSD_NFT x G_tilde_NM) over all sources

    Output
    ------
    - G_tilde_new_NM: Array [N, M]"""

    numerator = np.einsum("nft, ftm -> nm", PSD_NFT, X_tilde_FTM / (Y_tilde_FTM**2))
    denominator = np.einsum("nft, ftm -> nm", PSD_NFT, 1 / Y_tilde_FTM)

    G_tilde_new_NM = G_tilde_old_NM * np.sqrt(numerator / denominator)
    return G_tilde_new_NM


def update_Q_IP(
    Q_FMM: ArrayLike,
    XX_FTMM: ArrayLike,
    Y_tilde_FTM: ArrayLike,
) -> ArrayLike:
    """Update Q_FMM. Eq24

    Input
    -----
    - Q_old_FMM:    Array [F, M, M]     = Diagonalizer of G
    - XX_FTMM:      Array [F, T, M, M]  = X_FT * X_FT^H
    - Y_tilde_FTM:  Array [F, T, M]     = Sum of (PSD_NFT x G_tilde_NM) over all sources

    Output
    ------
    - Q_new_FMM: Array [F, M, M]
    """

    F, T, M = Y_tilde_FTM.shape

    for m in range(M):
        V_FMM = np.einsum("ftij, ft -> fij", XX_FTMM, 1 / Y_tilde_FTM[..., m]) / T
        tmp_FM = np.linalg.inv(Q_FMM @ V_FMM)[..., m]
        Q_FMM[:, m] = (
            tmp_FM
            / np.sqrt(np.einsum("fi, fij, fj -> f", tmp_FM.conj(), V_FMM, tmp_FM))[
                :, None
            ]
        ).conj()

    return Q_FMM


def update_Q_ISS(
    Q_old_FMM: ArrayLike,
    Qx_FTM: ArrayLike,
    Y_tilde_FTM: ArrayLike,
) -> ArrayLike:
    """Update Q_FMM. Eq25

    Input
    -----
    - Q_old_FMM:    Array [F, M, M] = Diagonalizer of G
    - Qx_FTM:       Array [F, T, M]
    - Y_tilde_FTM:  Array [F, T, M] = Sum of (PSD_NFT x G_tilde_NM) over all sources

    Output
    ------
    - Q_new_FMM: Array [F, M, M]
    """

    F, T, M = Y_tilde_FTM.shape

    for m in range(M):
        QxQx_FTM = Qx_FTM * Qx_FTM[:, :, m, None].conj()
        V_tmp_FxM = (QxQx_FTM[:, :, m, None] / Y_tilde_FTM).mean(axis=1)
        V_FxM = (QxQx_FTM / Y_tilde_FTM).mean(axis=1) / V_tmp_FxM
        V_FxM[:, m] = 1 - 1 / np.sqrt(V_tmp_FxM[:, m])
        Qx_new_FTM = Qx_FTM - np.einsum("fm, ft -> ftm", V_FxM, Qx_FTM[:, :, m])
        Q_new_FMM = Q_old_FMM - np.einsum("fi, fj -> fij", V_FxM, Q_old_FMM[:, m])

    return Q_new_FMM, Qx_new_FTM


def calculate_X_tilde(X_FTM: ArrayLike, Q_FMM: ArrayLike):
    """Calculate X_tilde_FTM. Eq31

    Input
    -----
    - X_FTM: Array [F, T, M] = Observed spectrogram
    - Q_FMM: Array [F, M, M] = Diagonalizer of G

    Output
    ------
    - Qx_FTM:      Array [F, T, M]
    - X_tilde_FTM: Array [F, T, M] = Power Spectral Density at each microphone
    """

    Qx_FTM = np.einsum("fmi, fti -> ftm", Q_FMM, X_FTM)
    X_tilde_FTM = np.abs(Qx_FTM) ** 2

    return Qx_FTM, X_tilde_FTM


def calculate_PSD(W_NFK: ArrayLike, H_NKT: ArrayLike):
    """Calculate PSD. NMF result

    Input
    -----
    - W_NFK
    - H_NKT

    Output
    ------
    - PSD_NFT
    """
    EPS = 1e-10
    return W_NFK @ H_NKT + EPS


def calculate_Y_tilde(G_tilde_NM: ArrayLike, PSD_NFT: ArrayLike):
    """Calculate Y_tilde_FTM. Eq31

    Input
    -----
    - G_tilde_old_NM:   Array [N, M]    = Diag coefficients of the diagonalized demixing matrix
    - PSD_NFT:          Array [N, F, T] = Power Spectral densities of the sources

    Output
    ------
    - Y_tilde_FTM:  Array [F, T, M] = Sum of (PSD_NFT x G_tilde_NM) over all sources
    """
    EPS = 1e-10
    Y_tilde_FTM = np.einsum("nft, nm -> ftm", PSD_NFT, G_tilde_NM) + EPS
    return Y_tilde_FTM


def calculate_log_likelihood(
    X_tilde_FTM: ArrayLike,
    Y_tilde_FTM: ArrayLike,
    Q_FMM: ArrayLike,
) -> float:
    """This function computes the log likelihood of FastMNMF2

    Input
    -----
    - X_tilde_FTM:      Array [F, T, M] = Power Spectral Density at each microphone
    - Y_tilde_FTM:      Array [F, T, M] = Sum of (PSD_NFT x G_tilde_NM) over all sources
    - Q_FMM: Array [F, M, M] = Diagonalizer of G

    Output
    ------
    - Log likelihood: Float
    """

    F, T, M = X_tilde_FTM.shape
    log_likelihood = (
        -(X_tilde_FTM / Y_tilde_FTM + np.log(Y_tilde_FTM)).sum()
        + T * (np.log(np.linalg.det(Q_FMM @ Q_FMM.transpose(0, 2, 1).conj()))).sum()
    ).real
    return log_likelihood


def update_all_params(
    X_FTM: ArrayLike,
    W_NFK: ArrayLike,
    H_NKT: ArrayLike,
    G_tilde_NM: ArrayLike,
    Q_FMM: ArrayLike,
    Qx_FTM: ArrayLike,
    X_tilde_FTM: ArrayLike,
    Y_tilde_FTM: ArrayLike,
    XX_FTMM: ArrayLike,
    index_iteration: int,
    algo: str = "IP",
    norm_interval: int = 10,
) -> tuple:
    """Update all parameters in the correct order

    Input
    -----
    - parameters to update
    - constant parameters and input

    Output
    ------
    - updated parameters
    """

    W_NFK = update_W(
        W_NFK,
        G_tilde_NM,
        X_tilde_FTM,
        Y_tilde_FTM,
        H_NKT,
    )
    PSD_NFT = calculate_PSD(W_NFK, H_NKT)
    Y_tilde_FTM = calculate_Y_tilde(G_tilde_NM, PSD_NFT)

    H_NKT = update_H(
        H_NKT,
        G_tilde_NM,
        X_tilde_FTM,
        Y_tilde_FTM,
        W_NFK,
    )
    PSD_NFT = calculate_PSD(W_NFK, H_NKT)
    Y_tilde_FTM = calculate_Y_tilde(G_tilde_NM, PSD_NFT)

    G_tilde_NM = update_G(
        G_tilde_NM,
        PSD_NFT,
        X_tilde_FTM,
        Y_tilde_FTM,
    )
    Y_tilde_FTM = calculate_Y_tilde(G_tilde_NM, PSD_NFT)

    if algo == "IP":
        Q_FMM = update_Q_IP(
            Q_FMM,
            XX_FTMM,
            Y_tilde_FTM,
        )
    elif algo == "ISS":
        Q_FMM, Qx_FTM = update_Q_ISS(
            Q_FMM,
            Qx_FTM,
            Y_tilde_FTM,
        )
    if index_iteration % norm_interval == 0:
        W_NFK, H_NKT, G_tilde_NM, Q_FMM = normalize(
            W_NFK,
            H_NKT,
            G_tilde_NM,
            Q_FMM,
        )
        Qx_FTM, X_tilde_FTM = calculate_X_tilde(X_FTM, Q_FMM)
        PSD_NFT = calculate_PSD(W_NFK, H_NKT)
        Y_tilde_FTM = calculate_Y_tilde(G_tilde_NM, PSD_NFT)

    else:
        Qx_FTM, X_tilde_FTM = calculate_X_tilde(X_FTM, Q_FMM)
    return W_NFK, H_NKT, G_tilde_NM, Q_FMM, Qx_FTM, X_tilde_FTM, Y_tilde_FTM


def update_all_params_split(
    X_FTM: ArrayLike,
    U_NLK: ArrayLike,
    T_NKO: ArrayLike,
    G_tilde_NM: ArrayLike,
    Q_FMM: ArrayLike,
    Qx_FTM: ArrayLike,
    X_tilde_FTM: ArrayLike,
    Y_tilde_FTM: ArrayLike,
    XX_FTMM: ArrayLike,
    E_NFL: ArrayLike,
    E_inv_NLF: ArrayLike,
    P_NOT: ArrayLike,
    P_inv_NTO: ArrayLike,
    index_iteration: int,
    algo: str = "IP",
    norm_interval: int = 10,
) -> tuple:
    """Update all parameters in the correct order with W and H split

    Input
    -----
    - parameters to update
    - constant parameters and input

    Output
    ------
    - updated parameters
    """
    H_NKT = np.einsum("nko, not -> nkt", T_NKO, P_NOT)
    U_NLK = update_W_split(
        E_NFL,
        U_NLK,
        E_inv_NLF,
        G_tilde_NM,
        X_tilde_FTM,
        Y_tilde_FTM,
        H_NKT,
    )
    W_NFK = np.einsum("nfl, nlk -> nfk", E_NFL, U_NLK)
    PSD_NFT = calculate_PSD(W_NFK, H_NKT)
    Y_tilde_FTM = calculate_Y_tilde(G_tilde_NM, PSD_NFT)

    T_NKO = update_H_split(
        T_NKO,
        P_NOT,
        P_inv_NTO,
        G_tilde_NM,
        X_tilde_FTM,
        Y_tilde_FTM,
        W_NFK,
    )
    H_NKT = np.einsum("nko, not -> nkt", T_NKO, P_NOT)
    PSD_NFT = calculate_PSD(W_NFK, H_NKT)
    Y_tilde_FTM = calculate_Y_tilde(G_tilde_NM, PSD_NFT)

    G_tilde_NM = update_G(
        G_tilde_NM,
        PSD_NFT,
        X_tilde_FTM,
        Y_tilde_FTM,
    )
    Y_tilde_FTM = calculate_Y_tilde(G_tilde_NM, PSD_NFT)

    if algo == "IP":
        Q_FMM = update_Q_IP(
            Q_FMM,
            XX_FTMM,
            Y_tilde_FTM,
        )
    elif algo == "ISS":
        Q_FMM, Qx_FTM = update_Q_ISS(
            Q_FMM,
            Qx_FTM,
            Y_tilde_FTM,
        )
    if index_iteration % norm_interval == 0:
        W_NFK, H_NKT, G_tilde_NM, Q_FMM = normalize(
            W_NFK,
            H_NKT,
            G_tilde_NM,
            Q_FMM,
        )
        Qx_FTM, X_tilde_FTM = calculate_X_tilde(X_FTM, Q_FMM)
        PSD_NFT = calculate_PSD(W_NFK, H_NKT)
        Y_tilde_FTM = calculate_Y_tilde(G_tilde_NM, PSD_NFT)

    else:
        Qx_FTM, X_tilde_FTM = calculate_X_tilde(X_FTM, Q_FMM)
    return U_NLK, T_NKO, G_tilde_NM, Q_FMM, Qx_FTM, X_tilde_FTM, Y_tilde_FTM


def separate(
    X_FTM: ArrayLike,
    Q_FMM: ArrayLike,
    PSD_NFT: ArrayLike,
    G_tilde_NM: ArrayLike,
    mic_index: int,
):
    """
    Return the separated spectrograms for the specified microphone

    Input
    -----
    - X_FTM:        Array [F, T, M] = Observed spectrogram
    - Q_FMM:        Array [F, M, M] = Diagonalizer of G
    - PSD_NFT:      Array [N, F, T] = Power Spectral densities of the sources
    - G_tilde_NM:   Array [N, M]    = Mixing matrix
    - mic_index:    int             = Index of the microphone to separate

    Output
    ------
    - X_separated_NFT: Array [N, F, T] = Spectrogram of the N separated sources
    """

    Y_NFTM = np.einsum("nft, nm -> nftm", PSD_NFT, G_tilde_NM)
    Y_tilde_FTM = Y_NFTM.sum(axis=0)
    Qx_FTM = np.einsum("fmi, fti -> ftm", Q_FMM, X_FTM)
    Qinv_FMM = np.linalg.inv(Q_FMM)

    separated_spec_NFT = np.einsum(
        "fj, ftj, nftj -> nft", Qinv_FMM[:, mic_index], Qx_FTM / Y_tilde_FTM, Y_NFTM
    )
    return separated_spec_NFT


def source_image():
    """
    Demix the multi-channel spectrogram of the sources

    Input
    -----
    - image_NFTM: Array [N, F, T, M] = Separated spectrograms of N sources for M microphones

    Output
    ------
    - source_NFT: Array [N, F, T] = Demixed spectrograms of the N sources
    """


def fast_MNMF2(
    X_FTM: ArrayLike,
    n_iter: int,
    n_microphones: int,
    n_sources: int,
    n_time_frames: int,
    n_freq_bins: int,
    n_basis: int,
    n_notes: int = 24,
    n_activations: int = 30,
    E_NFL: ArrayLike = None,
    P_NOT: ArrayLike = None,
    init: str = "circular",
    algo: str = "IP",
    split: bool = False,
    mic_index: int = None,
    show_progress: bool = False,
):
    """Main function of FastMNMF2

    Input
    -----
    - X_FTM:            Array [F, T, M] = Observed spectrogram
    - n_iter:           int             = Number of iterations
    - n_microphones:    int             = Number of microphones
    - n_sources:        int             = Number of sources
    - n_time_frames:    int             = Number of time frames
    - n_freq_bins:      int             = Number of frequency bins
    - n_basis:          int             = Number of basis functions
    - n_notes:          int             = Number of notes
    - n_activations:    int             = Number of activations
    - E_NFL:            Array [N, F, L] = Spectral patterns
    - P_NOT:            Array [N, O, T] = Time patterns
    - algo:             str             = Algorithm to use for Q update
    - split:            bool            = If True, update W and H as split matrices
    - mic_index:        int             = Index of the microphone to separate. All microphones are separated if None
    - show_progress:    bool            = If True, show the progress bar

    Output
    ------
    - separated_spec: Array [N, F, T]    = Spectrogram of the N separated sources for mic_index
    - separated_spec: Array [M, N, F, T] = Spectrogram separated for all microphones if mic_index is None
    """
    ############
    ### Init ###
    ############

    G_tilde_NM, Q_FMM = init_GQ(
        init_type=init,
        n_FFT=n_freq_bins,
        n_sources=n_sources,
        n_sensors=n_microphones,
    )

    if split:
        U_NLK, T_NKO = init_UT_split(
            n_sources,
            n_basis,
            n_notes,
            n_activations,
        )
        if E_NFL is None:
            E_NFL, _ = init_EP_split(
                n_sources,
                n_freq_bins,
                n_time_frames,
                n_notes,
                n_activations,
            )
        if P_NOT is None:
            _, P_NOT = init_EP_split(
                n_sources,
                n_freq_bins,
                n_time_frames,
                n_notes,
                n_activations,
            )
        E_inv_NLF, P_inv_NTO = inverse_EP(E_NFL, P_NOT)
        W_NFK = np.einsum("nfl, nlk -> nfk", E_NFL, U_NLK)
        H_NKT = np.einsum("nko, not -> nkt", T_NKO, P_NOT)
    else:
        W_NFK, H_NKT = init_WH(
            n_FFT=n_freq_bins,
            n_time_frames=n_time_frames,
            n_basis=n_basis,
            n_sources=n_sources,
        )
    G_tilde_NM /= G_tilde_NM.sum(axis=1)[:, None]
    W_NFK, H_NKT, G_tilde_NM, Q_FMM = normalize(
        W_NFK,
        H_NKT,
        G_tilde_NM,
        Q_FMM,
    )
    if algo == "IP":
        XX_FTMM = init_IP(X_FTM)

    if show_progress:
        likelihood = np.zeros(n_iter)

    Qx_FTM, X_tilde_FTM = calculate_X_tilde(X_FTM, Q_FMM)
    PSD_NFT = calculate_PSD(W_NFK, H_NKT)
    Y_tilde_FTM = calculate_Y_tilde(G_tilde_NM, PSD_NFT)

    if split:
        updatable_params = (
            U_NLK,
            T_NKO,
            G_tilde_NM,
            Q_FMM,
            Qx_FTM,
            X_tilde_FTM,
            Y_tilde_FTM,
        )
    else:
        updatable_params = (
            W_NFK,
            H_NKT,
            G_tilde_NM,
            Q_FMM,
            Qx_FTM,
            X_tilde_FTM,
            Y_tilde_FTM,
        )

    #################
    ### Main Loop ###
    #################

    if split:
        for k in range(n_iter):
            updatable_params = update_all_params_split(
                X_FTM,
                *updatable_params,
                XX_FTMM,
                E_NFL,
                E_inv_NLF,
                P_NOT,
                P_inv_NTO,
                k,
                algo,
                norm_interval=10,
            )
            if show_progress:
                # TODO: fix parameters
                likelihood[k] = calculate_log_likelihood(
                    updatable_params[5],
                    updatable_params[6],
                    updatable_params[3],
                )
                print(f"Iteration {k+1}/{n_iter} - Loss: {likelihood[k]}", end="\r")
    else:
        for k in range(n_iter):
            updatable_params = update_all_params(
                X_FTM,
                *updatable_params,
                XX_FTMM,
                k,
                algo,
                norm_interval=10,
            )
            if show_progress:
                likelihood[k] = calculate_log_likelihood(
                    updatable_params[5],
                    updatable_params[6],
                    updatable_params[3],
                )
                print(
                    f"Iteration {k+1}/{n_iter} - Likelihood: {likelihood[k]}", end="\r"
                )

    ##################
    ### Separation ###
    ##################
    if split:
        (
            U_NLK,
            T_NKO,
            G_tilde_NM,
            Q_FMM,
            Qx_FTM,
            X_tilde_FTM,
            Y_tilde_FTM,
        ) = updatable_params
    else:
        (
            W_NFK,
            H_NKT,
            G_tilde_NM,
            Q_FMM,
            Qx_FTM,
            X_tilde_FTM,
            Y_tilde_FTM,
        ) = updatable_params

    if mic_index is not None:
        separated_spec = separate(
            X_FTM, Q_FMM, PSD_NFT, G_tilde_NM, mic_index=mic_index
        )

    else:
        separated_spec = np.zeros(
            (n_microphones, n_sources, n_freq_bins, n_time_frames), dtype=np.complex_
        )
        for m in range(n_microphones):
            separated_spec[m] = separate(X_FTM, Q_FMM, PSD_NFT, G_tilde_NM, mic_index=m)

    if show_progress:
        plt.plot(likelihood, label="Log Likelihood")
        plt.legend()
        plt.show()

    return separated_spec, *updatable_params


def main():
    # N, F, T, L, O = 4, 32, 10, 3, 5
    # E, P = init_EP_split(N, F, T, L, O)
    # E_inv, P_inv = inverse_EP(E, P)
    # with np.printoptions(edgeitems=np.inf, linewidth=np.inf):
    #     print(E_inv @ E)
    #     print(P @ P_inv)

    F, T, M = 1024, 100, 4
    dummy_X = np.random.rand(F, T, M) * (1 + 1j)
    separated_spec, *updatable_params = fast_MNMF2(
        dummy_X,
        n_iter=20,
        n_sources=M - 1,
        n_microphones=M,
        n_time_frames=T,
        n_freq_bins=F,
        n_basis=32,
        n_notes=24,
        n_activations=10,
        split=True,
        show_progress=True,
    )
    return


if __name__ == "__main__":
    main()


"""
FastMNMF2 from pyroomacoustics
=========

Blind Source Separation using Fast Multichannel Nonnegative Matrix Factorization 2 (FastMNMF2)
"""


def fastmnmf2_pyroom(
    X,
    n_src=None,
    n_iter=30,
    n_components=32,
    mic_index=0,
    W0=None,
    accelerate=True,
    callback=None,
):
    """
    Implementation of FastMNMF2 algorithm presented in

    K. Sekiguchi, Y. Bando, A. A. Nugraha, K. Yoshii, T. Kawahara, *Fast Multichannel Nonnegative
    Matrix Factorization With Directivity-Aware Jointly-Diagonalizable Spatial
    Covariance Matrices for Blind Source Separation*, IEEE/ACM TASLP, 2020.
    [`IEEE <https://ieeexplore.ieee.org/abstract/document/9177266>`_]

    The code of FastMNMF2 with GPU support and more sophisticated initialization
    is available on  https://github.com/sekiguchi92/SoundSourceSeparation

    Parameters
    ----------
    X: ndarray (nframes, nfrequencies, nchannels)
        STFT representation of the observed signal
    n_src: int, optional
        The number of sound sources (default None).
        If None, n_src is set to the number of microphones
    n_iter: int, optional
        The number of iterations (default 30)
    n_components: int, optional
        Number of components in the non-negative spectrum (default 8)
    mic_index: int or 'all', optional
        The index of microphone of which you want to get the source image (default 0).
        If 'all', return the source images of all microphones
    W0: ndarray (nfrequencies, nchannels, nchannels), optional
        Initial value for diagonalizer Q (default None).
        If None, identity matrices are used for all frequency bins.
    accelerate: bool, optional
        If true, the basis and activation of NMF are updated simultaneously (default True)
    callback: func, optional
        A callback function called every 10 iterations, allows to monitor convergence

    Returns
    -------
    If mic_index is int, returns an (nframes, nfrequencies, nsources) array.
    If mic_index is 'all', returns an (nchannels, nframes, nfrequencies, nsources) array.
    """
    eps = 1e-10
    g_eps = 5e-2
    interval_update_Q = 1  # 2 may work as well and is faster
    interval_normalize = 10
    TYPE_FLOAT = X.real.dtype
    TYPE_COMPLEX = X.dtype

    # initialize parameter
    X_FTM = X.transpose(1, 0, 2)
    n_freq, n_frames, n_chan = X_FTM.shape
    XX_FTMM = np.matmul(X_FTM[:, :, :, None], X_FTM[:, :, None, :].conj())
    if n_src is None:
        n_src = X_FTM.shape[2]

    if W0 is not None:
        Q_FMM = W0
    else:
        Q_FMM = np.tile(np.eye(n_chan).astype(TYPE_COMPLEX), [n_freq, 1, 1])

    g_NM = np.ones([n_src, n_chan], dtype=TYPE_FLOAT) * g_eps
    for m in range(n_chan):
        g_NM[m % n_src, m] = 1

    for m in range(n_chan):
        mu_F = (Q_FMM[:, m] * Q_FMM[:, m].conj()).sum(axis=1).real
        Q_FMM[:, m] /= np.sqrt(mu_F[:, None])

    H_NKT = np.random.rand(n_src, n_components, n_frames).astype(TYPE_FLOAT)
    W_NFK = np.random.rand(n_src, n_freq, n_components).astype(TYPE_FLOAT)
    lambda_NFT = W_NFK @ H_NKT
    Qx_power_FTM = np.abs(np.einsum("fij, ftj -> fti", Q_FMM, X_FTM)) ** 2
    Y_FTM = np.einsum("nft, nm -> ftm", lambda_NFT, g_NM)

    def separate():
        Qx_FTM = np.einsum("fij, ftj -> fti", Q_FMM, X_FTM)
        Qinv_FMM = np.linalg.inv(Q_FMM)
        Y_NFTM = np.einsum("nft, nm -> nftm", lambda_NFT, g_NM)

        if mic_index == "all":
            return np.einsum(
                "fij, ftj, nftj -> itfn", Qinv_FMM, Qx_FTM / Y_NFTM.sum(axis=0), Y_NFTM
            )
        elif type(mic_index) is int:
            return np.einsum(
                "fj, ftj, nftj -> tfn",
                Qinv_FMM[:, mic_index],
                Qx_FTM / Y_NFTM.sum(axis=0),
                Y_NFTM,
            )
        else:
            raise ValueError("mic_index should be int or 'all'")

    # update parameters
    for epoch in range(n_iter):
        if callback is not None and epoch % 10 == 0:
            callback(separate())

        # update W and H (basis and activation of NMF)
        tmp1_NFT = np.einsum("nm, ftm -> nft", g_NM, Qx_power_FTM / (Y_FTM**2))
        tmp2_NFT = np.einsum("nm, ftm -> nft", g_NM, 1 / Y_FTM)

        numerator = np.einsum("nkt, nft -> nfk", H_NKT, tmp1_NFT)
        denominator = np.einsum("nkt, nft -> nfk", H_NKT, tmp2_NFT)
        W_NFK *= np.sqrt(numerator / denominator)

        if not accelerate:
            tmp1_NFT = np.einsum("nm, ftm -> nft", g_NM, Qx_power_FTM / (Y_FTM**2))
            tmp2_NFT = np.einsum("nm, ftm -> nft", g_NM, 1 / Y_FTM)
            lambda_NFT = W_NFK @ H_NKT + eps
            Y_FTM = np.einsum("nft, nm -> ftm", lambda_NFT, g_NM) + eps

        numerator = np.einsum("nfk, nft -> nkt", W_NFK, tmp1_NFT)
        denominator = np.einsum("nfk, nft -> nkt", W_NFK, tmp2_NFT)
        H_NKT *= np.sqrt(numerator / denominator)

        lambda_NFT = W_NFK @ H_NKT + eps
        Y_FTM = np.einsum("nft, nm -> ftm", lambda_NFT, g_NM) + eps

        # update g_NM (diagonal element of spatial covariance matrices)
        numerator = np.einsum("nft, ftm -> nm", lambda_NFT, Qx_power_FTM / (Y_FTM**2))
        denominator = np.einsum("nft, ftm -> nm", lambda_NFT, 1 / Y_FTM)
        g_NM *= np.sqrt(numerator / denominator)
        Y_FTM = np.einsum("nft, nm -> ftm", lambda_NFT, g_NM) + eps

        # udpate Q (joint diagonalizer)
        if (interval_update_Q <= 0) or (epoch % interval_update_Q == 0):
            for m in range(n_chan):
                V_FMM = (
                    np.einsum("ftij, ft -> fij", XX_FTMM, 1 / Y_FTM[..., m]) / n_frames
                )
                tmp_FM = np.linalg.solve(
                    np.matmul(Q_FMM, V_FMM), np.eye(n_chan)[None, m]
                )
                Q_FMM[:, m] = (
                    tmp_FM
                    / np.sqrt(
                        np.einsum("fi, fij, fj -> f", tmp_FM.conj(), V_FMM, tmp_FM)
                    )[:, None]
                ).conj()
                Qx_power_FTM = np.abs(np.einsum("fij, ftj -> fti", Q_FMM, X_FTM)) ** 2

        # normalize
        if (interval_normalize <= 0) or (epoch % interval_normalize == 0):
            phi_F = np.einsum("fij, fij -> f", Q_FMM, Q_FMM.conj()).real / n_chan
            Q_FMM /= np.sqrt(phi_F)[:, None, None]
            W_NFK /= phi_F[None, :, None]

            mu_N = g_NM.sum(axis=1)
            g_NM /= mu_N[:, None]
            W_NFK *= mu_N[:, None, None]

            nu_NK = W_NFK.sum(axis=1)
            W_NFK /= nu_NK[:, None]
            H_NKT *= nu_NK[:, :, None]

            lambda_NFT = W_NFK @ H_NKT + eps
            Qx_power_FTM = np.abs(np.einsum("fij, ftj -> fti", Q_FMM, X_FTM)) ** 2
            Y_FTM = np.einsum("nft, nm -> ftm", lambda_NFT, g_NM) + eps

    return separate(), W_NFK, H_NKT, Y_FTM, g_NM, Q_FMM

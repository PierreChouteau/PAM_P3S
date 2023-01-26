# This script is based on the work of Kouhei Sekiguchi, Member, IEEE, Yoshiaki Bando, Member, IEEE, Aditya Arie Nugraha, Member, IEEE,
# Kazuyoshi Yoshii, Member, IEEE, and Tatsuya Kawahara, Fellow, IEEE in the article "Fast Multichannel Nonnegative Matrix Factorization
# with Directivity-Aware Jointly-Diagonalizable Spatial Covariance Matrices for Blind Source Separation", 2020

# The source code of K. Sekiguchi can be found here: https://github.com/sekiguchi92/SoundSourceSeparation

# This script is a re-implementation of FastNMF2 to add other constraints and different initialization methods more freely

from numpy.typing import ArrayLike
import numpy as np
from matplotlib import pyplot as plt
import unittest


def init_fast_MNMF(
    init_type: str,
    n_FFT: int,
    n_time_frames: int,
    n_basis: int,
    n_sources: int,
    n_sensors: int,
) -> tuple:
    """Initialize FastMNMF2 parameters.

    Inputs:
    - init_type:                str = random, diagonal, circular or gradual
    - n_FFT         (allias F): int = STFT window length
    - n_time_frames (allias T): int = STFT number of time frames
    - n_basis       (allias K): int = Number of elements (cols for W, rows for H) in W and H
    - n_sources     (allias N): int = Number of instruments to separate from the mix
    - n_sensors     (allias M): int = Number of microphones

    Outputs:
    - W_NFK:        Array [N, F, K] = Spectral base of NMF
    - H_NKT:        Array [N, K, T] = Activation matrix of NMF
    - G_tilde_NM:   Array [N, M]    = Spatial Covariance matrix
    - Q_FMM:        Array [F, M, M] = Diagonalizer of G
    """
    init_type_dict = {"RANDOM": 0, "DIAGONAL": 1, "CIRCULAR": 2, "GRADUAL": 3}
    G_eps = 5e-2 # Norm coef for G
    try:
        ind = init_type_dict[init_type.upper()]
    except KeyError:
        print("Wrong init type name, random is chosen by default")
        ind = 0
    match ind:
        case 0:
            # Random init

            # G_tilde is a vector of size M = n_sensors
            G_tilde_NM = np.random.rand(n_sources, n_sensors)
            # Q is nFFTxMxM
            Q_NMM = np.random.rand(n_FFT, n_sensors, n_sensors)
        case 1:
            # Diagonal init
            G_tilde_NM = np.zeros((n_sources, n_sensors)) + G_eps
            Q_NMM = np.tile(np.eye(n_sensors, dtype=np.complex_), (n_FFT, 1, 1))
        case 2:
            # Circular init
            G_tilde_NM = ...
            Q_NMM = ...
        case 3:
            # Gradual init
            G_tilde_NM = ...
            Q_NMM = ...
    W_NFK = np.random.rand(n_sources, n_FFT, n_basis)
    H_NKT = np.random.rand(n_sources, n_basis, n_time_frames)
    return W_NFK, H_NKT, G_tilde_NM, Q_NMM


def update_W(
    W_old_NFK: ArrayLike,
    G_tilde_NM: ArrayLike,
    X_tilde_FTM: ArrayLike,
    Y_tilde_FTM: ArrayLike,
    H_NKT: ArrayLike,
) -> ArrayLike:
    """Update W. Eq34

    Input:
    - W_old_NFK:    Array [N, F, K] = Source model base matrix
    - G_tilde_NM:   Array [N, M]    = Diag coefficients of the diagonalized demixing matrix
    - X_tilde_FTM:  Array [F, T, M] = Power Spectral Density at each microphone
    - Y_tilde_FTM:  Array [F, T, M] = Sum of (PSD_NFT x G_tilde_NM) over all sources
    - H_NKT:        Array [N, K, T] = Source model activation matrix

    Output:
    - W_new_NFK: Array [N, F, K]
    """

    tmp1_NFT = np.einsum("nm, ftm -> nft", G_tilde_NM, X_tilde_FTM / (Y_tilde_FTM**2))
    tmp2_NFT = np.einsum("nm, ftm -> nft", G_tilde_NM, 1 / Y_tilde_FTM)

    numerator = np.einsum("nkt, nft -> nfk", H_NKT, tmp1_NFT)
    denominator = np.einsum("nkt, nft -> nfk", H_NKT, tmp2_NFT)

    W_new_NFK = W_old_NFK * np.sqrt(numerator / denominator)
    return W_new_NFK


def update_H(
    H_old_NKT: ArrayLike,
    G_tilde_NM: ArrayLike,
    X_tilde_FTM: ArrayLike,
    Y_tilde_FTM: ArrayLike,
    W_NFK: ArrayLike,
) -> ArrayLike:
    """Update H. Eq35

    Input:
    - H_old_NKT:    Array [N, K, T] = Source model activation matrix
    - G_tilde_NM:   Array [N, M]    = Diag coefficients of the diagonalized demixing matrix
    - X_tilde_FTM:  Array [F, T, M] = Power Spectral Density at each microphone
    - Y_tilde_FTM:  Array [F, T, M] = Sum of (PSD_NFT x G_tilde_NM) over all sources
    - W_NFK:        Array [N, F, K] = Source model base matrix

    Output:
    - H_new_NKT: Array [N, K, T]
    """

    tmp1_NFT = np.einsum("nm, ftm -> nft", G_tilde_NM, X_tilde_FTM / (Y_tilde_FTM**2))
    tmp2_NFT = np.einsum("nm, ftm -> nft", G_tilde_NM, 1 / Y_tilde_FTM)

    numerator = np.einsum("nfk, nft -> nkt", W_NFK, tmp1_NFT)
    denominator = np.einsum("nfk, nft -> nkt", W_NFK, tmp2_NFT)

    H_new_NKT = H_old_NKT * np.sqrt(numerator / denominator)
    return H_new_NKT


def update_G(
    G_tilde_old_NM: ArrayLike,
    PSD_NFT: ArrayLike,
    X_tilde_FTM: ArrayLike,
    Y_tilde_FTM: ArrayLike,
) -> ArrayLike:
    """Update G_tilde. Eq 36

    Input:
    - G_tilde_old_NM:   Array [N, M]    = Diag coefficients of the diagonalized demixing matrix
    - PSD_NFT:          Array [N, F, T] = Power Spectral densities of the sources
    - X_tilde_FTM:      Array [F, T, M] = Power Spectral Density at each microphone
    - Y_tilde_FTM:      Array [F, T, M] = Sum of (PSD_NFT x G_tilde_NM) over all sources

    Output:
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

    Input:
    - Q_old_FMM:    Array [F, M, M]     = Diagonalizer of G
    - XX_FTMM:      Array [F, T, M, M]  = X_FT * X_FT^H
    - Y_tilde_FTM:  Array [F, T, M]     = Sum of (PSD_NFT x G_tilde_NM) over all sources

    Output:
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

    Input:
    - Q_old_FMM:    Array [F, M, M] = Diagonalizer of G
    - Qx_FTM:       Array [F, T, M]
    - Y_tilde_FTM:  Array [F, T, M] = Sum of (PSD_NFT x G_tilde_NM) over all sources

    Output:
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

    Input:
    - X_FTM: Array [F, T, M] = Observed spectrogram
    - Q_FMM: Array [F, M, M] = Diagonalizer of G

    Output:
    - Qx_FTM:      Array [F, T, M]
    - X_tilde_FTM: Array [F, T, M] = Power Spectral Density at each microphone
    """

    Qx_FTM = np.einsum("fmi, fti -> ftm", Q_FMM, X_FTM)
    X_tilde_FTM = np.abs(Qx_FTM) ** 2

    return Qx_FTM, X_tilde_FTM


def calculate_PSD(W_NFK: ArrayLike, H_NKT: ArrayLike):
    """Calculate PSD. NMF result

    Input:
    - W_NFK
    - H_NKT

    Output:
    - PSD_NFT
    """
    EPS = 1e-6
    return W_NFK @ H_NKT + EPS


def calculate_Y_tilde(G_tilde_NM: ArrayLike, PSD_NFT: ArrayLike):
    """Calculate Y_tilde_FTM. Eq31

    Input:
    - G_tilde_old_NM:   Array [N, M]    = Diag coefficients of the diagonalized demixing matrix
    - PSD_NFT:          Array [N, F, T] = Power Spectral densities of the sources

    Output:
    - Y_tilde_FTM:  Array [F, T, M] = Sum of (PSD_NFT x G_tilde_NM) over all sources
    """
    EPS = 1e-6
    Y_tilde_FTM = np.einsum("nft, nm -> ftm", PSD_NFT, G_tilde_NM) + EPS
    return Y_tilde_FTM


def calculate_log_likelihood(
    X_tilde_FTM: ArrayLike,
    Y_tilde_FTM: ArrayLike,
    Q_FMM: ArrayLike,
) -> float:
    """This function computes the log likelihood of FastMNMF2

    Input:
    - X_tilde_FTM:      Array [F, T, M] = Power Spectral Density at each microphone
    - Y_tilde_FTM:      Array [F, T, M] = Sum of (PSD_NFT x G_tilde_NM) over all sources
    - Q_FMM: Array [F, M, M] = Diagonalizer of G

    Output:
    - Log likelihood: Float
    """

    F, T, M = X_tilde_FTM.shape
    log_likelihood = (
        -(X_tilde_FTM / Y_tilde_FTM + np.log(Y_tilde_FTM)).sum()
        + T * (np.log(np.linalg.det(Q_FMM @ Q_FMM.transpose(0, 2, 1).conj()))).sum()
    ).real
    return log_likelihood


def normalize(
    W_NFK: ArrayLike,
    H_NKT: ArrayLike,
    G_tilde_NM: ArrayLike,
    Q_FMM: ArrayLike,
):
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
    """Update all parameters in the correct order"""

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
    else:
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


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!! Beyond this point some functions are not up to date !!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def fast_MNMF(X: ArrayLike, init_type: str = "RANDOM", show: bool = True) -> tuple:
    """Input: STFT of a mixture of N sources recorded by M microphones
    Output: W, H, G, Q optimized by maximum likelihood"""

    # Number of itteration. Maybe replace it later by convergence criteria
    n_itteration = 50

    # Initialization
    W, H, G_tilde, Q = init_fast_MNMF(init_type)
    log_like_list = np.zeros(n_itteration)

    # Loop
    for k in range(n_itteration):
        W, H, G_tilde, Q = update_all_params(W, H, G_tilde, Q)
        log_like_list[k] = calculate_log_likelihood(X, W, H, G_tilde, Q)
    if show:
        fig = plt.figure(figsize=(20, 10))
        ax = fig.gca()
        ax.plot(log_like_list)
        plt.show()
    return W, H, G_tilde, Q


def main():
    return


class FastNMFtest(unittest.TestCase):
    def setUp(self):
        ...

    def test_default_widget_size(self):
        self.assertEqual()  # test , # expected value, # message


if __name__ == "__main__":
    main()

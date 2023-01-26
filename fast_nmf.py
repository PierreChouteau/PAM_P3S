# This script is based on the work of Kouhei Sekiguchi, Member, IEEE, Yoshiaki Bando, Member, IEEE, Aditya Arie Nugraha, Member, IEEE,
# Kazuyoshi Yoshii, Member, IEEE, and Tatsuya Kawahara, Fellow, IEEE in the article "Fast Multichannel Nonnegative Matrix Factorization
# with Directivity-Aware Jointly-Diagonalizable Spatial Covariance Matrices for Blind Source Separation", 2020

# The code of K. Sekiguchi can be found here: https://github.com/sekiguchi92/SoundSourceSeparation

# This script is a re-implementation of FastNMF2 to add other constraints and different initialization methods more freely

from numpy import typing
import numpy as np
from matplotlib import pyplot as plt
import unittest


def init_fast_MNMF(
    init_type: str,
    n_FFT: int,
    n_time_frames: int,
    n_bases: int,
    n_sources: int,
    n_sensors: int,
) -> tuple:
    """Initialize FastMNMF2 parameters.

    Inputs:
    - init_type:                str = random, diagonal, circular or gradual
    - n_FFT         (allias F): int = STFT window length
    - n_time_frames (allias T): int = STFT number of time frames
    - n_bases       (allias K): int = Number of elements (cols for W, rows for H) in W and H
    - n_sources     (allias N): int = Number of instruments to separate from the mix
    - n_sensors     (allias M): int = Number of microphones

    Outputs:
    - W_NFK:        Array [N, F, K] = Spectral base of NMF
    - H_NKT:        Array [N, K, T] = Activation matrix of NMF
    - G_TILDE_NM:   Array [N, M]    = Spatial Covariance matrix
    - Q_FMM:        Array [F, M, M] = Diagonalizer of G
    """
    init_type_dict = {"RANDOM": 0, "DIAGONAL": 1, "CIRCULAR": 2, "GRADUAL": 3}
    try:
        ind = init_type_dict[init_type.upper()]
    except KeyError:
        print("Wrong init type name, random is chosen by default")
        ind = 0
    match ind:
        case 0:
            # Random init

            # G_tilde is a vector of size M = n_sensors
            G_TILDE_NM = np.random.rand(n_sources, n_sensors)
            # Q is nFFTxMxM
            Q_NMM = np.random.rand(n_FFT, n_sensors, n_sensors)
        case 1:
            # Diagonal init
            G_TILDE_NM = ...
            Q_NMM = ...
        case 2:
            # Circular init
            G_TILDE_NM = ...
            Q_NMM = ...
        case 3:
            # Gradual init
            G_TILDE_NM = ...
            Q_NMM = ...
    W_NFK = np.random.rand(n_sources, n_FFT, n_bases)
    H_NKT = np.random.rand(n_sources, n_bases, n_time_frames)
    return W_NFK, H_NKT, G_TILDE_NM, Q_NMM


def update_W(
    W_OLD_NFK: typing.ArrayLike,
    G_TILDE_NM: typing.ArrayLike,
    X_TILDE_FTM: typing.ArrayLike,
    Y_TILDE_FTM: typing.ArrayLike,
    H_NKT: typing.ArrayLike,
) -> typing.ArrayLike:
    """Update W. Eq34

    Input:
    - W_OLD_NFK:    Array [N, F, K] = Source model base matrix
    - G_TILDE_NM:   Array [N, M]    = Diag coefficients of the diagonalized demixing matrix
    - X_TILDE_FTM:  Array [F, T, M] = Power Spectral Density at each microphone
    - Y_TILDE_FTM:  Array [F, T, M] = Sum of (PSD_NFT x G_TILDE_NM) over all sources
    - H_NKT:        Array [N, K, T] = Source model activation matrix

    Output:
    - W_NEW_NFK: Array [N, F, K]
    """

    tmp1_NFT = np.einsum("nm, ftm -> nft", G_TILDE_NM, X_TILDE_FTM / (Y_TILDE_FTM**2))
    tmp2_NFT = np.einsum("nm, ftm -> nft", G_TILDE_NM, 1 / Y_TILDE_FTM)

    numerator = np.einsum("nkt, nft -> nfk", H_NKT, tmp1_NFT)
    denominator = np.einsum("nkt, nft -> nfk", H_NKT, tmp2_NFT)

    W_NEW_NFK = W_OLD_NFK * np.sqrt(numerator / denominator)
    return W_NEW_NFK


def update_H(
    H_OLD_NKT: typing.ArrayLike,
    G_TILDE_NM: typing.ArrayLike,
    X_TILDE_FTM: typing.ArrayLike,
    Y_TILDE_FTM: typing.ArrayLike,
    W_NFK: typing.ArrayLike,
) -> typing.ArrayLike:
    """Update H. Eq35

    Input:
    - H_OLD_NKT:    Array [N, K, T] = Source model activation matrix
    - G_TILDE_NM:   Array [N, M]    = Diag coefficients of the diagonalized demixing matrix
    - X_TILDE_FTM:  Array [F, T, M] = Power Spectral Density at each microphone
    - Y_TILDE_FTM:  Array [F, T, M] = Sum of (PSD_NFT x G_TILDE_NM) over all sources
    - W_NFK:        Array [N, F, K] = Source model base matrix

    Output:
    - H_NEW_NKT: Array [N, K, T]
    """

    tmp1_NFT = np.einsum("nm, ftm -> nft", G_TILDE_NM, X_TILDE_FTM / (Y_TILDE_FTM**2))
    tmp2_NFT = np.einsum("nm, ftm -> nft", G_TILDE_NM, 1 / Y_TILDE_FTM)

    numerator = np.einsum("nfk, nft -> nkt", W_NFK, tmp1_NFT)
    denominator = np.einsum("nfk, nft -> nkt", W_NFK, tmp2_NFT)

    H_NEW_NKT = H_OLD_NKT * np.sqrt(numerator / denominator)
    return H_NEW_NKT


def update_G(
    G_TILDE_OLD_NM: typing.ArrayLike,
    PSD_NFT: typing.ArrayLike,
    X_TILDE_FTM: typing.ArrayLike,
    Y_TILDE_FTM: typing.ArrayLike,
) -> typing.ArrayLike:
    """Update G_tilde. Eq 36

    Input:
    - G_TILDE_OLD_NM:   Array [N, M]    = Diag coefficients of the diagonalized demixing matrix
    - PSD_NFT:          Array [N, F, T] = Power Spectral densities of the sources
    - X_TILDE_FTM:      Array [F, T, M] = Power Spectral Density at each microphone
    - Y_TILDE_FTM:      Array [F, T, M] = Sum of (PSD_NFT x G_TILDE_NM) over all sources

    Output:
    - G_TILDE_NEW_NM: Array [N, M]"""

    numerator = np.einsum("nft, ftm -> nm", PSD_NFT, X_TILDE_FTM / (Y_TILDE_FTM**2))
    denominator = np.einsum("nft, ftm -> nm", PSD_NFT, 1 / Y_TILDE_FTM)

    G_TILDE_NEW_NM = G_TILDE_OLD_NM * np.sqrt(numerator / denominator)
    return G_TILDE_NEW_NM


def update_Q_IP(
    Q_FMM: typing.ArrayLike,
    XX_FTMM: typing.ArrayLike,
    Y_TILDE_FTM: typing.ArrayLike,
) -> typing.ArrayLike:
    """Update Q_FMM. Eq24

    Input:
    - Q_OLD_FMM:    Array [F, M, M]     = Diagonalizer of G
    - XX_FTMM:      Array [F, T, M, M]  = X_FT * X_FT^H
    - Y_TILDE_FTM:  Array [F, T, M]     = Sum of (PSD_NFT x G_TILDE_NM) over all sources

    Output:
    - Q_NEW_FMM: Array [F, M, M]
    """

    F, T, M = Y_TILDE_FTM.shape

    for m in range(M):
        V_FMM = np.einsum("ftij, ft -> fij", XX_FTMM, 1 / Y_TILDE_FTM[..., m]) / T
        tmp_FM = np.linalg.inv(Q_FMM @ V_FMM)[..., m]
        Q_FMM[:, m] = (
            tmp_FM
            / np.sqrt(np.einsum("fi, fij, fj -> f", tmp_FM.conj(), V_FMM, tmp_FM))[
                :, None
            ]
        ).conj()

    return Q_FMM


def update_Q_ISS(
    Q_OLD_FMM: typing.ArrayLike,
    Qx_FTM: typing.ArrayLike,
    Y_TILDE_FTM: typing.ArrayLike,
) -> typing.ArrayLike:
    """Update Q_FMM. Eq25

    Input:
    - Q_OLD_FMM:    Array [F, M, M] = Diagonalizer of G
    - Qx_FTM:       Array [F, T, M]
    - Y_TILDE_FTM:  Array [F, T, M] = Sum of (PSD_NFT x G_TILDE_NM) over all sources

    Output:
    - Q_NEW_FMM: Array [F, M, M]
    """

    F, T, M = Y_TILDE_FTM.shape

    for m in range(M):
        QxQx_FTM = Qx_FTM * Qx_FTM[:, :, m, None].conj()
        V_tmp_FxM = (QxQx_FTM[:, :, m, None] / Y_TILDE_FTM).mean(axis=1)
        V_FxM = (QxQx_FTM / Y_TILDE_FTM).mean(axis=1) / V_tmp_FxM
        V_FxM[:, m] = 1 - 1 / np.sqrt(V_tmp_FxM[:, m])
        Qx_NEW_FTM = Qx_FTM - np.einsum("fm, ft -> ftm", V_FxM, Qx_FTM[:, :, m])
        Q_NEW_FMM = Q_OLD_FMM - np.einsum("fi, fj -> fij", V_FxM, Q_OLD_FMM[:, m])

    return Q_NEW_FMM, Qx_NEW_FTM


def calculate_X_TILDE(X_FTM: typing.ArrayLike, Q_FMM: typing.ArrayLike):
    """Calculate X_TILDE_FTM. Eq31

    Input:
    - X_FTM: Array [F, T, M] = Observed spectrogram
    - Q_FMM: Array [F, M, M] = Diagonalizer of G
    """

    Qx_FTM = np.einsum("fmi, fti -> ftm", Q_FMM, X_FTM)
    X_TILDE_FTM = np.abs(Qx_FTM) ** 2

    return Qx_FTM, X_TILDE_FTM


def calculate_PSD(W_NFK: typing.ArrayLike, H_NKT: typing.ArrayLike):
    """Calculate PSD. NMF result

    Input:
    - W_NFK
    - H_NKT

    Output:
    - PSD_NFT
    """
    EPS = 1e-6
    return W_NFK @ H_NKT + EPS


def calculate_Y_TILDE(G_TILDE_NM: typing.ArrayLike, PSD_NFT: typing.ArrayLike):
    """Calculate Y_TILDE_FTM. Eq31"""
    EPS = 1e-6
    Y_FTM = np.einsum("nft, nm -> ftm", PSD_NFT, G_TILDE_NM) + EPS
    return Y_FTM

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!! Beyond this point some functions are not up to date !!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def update_all_params(
    W_old: typing.ArrayLike,
    H_old: typing.ArrayLike,
    G_old: typing.ArrayLike,
    Q_old: typing.ArrayLike,
) -> tuple:
    """Update all parameters in the correct order"""
    W_new = update_W(W_old)
    H_new = update_W(H_old)
    G_new = update_W(G_old)
    Q_new = update_W(Q_old)
    return W_new, H_new, G_new, Q_new


def log_likelihood(
    X: typing.ArrayLike,
    W: typing.ArrayLike,
    H: typing.ArrayLike,
    G_tilde: typing.ArrayLike,
    Q: typing.ArrayLike,
) -> float:
    """This function computes the log likelihood of FastMNMF2"""
    log_like = 10
    return log_like


def fast_MNMF(
    X: typing.ArrayLike, init_type: str = "RANDOM", show: bool = True
) -> tuple:
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
        log_like_list[k] = log_likelihood(X, W, H, G_tilde, Q)
    if show:
        fig = plt.figure(figsize=(20, 10))
        ax = fig.gca()
        ax.plot(log_like_list)
        plt.show()
    return W, H, G_tilde, Q


def source_estimation(
    X: typing.ArrayLike,
    W: typing.ArrayLike,
    H: typing.ArrayLike,
    G_tilde: typing.ArrayLike,
    Q: typing.ArrayLike,
) -> typing.ArrayLike:
    """Input: MNMF matrices
    Output: Separated source images. source_images[n] is the spectrogram of estimated n^th source"""
    source_images = ...
    return source_images


def main():
    X = np.zeros((1025, 100, 4))
    W, H, G_tilde, Q = fast_MNMF(X)
    source_images = source_estimation(W, H, G_tilde, Q)
    return


class FastNMFtest(unittest.TestCase):
    def setUp(self):
        ...

    def test_default_widget_size(self):
        self.assertEqual()  # test , # expected value, # message


if __name__ == "__main__":
    main()

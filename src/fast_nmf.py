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


def init_fast_MNMF(
    init_type: str,
    n_FFT: int,
    n_time_frames: int,
    n_basis: int,
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
    - n_time_frames (allias T): int = STFT number of time frames
    - n_basis       (allias K): int = Number of elements (cols for W, rows for H) in W and H
    - n_sources     (allias N): int = Number of instruments to separate from the mix
    - n_sensors     (allias M): int = Number of microphones

    Outputs
    -------
    - W_NFK:        Array [N, F, K] = Spectral base of NMF
    - H_NKT:        Array [N, K, T] = Activation matrix of NMF
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
        print("Wrong init type name, random is chosen by default")
        ind = 0
    match ind:
        case 0:
            # Random init
            G_tilde_NM = np.random.rand(n_sources, n_sensors)
            Q_FMM = np.random.rand(n_FFT, n_sensors, n_sensors)
            W_NFK = np.random.rand(n_sources, n_FFT, n_basis)
            H_NKT = np.random.rand(n_sources, n_basis, n_time_frames)
        case 1:
            # Diagonal init
            G_tilde_NM = np.zeros((n_sources, n_sensors)) + G_eps
            Q_FMM = np.tile(np.eye(n_sensors, dtype=np.complex_), (n_FFT, 1, 1))
            W_NFK = np.random.rand(n_sources, n_FFT, n_basis)
            H_NKT = np.random.rand(n_sources, n_basis, n_time_frames)

            for n in range(min(n_sources, n_sensors)):
                G_tilde_NM[n, n] = 1
        case 2:
            # Circular init
            G_tilde_NM = np.zeros((n_sources, n_sensors)) + G_eps
            Q_FMM = np.tile(np.eye(n_sensors, dtype=np.complex_), (n_FFT, 1, 1))
            W_NFK = np.random.rand(n_sources, n_FFT, n_basis)
            H_NKT = np.random.rand(n_sources, n_basis, n_time_frames)

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
                W_NFK = np.random.rand(n_sources, n_FFT, n_basis)
                H_NKT = np.random.rand(n_sources, n_basis, n_time_frames)
            else:
                # Next updates with bigger K after ~50 itterations
                W_NFK = np.random.rand(n_sources, n_FFT, n_basis)
                H_NKT = np.random.rand(n_sources, n_basis, n_time_frames)
    return normalize(W_NFK, H_NKT, G_tilde_NM, Q_FMM)


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
    EPS = 1e-6
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
    EPS = 1e-6
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


def fast_MNMF2(
    X_FTM: ArrayLike,
    n_iter: int,
    n_microphones: int,
    n_sources: int,
    n_time_frames: int,
    n_freq_bins: int,
    n_basis: int,
    algo: str = "IP",
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
    - algo:             str             = Algorithm to use for Q update
    - mic_index:        int             = Index of the microphone to separate. All microphones are separated if None

    Output
    ------
    - separated_spec: Array [N, F, T] = Spectrogram of the N separated sources for mic_index
    - separated_spec: Array [M, N, F, T] = Spectrogram separated for all microphones if mic_index is None
    """
    ############
    ### Init ###
    ############

    W_NFK, H_NKT, G_tilde_NM, Q_FMM = init_fast_MNMF(
        init_type="circular",
        n_FFT=n_freq_bins,
        n_time_frames=n_time_frames,
        n_basis=n_basis,
        n_sources=n_sources,
        n_sensors=n_microphones,
    )
    if algo == "IP":
        XX_FTMM = init_IP(X_FTM)
    
    if show_progress:
        loss = np.zeros(n_iter)

    Qx_FTM, X_tilde_FTM = calculate_X_tilde(X_FTM, Q_FMM)
    PSD_NFT = calculate_PSD(W_NFK, H_NKT)
    Y_tilde_FTM = calculate_Y_tilde(G_tilde_NM, PSD_NFT)

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
            loss[k] = calculate_log_likelihood(updatable_params[5], updatable_params[6], updatable_params[3])
            print(f"Iteration {k+1}/{n_iter} - Loss: {loss[k]}")

    ###################
    ### Seaparation ###
    ###################
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
        plt.plot(loss, label="Loss")
        plt.legend()
        plt.show()

    return separated_spec


def main():
    return


if __name__ == "__main__":
    main()

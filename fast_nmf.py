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
    '''Initialize FastMNMF2 parameters.

    Inputs: 
    - init_type: str = random, diagonal, circular or gradual
    - n_FFT (allias F): int = STFT window length
    - n_time_frames (allias T): int = STFT number of time frames
    - n_bases (allias K): int = Number of elements (cols for W, rows for H) in W and H
    - n_sources (allias N): int = Number of instruments to separate from the mix
    - n_sensors (allias M): int = Number of microphones

    Outputs:
    - W: Array [N, F, K] = Spectral base of NMF
    - H: Array [N, K, T] = Activation matrix of NMF
    - G_tilde: Array [N, M] = Spatial Covariance matrix
    - Q: Array [F, M, M] = Diagonalizer of G
    '''
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
            G_tilde = np.random.rand(n_sources, n_sensors)
            # Q is nFFTxMxM
            Q = np.random.rand(n_FFT, n_sensors, n_sensors)
        case 1:
            # Diagonal init
            G_tilde = ...
            Q = np.identity(n_sensors)
        case 2:
            # Circular init
            G_tilde = ...
            Q = ...
        case 3:
            # Gradual init
            G_tilde = ...
            Q = ...
    W = np.random.rand(n_FFT, n_bases)
    H = np.random.rand(n_bases, n_time_frames)
    return W, H, G_tilde, Q


def update_W(W_old: typing.ArrayLike) -> typing.ArrayLike:
    '''Update W with constraints.
    
    Input:
    - W_old: Array [N, F, K]

    Output:
    - W_new: Array [N, F, K]

    Constraints:

    '''
    W_new = ...
    return W_new


def update_H(H_old: typing.ArrayLike) -> typing.ArrayLike:
    '''Update H with constraints.
    
    Input:
    - H_old: Array [N, K, T]

    Output:
    - H_new: Array [N, K, T]

    Constraints:
        
    '''
    H_new = ...
    return H_new


def update_G(G_old: typing.ArrayLike) -> typing.ArrayLike:
    '''Update G_tilde with constraints.
    
    Input:
    - G_old: Array [N, M]

    Output:
    - G_new: Array [N, M]

    Constraints:
        
    '''
    G_new = ...
    return G_new


def update_Q(Q_old: typing.ArrayLike) -> typing.ArrayLike:
    '''Update G_tilde with constraints.
    
    Input:
    - Q_old: Array [F, M, M]

    Output:
    - Q_new: Array [F, M, M]

    Constraints:
        
    '''
    Q_new = ...
    return Q_new


def update_all_params(
    W_old: typing.ArrayLike,
    H_old: typing.ArrayLike,
    G_old: typing.ArrayLike,
    Q_old: typing.ArrayLike,
) -> tuple:
    '''Update all parameters in the correct order'''
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

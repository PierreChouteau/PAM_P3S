import numpy as np
import pyroomacoustics as pra
from mir_eval.separation import bss_eval_sources


def compute_si_sdr(reference, estimation):
    """
    Fonction provenant du github https://github.com/fgnt/pb_bss et servant à estimer
    le si-sdr entre un signal de reference et un signal estimé


    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]
    Returns:
        SI-SDR
    [1] SDR– Half- Baked or Well Done?
    http://www.merl.com/publications/docs/TR2019-013.pdf
    >>> np.random.seed(0)
    >>> reference = np.random.randn(100)
    >>> si_sdr(reference, reference)
    inf
    >>> si_sdr(reference, reference * 2)
    inf
    >>> si_sdr(reference, np.flip(reference))
    -25.127672346460717
    >>> si_sdr(reference, reference + np.flip(reference))
    0.481070445785553
    >>> si_sdr(reference, reference + 0.5)
    6.3704606032577304
    >>> si_sdr(reference, reference * 2 + 1)
    6.3704606032577304
    >>> si_sdr([1., 0], [0., 0])  # never predict only zeros
    nan
    >>> si_sdr([reference, reference], [reference * 2 + 1, reference * 1 + 0.5])
    array([6.3704606, 6.3704606])
    """
    estimation, reference = np.broadcast_arrays(estimation, reference)

    assert reference.dtype == np.float64, reference.dtype
    assert estimation.dtype == np.float64, estimation.dtype

    reference_energy = np.sum(reference**2, axis=-1, keepdims=True)

    # This is $\alpha$ after Equation (3) in [1].
    optimal_scaling = (
        np.sum(reference * estimation, axis=-1, keepdims=True) / reference_energy
    )

    # This is $e_{\text{target}}$ in Equation (4) in [1].
    projection = optimal_scaling * reference

    # This is $e_{\text{res}}$ in Equation (4) in [1].
    noise = estimation - projection

    ratio = np.sum(projection**2, axis=-1) / np.sum(noise**2, axis=-1)
    return 10 * np.log10(ratio)


def compute_perf(y, ref):
    """
    Fonction permettant de calculer les performances de la méthode de séparation de sources
    suivant les critères SDR, SI-SDR, SIR et SAR.

    Inputs:
    ------------------------------------------------------

    y: numpy.ndarray (n_channels, n_sources, n_samples)
        Matrice contenant les signaux audios estimés

    ref: numpy.ndarray (n_channels, n_sources, n_samples)
        Matrice contenant les signaux audios de référence


    Outputs:
    ------------------------------------------------------

    sdr: list
        Liste contenant les valeurs de SDR pour chaque source

    si_sdr: list
        Liste contenant les valeurs de SI-SDR pour chaque source

    sir: list
        Liste contenant les valeurs de SIR pour chaque source

    sar: list
        Liste contenant les valeurs de SAR pour chaque source

    perm: list
        Liste contenant les permutations des sources estimées par rapport aux sources de référence

    """
    sdr = []
    si_sdr = []
    sir = []
    sar = []
    perm = []

    # On parcourt les microphones
    for i in range(y.shape[0]):

        # On prend le minimum entre le nombre de samples du signal de référence et celui de l'estimation
        m_ = np.minimum(y[i].shape[1], ref[i].shape[1])

        # On calcule les performances
        sdr_, sir_, sar_, perm_ = bss_eval_sources(ref[i, :, :m_], y[i, :, :m_])
        si_sdr_ = compute_si_sdr(ref[i, :, :m_], y[i, :, :m_])

        # On ajoute les performances pour tous les microphones
        sdr.append(sdr_)
        si_sdr.append(si_sdr_)
        sir.append(sir_)
        sar.append(sar_)
        perm.append(perm_)

    return sdr, si_sdr, sir, sar, perm

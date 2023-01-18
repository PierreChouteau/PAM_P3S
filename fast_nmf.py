from numpy import typing

def init_fast_MNMF(init_type: str) -> tuple:
    init_type_dict = {"RANDOM": 0, "DIAGONAL": 1, "CIRCULAR": 2, "GRADUAL": 3}
    try:
        ind = init_type_dict[init_type.upper()]
    except KeyError:
        print("Wrong init type name, random is chosen by default")
        ind = 0
    match ind:
        case 0:
            # Random init
            W, H, G_tilde, Q = ...
        case 1:
            # Diagonal init
            W, H, G_tilde, Q = ...
        case 2:
            # Circular init
            W, H, G_tilde, Q = ...
        case 3:
            # Gradual init
            W, H, G_tilde, Q = ...
    return W, H, G_tilde, Q


def update_W(W_old: typing.ArrayLike) -> typing.ArrayLike:
    W_new = ...
    return W_new


def update_H(H_old: typing.ArrayLike) -> typing.ArrayLike:
    H_new = ...
    return H_new


def update_G(G_old: typing.ArrayLike) -> typing.ArrayLike:
    G_new = ...
    return G_new


def update_Q(Q_old: typing.ArrayLike) -> typing.ArrayLike:
    Q_new = ...
    return Q_new


def update_all_params(
    W_old: typing.ArrayLike,
    H_old: typing.ArrayLike,
    G_old: typing.ArrayLike,
    Q_old: typing.ArrayLike,
) -> tuple:
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
    log_like = ...
    return log_like


def fast_MNMF(X: typing.ArrayLike, init_type: str = "RANDOM"):
    """This function """
    W, H, G_tilde, Q = init_fast_MNMF(init_type)
    return


def main():
    test()
    return


if __name__ == "__main__":
    main()

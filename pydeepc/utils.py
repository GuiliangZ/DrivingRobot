import numpy as np
from typing import NamedTuple, Tuple, Optional, Union
from numpy.typing import NDArray
import casadi as ca


class Data(NamedTuple):
    """
    Tuple that contains input/output data
    :param u: input data (T×M)
    :param y: output data (T×P)
    """
    u: NDArray[np.float64]
    y: NDArray[np.float64]


def create_hankel_matrix(data: NDArray[np.float64], order: int) -> NDArray[np.float64]:
    """
    Create an L-block Hankel matrix from TxM data.
    :param data:  T×M array
    :param order: number of block rows L
    :return:      (L·M)×(T-L+1) Hankel matrix
    """
    data = np.asarray(data, dtype=float)
    assert data.ndim == 2, "Data must be 2D"
    T, M = data.shape
    assert 1 <= order <= T, "order must be between 1 and T"
    cols = T - order + 1
    H = np.zeros((order * M, cols))
    for i in range(cols):
        H[:, i] = data[i:i+order, :].ravel()
    return H


def split_data(
    data: Data,
    Tini: int,
    horizon: int,
    explained_variance: Optional[float] = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Build Past/Future Hankel blocks for u and y:
      Up ∈ ℝ^{Tini·Mu×Nd}, Uf ∈ ℝ^{horizon·Mu×Nd}
      Yp ∈ ℝ^{Tini·My×Nd}, Yf ∈ ℝ^{horizon·My×Nd}
    where Nd = T - Tini - horizon + 1
    """
    assert Tini > 0 and horizon > 0, "Tini and horizon must be ≥1"
    Mu = data.u.shape[1]
    My = data.y.shape[1]
    # full Hankel of depth Tini+horizon
    Hu = create_hankel_matrix(data.u, Tini + horizon)
    Hy = create_hankel_matrix(data.y, Tini + horizon)
    if explained_variance is not None:
        # low-rank approx via SVD
        Hu = low_rank_matrix_approximation(Hu, explained_var=explained_variance)
        Hy = low_rank_matrix_approximation(Hy, explained_var=explained_variance)
    # split rows
    Up = Hu[: Tini * Mu, :]
    Uf = Hu[-horizon * Mu :, :]
    Yp = Hy[: Tini * My, :]
    Yf = Hy[-horizon * My :, :]
    return Up, Uf, Yp, Yf


def low_rank_matrix_approximation(
    X: NDArray[np.float64],
    explained_var: Optional[float] = 0.9,
    rank: Optional[int] = None,
    SVD: Optional[Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]] = None,
    **svd_kwargs
) -> NDArray[np.float64]:
    """
    Return a low-rank approximation of X via truncated SVD.
    :param X:           original matrix
    :param explained_var: fraction of variance to retain (0<explained_var≤1)
    :param rank:        explicit rank override
    :param SVD:         precomputed (U,S,V) tuple
    """
    assert X.ndim == 2, "X must be 2D"
    # compute SVD
    u, s, v = SVD if SVD is not None else np.linalg.svd(X, full_matrices=False, **svd_kwargs)
    if rank is None:
        # pick rank to cover explained_var
        var = s**2
        cumvar = np.cumsum(var) / np.sum(var)
        rank = int(np.searchsorted(cumvar, explained_var, side='right') + 1)
    assert 1 <= rank <= min(u.shape[1], v.shape[0]), "invalid rank"
    # reconstruct
    U_low = u[:, :rank]
    S_low = s[:rank]
    V_low = v[:rank, :]
    return (U_low * S_low) @ V_low
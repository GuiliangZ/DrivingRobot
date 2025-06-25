import numpy as np
from typing import NamedTuple, Optional, Tuple

# Define the Data class
class Data(NamedTuple):
    u: np.ndarray  # shape (T, Mu)
    y: np.ndarray  # shape (T, My)

# Create dummy data (149 time steps, 1 input/output feature)
T = 250
u = np.linspace(0, 1, T).reshape(-1, 1)           # shape (149, 1)
v = np.sin(np.linspace(0, 2*np.pi, T)).reshape(-1, 1)  # shape (149, 1)

data = Data(u=u, y=v)

# Create Hankel matrix function
def create_hankel_matrix(data: np.ndarray, order: int) -> np.ndarray:
    assert len(data.shape) == 2, "Data needs to be a matrix"
    T, M = data.shape
    assert T >= order and order > 0, "The number of data points needs to be larger than the order"
    H = np.zeros((order * M, T - order + 1))
    for idx in range(T - order + 1):
        H[:, idx] = data[idx:idx + order, :].flatten()
    return H

# split_data function (without low-rank approximation for now)
def split_data(data: Data, Tini: int, horizon: int, explained_variance: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert Tini >= 1, "Tini cannot be lower than 1"
    assert horizon >= 1, "Horizon cannot be lower than 1"
    assert explained_variance is None or 0 < explained_variance <= 1, "explained_variance should be in (0,1] or be none"
    
    Mu, My = data.u.shape[1], data.y.shape[1]
    Hu = create_hankel_matrix(data.u, Tini + horizon)
    Hy = create_hankel_matrix(data.y, Tini + horizon)

    Up, Uf = Hu[:Tini * Mu], Hu[-horizon * Mu:]
    Yp, Yf = Hy[:Tini * My], Hy[-horizon * My:]
    
    return Up, Uf, Yp, Yf

# Parameters
Tini = 50
horizon = 50

# Split
Up, Uf, Yp, Yf = split_data(data, Tini, horizon)

# Show shapes
print("Up shape:", Up.shape)
print("Uf shape:", Uf.shape)
print("Yp shape:", Yp.shape)
print("Yf shape:", Yf.shape)

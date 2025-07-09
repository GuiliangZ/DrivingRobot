import os
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

def load_timeseries(data_dir):
    """
    Read every .xlsx in data_dir, concatenating their 'u' and 'v_meas' columns.
    Returns
    -------
    u : np.ndarray, shape (T_total,)
    v : np.ndarray, shape (T_total,)
    """

    
    u_list, v_list = [], []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.xlsx'):
            continue
        df = pd.read_excel(os.path.join(data_dir, fname))
        u_list.append(df['u'].values)
        v_list.append(df['v_meas'].values)
    u = np.concatenate(u_list, axis=0)
    v = np.concatenate(v_list, axis=0)
    return u, v

def build_hankel(u: np.ndarray, v: np.ndarray, N_ini: int, N: int):
    """
    Build the block‐Hankel matrices:
      U_p ∈ R^(N_ini×K),  U_f ∈ R^(N   ×K)
      Y_p ∈ R^(N_ini×K),  Y_f ∈ R^(N   ×K)
    with K = T − (N_ini+N) + 1.
    """
    L = N_ini + N
    # shape (T − L + 1, L) → transpose → (L, K)
    U = sliding_window_view(u, window_shape=L).T
    Y = sliding_window_view(v, window_shape=L).T
    U_p = U[:N_ini, :]
    U_f = U[N_ini:, :]
    Y_p = Y[:N_ini, :]
    Y_f = Y[N_ini:, :]
    return U_p, U_f, Y_p, Y_f

def get_or_build_hankel(data_dir: str,
                        cache_path: str,
                        N_ini: int,
                        N: int):
    """
    If cache_path exists, load from it. Otherwise read the Excels,
    build Hankels and save to cache_path (.npz) for next time.
    """
    if os.path.isfile(cache_path):
        npz = np.load(cache_path)
        return npz['U_p'], npz['U_f'], npz['Y_p'], npz['Y_f']

    u, v = load_timeseries(data_dir)
    U_p, U_f, Y_p, Y_f = build_hankel(u, v, N_ini, N)
    np.savez(cache_path, U_p=U_p, U_f=U_f, Y_p=Y_p, Y_f=Y_f)
    return U_p, U_f, Y_p, Y_f

if __name__ == "__main__":
    # ─── USER PARAMETERS ──────────────────────────────────────────────
    DATA_DIR   = "deepc/dataForHankle"
    CACHE_FILE = "deepc/dataForHankle/hankel_dataset.npz"
    N_ini, N_f   = 50, 50
    # ───────────────────────────────────────────────────────────────────

    U_p, U_f, Y_p, Y_f = get_or_build_hankel(DATA_DIR, CACHE_FILE, N_ini, N_f)
    print("Shapes: U_p", U_p.shape, "U_f", U_f.shape,
          "Y_p", Y_p.shape, "Y_f", Y_f.shape)

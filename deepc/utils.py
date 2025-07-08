# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
# import torch.optim as optim
import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
from datetime import datetime
import threading
from pathlib import Path

import numpy as np
import scipy.io as sio
import pandas as pd

# import Jetson.GPIO as GPIO
# GPIO.cleanup()

import board
import busio
from adafruit_pca9685 import PCA9685

import cantools
import can
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver

from smbus2 import SMBus

# ────────────────────────── GLOBALS VARIABLES ─────────────────────────────
# ——— CP2112 I²C setup ———
PCA9685_ADDR = 0x40      # default PCA9685 address
# PCA9685 register addresses
MODE1_REG    = 0x00
PRESCALE_REG = 0xFE
LED0_ON_L    = 0x06     # base address for channel 0

def init_pca9685(bus: SMBus, freq_hz: float = 1000.0):
    """
    Reset PCA9685 and set PWM frequency.
    """
    prescale_val = int(round(25_000_000.0 / (4096 * freq_hz) - 1))
    bus.write_byte_data(PCA9685_ADDR, MODE1_REG, 0x10)
    time.sleep(0.01)
    bus.write_byte_data(PCA9685_ADDR, PRESCALE_REG, prescale_val)
    time.sleep(0.01)
    bus.write_byte_data(PCA9685_ADDR, MODE1_REG, 0x00)
    time.sleep(0.01)
    bus.write_byte_data(PCA9685_ADDR, MODE1_REG, 0xA1)
    time.sleep(0.01)

def set_duty_cycle(bus: SMBus, channel: int, percent: float):
    """
    Set one channel’s duty cycle (0–100%). Uses 12-bit resolution.
    """
    if not (0 <= channel <= 15):
        raise ValueError("channel must be in 0..15")
    if not (0.0 <= percent <= 100.0):
        raise ValueError("percent must be between 0.0 and 100.0")

    # Convert percentage → 12-bit count (0..4095)
    duty_count = int(percent * 4095 / 100)
    on_l  = 0
    on_h  = 0
    off_l = duty_count & 0xFF
    off_h = (duty_count >> 8) & 0x0F

    # Compute first-LED register for this channel
    reg = LED0_ON_L + 4 * channel
    # Write [ON_L, ON_H, OFF_L, OFF_H]
    bus.write_i2c_block_data(PCA9685_ADDR, reg, [on_l, on_h, off_l, off_h])

# ─────────────────────────── LOAD DRIVE-CYCLE .MAT ────────────────────────────
def load_drivecycle_mat_files(base_folder: str):
    """
    Scan the 'drivecycle/' folder under base_folder. Load each .mat file
    via scipy.io.loadmat. Return a dict:
       { filename_without_ext: { varname: numpy_array, ... }, ... }.

    We assume each .mat has exactly one “user variable” that is an N×2 array:
      column 0 = time (s), column 1 = speed (mph).
    """
    drivecycle_dir = Path(base_folder) / "drivecycle"
    if not drivecycle_dir.is_dir():
        raise FileNotFoundError(f"Cannot find directory: {drivecycle_dir}")

    mat_data = {}
    for mat_file in drivecycle_dir.glob("*.mat"):
        try:
            data_dict = sio.loadmat(mat_file)
        except NotImplementedError:
            print(f"Warning: '{mat_file.name}' might be MATLAB v7.3. Skipping.")
            continue

        key = mat_file.stem
        mat_data[key] = data_dict
        user_vars = [k for k in data_dict.keys() if not k.startswith("__")]
        print(f"[Main] Loaded '{mat_file.name}' → variables = {user_vars}")

    return mat_data

# ─────────────────────────── LOAD TIME SERIES input-output data for building Hankel matrix ────────────────────────────
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
    ud = u.reshape(-1, 1)
    yd = v.reshape(-1, 1)
    return ud, yd

# ──────────────────────────── USER CHOOSE CYCLE KEY ──────────────────────────────
def choose_cycle_key(all_cycles):
    """
    Print all available cycle keys and ask the user to pick one or more.
    Keeps prompting until at least one valid key is selected.
    Returns a list of chosen keys.
    """
    keys_list = list(all_cycles.keys())

    while True:
        print("\nAvailable drive cycles:")
        for idx, k in enumerate(keys_list, start=1):
            print(f"  [{idx}] {k}")

        sel = input("Select cycles (comma-separated indices or names): ").strip()
        tokens = [t.strip() for t in sel.split(",") if t.strip()]

        cycle_keys = []
        for t in tokens:
            if t.isdigit():
                i = int(t) - 1
                if 0 <= i < len(keys_list):
                    cycle_keys.append(keys_list[i])
            elif t in keys_list:
                cycle_keys.append(t)

        # Remove duplicates, preserve order
        cycle_keys = list(dict.fromkeys(cycle_keys))

        if cycle_keys:
            return cycle_keys
        else:
            print("  → No valid selection detected. Please try again.\n")

def choose_vehicleModelName():
    """
    Print all available Tesla vehicle models ans ask the user to pick one for logging purpose.
    Keeps prompting until one and only one valid key is selected.
    Available vehicle models are: Model S,X,3,Y,Truck,Taxi
    Returns a string of vehicle model name
    """
    models = ["Model_S", "Model_X", "Model_3", "Model_Y", "Truck", "Taxi"]
    while True:
        print("\n Available vehicle models:")
        for idx, m in enumerate(models, start=1):
            print(f"[{idx}] {m}")
        sel = input("Select one model [index or name(case sensitive)]: ").strip()
        chosen = None

        if sel.isdigit():
            i = int(sel) - 1
            if 0 <= i <len(models):
                chosen = models[i]
        else:
            for m in models:
                if sel.lower() == m.lower():
                    chosen = m
                    break
        if chosen:
            return chosen
        else:
            print("  → Invalid selection. Please enter exactly one valid index or model name.\n")

# ───────────────────────────── GAIN‐SCHEDULING PID VALUES─────────────────────────────────
def get_gains_for_speed(ref_speed: float):
    """
    Return (Kp, Ki, Kd, Kff) according to the current reference speed (kph).

    """
    spd = ref_speed
    kp_bp_spd = np.array([0,20,40,60,80,100,120,140], dtype=float)
    kp_vals = np.array([6,7,8,9,9,10,10,10], dtype=float)
    kp = float(np.interp(spd, kp_bp_spd, kp_vals))

    ki_bp_spd = np.array([0,20,40,60,80,100,120,140], dtype=float)
    ki_vals = np.array([1.5,1.6,1.7,1.9,2,2,2,2], dtype=float)
    ki = float(np.interp(spd, ki_bp_spd, ki_vals))

    # The baseline code doesn't use kd, - now the kd_vals are wrong and random, adjust when needed
    kd_bp_spd = np.array([0,20,40,60,80,100,120], dtype=float)
    kd_vals = np.array([6,7,8,9,10,10,10], dtype=float)
    kd = float(np.interp(spd, kd_bp_spd, kd_vals))
    kd = 0

    kff_bp_spd = np.array([0,3,4,60,80,100,120,140], dtype=float)
    kff_vals = np.array([4,4,3,3,3,3,3,3], dtype=float)
    kff = float(np.interp(spd, kff_bp_spd, kff_vals))

    return (kp, ki, kd, kff)

def get_gains_for_speed_slower_frequency(ref_speed: float, Ts: float):
    """
    This function is to adjust PID gains accordingly with change in system frequency Ts.
    Return (Kp, Ki, Kd, Kff) according to the current reference speed (kph).

    """
    Ts_original = 0.01
    mux = Ts_original/Ts
    spd = ref_speed
    kp_bp_spd = np.array([0,20,40,60,80,100,120,140], dtype=float)
    kp_vals = np.array([6,7,8,9,9,10,10,10], dtype=float)
    kp = float(np.interp(spd, kp_bp_spd, kp_vals))

    ki_bp_spd = np.array([0,20,40,60,80,100,120,140], dtype=float)
    ki_vals = np.array([1.5,1.6,1.7,1.9,2,2,2,2], dtype=float)
    ki = float(np.interp(spd, ki_bp_spd, ki_vals))
    ki = ki * mux

    # The baseline code doesn't use kd, - now the kd_vals are wrong and random, adjust when needed
    kd_bp_spd = np.array([0,20,40,60,80,100,120], dtype=float)
    kd_vals = np.array([6,7,8,9,10,10,10], dtype=float)
    kd = float(np.interp(spd, kd_bp_spd, kd_vals))
    kd = 0

    kff_bp_spd = np.array([0,3,4,60,80,100,120,140], dtype=float)
    kff_vals = np.array([4,4,3,3,3,3,3,3], dtype=float)
    kff = float(np.interp(spd, kff_bp_spd, kff_vals))

    return (kp, ki, kd, kff)

# ───────────────────────────── DeePC RELATED - HANKEL MATRIX ─────────────────────────────────
def hankel(x, L):
    """
        ------Construct Hankel matrix------
        x: data sequence (data_size, x_dim)
        L: row dimension of the hankel matrix
        T: data samples of data x
        return: H(x): hankel matrix of x  H(x): (x_dim*L, T-L+1)
                H(x) = [x(0)   x(1) ... x(T-L)
                        x(1)   x(2) ... x(T-L+1)
                        .       .   .     .
                        .       .     .   .
                        .       .       . .
                        x(L-1) x(L) ... x(T-1)]
                Hankel matrix of order L has size:  (x_dim*L, T-L+1)
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    T, x_dim = x.shape

    Hx = np.zeros((L * x_dim, T - L + 1))
    for i in range(L):
        Hx[i * x_dim:(i + 1) * x_dim, :] = x[i:i + T - L + 1, :].T  # x need transpose to fit the hankel dimension
    return Hx

def hankel_full(ud, yd, Tini, THorizon):
    """
    Build the full DeePC Hankel matrix once by stacking past and future blocks.

    Parameters:
    -----------
    ud : array_like, shape (T_data, u_dim)
        Historical input sequence.
    yd : array_like, shape (T_data, y_dim)
        Historical output sequence.
    Tini : int
        Number of past (initialization) steps.
    THorizon : int
        Prediction horizon (number of future steps).

    Returns:
    --------
    hankel_full_mtx : np.ndarray, shape ((u_dim + y_dim) * (Tini + THorizon), K)
        A stacked Hankel matrix containing:
            [ Up;  # past-input block
              Yp;  # past-output block
              Uf;  # future-input block
              Yf ] # future-output block
        where K = T_data - (Tini + THorizon) + 1 is the total number of columns. (Large number)
    """
    # Build block-Hankel for inputs and outputs
    Hud = hankel(ud, Tini + THorizon)
    Huy = hankel(yd, Tini + THorizon)

    u_dim = ud.shape[1]
    y_dim = yd.shape[1]

    # Slice into past (first Tini) and future (last THorizon)
    Up = Hud[: u_dim * Tini, :]
    Uf = Hud[u_dim * Tini : u_dim * (Tini + THorizon), :]
    Yp = Huy[: y_dim * Tini, :]
    Yf = Huy[y_dim * Tini : y_dim * (Tini + THorizon), :]
    print(f"Hankel full matrix with shape: Up{Up.shape}, Uf{Uf.shape},Yp{Yp.shape},Yf{Yf.shape}")
    return Up, Uf, Yp, Yf

def hankel_subBlocks(Up, Uf, Yp, Yf, Tini, THorizon, hankel_subB_size, hankel_idx):
    """
    hankel_subB_size:   The sub-hankel matrix for current optimization problem
    hankel_idx:         the current hankel matrix in the official run
    The sub-hankel matrix was chosen as hankel_idx as center, and front g_dim, and back g_dim data section.
      front g_dim for state estimation, and back g_dim for prediction. g_dim I'm leaving for 50 front and 50 back buffer by choosing g_dim = 100(hankel_subB_size=199)
    
    shape: Up, Uf, Tp, Tf - (Tini, g_dim)/(THorizon, g_dim)
    """
    # how many columns on each side of hankel_idx we want
    g_dim = hankel_subB_size - Tini - THorizon + 1

    # desired slice is [start:end] with width = end - start = g_dim
    half  = g_dim // 2
    start = hankel_idx - half
    end   = hankel_idx + half
    width = end - start

    # allocate zero‐padded output blocks
    Up_cur = np.zeros((Tini,            width), dtype=Up.dtype)
    Uf_cur = np.zeros((Tini,            width), dtype=Uf.dtype)
    Yp_cur = np.zeros((THorizon, width), dtype=Yp.dtype)
    Yf_cur = np.zeros((THorizon, width), dtype=Yf.dtype)

    # clamp source columns to [0, max_col)
    max_col = Up.shape[1]
    src_start = max(start,               0)
    src_end   = min(end,   max_col)

    # where in the padded block these columns should go
    dst_start = src_start - start        # if start<0, dst_start>0
    dst_end   = dst_start + (src_end - src_start)

    # copy the in-bounds slice into the zero blocks
    Up_cur[:,      dst_start:dst_end] = Up[:Tini,         src_start:src_end]
    Uf_cur[:,      dst_start:dst_end] = Uf[:Tini,         src_start:src_end]
    Yp_cur[:,      dst_start:dst_end] = Yp[:THorizon,     src_start:src_end]
    Yf_cur[:,      dst_start:dst_end] = Yf[:THorizon,     src_start:src_end]

    return Up_cur, Uf_cur, Yp_cur, Yf_cur


# def hankel_subBlocks(Up, Uf, Yp, Yf, Tini, THorizon, hankel_subB_size, hankel_idx):

#     g_dim = hankel_subB_size - Tini - THorizon + 1
#     Up_cur = Up[:Tini,         hankel_idx-g_dim:hankel_idx+g_dim]
#     Uf_cur = Uf[:Tini,         hankel_idx-g_dim:hankel_idx+g_dim]
#     Yp_cur = Yp[Tini:THorizon, hankel_idx-g_dim:hankel_idx+g_dim]
#     Yf_cur = Yf[Tini:THorizon, hankel_idx-g_dim:hankel_idx+g_dim]
#     return Up_cur, Uf_cur, Yp_cur, Yf_cur

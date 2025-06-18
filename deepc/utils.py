import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
from datetime import datetime
import threading
from pathlib import Path

import numpy as np
import scipy.io as sio
import pandas as pd

import Jetson.GPIO as GPIO
GPIO.cleanup()

import board
import busio
from adafruit_pca9685 import PCA9685

import cantools
import can
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver

# ─────────────────────────── LOAD Data for DeePC ────────────────────────────
class MultiSheetTimeSeriesDataset(Dataset):
    def __init__(self, xlsx_file, seq_len, normalize=False, cache_path=None):
        """
        Reads every sheet in xlsx_file (all assumed to have columns
        ['Adj_Time','Duty_Cyc','DynoSpeed_kph']), then builds sliding windows.
        """
        if cache_path and os.path.exists(cache_path):
            data = np.load(cache_path)
            self.X = data["X"]
            self.y = data["y"]
            self.X_mean = data["X_mean"]
            self.X_std = data["X_std"]
            self.y_mean = data["y_mean"]
            self.y_std = data["y_std"]
            print(f"Loaded from cache: {cache_path}")
            return
        # Read all sheets into a dict of DataFrames
        sheets = pd.read_excel(xlsx_file, sheet_name=None)  # requires openpyxl
        
        self.seq_len = seq_len
        X_windows = []
        y_windows  = []
        print("Start loading data...")
        for name, df in sheets.items():
             # -- drop ALL rows with duplicate Adj_Time --
            df = df[df['Adj_Time'].ne(df['Adj_Time'].shift())].reset_index(drop=True)
            pwm = df['Duty_Cyc'].to_numpy(dtype=np.float32)
            vel = df['DynoSpeed_kph'].to_numpy(dtype=np.float32)
            
            # for each sheet, build its own sliding windows
            for i in range(len(pwm) - seq_len):
                X_windows.append(pwm[i : i + seq_len])
                y_windows.append(vel[i : i + seq_len])
            print(f"DataSheetName: {name},  pwm shape: {pwm.shape} vel shape: {vel.shape}")
        
        # stack into arrays
        self.X = np.stack(X_windows)    # shape [N_total, seq_len]
        self.y = np.stack(y_windows)    # shape [N_total]
        print(f"[MultiSheetTimeSeriesDataset] X shape: {self.X.shape}, y shape: {self.y.shape}")

         # compute and apply normalization if requested
        if normalize:
            # feature-wise global stats
            self.X_mean = self.X.mean()
            self.X_std  = self.X.std()
            self.y_mean = self.y.mean()
            self.y_std  = self.y.std()

            # standardize
            self.X = (self.X - self.X_mean) / self.X_std
            self.y = (self.y - self.y_mean) / self.y_std

        else:
            # placeholders if not normalized
            self.X_mean = 0.0; self.X_std = 1.0
            self.y_mean = 0.0; self.y_std = 1.0

        # after building self.X, self.y:
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(cache_path, X=self.X, y=self.y, X_mean = self.X_mean, X_std = self.X_std,
                                 y_mean=self.y_mean, y_std=self.y_std)
            print(f"Saved cache → {cache_path}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # x is still a full window
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(-1)  # [seq_len,1]
        y = torch.from_numpy(self.y[idx])                 # now [seq_len]

        return x, y
    
    def inverse_target(self, y_norm):
        """Convert a normalized target back to original units."""
        return y_norm * self.y_std + self.y_mean

    def inverse_input(self, X_norm):
        """Convert normalized inputs back to original scale."""
        return X_norm * self.X_std + self.X_mean

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

# ──────────────────────────── CAN LISTENER THREAD ──────────────────────────────
def can_listener_thread(dbc_path: str, can_iface: str):
    """
    Runs in a background thread. Listens for 'Speed_and_Force' on can_iface,
    decodes it using KAVL_V3.dbc, and updates globals latest_speed & latest_force.
    """
    global latest_speed, latest_force, can_running

    try:
        db = cantools.database.load_file(dbc_path)
    except FileNotFoundError:
        print(f"[CAN⋅Thread] ERROR: Cannot find DBC at '{dbc_path}'. Exiting CAN thread.")
        return

    try:
        speed_force_msg = db.get_message_by_name('Speed_and_Force')
    except KeyError:
        print("[CAN⋅Thread] ERROR: 'Speed_and_Force' not found in DBC. Exiting CAN thread.")
        return

    try:
        bus = can.interface.Bus(channel=can_iface, interface='socketcan')
    except OSError:
        print(f"[CAN⋅Thread] ERROR: Cannot open CAN interface '{can_iface}'. Exiting CAN thread.")
        return

    print(f"[CAN⋅Thread] Listening on {can_iface} for ID=0x{speed_force_msg.frame_id:03X}…")
    while can_running:
        msg = bus.recv(timeout=1.0)
        if msg is None:
            continue
        if msg.arbitration_id != speed_force_msg.frame_id:
            continue
        try:
            decoded = db.decode_message(msg.arbitration_id, msg.data)
        except KeyError:
            continue

        s = decoded.get('Speed_kph')
        f = decoded.get('Force_N')
        if s is not None:
            if -0.1 < s < 0.1:
                s = 0.0
            else:
                latest_speed = float(s)
        if f is not None:
            latest_force = float(f)

    bus.shutdown()
    print("[CAN⋅Thread] Exiting CAN thread.")

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


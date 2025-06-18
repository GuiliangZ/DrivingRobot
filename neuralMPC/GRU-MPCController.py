#!/usr/bin/env python3
"""
GRU-MPC where use trained GRU NN as system model and use MPC to track the pre-defined speed profile precisely.
Runs at 100 Hz (T_s = 0.01 s) to track a reference speed profile, reading v_meas from CAN,
and writing a duty‐cycle (–15% to +100%) to a PCA9685 PWM board.

Required setup:
  sudo ip link set can0 up type can bitrate 500000 dbitrate 1000000 fd on
  pip install numpy scipy cantools python-can adafruit-circuitpython-pca9685
  sudo pip install Jetson.GPIO
"""

# !!!!!!!!!! Always run this command line in the terminal to start the CAN reading: !!!!!!!!!
# sudo ip link set can0 up type can bitrate 500000 dbitrate 1000000 fd on

import os
import time
from datetime import datetime
import threading
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.optimize import minimize
import pandas as pd

import Jetson.GPIO as GPIO
GPIO.cleanup()

import board
import busio
from adafruit_pca9685 import PCA9685

import cantools
import can

import torch
import torch.nn as nn
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

# ────────────────────────── GLOBALS FOR CAN THREAD ─────────────────────────────
latest_speed = None    # Measured speed (kph) from CAN
latest_force = None    # Measured force (N) from CAN (unused here)
can_running  = True    # Flag to stop the CAN thread on shutdown

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
            latest_speed = float(s) if abs(s) >= 0.1 else 0.0
        if f is not None:
            latest_force = float(f) if abs(f) >= 0.1 else 0.0

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

# ────────────────────────── GRU MODEL & MPC SETUP ───────────────────────────────

# ─── PyTorch GRU REGRESSOR DEFINITION & LOADING ────────────────────────────────
class GRURegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        # x: [batch, seq_len, input_size]
        out, h = self.gru(x, h)           # out: [batch, seq_len, hidden_size]
        out = self.fc(out)                # out: [batch, seq_len, output_size]
        return out, h

def load_model(path, device):
    # instantiate model with the exact same hyperparameters as training
    model = GRURegressor(input_size=1,
                         hidden_size=128,
                         num_layers=2,
                         output_size=1).to(device)
    state = torch.load(path, map_location=device)
    # if you saved state_dict:
    model.load_state_dict(state)
    model.eval()
    return model

# ─── SIMPLE RANDOM-SHOOTING MPC FUNCTION ───────────────────────────────────────

def mpc_step(model, elapsed_time, ref_time, ref_speed,
             horizon=10, num_candidates=30,
             u_min=-15.0, u_max=100.0, Ts=0.01, device='cpu'):
    """
    Random-shooting MPC:
      • samples `num_candidates` PWM sequences of length `horizon`
      • rolls them through the GRU model to predict speeds
      • computes sum-of-squared tracking error against the reference
      • returns the first control move of the best sequence.
    """
    # 1) build future reference vector
    ref_future = np.array([
        float(np.interp(elapsed_time + Ts*(i+1), ref_time, ref_speed))
        for i in range(horizon)
    ], dtype=np.float32)  # shape (horizon,)

    # 2) sample candidate sequences in [u_min, u_max]
    cands = np.random.uniform(u_min, u_max,
                              size=(num_candidates, horizon)).astype(np.float32)

    # 3) roll out each candidate, compute cost
    costs = np.zeros(num_candidates, dtype=np.float32)
    with torch.no_grad():
        for i in range(num_candidates):
            u_seq = torch.from_numpy(cands[i:i+1, :]) \
                       .unsqueeze(2).to(device)  # [1, horizon, 1]
            preds, _ = model(u_seq)    # [1, horizon, 1]
            v_pred = preds.squeeze(0).squeeze(1).cpu().numpy()  # (horizon,)
            costs[i] = np.sum((v_pred - ref_future)**2)
    # 4) pick best and return its first element
    best_idx = int(np.argmin(costs))
    return float(cands[best_idx, 0])

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load trained GRU model (full model saved)
model = torch.load("gru_regressor_0527.pth", map_location=device)
model.to(device)
model.eval()

def mpc_control(past_u, ref_times, ref_speeds, elapsed_time, Ts, model, Np):
    """
    Simple receding-horizon MPC: assume model takes input sequence of length Np and predicts Np future speeds.
    We optimize the first control move by fitting a constant-u sequence over horizon.
    """
    # future reference
    t_future = elapsed_time + np.arange(1, Np+1) * Ts
    r_future = np.interp(t_future, ref_times, ref_speeds)

    def cost(u_seq):
        # prepare input: shape [1, Np, 1]
        u_in = torch.tensor(u_seq[None, :, None], dtype=torch.float32, device=device)
        with torch.no_grad():
            v_pred = model(u_in).cpu().numpy().flatten()
        return np.sum((v_pred - r_future)**2)

    # init guess: keep previous input
    u0 = np.ones(Np, dtype=float) * past_u[-1]
    bounds = [(-15.0, 100.0)] * Np
    res = minimize(cost, u0, bounds=bounds, options={'maxiter': 10, 'disp': False})
    if res.success:
        return float(res.x[0])
    else:
        return float(u0[0])


# ─────────────────────────────── MAIN CONTROL ─────────────────────────────────
if __name__ == "__main__":
    # ─── PCA9685 PWM SETUP ──────────────────────────────────────────────────────
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c, address=0x40)
    pca.frequency = 1000  # 1 kHz PWM

    def set_duty(channel: int, percent: float, retries: int = 3, delay: float = 0.01):
        """
        Send a duty‐cycle % [0..100] to PCA9685 channel, with automatic retry on I/O errors.
        """
        pct = float(percent)
        if not 0.0 <= pct <= 100.0:
            raise ValueError("set_duty: percent must be in [0,100]")
        duty_val = int(pct * 0xFFFF / 100.0)

        for attempt in range(1, retries + 1):
            try:
                pca.channels[channel].duty_cycle = duty_val
                return
            except OSError as e:
                # Most likely an I²C bus lock-up or transient noise
                print(f"[I2C ERROR] Channel {channel}, attempt {attempt}/{retries}: {e}")
                time.sleep(delay)
        # If we get here, all retries failed. Skip this update but let the loop continue.
        print(f"[WARN] Could not set channel {channel} after {retries} retries.")

    # ─── START CAN LISTENER THREAD ───────────────────────────────────────────────
    DBC_PATH = '/home/guiliang/Desktop/DrivingRobot/KAVL_V3.dbc'
    CAN_INTERFACE = 'can0'

    can_thread = threading.Thread(
        target=can_listener_thread,
        args=(DBC_PATH, CAN_INTERFACE),
        daemon=True
    )
    can_thread.start()

    # ─── PARAMETERS SETUP ───────────────────────────────────────────────  
    # Load reference cycle from .mat(s)
    base_folder = ""
    all_cycles = load_drivecycle_mat_files(base_folder)
    # Prompt the user:
    cycle_keys = choose_cycle_key(all_cycles)

    # Sampling time (discrete sample time)
    Ts = 0.01  # 100 Hz

    # Add this to regulate the rate of change of pwm output u
    max_delta = 50.0             # maximum % change per 0.01 s tick
    max_speed = 140.0
    MPC_Horizon = 5  # MPC horizon

    for cycle_key in cycle_keys:
        cycle_data = all_cycles[cycle_key]
        print(f"\n[Main] Using reference cycle '{cycle_key}'")

        mat_vars = [k for k in cycle_data.keys() if not k.startswith("__")]
        if len(mat_vars) != 1:
            raise RuntimeError(f"Expected exactly one variable in '{cycle_key}', found {mat_vars}")
        varname = mat_vars[0]
        ref_array = cycle_data[varname]

        if ref_array.ndim != 2 or ref_array.shape[1] < 2:
            raise RuntimeError(f"Expected '{varname}' to be N×2 array. Got shape {ref_array.shape}")

        # Extract reference time (s) and speed (mph)
        ref_time  = ref_array[:, 0].astype(float).flatten()
        # ref_speed = ref_array[:, 1].astype(float).flatten()
        ref_speed_mph = ref_array[:, 1].astype(float).flatten()
        ref_speed = ref_speed_mph * 1.60934 #kph

        print(f"[Main] Reference loaded: shape = {ref_array.shape}")

        # Prepare logging
        log_data       = []
        # Record loop‐start time so we can log elapsed time from 0.0
        run_start      = datetime.now()
        t0             = time.time()
        next_time      = t0

        # Reset PID state
        prev_error     = 0.0
        u_prev         = [0.0] * MPC_Horizon
        # Track previous reference speed for derivative on ref (if needed)
        prev_ref_speed = None

        print(f"\n[Main] Starting cycle '{cycle_key}', duration={ref_time[-1]:.2f}s")

    # ─── MAIN 100 Hz CONTROL LOOP ─────────────────────────────────────────────────
        print("[Main] Entering 100 Hz control loop. Press Ctrl+C to exit.\n")
        try:
            while True:
                now = time.time()
                if now < next_time:
                    time.sleep(next_time - now)
                current_time = time.time()

                # Compute elapsed time since loop start
                elapsed_time = current_time - t0

                # ──  Interpolate reference speed at t and t+Ts ───────────────────
                if elapsed_time <= ref_time[0]:
                    rspd_now = ref_speed[0]
                elif elapsed_time >= ref_time[-1]:
                    rspd_now = ref_speed[-1]
                    break
                else:
                    rspd_now = float(np.interp(elapsed_time, ref_time, ref_speed))

                # ── Compute current error e[k] ────────
                v_meas = latest_speed if latest_speed is not None else 0.0
                F_meas = latest_force if latest_force is not None else 0.0
                e_k    = rspd_now - v_meas
  
                # ── Total output u[k], clipped to [-15, +100] ────────────────
                u_unclamped = mpc_control(u_prev, ref_time, ref_speed, elapsed_time, Ts, model, MPC_Horizon)
                u = float(np.clip(u_unclamped, -15.0, +100.0))

                # Add a vehicle speed limiter safety feature
                # ──  Rate‐of‐change limiting on u ──────────────────────────────
                # Allow u to move by at most ±max_delta from the previous cycle:
                lower_bound = u_prev - max_delta
                upper_bound = u_prev + max_delta
                u = float(np.clip(u, lower_bound, upper_bound))

                # enforce a vehicle speed limiter safety
                if latest_speed is not None and latest_speed >= max_speed:
                    u = 0.0

                # ──  Send PWM to PCA9685: accel (ch=0) if u>=0, else brake (ch=4) ──
                if u >= 0.0:
                    set_duty(4, 0.0)            # ensure brake channel is zero
                    set_duty(0, u)      # channel 0 = accelerator
                else:
                    set_duty(0, 0.0)            # ensure accel channel is zero
                    set_duty(4, -u)    # channel 4 = brake

                # ──  Debug printout ─────────────────────────────────────────────
                print(
                    f"[{elapsed_time:.3f}] "
                    f"v_ref={rspd_now:6.2f} kph"
                    f"v_meas={v_meas:6.2f} kph, e={e_k:+6.2f}, "
                    f"u={u:+6.2f}%,"
                    f"F_dyno={F_meas:6.2f} N"
                )

                # ──  Save state for next iteration ──────────────────────────────
                prev_error     = e_k
                prev_ref_speed = rspd_now
                u_prev         = u

                # ── 11) Schedule next tick at 100 Hz ───────────────────────────────
                next_time += Ts

                # 12) Append this tick’s values to log_data
                log_data.append({
                    "time":      elapsed_time,
                    "v_ref":     rspd_now,
                    "v_meas":    v_meas,
                    "error":     e_k,
                    "u":         u,

                })

        except KeyboardInterrupt:
            print("\n[Main] KeyboardInterrupt detected. Exiting…")

        finally:
            # Stop CAN thread and wait up to 1 s
            can_running = False
            print("CAN_Running Stops!!!")
            can_thread.join(timeout=1.0)

            # Zero out all PWM channels before exiting
            for ch in range(16):
                pca.channels[ch].duty_cycle = 0
            # pca.deinit()
            print("[Main] pca board PWM signal cleaned up and set back to 0.")
                # ── Save log_data to Excel ───────────────────────────────────
            if log_data:
                df = pd.DataFrame(log_data)
                df['cycle_name']   = cycle_key
                datetime = datetime.now()
                df['run_datetime'] = datetime.strftime("%Y-%m-%d %H:%M:%S")
                # Build a descriptive filename
                timestamp_str = datetime.strftime("%Y%m%d_%H%M%S")
                excel_filename = f"DriveRobot_log_{cycle_key}_{timestamp_str}.xlsx"
                        # Ensure the subfolder exists
                log_dir = os.path.join(base_folder, "Log_DriveRobot")
                os.makedirs(log_dir, exist_ok=True)     
                excel_path = os.path.join(log_dir, excel_filename)

                df.to_excel(excel_path, index=False)
                print(f"[Main] Saved log to '{excel_path}'")

        print(f"[Main] Finish Running {cycle_key}, take a 5 second break...")
        time.sleep(5)

    pca.deinit()
    print("[Main] pca board PWM signal cleaned up and exited.")
    print("[Main] Cleaned up and exited.")

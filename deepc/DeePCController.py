#!/usr/bin/env python3
"""
DeePC - Data-Enabled Predictive Control controller use data directly in the optimization framework
to track the pre-defined speed profile precisely.
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
import deepctools as dpc

from utils import *

# ────────────────────────── GLOBALS FOR CAN THREAD ─────────────────────────────
latest_speed = None    # Measured speed (kph) from CAN
latest_force = None    # Measured force (N) from CAN (unused here)
can_running  = True    # Flag to stop the CAN thread on shutdown

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

    # Sampling time (Simulink discrete sample time)
    Ts = 0.01  # 100 Hz

    # Add this to regulate the rate of change of pwm output u
    max_delta = 50.0             # maximum % change per 0.01 s tick

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
        I_state        = 0.0
        D_f_state      = 0.0
        u_prev         = 0.0
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

                # ── 1) Interpolate reference speed at t and t+Ts ───────────────────
                if elapsed_time <= ref_time[0]:
                    rspd_now = ref_speed[0]
                elif elapsed_time >= ref_time[-1]:
                    # rspd_now = ref_speed[-1]
                    rspd_now = 0.0
                    break
                else:
                    rspd_now = float(np.interp(elapsed_time, ref_time, ref_speed))

                # ── 2) Compute current error e[k] ────────
                v_meas = latest_speed if latest_speed is not None else 0.0
                F_meas = latest_force if latest_force is not None else 0.0
                e_k    = rspd_now - v_meas
  





  
                # ── 3) Controller that generates u_unclamped ────────

                # --- a) Create a Deepc class ----------
                deepc = DeePC(data, Tini = T_INI, horizon = HORIZON)

                # --- b) build an optimization problem -----
                deepc.build_problem(        
                        build_loss = loss_callback,
                        build_constraints = constraints_callback,
                        lambda_g = LAMBDA_G_REGULARIZER,
                        lambda_y = LAMBDA_Y_REGULARIZER,
                        lambda_u = LAMBDA_U_REGULARIZER)

                # --- c) get the optimal control signal ----
                u_optimal = deepc.solve(data_ini = data_ini, warm_start=True, solver=cp.ECOS)

                # --- d) choose only the first control input as mpc input --- 





                # ── 7) Total output u[k], clipped to [-15, +100] ────────────────
                u_unclamped = 0
                u = float(np.clip(u_unclamped, -15.0, +100.0))

                # Add a vehicle speed limiter safety feature
                # ──  Rate‐of‐change limiting on u ──────────────────────────────
                # Allow u to move by at most ±max_delta from the previous cycle:
                lower_bound = u_prev - max_delta
                upper_bound = u_prev + max_delta
                u = float(np.clip(u, lower_bound, upper_bound))

                # enforce a vehicle speed limiter safety
                if latest_speed is not None and latest_speed >= 140.0:
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

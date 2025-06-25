#!/usr/bin/env python3
"""
DeePC - Data-Enabled Predictive Control controller use data directly in the optimization framework
to track the pre-defined speed profile precisely.
Runs at 100 Hz (T_s = 0.01 s) to track a reference speed profile, reading v_meas from CAN,
and writing a duty‐cycle (–15% to +100%) to a PCA9685 PWM board.

Required setup:
  pip install numpy scipy cantools python-can adafruit-circuitpython-pca9685
  sudo pip install Jetson.GPIO
  # !!!!!!!!!! Always run this command line in the terminal to start the CAN reading: !!!!!!!!!
    sudo ip link set can0 up type can bitrate 500000 dbitrate 1000000 fd on
    sudo ip link set can1 up type can bitrate 500000 dbitrate 1000000 fd on
    sudo ip link set can2 up type can bitrate 500000 dbitrate 1000000 fd on
"""
import os
import time
from datetime import datetime
import threading
from pathlib import Path

import numpy as np
import scipy.io as sio
import pandas as pd
from collections import deque

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
from deepctools.util import *
from utils import *
import DeePCAcados as dpcAcados

# ────────────────────────── GLOBALS FOR CAN THREAD ─────────────────────────────
latest_speed = None    # Measured speed (kph) from CAN
latest_force = None    # Measured force (N) from CAN (unused here)
dyno_can_running  = True    # Flag to stop the CAN thread on shutdown
veh_can_running  = True 
BMS_socMin = None     # Measured current vehicle SOC from Vehicle CAN
dyno_can_running  = False   # For temperal debugging
veh_can_running  = False 



# ─────────────────────────────── MAIN CONTROL ─────────────────────────────────
if __name__ == "__main__":
    # ─── DeePC Acados SETUP ──────────────────────────────────────────────────────
    # DeePC paramters    
    PROJECT_DIR = Path(__file__).resolve().parent 
    DATA_DIR   = PROJECT_DIR / "dataForHankle"                                   # Hankel matrix data loading location
    CACHE_FILE = os.path.join(DATA_DIR, "hankel_dataset.npz")       # Cache the previously saved Hankel matrix
    s = 1                       # How many steps before we solve again the DeePC problem - how many control input used per iteration
    Tini = 50                   # Size of the initial set of data - 0.5s bandwidth
    THorizon = 50               # Prediction Horizon length - Np
    lambda_g = 1                # g regularizer (see DeePC paper, eq. 8)
    lambda_y = 1                # y regularizer (see DeePC paper, eq. 8)
    lambda_u = 0                # u regularizer
    u_dim = 1                                   # the dimension of control inputs - DR case: 1 - PWM input
    y_dim = 1                                   # the dimension of controlled outputs - DR case: 1 -Dyno speed output
    Q = np.diag(np.tile(1, THorizon))           # the weighting matrix of controlled outputs y - Shape(THorizon, THorizon)-diagonal matrix
    R = np.diag(np.tile(1, THorizon))           # the weighting matrix of control inputs u - Shape(THorizon, THorizon)-diagonal matrix
    ineqconidx = {'u': [0], 'y':[0]}                                            # specify the wanted constraints for u and y - [0] means first channel which we only have 1 channel in DR project
    ineqconbd={'lbu': np.array([-15]), 'ubu': ([100]),
               'lby': np.array([0]), 'uby': np.array([140])}                    # specify the bounds for u and y

    if os.path.isfile(CACHE_FILE):
        npz = np.load(CACHE_FILE)
        ud, yd = npz['ud'], npz['yd']
    ud, yd = load_timeseries(DATA_DIR)          # history data collected offline to construct Hankel matrix; size (T, ud/yd)
    T = ud.shape[0]                             # the length of offline collected data
    np.savez(CACHE_FILE, ud=ud, yd=yd)
    # init deepc tools
    dpc_args = [u_dim, y_dim, T, Tini, THorizon, ud, yd, Q, R]
    dpc_kwargs = dict(lambda_g=lambda_g,
                      lambda_y=lambda_y,
                      sp_change=True,
                      ineqconidx=ineqconidx,
                      ineqconbd=ineqconbd
                      )
    dpc = dpc.deepctools(*dpc_args, **dpc_kwargs)

    # init and formulate deepc solver
    dpc_opts = {                            # cs.nlpsol solver parameters
        'ipopt.max_iter': 100,  # 50
        'ipopt.tol': 1e-5,
        'ipopt.print_level': 1,
        'print_time': 0,
        'ipopt.acceptable_tol': 1e-8,
        'ipopt.acceptable_obj_change_tol': 1e-6,
    }

    dpc.init_DeePCAcadosSolver(pts=dpc_opts)
    # dpc.init_DeePCsolver(pts=dpc_opts)            # Those solver are available as part of the deepctools, but may be slower than DeePCAcados for real time application
    # dpc.init_RDeePCsolver(pts=dpc_opts)

    # ─── PCA9685 PWM SETUP ──────────────────────────────────────────────────────
    # i2c = busio.I2C(board.SCL, board.SDA)
    # pca = PCA9685(i2c, address=0x40)
    # pca.frequency = 1000  # 1 kHz PWM                     # Comment out for temperal debugging
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
    DYNO_DBC_PATH = '/home/guiliang/Desktop/DrivingRobot/KAVL_V3.dbc'
    DYNO_CAN_INTERFACE = 'can1'
    VEH_DBC_PATH = '/home/guiliang/Desktop/DrivingRobot/vehBus.dbc'
    VEH_CAN_INTERFACE = 'can2'
    if dyno_can_running:
        dyno_can_thread = threading.Thread(
            target=dyno_can_listener_thread,
            args=(DYNO_DBC_PATH, DYNO_CAN_INTERFACE),
            daemon=True
        )
        dyno_can_thread.start()
    if veh_can_running:
        veh_can_thread = threading.Thread(
            target=veh_can_listener_thread,
            args=(VEH_DBC_PATH, VEH_CAN_INTERFACE),
            daemon=True
        )
        veh_can_thread.start()

    # ─── System Setup ────────────────────────────────────────────────
    base_folder = ""
    all_cycles = load_drivecycle_mat_files(base_folder) # Load reference cycle from .mat(s)
    cycle_keys = choose_cycle_key(all_cycles)           # Prompt the user to choose multiple drive cycles the user wish to test
    veh_modelName = choose_vehicleModelName()           # Prompt the user to choose the model of testing vehicle for logging purpose
    max_delta = 50.0            # maximum % change per 0.01 s tick - regulate the rate of change of pwm output u
    SOC_CycleStarting = 0.0     # Managing Vehicle SOC
    SOC_Stop = 2.2              # Stop the test at SOC 2.2% so the vehicle doesn't go completely drained that it cannot restart/charge

    Ts = 0.01                   # 100 Hz main control loop updating rate - Sampling time 

    for idx, cycle_key in enumerate(cycle_keys):
        # ----------------Stop the test if the vehicle SOC is too low to prevent draining the vehicle---------------------
        if BMS_socMin is not None and BMS_socMin <= SOC_Stop:
            break
        else:
            SOC_CycleStarting = BMS_socMin

        # ----------------Loading current cycle data----------------------------------------------------------------------
        cycle_data = all_cycles[cycle_key]
        print(f"\n[Main] Using reference cycle '{cycle_key}'")
        mat_vars = [k for k in cycle_data.keys() if not k.startswith("__")]
        if len(mat_vars) != 1:
            raise RuntimeError(f"Expected exactly one variable in '{cycle_key}', found {mat_vars}")
        varname = mat_vars[0]
        ref_array = cycle_data[varname]
        if ref_array.ndim != 2 or ref_array.shape[1] < 2:
            raise RuntimeError(f"Expected '{varname}' to be N×2 array. Got shape {ref_array.shape}")

        # -----------------Extract reference time (s) and speed (mph)--------------------------------------------------------
        ref_time  = ref_array[:, 0].astype(float).flatten()
        ref_speed_mph = ref_array[:, 1].astype(float).flatten()             # All the drive cycle .mat data file speed are in MPH
        ref_speed = ref_speed_mph * 1.60934                                 # now in kph
        print(f"[Main] Reference loaded: shape = {ref_array.shape}")
        ref_horizon_speed = ref_speed[:THorizon].reshape(-1,1)                    # Prepare reference speed horizon for DeePC - Length 

        # -----------------Reset states--------------------------------------------------------------------------------------
        prev_error     = 0.0
        u_prev         = 0.0
        prev_ref_speed = None                   # Track previous reference speed 
        u_history      = deque([0.0]*Tini,maxlen=Tini)     # Record the history of control input for DeePC generating u_ini
        spd_history    = deque([0.0]*Tini,maxlen=Tini)     # Record the history of control input for DeePC generating y_ini
        u_init = np.array(u_history).reshape(-1, 1)  # shape (<=Tini,1)
        y_init = np.array(spd_history).reshape(-1, 1)
        log_data       = []                     # Prepare logging
        # Record loop‐start time so we can log elapsed time from 0.0
        next_time      = time.time()
        now            = time.time()
        print(f"\n[Main] Starting cycle '{cycle_key}' on {veh_modelName}, duration={ref_time[-1]:.2f}s")

    # ─── MAIN 100 Hz CONTROL LOOP ─────────────────────────────────────────────────
        print("[Main] Entering 100 Hz control loop. Press Ctrl+C to exit.\n")
        try:
            while True:
                now = time.time()
                if now < next_time:
                    time.sleep(next_time - now)
                elapsed_time = now - next_time                 # Compute elapsed time since loop start

                # ── Interpolate reference speed at t and t+Ts ───────────────────
                if elapsed_time <= ref_time[0]:
                    rspd_now = ref_speed[0]
                elif elapsed_time >= ref_time[-1]:
                    rspd_now = 0.0
                    break
                else:
                    rspd_now = float(np.interp(elapsed_time, ref_time, ref_speed))
                    
                # -- Interpolate reference speed for DeePC ref_horizon_speed -------------------------
                t_future = elapsed_time + Ts * np.arange(THorizon)      # look 0.01 * THorizon s ahead of time
                if t_future[-1] >= ref_time[-1]:                        # if the last future time is beyond your reference horizon...
                    valid_mask = t_future <= ref_time[-1]               # build a boolean mask of all valid future times
                    THorizon = int(valid_mask.sum())                    # shrink THorizon to only those valid steps - !! Horizon will change in last few steps
                    t_future = t_future[valid_mask]             
                ref_horizon_speed = np.interp(t_future, ref_time, ref_speed)
                ref_horizon_speed = ref_horizon_speed.reshape(-1, 1)
                
                # ── Compute current error e[k] and future error e_fut[k] ────────
                v_meas = latest_speed if latest_speed is not None else 0.0
                F_meas = latest_force if latest_force is not None else 0.0
                e_k    = rspd_now - v_meas
                
                # ── Implementing real time acados based solver for DeePC ────────
                u_opt, g_opt, t_deepc = dpc.solver_step(uini=u_init, yini=y_init, yref=ref_horizon_speed)         #Generate a time series of "optimal" control input given v_ref and previous u and v_dyno(for implicit state estimation)

                # ──  Add a vehicle speed limiter safety feature (Theoretically don't need this part because all the control contraints are baked in the DeePC formulation) ────────
                u_unclamped = u_opt[0]# Total output u[k], clipped to [-15, +100] 
                u = float(np.clip(u_unclamped, -15.0, +100.0))
                lower_bound = u_prev - max_delta
                upper_bound = u_prev + max_delta                        # Allow u to move by at most ±max_delta from the previous cycle:
                u = float(np.clip(u, lower_bound, upper_bound))         # Rate‐of‐change limiting on u
                if latest_speed is not None and latest_speed >= 140.0:  # enforce a vehicle speed limiter safety
                    u = 0.0
                    break

                # ──  Send PWM to PCA9685: accel (ch=0) if u>=0, else brake (ch=4) ──
                if u >= 0.0:
                    set_duty(4, 0.0)            # ensure brake channel is zero
                    set_duty(0, u)              # channel 0 = accelerator
                else:
                    set_duty(0, 0.0)            # ensure accel channel is zero
                    set_duty(4, -u)             # channel 4 = brake

                # ──  Debug printout ─────────────────────────────────────────────
                print(
                    f"[{elapsed_time:.3f}] "
                    f"v_ref={rspd_now:6.2f} kph, "
                    f"v_meas={v_meas:6.2f} kph, e={e_k:+6.2f} kph,"
                    f"u={u:+6.2f}%,"
                    f"F_dyno={F_meas:6.2f} N,"
                    f"BMS_socMin={BMS_socMin:6.2f} %,"
                    f"SOC_CycleStarting={SOC_CycleStarting} %,"
                    f"t_deepc={t_deepc:6.2f} s"
                )

                # ── 10) Save state for next iteration ──────────────────────────────
                prev_error     = e_k
                prev_ref_speed = rspd_now
                u_prev         = u
                # record Tinit length of historical data for state estimation
                u_history.append(u)                         
                spd_history.append(v_meas)
                u_init = np.array(u_history).reshape(-1, 1)  # shape (Tini,1)
                y_init = np.array(spd_history).reshape(-1, 1)

                # ── 11) Schedule next tick at 100 Hz ───────────────────────────────
                next_time += Ts

                # 12) Append this tick’s values to log_data
                log_data.append({
                    "time":      elapsed_time,
                    "v_ref":     rspd_now,
                    "v_meas":    v_meas,
                    "u":         u,
                    "error":     e_k,
                    "t_deepc":   t_deepc,
                    "BMS_socMin":BMS_socMin,
                    "SOC_CycleStarting":SOC_CycleStarting,

                })
                if BMS_socMin <= SOC_Stop:
                    break

        except KeyboardInterrupt:
            print("\n[Main] KeyboardInterrupt detected. Exiting…")

        finally:
            for ch in range(16):
                pca.channels[ch].duty_cycle = 0         # Zero out all PWM channels before exiting
            print("[Main] pca board PWM signal cleaned up and set back to 0.")
                # ── Save log_data to Excel ───────────────────────────────────
            if log_data:
                df = pd.DataFrame(log_data)
                df['cycle_name']   = cycle_key
                datetime = datetime.now()
                df['run_datetime'] = datetime.strftime("%Y-%m-%d %H:%M:%S")
                timestamp_str = datetime.strftime("%H%M_%m%d")
                excel_filename = f"{timestamp_str}_DR_log_{veh_modelName}_{cycle_key}_{SOC_CycleStarting}%Start.xlsx"
                log_dir = os.path.join(base_folder, "Log_DriveRobot")
                os.makedirs(log_dir, exist_ok=True)     
                excel_path = os.path.join(log_dir, excel_filename)
                df.to_excel(excel_path, index=False)
                print(f"[Main] Saved log to '{excel_path}' as {excel_filename}")
        next_cycle = cycle_keys[idx+1] if idx+1 < len(cycle_keys) else None
        remaining_cycle = cycle_keys[idx+1:]
        print(f"[Main] Finish Running {cycle_key} on {veh_modelName}, Next running cycle {next_cycle}, take a 5 second break...")
        print(f"Current SOC: {BMS_socMin}%, system will stop at SOC: {SOC_Stop}% ")
        print(f"[Main] Plan to run the following cycles: {remaining_cycle}")
        time.sleep(5)

    # Stop CAN thread and wait up to 1 s
    dyno_can_running = False
    veh_can_running = False
    print("All CAN_Running Stops!!!")
    dyno_can_thread.join(timeout=1.0)
    veh_can_thread.join(timeout=1.0)
    pca.deinit()
    print("[Main] pca board PWM signal cleaned up and exited.")
    print("[Main] Cleaned up and exited.")

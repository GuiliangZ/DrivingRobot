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
# CAN setup
sudo modprobe can       # core CAN support
sudo modprobe can_raw   # raw-socket protocol
sudo modprobe can_dev   # network interface “canX” support
sudo modprobe kvaser_usb
sudo ip link set can0 down                         # if it was already up
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up
sudo ip link set can1 down
sudo ip link set can1 type can bitrate 500000
sudo ip link set can1 up
    i2cdetect -l
    ls /sys/class/net/ | grep can
    """
import psutil, os
import time
from datetime import datetime
import threading
from pathlib import Path

import numpy as np
import scipy.io as sio
import pandas as pd
from collections import deque

# import Jetson.GPIO as GPIO
# GPIO.cleanup()
# import board
# import busio
# from adafruit_pca9685 import PCA9685

import cantools
import can
import casadi as cs
from acados_template import AcadosOcp, AcadosOcpSolver
import deepctools as dpc
from deepctools.util import *
from utils_deepc import *
import DeePCAcados as dpcAcados

# ────────────────────────── GLOBALS FOR CAN THREAD ─────────────────────────────
algorithm_name = "DeePCAcados_Comp"
latest_speed = None                                 # Measured speed (kph) from CAN
latest_force = None                                 # Measured force (N) from CAN (unused here)
dyno_can_running  = True                            # Flag to stop the CAN thread on shutdown
veh_can_running  = True 
BMS_socMin = None                                   # Measured current vehicle SOC from Vehicle CAN
# dyno_can_running  = False                         # For temperal debugging
# veh_can_running  = False 
CP2112_BUS   = 17         # e.g. /dev/i2c-3

# ──────────────────────────── CAN LISTENER THREAD ──────────────────────────────
def dyno_can_listener_thread(dbc_path: str, can_iface: str):
    """
    Runs in a background thread. Listens for 'Speed_and_Force' on can_iface,
    decodes it using KAVL_V3.dbc, and updates globals latest_speed & latest_force.
    """
    global latest_speed, latest_force, dyno_can_running

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
    while dyno_can_running:
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
                latest_speed = int(round(s))
            else:
                latest_speed = float(s)
        if f is not None:
            latest_force = float(f)

    bus.shutdown()
    print("[CAN⋅Thread] Exiting CAN thread.")

def veh_can_listener_thread(dbc_path: str, can_iface: str):
    """
    Runs in a background thread. Listens for 'BMS_socMin' on can_iface,
    decodes it using vehBus.dbc, and updates globals BMS_socMin.
    """
    global BMS_socMin, veh_can_running

    try:
        db = cantools.database.load_file(dbc_path)
    except FileNotFoundError:
        print(f"[CAN⋅Thread] ERROR: Cannot find DBC at '{dbc_path}'. Exiting CAN thread.")
        return

    try:
        bms_soc_msg = db.get_message_by_name('BMS_socStatus')
    except KeyError:
        print("[CAN⋅Thread] ERROR: 'BMS_socStatus' not found in DBC. Exiting CAN thread.")
        return

    try:
        bus = can.interface.Bus(channel=can_iface, interface='socketcan')
    except OSError:
        print(f"[CAN⋅Thread] ERROR: Cannot open CAN interface '{can_iface}'. Exiting CAN thread.")
        return

    print(f"[CAN⋅Thread] Listening on {can_iface} for ID=0x{bms_soc_msg.frame_id:03X}…")
    while veh_can_running:
        msg = bus.recv(timeout=3.0)
        if msg is None:
            continue
        if msg.arbitration_id != bms_soc_msg.frame_id:
            continue
        try:
            decoded = db.decode_message(msg.arbitration_id, msg.data)
        except KeyError:
            continue

        BMS_socMin = decoded.get('BMS_socMin')
        if BMS_socMin is not None:
            if BMS_socMin <= 2:
                BMS_socMin = int(round(BMS_socMin))
            else:
                BMS_socMin = float(BMS_socMin)
    bus.shutdown()
    print("[CAN⋅Thread] Exiting CAN thread.")

# ─────────────────────────────── MAIN CONTROL ─────────────────────────────────
if __name__ == "__main__":
    # System parameters
    max_delta = 50.0                                    # maximum % change per 0.01 s(Ts) tick - regulate the rate of change of pwm output u
    SOC_CycleStarting = 0.0                             # Managing Vehicle SOC - record SOC at every starting point of the cycle
    SOC_Stop = 2.2                                      # Stop the test at SOC 2.2% so the vehicle doesn't go completely drained that it cannot restart/charge

    
    Ts = 0.1                                           # 100 Hz main control loop updating rate - Sampling time 
                # (if the actual loop time is larger than this Ts, the loop will updated at the actual computation time. 
                # It will only be at this control frequency when the loop is fast enough to finish computation under this time)
    
    # ─── DeePC Acados SETUP ──────────────────────────────────────────────────────
    # DeePC paramters    
    PROJECT_DIR = Path(__file__).resolve().parent 
    DATA_DIR   = PROJECT_DIR / "dataForHankle" / "smallDataSet"                 # Hankel matrix data loading location
    CACHE_FILE_Ori_DATA = os.path.join(DATA_DIR, "hankel_dataset.npz")          # Cache the previously saved SISO data
    CACHE_FILE_HANKEL_DATA = os.path.join(DATA_DIR, "hankel_matrix.npz")        # Cache the previously saved Hankel matrix
    if os.path.isfile(CACHE_FILE_Ori_DATA):
        print(f"[Main] Using cached input output data from {CACHE_FILE_Ori_DATA}")
        npz = np.load(CACHE_FILE_Ori_DATA, allow_pickle=True)
        ud, yd = npz['ud'], npz['yd']
    else:
        print("[Main] Start to load the fresh offline data for building hankel matrix... this may take a while")
        ud, yd = load_timeseries(DATA_DIR)             # history data collected offline to construct Hankel matrix; size (T, ud/yd)
        np.savez(CACHE_FILE_Ori_DATA, ud=ud, yd=yd)
        print(f"[Main] Finished loading data for hankel matrix, and saved to {CACHE_FILE_Ori_DATA}")

    # init deepc tools
    print("[Main] Initiate DeePC setup and compile procedure..")
    u_dim       = 1                                    # the dimension of control inputs - DR case: 1 - PWM input
    y_dim       = 1                                    # the dimension of controlled outputs - DR case: 1 -Dyno speed output
    # s = 1                                            # How many steps before we solve again the DeePC problem - how many control input used per iteration
    # DeePC related hyperparameters to tune
    recompile_solver = True                            # True to recompile the acados solver at change of following parameters. False to use the previously compiled solver
    use_data_for_hankel_cached = False                 # when want to load new excel data for building hankel matrix
    use_hankel_cached = False
    # Flip those logic to reuse what has already compiled to save time
    recompile_solver = False
    # use_data_for_hankel_cached = True                  # True to reuse the .npz file build from excel sheet
    # use_hankel_cached = True
    # Tini        = 30                                 # Size of the initial set of data       - 0.5s(5s) bandwidth (50)
    # THorizon    = 30                                 # Prediction Horizon length - Np        - 0.5s(5s) bandwidth (50) 
    # hankel_subB_size = 129                           # hankel sub-Block column size at each run-time step (199-299)!!! very important hyperparameter to tune. When 
    Tini        = 10                                   # Size of the initial set of data       - 0.5s(5s) bandwidth (50)
    THorizon    = 10                                   # Prediction Horizon length - Np        - 0.5s(5s) bandwidth (50) 
    hankel_subB_size = 40 
    Q_val = 1000                                         # the weighting matrix of controlled outputs y
    R_val = 1                                          # the weighting matrix of control inputs u
    lambda_g_val = 50                                  # the weighting matrix of norm of operator g
    lambda_y_val = 10                                  # the weighting matrix of mismatch of controlled output y
    lambda_u_val = 10                                  # the weighting matrix of mismatch of controlled output u
    T            = hankel_subB_size                    # the length of offline collected data - In my problem, OCP only see moving window of data which is same as "hankel_subB_size"
    g_dim        = T-Tini-THorizon+1                   # g_dim=T-Tini-Np+1 [Should g_dim >= u_dim * (Tini + Np)]

    # Smaller values (e.g., 0.05) result in slower decay, while larger values (e.g., 0.2) focus more on the first few steps.
    decay_rate_q = 0.1                                  # decay rate for Q weights
    decay_rate_r = 0.1                                  # decay rate for R weights
    # Define Q and R with exponential decay
    q_weights = Q_val * np.exp(-decay_rate_q * np.arange(THorizon))
    r_weights = R_val * np.exp(-decay_rate_r * np.arange(THorizon))
    Q = np.diag(q_weights)                              # Shape (THorizon, THorizon)
    R = np.diag(r_weights)                              # Shape (THorizon, THorizon)
    # Q           = np.diag(np.tile(Q_val, THorizon))                               # the weighting matrix of controlled outputs y - Shape(THorizon, THorizon)-diagonal matrix
    # R           = np.diag(np.tile(R_val, THorizon))                               # the weighting matrix of control inputs u - Shape(THorizon, THorizon)-diagonal matrix
    lambda_g    = np.diag(np.tile(lambda_g_val, g_dim))                           # weighting of the regulation of g (eq. 8) - shape(T-L+1, T-L+1)
    lambda_y    = np.diag(np.tile(lambda_y_val, Tini))                            # weighting matrix of noise of y (eq. 8) - shape(dim*Tini, dim*Tini)
    lambda_u  = np.diag(np.tile(lambda_u_val, Tini))                              # weighting matrix of noise of u - shape(dim*Tini, dim*Tini)

    #DeePC_kickIn_time = 100                          # because we need to build hankel matrix around the current time point, should be half of the hankel_subB_size
    if os.path.isfile(CACHE_FILE_HANKEL_DATA) and use_hankel_cached:
        print(f"[Main] Using cached hankel matrix data from {CACHE_FILE_HANKEL_DATA}")
        npz_hankel = np.load(CACHE_FILE_HANKEL_DATA)
        Up, Uf, Yp, Yf = npz_hankel['Up'], npz_hankel['Uf'], npz_hankel['Yp'], npz_hankel['Yf']
        # print(f"Up_cur shape{Up.shape} value: {Up}, "
        # f"Uf_cur shape{Uf.shape} value: {Uf}, "
        # f"Yp_cur shape{Yp.shape} value: {Yp}, "
        # f"Yf_cur shape{Uf.shape} value: {Uf}, ")
    else:
        print("[Main] Start to make hankel matrix data from cache... this may take a while")
        Up, Uf, Yp, Yf = hankel_full(ud, yd, Tini, THorizon)
        np.savez(CACHE_FILE_HANKEL_DATA, Up=Up, Uf=Uf, Yp=Yp, Yf=Yf)
        print(f"[Main] Finished making data for hankel matrix with shape Up{Up.shape}, Uf{Uf.shape}, Yp{Yp.shape}, Yf{Yf.shape}, and saved to {CACHE_FILE_HANKEL_DATA}")

    # Added a constraint to regulated the rate of change of control input u
    ineqconidx  = {'u': [0], 'y':[0], 'du':[0]}                                     # specify the wanted constraints for u and y - [0] means first channel which we only have 1 channel in DR project
    ineqconbd   ={'lbu': np.array([-15]), 'ubu': ([100]),                           # specify the bounds for u and y
                    'lby': np.array([0]), 'uby': np.array([140]),
                    'lbdu': np.array([-10]), 'ubdu': np.array([1.2])}               # lower and upper bound for change of control input - can find the approximate range from baseline data for 100 Hz             
    # ineqconidx = {'u': list(range(u_dim)), 'y': list(range(y_dim))}
    # ineqconbd = {
    #     'lbu': np.tile([-30], Tini),  # Apply -30 to all u steps
    #     'ubu': np.tile([100], Tini),  # Apply 100 to all u steps
    #     'lby': np.tile([0], Tini),    # Apply 0 to all y steps
    #     'uby': np.tile([140], Tini),  # Apply 140 to all y steps
    #     # 'lbdu': np.tile([-10], Tini), # Add Delta u constraints
    #     # 'ubdu': np.tile([10], Tini)   # Add Delta u constraints
    # }

    dpc_args = [u_dim, y_dim, T, Tini, THorizon]                                    # THorizon is Np in dpc class
    dpc_kwargs = dict(ineqconidx=ineqconidx, ineqconbd=ineqconbd)
    dpc = dpcAcados.deepctools(*dpc_args, **dpc_kwargs)

    # init and formulate deepc solver
    dpc.init_DeePCAcadosSolver(recompile_solver=recompile_solver, ineqconidx=ineqconidx, ineqconbd=ineqconbd) # Use acados solver
    # dpc_opts = {                            # cs.nlpsol solver parameters - not used in acados
    #     'ipopt.max_iter': 100,  # 50
    #     'ipopt.tol': 1e-5,
    #     'ipopt.print_level': 1,
    #     'print_time': 0,
    #     'ipopt.acceptable_tol': 1e-8,
    #     'ipopt.acceptable_obj_change_tol': 1e-6,
    # }
    # Specify what solver wanted to use - # Those solver are available as part of the deepctools, but may be slower than DeePCAcados for real time application
    # dpc.init_DeePCsolver(uloss='u', ineqconidx=ineqconidx, ineqconbd=ineqconbd, opts=dpc_opts)            
    # dpc.init_RDeePCsolver(uloss='u', ineqconidx=ineqconidx, ineqconbd=ineqconbd, opts=dpc_opts)
    # dpc.init_FullRDeePCsolver(uloss='u', ineqconidx=ineqconidx, ineqconbd=ineqconbd, opts=dpc_opts)
    print("[Main] Finished compiling DeePC problem, starting the nominal system setup procedure!")

    # ─── PCA9685 PWM SETUP ──────────────────────────────────────────────────────
    bus = SMBus(CP2112_BUS)

    # ─── START CAN LISTENER THREAD ───────────────────────────────────────────────
    DYNO_DBC_PATH = '/home/guiliang/Desktop/DrivingRobot/KAVL_V3.dbc'
    DYNO_CAN_INTERFACE = 'can0'
    VEH_DBC_PATH = '/home/guiliang/Desktop/DrivingRobot/vehBus.dbc'
    VEH_CAN_INTERFACE = 'can1'
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

    for idx, cycle_key in enumerate(cycle_keys):
        # ----------------Stop the test if the vehicle SOC is too low to prevent draining the vehicle---------------------
        if BMS_socMin is not None and BMS_socMin <= SOC_Stop:
            break
        else:
            SOC_CycleStarting = BMS_socMin
            if SOC_CycleStarting is not None:
                SOC_CycleStarting = round(SOC_CycleStarting, 2)
            else:
                SOC_CycleStarting = 0.0

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
        ref_speed_mph = ref_array[:, 1].astype(float).flatten()         # All the drive cycle .mat data file speed are in MPH
        ref_speed = ref_speed_mph * 1.60934                             # now in kph
        print(f"[Main] Reference loaded: shape = {ref_array.shape}")
        ref_horizon_speed = ref_speed[:THorizon].reshape(-1,1)          # Prepare reference speed horizon for DeePC - Length 

        # -----------------Reset states--------------------------------------------------------------------------------------
        loop_count     = 0
        hankel_idx     = 0
        prev_error     = 0.0
        u_prev         = 0.0
        g_prev         = None                                           # Record DeePC decision matrix g for solver hot start   
        prev_ref_speed = None                                           # Track previous reference speed 
        u_history      = deque([0.0]*Tini,maxlen=Tini)                  # Record the history of control input for DeePC generating u_ini
        spd_history    = deque([0.0]*Tini,maxlen=Tini)                  # Record the history of control input for DeePC generating y_ini
        u_init = np.array(u_history).reshape(-1, 1)                     # shape (u_dim*Tini,1)
        y_init = np.array(spd_history).reshape(-1, 1)
        log_data       = []                                             # Prepare logging
        # Record loop‐start time so we can log elapsed time from 0.0
        next_time      = time.perf_counter()
        t0             = time.perf_counter()
        print(f"\n[Main] Starting cycle '{cycle_key}' on {veh_modelName}, duration={ref_time[-1]:.2f}s")

        # ----------------Real-time Effort (try to avoid system lags - each loop more than 10ms)
        # For real-time effort - put into kernel - linux 5.15.0-1087-realtime for strict time update - but this kernel doesn't have wifi and nvidia drive
        SCHED_FIFO = os.SCHED_FIFO
        priority = 99
        param = os.sched_param(priority)
        try:
            os.sched_setscheduler(0, SCHED_FIFO, param)
            print(f"[Main] Real-time scheduling enabled: FIFO, priority={priority}")
        except:
            print("Need to run as root (or have CAP_SYS_NICE)")

    # ─── MAIN 100 Hz CONTROL LOOP ─────────────────────────────────────────────────
        print("[Main] Entering 100 Hz control loop. Press Ctrl+C to exit.\n")
        try:
            while True:
                loop_start = time.perf_counter()
                sleep_for = next_time - loop_start
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_time = time.perf_counter()
                elapsed_time = loop_start - t0                          # Compute elapsed time since loop start

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
                # print(
                #     f"The ref_horizon_speed for DeePC with shape{ref_horizon_speed.shape} is: {ref_horizon_speed.flatten().ravel()}"
                #     f"The uini for DeePC with shape{u_init.shape} is {u_init.flatten().ravel()}"
                #     f"The yini for DeePC with shape{y_init.shape} is {y_init.flatten().ravel()}")
                
                # ── Compute current error e[k] and future error e_fut[k] ────────
                v_meas = latest_speed if latest_speed is not None else 0.0
                F_meas = latest_force if latest_force is not None else 0.0
                e_k    = rspd_now - v_meas
                
                # ── Implementing real time acados based solver for DeePC ────────
                # if hankel_idx == DeePC_kickIn_time (initial time where DeePC kicks in): # At start: can build a local hankel matrix now, start to use DeePC
                # if hankel_idx+THorizon >= DeePC_stop_time  # At the end, hankel matrix exceed the full length of reference data
                Up_cur, Uf_cur, Yp_cur, Yf_cur = hankel_subBlocks(Up, Uf, Yp, Yf, Tini, THorizon, hankel_subB_size, hankel_idx)
                # print(
                    # f"Up_cur shape{Up_cur.shape} value: {Up_cur}, "
                    #   f"Uf_cur shape{Uf_cur.shape} value: {Uf_cur}, "
                    #   f"Yp_cur shape{Yp_cur.shape} value: {Yp_cur}, "
                    #   f"Yf_cur shape{Uf_cur.shape} value: {Uf_cur}, "
                    #   f"hankel_idx {hankel_idx}")
                
                u_opt, g_opt, t_deepc, exist_feasible_sol = dpc.acados_solver_step(uini=u_init, yini=y_init, yref=ref_horizon_speed,           # For real-time Acados solver-Generate a time series of "optimal" control input given v_ref and previous u and v_dyno(for implicit state estimation)
                                                                Up_cur=Up_cur, Uf_cur=Uf_cur, Yp_cur=Yp_cur, Yf_cur=Yf_cur, Q_val=Q, R_val=R,
                                                                lambda_g_val=lambda_g, lambda_y_val=lambda_y, lambda_u_val=lambda_u, g_prev = g_prev)   

                # ──  Add a vehicle speed limiter safety feature (Theoretically don't need this part because all the control contraints are baked in the DeePC formulation) ────────
                u_unclamped = u_opt[0]       
                u = float(np.clip(u_unclamped, -15.0, +100.0))          # Total output u[k], clipped to [-15, +100]
                lower_bound = u_prev - max_delta
                upper_bound = u_prev + max_delta                        # Allow u to move by at most ±max_delta from the previous cycle:
                u = float(np.clip(u, lower_bound, upper_bound))         # Rate‐of‐change limiting on u
                if latest_speed is not None and latest_speed >= 140.0:  # enforce a vehicle speed limiter safety
                    u = 0.0
                    break

                # ──  Send PWM to PCA9685: accel (ch=0) if u>=0, else brake (ch=4) ──
                if u >= 0.0:
                    set_duty_cycle(bus, 4, 0.0)                                    # ensure brake channel is zero
                    set_duty_cycle(bus, 0, u)                                      # channel 0 = accelerator
                else:
                    set_duty_cycle(bus, 0, 0.0)                                    # ensure brake channel is zero
                    set_duty_cycle(bus, 4, -u)                                     # channel 4 = brake

                actual_elapsed_time = round((time.perf_counter() - loop_start)*1000,3)
                actual_control_frequency = 1/(actual_elapsed_time / 1000)
                # ──  Debug printout ─────────────────────────────────────────────
                print(
                    f"[{elapsed_time:.3f}] "
                    f"v_ref={rspd_now:6.2f}kph, "
                    f"v_meas={v_meas:6.2f}kph, e={e_k:+6.2f}kph, "
                    f"u={u:6.2f}%, "
                    f"F_dyno={F_meas:6.2f} N, "
                    f"BMS_socMin={BMS_socMin:6.2f} %,"
                    f"SOC_CycleStarting AT={SOC_CycleStarting:6.2f} %, "
                    f"t_deepc={t_deepc:6.3f} ms, "
                    f"actual_elapsed_time_per_loop={actual_elapsed_time:6.3f} ms, "
                    f"actual_control_frequency={actual_control_frequency:6.3f} Hz"
                )

                # print(f"loop_count: {loop_count}")

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
                g_prev = g_opt
                loop_count += 1
                # Update hankel_idx: because this is not ROTS system, 
                # there's lags, it's not running exactly Ts per loop, 
                # make hankel index correspond to the first 4 digits of elapsed_time

                # Important parameter to tune to make sure it synced with data - related to control frequency
                s = f"{elapsed_time:.3f}"               # "20.799"
                digits = s.replace(".", "")             # "20799"
                hankel_idx = int(digits[:4])            # 2079
               
                # 12) Append this tick’s values to log_data
                log_data.append({
                    "time":      elapsed_time,
                    "v_ref":     rspd_now,
                    "v_meas":    v_meas,
                    "u":         u,
                    "error":     e_k,
                    "t_deepc(ms)":   t_deepc,
                    "BMS_socMin":BMS_socMin,
                    "SOC_CycleStarting":SOC_CycleStarting,
                })
                if BMS_socMin <= SOC_Stop:
                    break

        except KeyboardInterrupt:
            print("\n[Main] KeyboardInterrupt detected. Exiting…")

        finally:
            for ch in range(16):
                set_duty_cycle(bus, channel=ch, percent=0.0)                              # Zero out all PWM channels before exiting
            print("[Main] pca board PWM signal cleaned up and set back to 0.")
                # ── Save log_data to Excel ───────────────────────────────────
            if log_data:
                df = pd.DataFrame(log_data)
                df['cycle_name']   = cycle_key
                datetime = datetime.now()
                df['run_datetime'] = datetime.strftime("%Y-%m-%d %H:%M:%S")
                timestamp_str = datetime.strftime("%H%M_%m%d")
                excel_filename = f"{timestamp_str}_DR_log_{veh_modelName}_{cycle_key}_Start{SOC_CycleStarting}%_{algorithm_name}_Ts{Ts}_Q{Q_val}_R{R_val}_Tini{Tini}_gDim{g_dim}_λg{lambda_g_val}_λu{lambda_u_val}__λy{lambda_y_val}.xlsx"
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
    bus.close()
    print("[Main] pca board PWM signal cleaned up and exited.")
    print("[Main] Cleaned up and exited.")

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

def load_timeseries(data_dir):
    """
    Read every .xlsx in data_dir, concatenating their 'u' and 'v_meas' columns.
    Returns
    -------
    u : np.ndarray, shape (T_total,)
    v : np.ndarray, shape (T_total,)
    """
    u_list, v_list, vref_list = [], [], []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.xlsx'):
            continue
        df = pd.read_excel(os.path.join(data_dir, fname))
        u_list.append(df['u'].values)
        v_list.append(df['v_meas'].values)
        vref_list.append(df['v_ref'].values)
    u = np.concatenate(u_list, axis=0)
    v = np.concatenate(v_list, axis=0)
    vref = np.concatenate(vref_list, axis=0)
    ud = u.reshape(-1, 1)
    yd = v.reshape(-1, 1)
    vref = vref.reshape(-1,1)
    return ud, yd, vref

# System parameters
algorithm_name = "deepc_toy_example"
Ts = 0.1                                            # 100 Hz main control loop updating rate - Sampling time 
Tini        = 20                                    # Size of the initial set of data       - 0.5s(5s) bandwidth (50)
THorizon    = 20                                    # Prediction Horizon length - Np        - 0.5s(5s) bandwidth (50) 
hankel_subB_size = 80                               # >(Tini+THorizon)*2-hankel sub-Block column size at each run-time step (199-299)!!! very important hyperparameter to tune. When 
# Tini        = 20                                  # Size of the initial set of data       - 0.5s(5s) bandwidth (50)
# THorizon    = 20                                  # Prediction Horizon length - Np        - 0.5s(5s) bandwidth (50) 
# hankel_subB_size = 89                             # >(Tini+THorizon)*2-hankel sub-Block column size at each run-time step (199-299)!!! very important hyperparameter to tune. When 

# Q_val = 10                                       # the weighting matrix of controlled outputs y
# R_val = 50                                         # the weighting matrix of control inputs u
# lambda_g_val = 70                                # the weighting matrix of norm of operator g
# lambda_y_val = 10                                 # the weighting matrix of mismatch of controlled output y
# lambda_u_val= 5                                  # the weighting matrix of mismatch of controlled output u
Q_val = 10                                       # the weighting matrix of controlled outputs y
R_val = 10                                         # the weighting matrix of control inputs u
lambda_g_val = 1                                # the weighting matrix of norm of operator g
lambda_y_val = 1                                 # the weighting matrix of mismatch of controlled output y
lambda_u_val= 1                                  # the weighting matrix of mismatch of controlled output u
T           = hankel_subB_size                    # the length of offline collected data - In my problem, OCP only see moving window of data which is same as "hankel_subB_size"
g_dim       = T-Tini-THorizon+1                   # g_dim=T-Tini-Np+1 [Should g_dim >= u_dim * (Tini + Np)]
u_dim       = 1                                   # the dimension of control inputs - DR case: 1 - PWM input
y_dim       = 1                                   # the dimension of controlled outputs - DR case: 1 -Dyno speed output

recompile_solver = True                           # True to recompile the acados solver at change of following parameters. False to use the previously compiled solver
use_data_for_hankel_cached = False                  # when want to load new excel data for building hankel matrix
use_hankel_cached = False
use_data_for_hankel_cached_sim = False                   # True to reuse the .npz file build from excel sheet
use_hankel_cached_sim = False

# Flip those logic to reuse what has already compiled to save time
# recompile_solver = False
# use_data_for_hankel_cached = True                   # True to reuse the .npz file build from excel sheet
# use_hankel_cached = True
# use_data_for_hankel_cached_sim = True                   # True to reuse the .npz file build from excel sheet
# use_hankel_cached_sim = True

# Since the g_dim is too big, if use original deepctools, the matrix become untractable, need to use casadi representation to formulate the problem
lambda_g    = np.diag(np.tile(lambda_g_val, g_dim))                           # weighting of the regulation of g (eq. 8) - shape(T-L+1, T-L+1)
lambda_y    = np.diag(np.tile(lambda_y_val, Tini))                            # weighting matrix of noise of y (eq. 8) - shape(dim*Tini, dim*Tini)
lambda_u  = np.diag(np.tile(lambda_u_val, Tini))                            # weighting matrix of noise of u - shape(dim*Tini, dim*Tini)
Q           = np.diag(np.tile(Q_val, THorizon))                               # the weighting matrix of controlled outputs y - Shape(THorizon, THorizon)-diagonal matrix
R           = np.diag(np.tile(R_val, THorizon))                               # the weighting matrix of control inputs u - Shape(THorizon, THorizon)-diagonal matrix

# Added a constraint to regulated the rate of change of control input u
ineqconidx  = {'u': [0], 'y':[0], 'du':[0]}                                     # specify the wanted constraints for u and y - [0] means first channel which we only have 1 channel in DR project
# ineqconidx  = {'u': [0], 'y':[0]} 
ineqconbd   ={'lbu': np.array([-30]), 'ubu': ([100]),                           # specify the bounds for u and y
                'lby': np.array([0]), 'uby': np.array([140]),
                'lbdu': np.array([-10]), 'ubdu': np.array([1.5])}               # lower and upper bound for change of control input - can find the approximate range from baseline data for 100 Hz             
# ineqconidx = {'u': list(range(u_dim)), 'y': list(range(y_dim))}
# ineqconbd = {
#     'lbu': np.tile([-30], Tini),  # Apply -30 to all u steps
#     'ubu': np.tile([100], Tini),  # Apply 100 to all u steps
#     'lby': np.tile([0], Tini),    # Apply 0 to all y steps
#     'uby': np.tile([140], Tini),  # Apply 140 to all y steps
#     # 'lbdu': np.tile([-10], Tini), # Add Delta u constraints
#     # 'ubdu': np.tile([10], Tini)   # Add Delta u constraints
# }

print("[Main] Initiate DeePC setup and compile procedure..")
PROJECT_DIR = Path().resolve()
# PROJECT_DIR = Path(__file__).resolve().parent 
DATA_DIR   = PROJECT_DIR / "dataForHankle" / "smallDataSet"                 # Hankel matrix data loading location
CACHE_FILE_Ori_DATA = os.path.join(DATA_DIR, "hankel_dataset.npz")          # Cache the previously saved SISO data
CACHE_FILE_HANKEL_DATA = os.path.join(DATA_DIR, "hankel_matrix.npz")        # Cache the previously saved Hankel matrix
DATA_DIR_Sim   = PROJECT_DIR / "dataForHankle" / "SimulateDR"                 # Hankel matrix data loading location
CACHE_FILE_Ori_DATA_Sim = os.path.join(DATA_DIR_Sim, "hankel_dataset_simulate.npz")          # Cache the previously saved SISO data
CACHE_FILE_HANKEL_DATA_Sim = os.path.join(DATA_DIR_Sim, "hankel_matrix_simulate.npz")        # Cache the previously saved Hankel matrix
print(DATA_DIR)

if os.path.isfile(CACHE_FILE_Ori_DATA) and use_data_for_hankel_cached:
    print(f"[Main] Using cached input output data from {CACHE_FILE_Ori_DATA}")
    npz = np.load(CACHE_FILE_Ori_DATA, allow_pickle=True)
    ud, yd = npz['ud'], npz['yd']
else:
    print("[Main] Start to load the fresh offline data for building hankel matrix... this may take a while")
    ud, yd, vref = load_timeseries(DATA_DIR)          # history data collected offline to construct Hankel matrix; size (T, ud/yd)
    np.savez(CACHE_FILE_Ori_DATA, ud=ud, yd=yd)
    print(f"[Main] Finished loading data for hankel matrix, and saved to {CACHE_FILE_Ori_DATA}")
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

if os.path.isfile(CACHE_FILE_Ori_DATA_Sim) and use_data_for_hankel_cached_sim:
    print(f"[Main] Using cached input output data from {CACHE_FILE_Ori_DATA_Sim}")
    npz_sim = np.load(CACHE_FILE_Ori_DATA_Sim, allow_pickle=True)
    ud_sim, yd_sim = npz_sim['ud_sim'], npz_sim['yd_sim']
else:
    print("[Main] Start to load the fresh offline data for building hankel matrix... this may take a while")
    ud_sim, yd_sim, vref_sim = load_timeseries(DATA_DIR_Sim)          # history data collected offline to construct Hankel matrix; size (T, ud/yd)
    np.savez(CACHE_FILE_Ori_DATA_Sim, ud_sim=ud_sim, yd_sim=yd_sim)
    print(f"[Main] Finished loading data for hankel matrix, and saved to {CACHE_FILE_Ori_DATA_Sim}")
if os.path.isfile(CACHE_FILE_HANKEL_DATA_Sim) and use_hankel_cached_sim:
    print(f"[Main] Using cached hankel matrix data from {CACHE_FILE_HANKEL_DATA_Sim}")
    npz_hankel_sim = np.load(CACHE_FILE_HANKEL_DATA_Sim)
    Up_sim, Uf_sim, Yp_sim, Yf_sim = npz_hankel_sim['Up'], npz_hankel_sim['Uf'], npz_hankel_sim['Yp'], npz_hankel_sim['Yf']
    # print(f"Up_cur shape{Up.shape} value: {Up}, "
    # f"Uf_cur shape{Uf.shape} value: {Uf}, "
    # f"Yp_cur shape{Yp.shape} value: {Yp}, "
    # f"Yf_cur shape{Uf.shape} value: {Uf}, ")
else:
    print("[Main] Start to make hankel matrix data from cache... this may take a while")
    Up_sim, Uf_sim, Yp_sim, Yf_sim = hankel_full(ud_sim, yd_sim, Tini, THorizon)
    np.savez(CACHE_FILE_HANKEL_DATA_Sim, Up=Up_sim, Uf=Uf_sim, Yp=Yp_sim, Yf=Yf_sim)
    print(f"[Main] Finished making data for hankel matrix with shape Up_sim{Up_sim.shape}, Uf_sim{Uf_sim.shape}, Yp_sim{Yp_sim.shape}, Yf_sim{Yf_sim.shape}, and saved to {CACHE_FILE_Ori_DATA_Sim}")

print(f"[Main] Finished making data for hankel matrix with shape Up{Up.shape}, Uf{Uf.shape}, Yp{Yp.shape}, Yf{Yf.shape}, and saved to {CACHE_FILE_HANKEL_DATA}")
print(f"[Main] Finished making data for hankel matrix with shape Up_sim{Up_sim.shape}, Uf_sim{Uf_sim.shape}, Yp_sim{Yp_sim.shape}, Yf_sim{Yf_sim.shape}, and saved to {CACHE_FILE_Ori_DATA_Sim}")
# print(f"vref:{vref}")
# ─── System Setup ────────────────────────────────────────────────
base_folder = "../"
# base_folder = ""
all_cycles = load_drivecycle_mat_files(base_folder) # Load reference cycle from .mat(s)
cycle_key = 8                     # 8 for WLTP cycle
veh_modelName = 3
# ----------------Loading current cycle data----------------------------------------------------------------------
keys = list(all_cycles.keys())
idx = int(cycle_key) - 1
cycle_key = keys[idx]
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

# -----------------Reset states--------------------------------------------------------------------------------------
loop_count     = 0
hankel_idx     = 1
prev_error     = 0.0
t_deepc        = 0.0
u_prev         = 0.0
g_prev         = None                                           # Record DeePC decision matrix g for solver hot start   
prev_ref_speed = None                                           # Track previous reference speed 
exist_feasible_sol = False
log_data       = [] 

u_history      = deque([0.0]*Tini,maxlen=Tini)                  # Record the history of control input for DeePC generating u_ini
spd_history    = deque([0.0]*Tini,maxlen=Tini)                  # Record the history of control input for DeePC generating y_ini
u_init = np.array(u_history).reshape(-1, 1)                     # shape (u_dim*Tini,1)
y_init = np.array(spd_history).reshape(-1, 1)

# Record loop‐start time so we can log elapsed time from 0.0
next_time      = time.perf_counter()
t0             = time.perf_counter()
# --------------- Start of the control loop --------------------------------------
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
        # Debug Purpose
        # print(
        #     f"The ref_horizon_speed for DeePC with shape{ref_horizon_speed.shape} is: {ref_horizon_speed.flatten().ravel()}"
        #     f"The uini for DeePC with shape{u_init.shape} is {u_init.flatten().ravel()}"
        #     f"The yini for DeePC with shape{y_init.shape} is {y_init.flatten().ravel()}")
        
        # ── Implementing real time acados based solver for DeePC ────────
        # if hankel_idx == DeePC_kickIn_time (initial time where DeePC kicks in): # At start: can build a local hankel matrix now, start to use DeePC
        # if hankel_idx+THorizon >= DeePC_stop_time  # At the end, hankel matrix exceed the full length of reference data
        Up_cur, Uf_cur, Yp_cur, Yf_cur = hankel_subBlocks(Up, Uf, Yp, Yf, Tini, THorizon, hankel_subB_size, hankel_idx)
        Up_cur_sim, Uf_cur_sim, Yp_cur_sim, Yf_cur_sim = hankel_subBlocks(Up_sim, Uf_sim, Yp_sim, Yf_sim, Tini, THorizon, hankel_subB_size, hankel_idx)

        u_init = Up_cur_sim[:,0:1]
        y_init = Yp_cur_sim[:,0:1]
        vref_cur = vref_sim[hankel_idx:hankel_idx+THorizon,0:1]
        rspd_now = vref_cur[0,0]
        # Debug purpose
        # print(f"u_init:{u_init}, \y_init:{y_init}")
        # print(f"vref_sim shap: {vref_sim.shape}, hankel_idx:{hankel_idx}")        
        # print(f"rspd_now: {rspd_now}, vref_cur:{vref_cur}")
        # print(
        #     f"Up_cur shape{Up_cur.shape} value: {Up_cur}, "
        #       f"Uf_cur shape{Uf_cur.shape} value: {Uf_cur}, "
        #       f"Yp_cur shape{Yp_cur.shape} value: {Yp_cur}, "
        #       f"Yf_cur shape{Yf_cur.shape} value: {Yf_cur}, "
        #       f"hankel_idx {hankel_idx}")
        u_opt, g_opt, t_deepc, exist_feasible_sol, cost = dpc.acados_solver_step(uini=u_init, yini=y_init, yref=vref_cur,           # For real-time Acados solver-Generate a time series of "optimal" control input given v_ref and previous u and v_dyno(for implicit state estimation)
                                                        Up_cur=Up_cur, Uf_cur=Uf_cur, Yp_cur=Yp_cur, Yf_cur=Yf_cur, Q_val=Q, R_val=R,
                                                        lambda_g_val=lambda_g, lambda_y_val=lambda_y, lambda_u_val=lambda_u, g_prev = g_prev)         

        if np.all(u_opt == 0):
            g_prev = None
        else:
            g_prev = g_opt
        u = u_opt[0]       
        actual_elapsed_time = round((time.perf_counter() - loop_start)*1000,3)
        # print(f"u: {u}, g_opt:{g_opt}, exist_feasible_sol:{exist_feasible_sol}, u_opt:{u_opt}")
        actual_control_frequency = 1/(actual_elapsed_time / 1000)
        # ──  Debug printout ─────────────────────────────────────────────
        print(
            f"[{elapsed_time:.3f}] "
            f"v_ref={rspd_now:6.2f}kph, "
            f"u={u:6.2f}%, "
            f"t_deepc={t_deepc:6.3f} ms, "
            f"actual_elapsed_time_per_loop={actual_elapsed_time:6.3f} ms, "
            f"actual_control_frequency={actual_control_frequency:6.3f} Hz, "
            f"DeePC exist_feasible_sol={exist_feasible_sol}, "
            f"hankel_idx={hankel_idx}",
            # f"g_opt= {g_opt}",
            f"u_opt = {u_opt}",
            f"DeePC_Cost = {cost}"
        )

        # print(f"loop_count: {loop_count}")

        # ── 10) Save state for next iteration ──────────────────────────────
        prev_ref_speed = rspd_now
        u_prev         = u
        # record Tinit length of historical data for state estimation
        u_history.append(u)                         

        # ── 11) Schedule next tick at 100 Hz ───────────────────────────────
        next_time += Ts
        
        loop_count += 1
        # Update hankel_idx: because this is not ROTS system, 
        # there's lags, it's not running exactly Ts per loop, 
        # make hankel index correspond to the first 4 digits of elapsed_time

        # s = f"{elapsed_time:.3f}"               # "20.799"
        # digits = s.replace(".", "")             # "20799"
        # hankel_idx = int(digits[:4])            # 2079
        hankel_idx += 1

        # 12) Append this tick’s values to log_data
        log_data.append({
            "time":      elapsed_time,
            "v_ref":     rspd_now,
            # "v_meas":    v_meas,
            "u":         u,
            # "error":     e_k,
            "t_deepc(ms)":   t_deepc,
            "exist_feasible_sol":exist_feasible_sol,
            "actual_elapsed_time":actual_elapsed_time,
            "hankel_idx": hankel_idx,
            "vref":ref_horizon_speed,
            "u_init": u_init,
            "y_init": y_init,
            "Up_cur": Up_cur,
            "Uf_cur": Uf_cur,
            "Yp_cur": Yp_cur,
            "Yf_cur": Yf_cur,
            "g_opt" : g_opt,
            "u_opt" : u_opt,
            "DeePC_Cost" : cost,
        })
        if elapsed_time >50:
            break

finally:
    # ── Save log_data to Excel ───────────────────────────────────
    df = pd.DataFrame(log_data)
    df['cycle_name']   = cycle_key
    datetime = datetime.now()
    df['run_datetime'] = datetime.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_str = datetime.strftime("%H%M_%m%d")
    excel_filename = f"{timestamp_str}_DR_log_{veh_modelName}_{cycle_key}_{algorithm_name}_Ts{Ts}_Q{Q_val}_R{R_val}_Tini{Tini}_gDim{g_dim}_λg{lambda_g_val}_λu{lambda_u_val}__λy{lambda_y_val}.xlsx"
    log_dir = os.path.join(base_folder, "deepc", "toy_example_Log_DriveRobot")
    os.makedirs(log_dir, exist_ok=True)     
    excel_path = os.path.join(log_dir, excel_filename)

    df.to_excel(excel_path, index=False)
    print(f"[Main] Saved log to '{excel_path}'")

















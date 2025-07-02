#!/usr/bin/env python3
"""
BaseLine_MultiDC_VehCAN.py

Discrete PID + Feedforward controller converted from Simulink to Python.
Runs at 100 Hz (T_s = 0.01 s) to track a reference speed profile, reading v_meas from Dyno_CAN, BMS_socMIN from Veh_CAN
and writing a duty‐cycle (–15% to +100%) to a PCA9685 PWM board.
Has the ability to run multiple drive cycles consecutively and save all the data. 

PID structure (per Simulink):
  e[k]       = ref_speed(t)    - v_meas(t)
  e_fut[k]   = ref_speed(t+T_s) - v_meas(t)

  P_k        = Kp * e[k]

  D_k        = Kd * (e[k] - e[k-1]) / T_s
  D_f[k]     = D_f[k-1] + [T_s / (T_s + Tf)] * (D_k - D_f[k-1])

  if v_meas > 0:
    I_k      = I_{k-1} + Ki * T_s * e[k]
    I_out    = I_k
  else:
    I_k      = I_{k-1}      (hold integrator)
    I_out    = 0

  FF_k       = Kff * e_fut[k]

  u[k]       = P_k + I_out + D_f[k] + FF_k
  u[k]       clipped to [-15, +100]

  If u[k] ≥ 0 → PWM accel channel (0) = u, brake (4) = 0
  If u[k] <  0 → PWM accel channel (0) = 0, brake (4) = |u|

Run:
  python3 BaseLine_MultiDC_VehCAN.py

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

import Jetson.GPIO as GPIO
GPIO.cleanup()

import board
import busio
from adafruit_pca9685 import PCA9685

import cantools
import can

# ────────────────────────── GLOBALS FOR CAN THREAD ─────────────────────────────
latest_speed = None           # Measured speed (kph) from CAN
latest_force = None           # Measured force (N) from CAN (unused here)
dyno_can_running  = True      # Flag to stop the CAN thread on shutdown
veh_can_running  = True 
BMS_socMin = None             # Measured current vehicle SOC from Vehicle CAN
# dyno_can_running  = False   # For temperal debugging
# veh_can_running  = False 


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

# ───────────────────────────── GAIN‐SCHEDULING ─────────────────────────────────
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
    
    # ─── Loading System Parameters ────────────────────────────────────────────────
    T_f = 5000.0                # Derivative filter coefficient. Formula uses: D_f[k] = D_f[k-1] + (T_s / (T_s + T_f)) * (D_k - D_f[k-1])
    FeedFwdTime = 0.65          # feedforward reference speed time
    max_delta = 50.0            # maximum % change per 0.01 s tick - regulate the rate of change of pwm output u
    SOC_CycleStarting = 0.0     # Managing Vehicle SOC
    SOC_Stop = 97.9             # Stop the test at SOC 2.2% so the vehicle doesn't go completely drained that it cannot restart/charge

    Ts = 0.1                   # 100 Hz main control loop updating rate - Sampling time 

    for idx, cycle_key in enumerate(cycle_keys):
        # Stop the test if the vehicle SOC is too low to prevent draining the vehicle
        if BMS_socMin is not None and BMS_socMin <= SOC_Stop:
            break
        else:
            SOC_CycleStarting = BMS_socMin

        #Loading current cycle data
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
        ref_speed_mph = ref_array[:, 1].astype(float).flatten()         # All the drive cycle .mat data file speed are in MPH
        ref_speed = ref_speed_mph * 1.60934                             # now ref speed in kph
        print(f"[Main] Reference loaded: shape = {ref_array.shape}")

        # Reset PID state
        prev_error     = 0.0
        I_state        = 0.0
        D_f_state      = 0.0
        u_prev         = 0.0
        prev_ref_speed = None                                           # Track previous reference speed for derivative on ref (if needed)
        log_data       = []                                             # Prepare logging
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

                # ── 1) Interpolate reference speed at t and t+Ts ───────────────────
                if elapsed_time <= ref_time[0]:
                    rspd_now = ref_speed[0]
                elif elapsed_time >= ref_time[-1]:
                    rspd_now = 0.0
                    break
                else:
                    rspd_now = float(np.interp(elapsed_time, ref_time, ref_speed))

                t_future = elapsed_time + FeedFwdTime
                if t_future <= ref_time[0]:
                    rspd_fut = ref_speed[0]
                elif t_future >= ref_time[-1]:
                    rspd_fut = 0.0
                else:
                    rspd_fut = float(np.interp(t_future, ref_time, ref_speed))

                # ── 2) Compute current error e[k] and future error e_fut[k] ────────
                v_meas = latest_speed if latest_speed is not None else 0.0      # v_meas is in kph
                F_meas = latest_force if latest_force is not None else 0.0
                e_k    = rspd_now - v_meas
                e_fut  = rspd_fut - v_meas

                # in the matlab baseline code, need to convert both spderror and futureSpderror from kph to mph to choose PID gains
                e_k = e_k * 0.621371
                e_fut = e_fut * 0.621371
                Kp, Ki, Kd, Kff = get_gains_for_speed(rspd_now)

                # ── 3) P‐term ───────────────────────────────────────────────────────
                P_term = Kp * e_k

                # ── 4) D‐term (discrete derivative + first‐order low‐pass filter) ──
                D_k = Kd * (e_k - prev_error) / Ts

                #    Low‐pass filter: D_f[k] = D_f[k-1] + (Ts/(Ts + T_f)) * (D_k - D_f[k-1])
                alpha = Ts / (Ts + T_f)
                D_f_state = D_f_state + alpha * (D_k - D_f_state)
                D_term = D_f_state

                # ── 5) I‐term (discrete integrator with freeze if v_meas <= 0) ─────
                if v_meas > 0.1 and rspd_now > 0.1:
                    I_state = I_state + Ki * Ts * e_k
                    I_out   = I_state
                else:
                    # Freeze integrator, but force I_out = 0
                    I_out = 0.0
                    I_state = 0.0

                # ── 6) FF‐term ─────────────────────────────────────────────────────
                FF_term = Kff * e_fut

                # ── 7) Total output u[k], clipped to [-15, +100] ────────────────
                u_unclamped = P_term + I_out + D_term + FF_term
                u = float(np.clip(u_unclamped, -15.0, +100.0))

                # ── 8) Add a vehicle speed limiter safety feature ──────────────────────────────
                lower_bound = u_prev - max_delta
                upper_bound = u_prev + max_delta                        # Allow u to move by at most ±max_delta from the previous cycle:
                u = float(np.clip(u, lower_bound, upper_bound))         # Rate‐of‐change limiting on u
                if latest_speed is not None and latest_speed >= 140.0:  # enforce a vehicle speed limiter safety
                    u = 0.0
                    break

                # ── 8) Send PWM to PCA9685: accel (ch=0) if u>=0, else brake (ch=4) ──
                if u >= 0.0:
                    set_duty(4, 0.0)                                    # ensure brake channel is zero
                    set_duty(0, u)                                      # channel 0 = accelerator
                else:
                    set_duty(0, 0.0)                                    # ensure accel channel is zero
                    set_duty(4, -u)                                     # channel 4 = brake

                # ── 9) Debug printout ─────────────────────────────────────────────
                print(
                    f"[{elapsed_time:.3f}] "
                    f"v_ref={rspd_now:6.2f} kph, "
                    f"v_meas={v_meas:6.2f} kph, e={e_k:+6.2f}, e_fut={e_fut:+6.2f}, "
                    f"P={P_term:+6.2f}, I={I_out:+6.2f}, D={D_term:+6.2f}, FF={FF_term:+6.2f}, "
                    f"u={u:+6.2f}%,"
                    f"F_dyno={F_meas:6.2f} N,"
                    f"BMS_socMin={BMS_socMin:6.2f} %,"
                    f"SOC_CycleStarting={SOC_CycleStarting} %"
                )

                # ── 10) Save state for next iteration ──────────────────────────────
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
                    "u":         u,
                    "BMS_socMin":BMS_socMin,
                    "SOC_CycleStarting":SOC_CycleStarting,
                    "error":     e_k,
                    "error_fut": e_fut,
                    "P_term":    P_term,
                    "I_term":    I_out,
                    "D_term":    D_term,
                    "FF_term":   FF_term,
                    "Kp":        Kp,
                    "Ki":        Ki,
                    "Kd":        Kd,
                    "Kff":       Kff,
                })
                if BMS_socMin <= SOC_Stop:
                    break

        except KeyboardInterrupt:
            print("\n[Main] KeyboardInterrupt detected. Exiting…")

        finally:
            for ch in range(16):
                pca.channels[ch].duty_cycle = 0                                 # Zero out all PWM channels before exiting
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


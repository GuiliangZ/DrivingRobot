#!/usr/bin/env python3
import os
import threading, time
from pathlib import Path

import numpy as np
import scipy.io as sio

import Jetson.GPIO as GPIO
GPIO.cleanup()

import board
import busio
from adafruit_pca9685 import PCA9685

import cantools
import can

# ─── STEP 0: Load all .mat files (as before) ─────────────────────────────────
def load_drivecycle_mat_files(base_folder: str):
    """
    Scans the 'drivecycle' subfolder under base_folder, finds all .mat files,
    and loads each one using scipy.io.loadmat.
    Returns a dict mapping each .mat filename (without extension) to its variable dict.
    """
    drivecycle_dir = Path(base_folder) / "drivecycle"
    if not drivecycle_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {drivecycle_dir}")

    mat_data = {}
    for mat_file in drivecycle_dir.glob("*.mat"):
        try:
            data_dict = sio.loadmat(mat_file)
        except NotImplementedError:
            print(f"Warning: {mat_file.name} might be v7.3. Skipping scipy.loadmat.")
            continue

        key = mat_file.stem
        mat_data[key] = data_dict
        names = [k for k in data_dict.keys() if not k.startswith("__")]
        print(f"Loaded '{mat_file.name}' → variables = {names}")
    return mat_data

# ─── GAIN SCHEDULING FUNCTION ─────────────────────────────────────────────────
def get_pid_gains(setpoint_speed: float):
    """
    Return (Kp, Ki, Kd) based on the current setpoint_speed (kph).
    This is a simple piecewise schedule. Feel free to refine or replace
    with interpolation, lookup‐table, or polynomial fit of your choice.
    """
    if setpoint_speed < 20.0:
        # At low speeds (<20 kph), use aggressive gains
        return (0.8, 0.2, 0.05)
    elif setpoint_speed < 50.0:
        # Mid‐range speeds (20–50 kph), moderate gains
        return (0.5, 0.1, 0.04)
    else:
        # High speeds (≥50 kph), smaller gains for smoother control
        return (0.3, 0.05, 0.02)

# ─── PID CONTROLLER CLASS ─────────────────────────────────────────────────────
class PIDController:
    def __init__(self, Kp: float, Ki: float, Kd: float, output_limits=(None, None)):
        """
        A basic PID controller with optional output limits.
        Kp, Ki, Kd: Initial gains.
        output_limits: (min_output, max_output), e.g. (-100.0, +100.0) for percent duty.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

        self.min_output, self.max_output = output_limits

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def update_gains(self, Kp: float, Ki: float, Kd: float):
        """
        Dynamically replace the PID gains. We keep integral and last_error
        so the controller doesn’t jump—only the proportional/integral/derivative
        scaling changes. If you want to zero the integral when the gains 
        change drastically, you could call self.integral = 0 here, but we'll
        leave it continuous in this example.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def __call__(self, setpoint: float, measurement: float, current_time: float) -> float:
        """
        Compute PID output given a setpoint, current measurement, and timestamp.
        Returns a control value (un‐clamped); clamping is done below.
        """
        error = setpoint - measurement

        # Initialize last_time if this is the first call
        if self.last_time is None:
            dt = 0.0
        else:
            dt = current_time - self.last_time

        # Proportional term
        P = self.Kp * error

        # Integral term
        if dt > 0.0:
            self.integral += error * dt
        I = self.Ki * self.integral

        # Derivative term
        if dt > 0.0:
            derivative = (error - self.last_error) / dt
        else:
            derivative = 0.0
        D = self.Kd * derivative

        # Raw PID output
        output = P + I + D

        # Clamp to output_limits (anti‐windup)
        if (self.max_output is not None and output > self.max_output) or \
           (self.min_output is not None and output < self.min_output):
            output = np.clip(output, self.min_output, self.max_output)

        # Save for next iteration
        self.last_error = error
        self.last_time = current_time

        return output

# ─── MAIN SCRIPT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1) Locate base_folder & load all drivecycle .mat files
    base_folder = os.path.dirname(os.path.abspath(__file__))
    all_cycles = load_drivecycle_mat_files(base_folder)

    # 2) Pick one drivecycle as the reference (e.g. 'CYC_WLTP' or similar).
    #    We assume each .mat has exactly one N×2 array: column[0]=time, column[1]=speed.
    example_key = next(iter(all_cycles))
    example_data = all_cycles[example_key]
    print(f"\nExample contents of '{example_key}':")

    mat_vars = [k for k in example_data.keys() if not k.startswith("__")]
    if len(mat_vars) != 1:
        raise RuntimeError(f"Expected exactly one user‐variable in '{example_key}', but found {mat_vars}")
    varname = mat_vars[0]
    ref_array = example_data[varname]

    if ref_array.ndim != 2 or ref_array.shape[1] < 2:
        raise RuntimeError(f"Expected '{varname}' in '{example_key}' to be an N×2 array. Got shape = {ref_array.shape}")

    # Split into time vs. speed
    ref_time = ref_array[:, 0].astype(float).flatten()   # e.g. seconds
    ref_speed = ref_array[:, 1].astype(float).flatten()  # e.g. kph

    print(f"  → Found reference array '{varname}' with shape {ref_array.shape}")
    print(f"     First 5 time stamps: {ref_time[:5]}")
    print(f"     First 5 speeds:      {ref_speed[:5]}")

    # 3) Instantiate the PID controller with some initial gains (they'll be overwritten
    #    by gain‐scheduling on each iteration anyway). Set output_limits = (-100, +100).
    pid = PIDController(Kp=0.5, Ki=0.1, Kd=0.05, output_limits=(-100.0, +100.0))
    pid.reset()

    # 4) PCA9685 PWM setup (same as before)
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c, address=0x40)
    pca.frequency = 1000  # up to ~1.6kHz

    def set_duty(channel: int, percent: float):
        """
        channel: 0–15
        percent: 0.0 to 100.0 (we clamp before calling this, so just assume valid).
        """
        pct = float(percent)
        if pct < 0.0 or pct > 100.0:
            raise ValueError("set_duty: percent must be between 0.0 and 100.0")
        duty_16bit = int(pct * 65535 / 100.0)
        pca.channels[channel].duty_cycle = duty_16bit

    # 5) Set up CAN + DBC (unchanged)
    DBC_PATH = '/home/guiliang/Desktop/DriveRobot/KAVL_V3.dbc'
    try:
        db = cantools.database.load_file(DBC_PATH)
    except FileNotFoundError:
        print(f"ERROR: Cannot find DBC at '{DBC_PATH}'.")
        raise

    try:
        speed_force_msg = db.get_message_by_name('Speed_and_Force')
    except KeyError:
        print("ERROR: 'Speed_and_Force' not found in the DBC.")
        raise

    CAN_INTERFACE = 'can0'
    try:
        bus = can.interface.Bus(channel=CAN_INTERFACE, bustype='socketcan')
    except OSError:
        print(f"ERROR: Cannot open CAN interface '{CAN_INTERFACE}'.")
        print("Make sure you have run something like:\n"
              "    sudo ip link set can0 up type can bitrate 500000 dbitrate 1000000 fd on")
        raise

    print(f"Listening on {CAN_INTERFACE} for ID=0x{speed_force_msg.frame_id:03X} ('Speed_and_Force')…")
    print("Press Ctrl+C to exit.\n")

    # 6) Prepare history & timing state
    force_history = []
    speed_history = []
    latest_force = None
    latest_speed = None

    previous_time = None  # for computing dt

    try:
        while True:
            # ─── wait for next CAN message (timeout=1s) ─────────────────────────
            msg = bus.recv(timeout=1.0)
            if msg is None:
                continue
            if msg.arbitration_id != speed_force_msg.frame_id:
                continue

            try:
                decoded = db.decode_message(msg.arbitration_id, msg.data)
            except KeyError:
                continue

            force_value = decoded.get('Force_N')
            speed_value = decoded.get('Speed_kph')
            if force_value is None or speed_value is None:
                continue

            # Update “live” variables
            latest_force = float(force_value)
            latest_speed = float(speed_value)
            force_history.append(latest_force)
            speed_history.append(latest_speed)

            # Current timestamp (seconds, float)
            current_time = time.time()

            # ─── Interpolate desired setpoint from reference curve ──────────────
            if current_time <= ref_time[0]:
                desired_speed = ref_speed[0]
            elif current_time >= ref_time[-1]:
                desired_speed = ref_speed[-1]
            else:
                desired_speed = np.interp(current_time, ref_time, ref_speed)

            # ─── GAIN SCHEDULING: pick new PID gains based on desired_speed ─────
            Kp_new, Ki_new, Kd_new = get_pid_gains(desired_speed)
            pid.update_gains(Kp_new, Ki_new, Kd_new)

            # ─── Compute dt for PID ──────────────────────────────────────────────
            if previous_time is None:
                dt = 0.0
            else:
                dt = current_time - previous_time

            # ─── Compute PID output with updated gains ──────────────────────────
            pwm_output = pid(desired_speed, latest_speed, current_time)

            # ─── Send PWM to PCA9685 ───────────────────────────────────────────
            # If pwm_output ≥ 0: accelerator on channel=0; else brake on channel=4.
            if pwm_output >= 0:
                set_duty(0, pwm_output)      # channel 0 = accelerator
                set_duty(4, 0.0)            # ensure brake channel is zero
            else:
                set_duty(0, 0.0)            # ensure accel channel is zero
                set_duty(4, -pwm_output)    # channel 4 = brake

            # ─── Print status ───────────────────────────────────────────────────
            print(
                f"[{current_time:.3f}] "
                f"Setpoint = {desired_speed:.2f} kph,  "
                f"Measured = {latest_speed:.2f} kph,  "
                f"Error = {desired_speed - latest_speed:+.2f} kph,  "
                f"Gains = (Kp={pid.Kp:.3f},Ki={pid.Ki:.3f},Kd={pid.Kd:.3f}),  "
                f"PID→ {pwm_output:+.1f}%"
            )

            previous_time = current_time

    except KeyboardInterrupt:
        print("\nInterrupted by user. Shutting down…")
    finally:
        bus.shutdown()
        # Zero out all PWM channels and deinit
        for ch in range(16):
            pca.channels[ch].duty_cycle = 0
        pca.deinit()

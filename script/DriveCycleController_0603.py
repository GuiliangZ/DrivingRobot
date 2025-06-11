#!/usr/bin/env python3
"""
Fixed‐frequency, gain‐scheduled PID control with integrator windup protection.
Separates CAN reception (in a background thread) from a 100 Hz control loop.
Loads a two‐column (time, speed) reference from a .mat file, interpolates the
desired setpoint by wall‐clock time, and drives a PCA9685 PWM board to follow it.
"""

import os
import time
import threading
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

# ─── GLOBALS ─────────────────────────────────────────────────────────────────
latest_speed = None   # Most recent measured speed (kph) from CAN
latest_force = None   # Most recent measured force (N) from CAN
can_running = True    # Flag to let CAN thread know when to stop

# ─── STEP 0: Load .mat files from drivecycle folder ──────────────────────────
def load_drivecycle_mat_files(base_folder: str):
    """
    Scans "drivecycle/" under base_folder, loads every .mat via scipy.io.loadmat,
    and returns a dict: { filename_without_ext: { varname: array, ... }, ... }.
    """
    drivecycle_dir = Path(base_folder) / "drivecycle"
    if not drivecycle_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {drivecycle_dir}")

    mat_data = {}
    for mat_file in drivecycle_dir.glob("*.mat"):
        try:
            data_dict = sio.loadmat(mat_file)
        except NotImplementedError:
            print(f"Warning: {mat_file.name} might be MATLAB v7.3. Skipping scipy.loadmat.")
            continue

        key = mat_file.stem
        mat_data[key] = data_dict
        user_vars = [k for k in data_dict.keys() if not k.startswith("__")]
        print(f"Loaded '{mat_file.name}' → variables = {user_vars}")
    return mat_data

# ─── GAIN SCHEDULING FUNCTION ─────────────────────────────────────────────────
# Example gain‐table (speed_kph, Kp, Ki, Kd)
gain_table = np.array([
    [   0.0,  1.0, 0.3, 0.1 ],
    [  30.0,  0.7, 0.2, 0.05],
    [  60.0,  0.4, 0.1, 0.02],
])
def get_pid_gains(setpoint_speed):
    # Extract columns
    speeds = gain_table[:, 0]
    Kp_vals = gain_table[:, 1]
    Ki_vals = gain_table[:, 2]
    Kd_vals = gain_table[:, 3]
    # Interpolate each gain
    Kp_interp = np.interp(setpoint_speed, speeds, Kp_vals)
    Ki_interp = np.interp(setpoint_speed, speeds, Ki_vals)
    Kd_interp = np.interp(setpoint_speed, speeds, Kd_vals)
    return (Kp_interp, Ki_interp, Kd_interp)


# ─── PID CONTROLLER WITH ANTI‐WINDUP ───────────────────────────────────────────
class PIDController:
    def __init__(self, Kp: float, Ki: float, Kd: float, output_limits=(None, None)):
        """
        A PID controller with integrator windup protection.
        Kp, Ki, Kd: Initial gains
        output_limits: (min_output, max_output), e.g. (-100.0, +100.0) for percent duty
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

        self.min_output, self.max_output = output_limits

    def reset(self):
        """Zero out integral and derivative history."""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def update_gains(self, Kp: float, Ki: float, Kd: float):
        """
        Change PID gains on‐the‐fly. To reduce abrupt integral jumps,
        you could scale or zero the integrator here if desired.
        """
        # Optionally, you could scale integral to new Ki:
        # if self.Ki != 0 and Ki != 0:
        #     self.integral = self.integral * (self.Ki / Ki)
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def __call__(self, setpoint: float, measurement: float, current_time: float) -> float:
        """
        Compute PID output with anti‐windup. Returns a value (unclamped in code,
        but effectively clamped inside) in the same units as output_limits.
        """
        error = setpoint - measurement

        # Δt since last call
        if self.last_time is None:
            dt = 0.0
        else:
            dt = current_time - self.last_time

        # Proportional
        P = self.Kp * error

        # Derivative
        if dt > 0.0:
            derivative = (error - self.last_error) / dt
        else:
            derivative = 0.0
        D = self.Kd * derivative

        # Integral with anti‐windup: only integrate if it won't drive the output further
        if dt > 0.0:
            # Compute potential new integral
            potential_integral = self.integral + error * dt
            I_candidate = self.Ki * potential_integral
            output_no_I = P + D
            output_with_I = output_no_I + I_candidate

            # If adding I_candidate would exceed limits, skip integrating this step
            if (self.max_output is not None and output_with_I > self.max_output) or \
               (self.min_output is not None and output_with_I < self.min_output):
                # Do NOT update self.integral; keep old integral
                I = self.Ki * self.integral
            else:
                # Accept the new integral
                self.integral = potential_integral
                I = I_candidate
        else:
            # First call or dt=0 → no integration
            I = self.Ki * self.integral

        # Raw output
        raw_output = P + I + D

        # Clamp to output_limits
        if self.max_output is not None and raw_output > self.max_output:
            output = self.max_output
        elif self.min_output is not None and raw_output < self.min_output:
            output = self.min_output
        else:
            output = raw_output

        # Save for next iteration
        self.last_error = error
        self.last_time = current_time

        return output

# ─── CAN LISTENER THREAD ──────────────────────────────────────────────────────
def can_listener_thread(DBC_PATH: str, can_iface: str):
    """
    Runs in a background thread. Listens for 'Speed_and_Force' messages on can_iface.
    Updates globals latest_speed and latest_force whenever a new CAN frame arrives.
    """
    global latest_speed, latest_force, can_running

    # Load DBC and prepare to decode
    try:
        db = cantools.database.load_file(DBC_PATH)
    except FileNotFoundError:
        print(f"ERROR: Cannot find DBC at '{DBC_PATH}'. Exiting CAN listener.")
        return

    try:
        speed_force_msg = db.get_message_by_name('Speed_and_Force')
    except KeyError:
        print("ERROR: 'Speed_and_Force' not found in the DBC. Exiting CAN listener.")
        return

    try:
        bus = can.interface.Bus(channel=can_iface, bustype='socketcan')
    except OSError:
        print(f"ERROR: Cannot open CAN interface '{can_iface}'. Exiting CAN listener.")
        return

    print(f"[CAN Thread] Listening on {can_iface} for ID=0x{speed_force_msg.frame_id:03X}…")

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
            latest_speed = float(s)
        if f is not None:
            latest_force = float(f)

    bus.shutdown()
    print("[CAN Thread] Stopped.")

# ─── MAIN CONTROL LOOP ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Locate base_folder & load drivecycle MAT files
    base_folder = os.path.dirname(os.path.abspath(__file__))
    all_cycles = load_drivecycle_mat_files(base_folder)

    # 2) Pick one reference cycle (assumes exactly one N×2 array inside)
    example_key = next(iter(all_cycles))
    example_data = all_cycles[example_key]
    print(f"\n[Main] Using reference from '{example_key}'")

    mat_vars = [k for k in example_data.keys() if not k.startswith("__")]
    if len(mat_vars) != 1:
        raise RuntimeError(f"Expected exactly one user‐variable in '{example_key}', but found {mat_vars}")
    varname = mat_vars[0]
    ref_array = example_data[varname]

    if ref_array.ndim != 2 or ref_array.shape[1] < 2:
        raise RuntimeError(f"Expected '{varname}' in '{example_key}' to be N×2. Got shape {ref_array.shape}")

    # Split into (time, speed)
    ref_time = ref_array[:, 0].astype(float).flatten()   # e.g., seconds
    ref_speed = ref_array[:, 1].astype(float).flatten()  # e.g., kph

    print(f"[Main] Reference '{varname}' shape = {ref_array.shape}")
    print(f"[Main] Time samples: {ref_time[:5]} ... {ref_time[-5:]}")
    print(f"[Main] Speed samples: {ref_speed[:5]} ... {ref_speed[-5:]}")

    # 3) Instantiate PID controller with dummy initial gains;
    #    they will be overwritten immediately at runtime by gain scheduling.
    pid = PIDController(Kp=0.5, Ki=0.1, Kd=0.05, output_limits=(-100.0, +100.0))
    pid.reset()

    # 4) PCA9685 PWM setup
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c, address=0x40)
    pca.frequency = 1000  # up to ∼1.6 kHz

    def set_duty(channel: int, percent: float):
        """
        channel: 0–15
        percent: 0.0 to 100.0 (must be within this range).
        """
        pct = float(percent)
        if pct < 0.0 or pct > 100.0:
            raise ValueError("set_duty: percent must be between 0 and 100")
        duty_16bit = int(pct * 65535 / 100.0)
        pca.channels[channel].duty_cycle = duty_16bit

    # 5) Start CAN listener thread
    DBC_PATH = '/home/guiliang/Desktop/DriveRobot/KAVL_V3.dbc'
    CAN_INTERFACE = 'can0'

    can_thread = threading.Thread(
        target=can_listener_thread,
        args=(DBC_PATH, CAN_INTERFACE),
        daemon=True
    )
    can_thread.start()

    # 6) Fixed‐frequency control loop: e.g. 100 Hz → dt_target = 0.01 s
    dt_target = 0.01  # seconds → 100 Hz
    next_time = time.time()

    print("[Main] Entering 100 Hz control loop. Press Ctrl+C to exit.\n")
    try:
        while True:
            now = time.time()
            if now < next_time:
                time.sleep(next_time - now)
            current_time = time.time()

            # Interpolate reference setpoint for this timestamp
            if current_time <= ref_time[0]:
                desired_speed = ref_speed[0]
            elif current_time >= ref_time[-1]:
                desired_speed = ref_speed[-1]
            else:
                desired_speed = np.interp(current_time, ref_time, ref_speed)

            # If we have no measurement yet, skip PID and send zero
            if latest_speed is None:
                pwm_output = 0.0
            else:
                # Gain scheduling based on desired_speed
                Kp_new, Ki_new, Kd_new = get_pid_gains(desired_speed)
                pid.update_gains(Kp_new, Ki_new, Kd_new)

                # Compute PID output
                pwm_output = pid(desired_speed, latest_speed, current_time)

            # Send PWM to PCA9685:
            #  - pwm_output ≥ 0 → accelerator on channel 0, brake off
            #  - pwm_output <  0 → brake on channel 4, accelerator off
            if pwm_output >= 0:
                set_duty(0, pwm_output)
                set_duty(4, 0.0)
            else:
                set_duty(0, 0.0)
                set_duty(4, -pwm_output)

            # Print status for debugging
            err = (desired_speed - latest_speed) if latest_speed is not None else 0.0
            print(
                f"[{current_time:.3f}] "
                f"Setpoint={desired_speed:.2f} kph, "
                f"Measured={latest_speed if latest_speed is not None else 0.0:.2f} kph, "
                f"Error={err:+.2f} kph, "
                f"Gains=(Kp={pid.Kp:.3f}, Ki={pid.Ki:.3f}, Kd={pid.Kd:.3f}), "
                f"PWM={pwm_output:+.1f}%"
            )

            # Schedule next tick
            next_time += dt_target

    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt detected. Exiting...")

    finally:
        # Signal CAN thread to stop
        can_running = False
        can_thread.join(timeout=1.0)

        # Zero out all PWM channels
        for ch in range(16):
            pca.channels[ch].duty_cycle = 0
        pca.deinit()
        print("[Main] PWM channels zeroed and PCA9685 deinitialized. Goodbye.")

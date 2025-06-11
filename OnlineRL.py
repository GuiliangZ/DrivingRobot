#!/usr/bin/env python3
"""
rl_speed_tracker.py

On‐hardware (Jetson Orin NX) DDPG‐style RL agent to track a reference speed profile at 100 Hz.
This script replaces the PI+FF controller with a small Actor‐Critic network that learns on the fly.

How it works:
1. CAN reception runs in a background thread, updating `latest_speed` each time a Speed_and_Force
   frame arrives.
2. The main loop runs at exactly 100 Hz (T_s = 0.01 s):
     • Interpolate reference speed profile at t and t+T_s, compute a_ref_now and a_ref_fut
     • Compute measured acceleration a_meas from CAN data
     • Form state vector s_t = [v_meas, a_ref_now, a_ref_fut]
     • If not first step, store (s_{t-1}, a_{t-1}, r_{t-1}, s_t) into replay buffer, where
         r_{t-1} = -|v_meas - ref_speed_{t-1}|
     • Sample action a_t = actor(s_t) + exploration_noise
     • Clip a_t ∈ [–100, +100], and send as PWM to PCA9685 (ch 0 for accel if ≥0, ch 4 for brake if <0)
     • After enough samples, run one gradient update on both Critic and Actor using a minibatch.
3. When you press Ctrl+C, everything shuts down (CAN thread stops, PCA9685 zeros out).

Before running:
  sudo ip link set can0 up type can bitrate 500000 dbitrate 1000000 fd on
  pip install numpy scipy torch cantools python-can adafruit-circuitpython-pca9685
  sudo pip install Jetson.GPIO

Author: ChatGPT (adapted to your Simulink‐style blocks + RL)
Date: 2025-06-03
"""

import os
import time
import threading
from pathlib import Path
import random
from collections import deque

import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.optim as optim

import Jetson.GPIO as GPIO
GPIO.cleanup()

import board
import busio
from adafruit_pca9685 import PCA9685

import cantools
import can

# ────────────────────────── GLOBALS FOR CAN THREAD ─────────────────────────────
latest_speed = None   # (kph) updated by CAN thread
latest_force = None   # (N)   updated by CAN thread (unused in RL)
can_running = True    # flag to stop CAN thread gracefully

# ───────────────────────── LOAD REFERENCE PROFILE ───────────────────────────────
def load_drivecycle_mat_files(base_folder: str):
    """
    Scan 'drivecycle/' under base_folder, load each .mat with scipy.io.loadmat.
    Return dict { filename_wo_ext: { varname: array, ... }, ... }.
    Assumes each .mat has exactly one user variable that is an N×2 array:
      col 0 = time [s], col 1 = speed [kph].
    """
    drivecycle_dir = Path(base_folder) / "drivecycle"
    if not drivecycle_dir.is_dir():
        raise FileNotFoundError(f"Cannot find 'drivecycle' folder at {drivecycle_dir}.")

    mat_data = {}
    for mat_file in drivecycle_dir.glob("*.mat"):
        try:
            data_dict = sio.loadmat(mat_file)
        except NotImplementedError:
            print(f"[Main] Warning: '{mat_file.name}' might be MATLAB v7.3. Skipping.")
            continue
        key = mat_file.stem
        mat_data[key] = data_dict
        user_vars = [k for k in data_dict.keys() if not k.startswith("__")]
        print(f"[Main] Loaded '{mat_file.name}' → variables = {user_vars}")
    return mat_data

# ──────────────────────────── DDPG COMPONENTS ───────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    """Simple 2-layer MLP → tanh → scale to [–1, +1]."""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()   # final output in [–1, +1]
        )

    def forward(self, x):
        # x: [batch, state_dim]
        return self.net(x)  # in [–1, +1]

class Critic(nn.Module):
    """Q-network: input = [state, action], output = scalar Q-value."""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, a):
        # x: [batch, state_dim], a: [batch, action_dim]
        xa = torch.cat([x, a], dim=1)
        return self.net(xa)   # [batch, 1]

class ReplayBuffer:
    """Circular replay buffer for DDPG transitions."""
    def __init__(self, max_size=100_000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state):
        # each: state, next_state = np.array([...], dtype=np.float32)
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            np.vstack(states),
            np.vstack(actions),
            np.vstack(rewards),
            np.vstack(next_states)
        )

    def __len__(self):
        return len(self.buffer)

# ────────────────────────── CAN LISTENER THREAD ─────────────────────────────────
def can_listener_thread(dbc_path: str, can_iface: str):
    """
    Runs in a background thread. Listens for 'Speed_and_Force' on can_iface,
    decodes it, and updates globals latest_speed & latest_force.
    """
    global latest_speed, latest_force, can_running

    try:
        db = cantools.database.load_file(dbc_path)
    except FileNotFoundError:
        print(f"[CAN⋅Thread] ERROR: Cannot find DBC at '{dbc_path}'")
        return

    try:
        speed_force_msg = db.get_message_by_name('Speed_and_Force')
    except KeyError:
        print("[CAN⋅Thread] ERROR: 'Speed_and_Force' not in DBC")
        return

    try:
        bus = can.interface.Bus(channel=can_iface, bustype='socketcan')
    except OSError:
        print(f"[CAN⋅Thread] ERROR: Cannot open CAN interface '{can_iface}'")
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
            latest_speed = float(s)
        if f is not None:
            latest_force = float(f)

    bus.shutdown()
    print("[CAN⋅Thread] Exiting…")

# ──────────────────────────────── MAIN SCRIPT ──────────────────────────────────
if __name__ == "__main__":
    # ━━ 1) Load reference from .mat ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    base_folder = os.path.dirname(os.path.abspath(__file__))
    all_cycles = load_drivecycle_mat_files(base_folder)

    # Pick the first key as the reference (N×2: [time, speed])
    cycle_key = next(iter(all_cycles))
    cycle_data = all_cycles[cycle_key]
    print(f"\n[Main] Using reference cycle '{cycle_key}'")

    mat_vars = [k for k in cycle_data.keys() if not k.startswith("__")]
    if len(mat_vars) != 1:
        raise RuntimeError(f"Expected exactly one variable in '{cycle_key}', found {mat_vars}")
    varname = mat_vars[0]
    ref_array = cycle_data[varname]

    if ref_array.ndim != 2 or ref_array.shape[1] < 2:
        raise RuntimeError(f"'{varname}' is not N×2. Got shape {ref_array.shape}")

    # Extract time and speed vectors
    ref_time = ref_array[:,0].astype(float).flatten()   # [s]
    ref_speed = ref_array[:,1].astype(float).flatten()  # [kph]

    print(f"[Main] Reference loaded: shape = {ref_array.shape}")
    # Print first/last few for sanity
    print(f"[Main] ref_time[:5] = {ref_time[:5]}")
    print(f"[Main] ref_speed[:5] = {ref_speed[:5]}")
    print(f"[Main] ref_time[-5:] = {ref_time[-5:]}")
    print(f"[Main] ref_speed[-5:] = {ref_speed[-5:]}\n")

    # ━━ 2) Initialize DDPG Actor, Critic, ReplayBuffer ━━━━━━━━━━━━━━━━━━━━━━━━━━
    state_dim  = 3   # [v_meas, a_ref_now, a_ref_fut]
    action_dim = 1   # single scalar duty‐cycle in [–100, +100]

    # Hyperparams (tune as needed)
    hidden_dim   = 64
    lr_actor     = 1e-4
    lr_critic    = 1e-3
    gamma        = 0.99
    tau          = 0.005       # soft update for target networks
    buffer_size  = 100_000
    batch_size   = 64
    exploration_noise_std = 0.2  # initial Gaussian sigma (scaled to action range)
    exploration_noise_decay = 0.9999

    # Create networks & targets
    actor       = Actor(state_dim, action_dim, hidden_dim).to(device)
    actor_target= Actor(state_dim, action_dim, hidden_dim).to(device)
    critic      = Critic(state_dim, action_dim, hidden_dim).to(device)
    critic_target= Critic(state_dim, action_dim, hidden_dim).to(device)

    # Copy weights from main nets to targets
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    # Optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
    critic_optimizer= optim.Adam(critic.parameters(), lr=lr_critic)

    # Replay buffer
    replay_buffer = ReplayBuffer(max_size=buffer_size)

    # Function to softly update target networks
    def soft_update(net, net_target, tau):
        for param, param_tgt in zip(net.parameters(), net_target.parameters()):
            param_tgt.data.copy_(tau * param.data + (1.0 - tau) * param_tgt.data)

    # Training step (one gradient update for both Critic & Actor)
    def ddpg_update():
        if len(replay_buffer) < batch_size:
            return

        # Sample batch
        states_np, actions_np, rewards_np, next_states_np = replay_buffer.sample(batch_size)
        states      = torch.tensor(states_np, dtype=torch.float32, device=device)
        actions     = torch.tensor(actions_np, dtype=torch.float32, device=device)
        rewards     = torch.tensor(rewards_np, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states_np, dtype=torch.float32, device=device)

        # Critic update:
        # Compute Q(s,a)
        Q_vals = critic(states, actions)                           # [batch,1]
        with torch.no_grad():
            # next_actions = actor_target(next_states)
            next_actions = actor_target(next_states)
            # scale from [–1,1]→[–100,100]
            next_actions = next_actions * 100.0
            Q_next = critic_target(next_states, next_actions).squeeze(1)   # [batch]
            target_Q = rewards.squeeze(1) + gamma * Q_next
        # Current critic value
        Q_current = Q_vals.squeeze(1)
        critic_loss = nn.MSELoss()(Q_current, target_Q)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Actor update (maximize Q under current Critic → minimize –Q)
        # Freeze Critic parameters for actor update
        for p in critic.parameters():
            p.requires_grad = False

        # re‐compute actor output on 'states'
        actor_out = actor(states)          # in [–1,1]
        actor_out_scaled = actor_out * 100.0
        actor_loss = -critic(states, actor_out_scaled).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # Unfreeze Critic
        for p in critic.parameters():
            p.requires_grad = True

        # Soft‐update targets
        soft_update(actor, actor_target, tau)
        soft_update(critic, critic_target, tau)

    # ━━ 3) PCA9685 SETUP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    i2c = busio.I2C(board.SCL, board.SDA)
    pca = PCA9685(i2c, address=0x40)
    pca.frequency = 1000  # 1 kHz PWM

    def set_duty(channel: int, percent: float):
        """
        Send a duty‐cycle % to PCA9685 channel.
        percent ∈ [0,100]. Converts to 16-bit register.
        """
        pct = float(percent)
        if pct < 0.0 or pct > 100.0:
            raise ValueError("set_duty: percent must be in [0,100]")
        duty_val = int(pct * 65535 / 100.0)
        pca.channels[channel].duty_cycle = duty_val

    # ━━ 4) START CAN LISTENER THREAD ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    DBC_PATH = '/home/guiliang/Desktop/DriveRobot/KAVL_V3.dbc'
    CAN_INTERFACE = 'can0'

    can_thread = threading.Thread(
        target=can_listener_thread,
        args=(DBC_PATH, CAN_INTERFACE),
        daemon=True
    )
    can_thread.start()

    # ━━ 5) MAIN 100 Hz LOOP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Ts = 0.01  # 100 Hz
    next_time = time.time()

    prev_ref_speed = None
    prev_state     = None
    prev_action    = None

    # Exploration noise parameters (Gaussian)
    noise_std = exploration_noise_std  # will decay slowly

    print("[Main] Entering 100 Hz RL control loop. Press Ctrl+C to stop.\n")
    try:
        while True:
            now = time.time()
            if now < next_time:
                time.sleep(next_time - now)
            current_time = time.time()

            # 1) Interpolate ref speed at t (rspd_now) and t+Ts (rspd_fut)
            if current_time <= ref_time[0]:
                rspd_now = ref_speed[0]
            elif current_time >= ref_time[-1]:
                rspd_now = ref_speed[-1]
            else:
                rspd_now = float(np.interp(current_time, ref_time, ref_speed))

            t_future = current_time + Ts
            if t_future <= ref_time[0]:
                rspd_fut = ref_speed[0]
            elif t_future >= ref_time[-1]:
                rspd_fut = ref_speed[-1]
            else:
                rspd_fut = float(np.interp(t_future, ref_time, ref_speed))

            # 2) Compute reference accel now & future (difference quotient)
            if prev_ref_speed is None:
                a_ref_now = 0.0
                a_ref_fut = 0.0
            else:
                a_ref_now = (rspd_now - prev_ref_speed) / Ts
                a_ref_fut = (rspd_fut - rspd_now) / Ts

            prev_ref_speed = rspd_now

            # 3) Grab CAN speed (v_meas) and compute measured accel (a_meas)
            v_meas = latest_speed if latest_speed is not None else 0.0
            if prev_state is None:
                # no previous measured speed to difference with
                a_meas = 0.0
            else:
                # But we can store previous measured speed inside prev_state[0]
                prev_v_meas = prev_state[0]  # previous state's v_meas
                a_meas = (v_meas - prev_v_meas) / Ts

            # 4) Build current state vector s_t = [v_meas, a_ref_now, a_ref_fut]
            s_cur = np.array([v_meas, a_ref_now, a_ref_fut], dtype=np.float32)

            # 5) If we have (prev_state, prev_action) from last tick, build reward & store transition
            if prev_state is not None and prev_action is not None:
                # Reward: negative absolute speed‐error at this tick
                # (we compare current v_meas to the previous reference speed prev_ref_speed)
                # Actually, prev_ref_speed has been updated → we saved it at the end of last loop, so:
                #    reward = -| v_meas - prev_ref_speed |
                # But prev_ref_speed was set to CURRENT iteration's ref_speed. We want the ref_speed used in prev iteration:
                # We can simply store prev_ref_speed separately:
                pass  # We’ll compute reward below, after we define prev_ref_rd (see code)

            # To fix the small off-by-one, let’s keep a separate variable:
            #   prev_ref_rd = reference speed used at the end of previous loop
            # So on loop iteration N:
            #   - prev_ref_rd holds the ref_speed computed on iteration N−1
            #   - Now v_meas is the measurement at iteration N
            #   → reward_{N−1} = −| v_meas − prev_ref_rd |

            # We’ll store prev_ref_rd at the bottom of each loop. For the very first iteration, prev_ref_rd = None.

            # Compute reward for the previous transition (if it exists):
            if 'prev_ref_rd' in locals() and prev_ref_rd is not None and prev_state is not None and prev_action is not None:
                r_prev = -abs(v_meas - prev_ref_rd)
                # Store (prev_state, prev_action, r_prev, s_cur) in replay buffer
                replay_buffer.push(
                    prev_state.reshape(1, -1),           # shape [1, state_dim]
                    np.array([[prev_action]], dtype=np.float32),  # shape [1, action_dim]
                    np.array([[r_prev]], dtype=np.float32),       # shape [1, 1]
                    s_cur.reshape(1, -1)                # shape [1, state_dim]
                )

            # 6) Select a_t = actor(s_cur) + exploration_noise
            # Convert s_cur to torch:
            s_torch = torch.tensor(s_cur.reshape(1, -1), dtype=torch.float32, device=device)
            with torch.no_grad():
                a_raw = actor(s_torch).cpu().numpy()[0, 0]    # in [–1, +1]
            # add Gaussian noise (scaled to action range)
            noise = np.random.normal(0.0, noise_std)
            a_noisy = np.clip(a_raw + noise, -1.0, 1.0)
            a_t = float(a_noisy * 100.0)   # scale to [–100, +100]

            # very slowly decay exploration noise
            noise_std *= exploration_noise_decay
            noise_std = max(noise_std, 0.01)

            # 7) Send PWM to PCA9685
            if a_t >= 0.0:
                set_duty(0, a_t)   # accelerator channel
                set_duty(4, 0.0)   # brake = 0
            else:
                set_duty(0, 0.0)
                set_duty(4, -a_t)  # brake channel

            # 8) Now we have new prev_state, prev_action, prev_ref_rd for next iteration:
            prev_state   = s_cur.copy()
            prev_action  = a_t
            prev_ref_rd  = rspd_now

            # 9) Do one RL update step (if enough samples)
            ddpg_update()

            # 10) Debug printout
            print(
                f"[{current_time:.3f}] "
                f"v_meas={v_meas:6.2f} kph, a_meas={a_meas:+5.2f} kph/s, "
                f"a_ref={a_ref_now:+5.2f} kph/s, a_ref_fut={a_ref_fut:+5.2f} kph/s, "
                f"action={a_t:+6.2f}%  noise={noise:+.3f}  buffer_len={len(replay_buffer)}"
            )

            # 11) Schedule next 100 Hz tick
            next_time += Ts

    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt → shutting down…")

    finally:
        # Stop CAN thread
        can_running = False
        can_thread.join(timeout=1.0)

        # Zero out PWM and deinit PCA9685
        for ch in range(16):
            pca.channels[ch].duty_cycle = 0
        pca.deinit()

        print("[Main] Cleaned up. Goodbye.")

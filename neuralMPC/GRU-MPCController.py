#!/usr/bin/env python3
"""
GRU-MPC where use trained GRU NN as system model and use MPC to track the pre-defined speed profile precisely.
Runs at 100 Hz (T_s = 0.01 s) to track a reference speed profile, reading v_meas from CAN,
and writing a duty‐cycle (–15% to +100%) to a PCA9685 PWM board.

Required setup:
  pip install numpy scipy cantools python-can adafruit-circuitpython-pca9685
  sudo pip install Jetson.GPIO
    sudo ip link set can0 up type can bitrate 500000 dbitrate 1000000 fd on
    sudo ip link set can1 up type can bitrate 500000 dbitrate 1000000 fd on

"""

# !!!!!!!!!! Always run this command line in the terminal to start the CAN reading: !!!!!!!!!
# sudo ip link set can0 up type can bitrate 500000 dbitrate 1000000 fd on
# sudo ip link set can1 up type can bitrate 500000 dbitrate 1000000 fd on

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
import l4casadi as l4c
from acados_template import AcadosSimSolver, AcadosOcpSolver, AcadosSim, AcadosOcp, AcadosModel
# from acados_settings import AcadosCustomOcp

# ────────────────────────── GLOBALS FOR CAN THREAD ─────────────────────────────
latest_speed = None    # Measured speed (kph) from CAN
latest_force = None    # Measured force (N) from CAN (unused here)
dyno_can_running  = True# Flag to stop the CAN thread on shutdown
# veh_can_running  = True
veh_can_running = False
dyno_can_running = False
BMS_socMin = None     # Measured current vehicle SOC from Vehicle CAN

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
                s = 0.0
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
        msg = bus.recv(timeout=1.0)
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
            if -0.1 < BMS_socMin < 0.1:
                BMS_socMin = 0.0
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

# ────────────────────────── GRU MODEL & MPC SETUP ───────────────────────────────
# ------- GRU Model struture - need to updated with the newly trained model -------
# class DeepGRURegressor0527(nn.Module):
#     def __init__(
#         self,
#         input_size: int = 1,
#         hidden_size: int = 64,
#         num_layers: int = 6,
#         dropout: float = 0.3,
#         bidirectional: bool = False,
#         fc_hidden_dims: list[int] = [512, 256, 64, 32, 32]
#     ):

#         super().__init__()
#         self.bidirectional = bidirectional
#         self.gru = nn.GRU(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout if num_layers > 1 else 0.0,
#             bidirectional=bidirectional,
#         )
#         self.gru_cell = torch.nn.GRUCell(input_size, hidden_size)
        

#         # Determine the input dimension to the first FC layer
#         fc_in_dim = hidden_size * (2 if bidirectional else 1)

#         # Build MLP head
#         layers = []
#         for h_dim in fc_hidden_dims:
#             layers += [
#                 nn.Linear(fc_in_dim, h_dim),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(dropout),
#             ]
#             fc_in_dim = h_dim
#         layers.append(nn.Linear(fc_in_dim, 1))

#         self.fc = nn.Sequential(*layers)

#     def forward(self, x: torch.Tensor, hidden:torch.Tensor = None ) -> tuple[torch.Tensor, torch.Tensor]:

#         if isinstance(x, list) and isinstance(x[0], ca.MX):
#             # x = [ MX(batch, input_size) for _ in range(seq_len) ]
#             batch, input_size = x[0].size()       # MX.size() still works
#             seq_len = len(x)

#             # 1) horizontally cat all timesteps into (batch, seq_len*input_size)
#             flat = ca.horzcat(*[ ca.reshape(xi, batch, input_size) for xi in x ])
#             # 2) reshape back into 3D (batch, seq_len, input_size)
#             x = ca.reshape(flat, batch, seq_len, input_size)

#         # now both branches have x as a "3D Tensor" (either torch.Tensor or MX)
#         B, T, D = x.size()    # MX.size() works too
#         out,_ = self.gru(x)
#         y_seq = self.fc(out)         # (B, seq_len, 1)
#         y_seq = y_seq.squeeze(-1)    # (B, seq_len)
#         return y_seq, hidden
# def sync_cells_from_gru(model):
#     nl = model.num_layers
#     nd = model.num_directions  # 1 or 2
#     for layer in range(nl):
#         # these are the tensors in the trained nn.GRU
#         wi = getattr(model.gru, f'weight_ih_l{layer}')
#         wh = getattr(model.gru, f'weight_hh_l{layer}')
#         bi = getattr(model.gru, f'bias_ih_l{layer}')
#         bh = getattr(model.gru, f'bias_hh_l{layer}')

#         for d in range(nd):
#             idx = layer*nd + d
#             cell = model.gru_cells[idx]
#             cell.weight_ih.data.copy_(wi)
#             cell.weight_hh.data.copy_(wh)
#             cell.bias_ih.data.copy_(bi)
#             cell.bias_hh.data.copy_(bh)

# class DeepGRURegressor0527(nn.Module):
#     def __init__(
#         self,
#         input_size: int = 1,
#         hidden_size: int = 64,
#         num_layers: int = 6,
#         dropout: float = 0.3,
#         bidirectional: bool = False,
#         fc_hidden_dims: list[int] = [512, 256, 64, 32, 32]
#     ):
#         super().__init__()
#         self.input_size   = input_size
#         self.hidden_size  = hidden_size
#         self.num_layers   = num_layers
#         self.bidirectional= bidirectional
#         self.num_directions = 2 if bidirectional else 1

#         # -- standard GRU for torch.Tensor inputs --
#         self.gru = nn.GRU(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=(dropout if num_layers > 1 else 0.0),
#             bidirectional=bidirectional,
#         )

#         # -- one GRUCell per layer & direction for MX unrolling --
#         cell_input_sizes = [input_size] + \
#             [hidden_size * self.num_directions] * (num_layers-1)
#         self.gru_cells = nn.ModuleList()
#         for inp_dim in cell_input_sizes:
#             for _ in range(self.num_directions):
#                 self.gru_cells.append(nn.GRUCell(inp_dim, hidden_size))

#         # -- build the same FC head --
#         fc_in_dim = hidden_size * self.num_directions
#         layers = []
#         for h_dim in fc_hidden_dims:
#             layers += [
#                 nn.Linear(fc_in_dim, h_dim),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(dropout),
#             ]
#             fc_in_dim = h_dim
#         layers.append(nn.Linear(fc_in_dim, 1))
#         self.fc = nn.Sequential(*layers)

#     def forward(self, x, hidden=None):
#         # ── CasADi MX path: x is a list of MX(batch,input_size) ──────────────
#         if isinstance(x, list) and isinstance(x[0], ca.MX):
#             B = x[0].size(1)        # batch
#             T = len(x)              # seq_len

#             # initialize hidden states as zeros if not provided
#             if hidden is None:
#                 # casadi: start from zeros
#                 h_list = [
#                     ca.MX.zeros(B, self.hidden_size)
#                     for _ in range(self.num_layers * self.num_directions)
#                 ]

#             elif isinstance(hidden, list) and isinstance(hidden[0], ca.MX):
#                 # casadi: adapter already gave you a list of MX for each layer
#                 h_list = hidden

#             # elif isinstance(hidden, ca.MX):
#             #     # torch path (unlikely here), or single MX you need to unpack
#             #     L = self.num_layers * self.num_directions
#             #     flat_h = ca.reshape(hidden, L, B * self.hidden_size)  # 2-D reshape
#             #     h_list = [
#             #         ca.reshape(flat_h[i, :], self.hidden_size, B)
#             #         for i in range(L)
#             #     ]
#             elif isinstance(hidden, ca.MX):
#                 # hidden is an MX of shape (L, B, H)
#                 L = self.num_layers * self.num_directions
#                 # slice out each (B×H) layer without any reshape
#                 h_list = [ hidden[i, :, :] for i in range(L) ]
#             else:
#                 raise TypeError(f"Unexpected hidden type {type(hidden)}")

#             outputs = []
#             # unroll each timestep
#             for xi in x:   # xi: MX(B, input_size)
#                 inp = xi
#                 next_h = []
#                 cell_idx = 0

#                 # loop over layers
#                 for layer in range(self.num_layers):
#                     # for each direction
#                     dir_h = []
#                     for d in range(self.num_directions):
#                         h_prev = h_list[cell_idx]
#                         h_new  = self.gru_cells[cell_idx](inp, h_prev)
#                         dir_h.append(h_new)
#                         cell_idx += 1
#                     # if bidirectional, you’d have to run reverse pass too;
#                     # here we assume unidirectional for CasADi
#                     # and just concatenate the forward outputs
#                     inp = dir_h[0] if self.num_directions == 1 else ca.vertcat(*dir_h)
#                     next_h += dir_h

#                 h_list = next_h
#                 # run MLP head on the last layer’s output
#                 y_t = self.fc(inp)           # MX(B,1)
#                 outputs.append(y_t)

#             # build y_seq: stack T outputs into MX of shape (B, T)
#             # first vertcat into a (B*T,1) column
#             flat_y = ca.vertcat(*[
#                 ca.reshape(o, B, 1) for o in outputs
#             ])
#             # then reshape into 2D (B, T)
#             y_seq = ca.reshape(flat_y, B, T)

#             # also pack h_list back into one MX tensor of shape (L, B, H)
#             flat_h = ca.vertcat(*[
#                 ca.reshape(h_i, 1, B*self.hidden_size) for h_i in h_list
#             ])  # size (L, B*H)
#             S = ca.Sparsity.dense([self.num_layers * self.num_directions,
#                        B,
#                        self.hidden_size])
#             hidden_out = ca.reshape(flat_h, S)

#             return y_seq, hidden_out

#         # ── PyTorch path: x is a Tensor(B, T, D) ────────────────────────────
#         else:
#             out, h_out = self.gru(x, hidden)
#             y_seq = self.fc(out).squeeze(-1)  # → Tensor(B, T)
#             return y_seq, h_out

class DeepGRURegressor0527(nn.Module):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 6,
        dropout: float = 0.3,
        bidirectional: bool = False,
        fc_hidden_dims: list[int] = [512, 256, 64, 32, 32]
    ):
        """
        A deep GRU-based regressor with configurable GRU depth, directionality,
        and an MLP head of arbitrary hidden dimensions.
        
        Args:
            input_size:    Number of features in the input sequence.
            hidden_size:   Number of features in the GRU hidden state.
            num_layers:    Number of stacked GRU layers.
            dropout:       Dropout probability between GRU layers and in MLP.
            bidirectional: If True, uses a bidirectional GRU (doubles hidden output).
            fc_hidden_dims: List of hidden sizes for the MLP head.
        """
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Determine the input dimension to the first FC layer
        fc_in_dim = hidden_size * (2 if bidirectional else 1)

        # Build MLP head
        layers = []
        for h_dim in fc_hidden_dims:
            layers += [
                nn.Linear(fc_in_dim, h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            fc_in_dim = h_dim
        layers.append(nn.Linear(fc_in_dim, 1))

        self.fc = nn.Sequential(*layers)

    def forward(self, x, hidden=None):
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch, seq_len, input_size)
        Returns:
            preds: Tensor of shape (batch,) with the regression output.
            hidden: (num_layers, batch, hidden_size) or None
        """
        # GRU returns: output (batch, seq_len, num_directions*hidden_size), h_n
        
        # if torch.is_tensor(x):
        #     B = x.size(0)
        # else:
        #     B = x.size(1)   # casadi.MX batch dim
        # if hidden is not None and hidden.size(1) == B:
        #     hidden = hidden.detach()
        # else:
        #     hidden = None

        # if isinstance(x, list):
        #     # x is probably something like [[…], […], …]
        #     x = torch.tensor(x, dtype=torch.float32,
        #                     device=next(self.parameters()).device)
        # if hidden is not None and isinstance(hidden, list):
        #     hidden = torch.tensor(hidden, dtype=torch.float32,
        #                         device=next(self.parameters()).device)
        breakpoint()
        out,_ = self.gru(x[0],hidden)
        breakpoint()
        y_seq = self.fc(out)         # (B, seq_len, 1)
        y_seq = y_seq.squeeze(-1)    # (B, seq_len)
        return y_seq, hidden


# ------- Acados based real-time MPC ----------------
class MPC:
    def __init__(self, model_gru, MPC_horizon, Q, R, N, Ts, decay_rate_ref):
        self.MPC_horizon = MPC_horizon
        self.model_gru = model_gru
        self.Q = Q
        self.R = R
        self.N = N
        self.Ts = Ts
        self.decay_rate_ref = decay_rate_ref
        # self.model_CasADi = l4c.realtime.RealTimeL4CasADi(self.model_gru, approximation_order=2)
        # self.model_CasADi = l4c.L4CasADi(self.model_gru,  batched=True, device='cpu')
        # self.solver_option = solver_option
        self._make_ocp()
        self._make_solver()

    def _make_ocp(self):
        batch_size  = 1
        seq_len     = 50                               # one time‐step per call
        input_size  = self.model_gru.gru.input_size        # e.g. 1 if you’re feeding in scalar duty-cycle
        hidden_size = self.model_gru.gru.hidden_size       # e.g. 64 or whatever you trained with
        num_layers  = self.model_gru.gru.num_layers  
        # print(f"input_size{input_size}, hidden_size{hidden_size}, num_layers{num_layers}")
        model_CasADi = l4c.realtime.RealTimeL4CasADi(self.model_gru, approximation_order=2)
        
        
        # x_sym      = ca.MX.sym("x",  batch_size, seq_len, input_size)    #   x ≡ your sequence input  shape (batch, seq_len, input_size)
        # hidden_sym = ca.MX.sym("h0", num_layers, batch_size, hidden_size)    #   h0 ≡ your initial hidden state shape (num_layers, batch, hidden_size)
        # casadi_model = self.model_CasADi.model(x_sym, hidden_sym)
        # casadi_model = self.model_CasADi.model()
        

        u_sym   = ca.MX.sym('u',   batch_size, seq_len, input_size)      # your control / input
        h_sym   = ca.MX.sym('h',   num_layers, batch_size, hidden_size) # your hidden state
        # single‐step call: returns (y_seq, h_next)
        _, h_next = model_CasADi.model(u_sym, h_sym)

        # ─── 2) Create an AcadosModel with discrete dynamics ──────────────────────────
        casadi_model = AcadosModel()
        casadi_model.name = 'gru_ocp'
        # states and controls
        casadi_model.x   = h_sym              # hidden state is your “x”
        casadi_model.u   = u_sym              # input is your “u”
        casadi_model.z   = []                 # no algebraic states
        # discrete‐time update
        casadi_model.f_impl_expr = None       # we’re not using implicit integrator here
        casadi_model.f_expl_expr = h_next     # h_{k+1} = h_next(u_k, h_k)



        # Directly assign the wraped model
        ocp = AcadosOcp()
        ocp.model = casadi_model
        ocp.dims.N = self.MPC_horizon
        ocp.model.T = self.Ts
        ocp.model.name = "gru_mpc"

        # quadratic cost using Linear least square
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS" # terminal stage cost function-typically u doesn't matter ony care about the state

        # Deal with residuals - error vector punishing in the least-square cost - specifically for "LINEAR_LS"
        ocp.cost.Vx = np.stack([[1.0], [0.0]]) 
        ocp.cost.Vu = np.stack([[0.0], [1.0]])
        ocp.cost.W = np.diag([self.Q, self.R])
        ocp.cost.W_e = np.array([[self.Q]])
        ocp.cost.yref = np.zeros(2)
        ocp.cost.yref_e = np.zeros(1)

        # Input bound for optimization problem u => [-15,100]
        ocp.constraints.lbu = np.array([-15.0])
        ocp.constraints.ubu = np.array([100.0])
        ocp.constraints.idxbu = np.array([0])       # bound the first input, my system is SISO, so only have one input(pwm)

        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP_RTI' # if self.solver_options is None else solver_options["solver_type"]

        self.ocp = ocp

    def _make_solver(self):
        # generate & compile once
        self.ocp.code_export_directory = 'acados_ocp_deeprnn'
        solver = AcadosOcpSolver(self.ocp, json_file=None, generate=False, build=False)
        solver.generate()
        solver.build()
        # self.ocp.generate_code()
        # self.ocp.compile()
        jsonf = os.path.join(self.ocp.code_export_directory,
                              f'acados_ocp_{self.model.name}.json')
        self.solver = AcadosOcpSolver(self.ocp, json_file = jsonf)

    def solve(self, x0: np.ndarray, x_ref: np.ndarray):
        """
        x0: shape (nx,) initial state
        x_ref: shape (N,) desired trajectory for state
        returns: optimal u[0]
        """
        self.solver.set(0,'lbx', x0)    # the optimization only varies uk and subsequent state x1,...xN,
        self.solver.set(0,'ubx', x0)    # The initial condition is held fixed

        # use weight to emphaize more on tracking the vel_ref close to current time stage
        decay_rate = self.decay_rate_ref ** np.arange(self.N)
        # at time step k, penalize deviation of the state from x_ref[k], and penalize the control from drifting away from zero
        for k in range(self.N):
            # wk = decay_rate[k] * self.Q         # use decayed weight for speed tracking
            # Wk = np.diag([wk, self.R])           # [ state‐weight, input‐weight ]
            # self.solver.set(k,   'W',   Wk)   # <-- override the stage cost weight
            self.solver.set(k,   'yref', np.array([x_ref[k], 0.0]))
        # terminal: at time k = N, compare state x_N with x_ref[N], compute the terminal cost
        self.solver.set(self.N, 'yref_e', np.array([x_ref[-1]]))
        
        status = self.solver.solve()
        if status != 0:
            print("Warning: acados returned status", status)
        return self.solver.get(0, 'u') # only use the first control among a series of solved control 

# ─────────────────────────────── MAIN CONTROL ─────────────────────────────────
if __name__ == "__main__":
    # ─── PCA9685 PWM SETUP ──────────────────────────────────────────────────────
    # i2c = busio.I2C(board.SCL, board.SDA)
    # pca = PCA9685(i2c, address=0x40)
    # pca.frequency = 1000  # 1 kHz PWM

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

    # ─── PARAMETERS SETUP ───────────────────────────────────────────────  
    # Load reference cycle from .mat(s)
    base_folder = ""
    all_cycles = load_drivecycle_mat_files(base_folder)
    # Prompt the user:
    cycle_keys = choose_cycle_key(all_cycles)
    veh_modelName = choose_vehicleModelName()

    # Sampling time (discrete sample time)
    Ts = 0.01  # 100 Hz
    # Add this to regulate the rate of change of pwm output u
    max_delta = 30.0             # maximum % change per 0.01 s tick
    max_speed = 140.0

    # ─── MPC SETUP ───────────────────────────────────────────────  
    MPC_horizon = 10        # MPC horizon
    Q = 10.0                # penalize speed error
    R = 1.0                 # penalize control effort
    decay_rate_ref = 0.99   # currently not using
    # -------- Load the previously trained GRU model --------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = os.path.join("neuralMPC/save", "gru_regressor_0527.pth")
    model_gru = DeepGRURegressor0527()                                      # 1) re-create the model with the same architecture
    state_dict = torch.load(save_path, map_location=device, weights_only=True,)
    model_gru.load_state_dict(state_dict, strict=False)   # 2) load weights   
    model_gru.to("cpu").eval() # casadi only works on CPU tensors not on GPU
    print("Model reloaded and ready for inference on", device)
    mpc = MPC(model_gru=model_gru, MPC_horizon=MPC_horizon, Q=Q, R=R, N=MPC_horizon, Ts=Ts, decay_rate_ref=decay_rate_ref) # Setup the solver for MPC 
    


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
        log_data  = []
        times_mpc = []
        # Record loop‐start time so we can log elapsed time from 0.0
        run_start      = datetime.now()
        t0             = time.time()
        next_time      = t0

        # Reset PID state
        prev_error     = 0.0
        u_prev         = [0.0] * MPC_horizon
        # Track previous reference speed for derivative on ref (if needed)
        prev_ref_speed = None

    # ─── MAIN 100 Hz CONTROL LOOP ─────────────────────────────────────────────────
        print(f"\n[Main] Starting cycle '{cycle_key}' on {veh_modelName}, duration={ref_time[-1]:.2f}s")
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
                    # rspd_now = ref_speed[-1]
                    rspd_now = 0.0
                    break
                else:
                    rspd_now = float(np.interp(elapsed_time, ref_time, ref_speed))

                # ── Compute current error e[k] ────────
                v_meas = latest_speed if latest_speed is not None else 0.0
                F_meas = latest_force if latest_force is not None else 0.0
                e_k    = rspd_now - v_meas
  
                # ---- Implement MPC controller -------- 
                # 1) Build the reference over the next N steps
                mpc_horizon_times = elapsed_time + np.arange(MPC_horizon) * Ts
                mpc_horizon_times = np.minimum(mpc_horizon_times, ref_time[-1])
                speed_ref_window_mpc = np.interp(mpc_horizon_times, ref_time, ref_speed)

                # 2) Load ref and initial state into the OCP
                start_pc = time.perf_counter()
                u_out_mpc = mpc.solve(v_meas, speed_ref_window_mpc)
                end_pc   = time.perf_counter()
                times_mpc.append(end_pc - start_pc)
                print(f"MPC solve took {(end_pc - start_pc)*1e3:.3f} ms")


                # ── Total output u[k], clipped to [-15, +100] ────────────────
                # u_unclamped = mpc_control(u_prev, ref_time, ref_speed, elapsed_time, Ts, model, MPC_Horizon)
                u_unclamped = u_out_mpc
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
                    f"BMS_socMin={BMS_socMin:6.2f} %"
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
                    "BMS_socMin":BMS_socMin,

                })
            print("mean:",  1e3*np.mean(times_mpc),  "ms")
            print("max :",  1e3*np.max(times_mpc),   "ms")
            print("99% pctl:", 1e3*np.percentile(times_mpc,99), "ms")

        except KeyboardInterrupt:
            print("\n[Main] KeyboardInterrupt detected. Exiting…")

        finally:
            # Stop CAN thread and wait up to 1 s
            dyno_can_running = False
            veh_can_running = False
            print("All CAN_Running Stops!!!")
            dyno_can_thread.join(timeout=1.0)
            veh_can_thread.join(timeout=1.0)

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
                timestamp_str = datetime.strftime("%m%d_%H%M")
                excel_filename = f"DR_log_{veh_modelName}_{cycle_key}_{timestamp_str}.xlsx"
                        # Ensure the subfolder exists
                log_dir = os.path.join(base_folder, "Log_DriveRobot")
                os.makedirs(log_dir, exist_ok=True)     
                excel_path = os.path.join(log_dir, excel_filename)

                df.to_excel(excel_path, index=False)
                print(f"[Main] Saved log to '{excel_path}' as {excel_filename}")

        print(f"[Main] Finish Running {cycle_key}, take a 5 second break...")
        time.sleep(5)

    pca.deinit()
    print("[Main] pca board PWM signal cleaned up and exited.")
    print("[Main] Cleaned up and exited.")

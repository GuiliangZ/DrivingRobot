import os
import time
from datetime import datetime
import threading
from pathlib import Path

import numpy as np
import scipy.io as sio
import pandas as pd

import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver
import deepctools as dpc
from utils import *

'''
The deepc file that contain everything related to deepc algorithms


'''


# ─── LOAD DATA FOR DeePC ───────────────────────────────────────────────
xlsx_path="PWMDynoSpeedNewsMCT.xlsx"
seq_len = 50
ds = MultiSheetTimeSeriesDataset(xlsx_path, seq_len, normalize=True, cache_path="save/timeseries_dataset0527.npz")



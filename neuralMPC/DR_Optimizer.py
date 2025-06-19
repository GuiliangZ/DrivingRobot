import casadi as cs
import numpy as np
import torch
import torch.nn as nn
import l4casadi as l4c
from acados_template import AcadosSimSolver, AcadosOcpSolver, AcadosSim, AcadosOcp, AcadosModel
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class PyTorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = torch.nn.Linear(2, 512)

        hidden_layers = []
        for i in range(5):
            hidden_layers.append(torch.nn.Linear(512, 512))

        self.hidden_layer = torch.nn.ModuleList(hidden_layers)
        self.out_layer = torch.nn.Linear(512, 2)

        # Model is not trained -- setting output to zero
        with torch.no_grad():
            self.out_layer.bias.fill_(0.)
            self.out_layer.weight.fill_(0.)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layer:
            x = torch.tanh(layer(x))
        x = self.out_layer(x)
        return x


class DoubleIntegratorWithLearnedDynamics:
    def __init__(self, learned_dyn):
        self.learned_dyn = learned_dyn

    def model(self):
        s = cs.MX.sym('s', 1)
        s_dot = cs.MX.sym('s_dot', 1)
        s_dot_dot = cs.MX.sym('s_dot_dot', 1)
        u = cs.MX.sym('u', 1)
        x = cs.vertcat(s, s_dot)
        x_dot = cs.vertcat(s_dot, s_dot_dot)

        res_model = self.learned_dyn(x)
        p = self.learned_dyn.get_sym_params()
        parameter_values = self.learned_dyn.get_params(np.array([0, 0]))

        f_expl = cs.vertcat(
            s_dot,
            u
        ) + res_model

        x_start = np.zeros((2))

        # store to struct
        model = cs.types.SimpleNamespace()
        model.x = x
        model.xdot = x_dot
        model.u = u
        model.z = cs.vertcat([])
        model.p = p
        model.parameter_values = parameter_values
        model.f_expl = f_expl
        model.x_start = x_start
        model.constraints = cs.vertcat([])
        model.name = "wr"

        return model

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

    def forward(self, x: torch.Tensor, hidden:torch.Tensor = None ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch, seq_len, input_size)
        Returns:
            preds: Tensor of shape (batch,) with the regression output.
            hidden: (num_layers, batch, hidden_size) or None
        """
        # GRU returns: output (batch, seq_len, num_directions*hidden_size), h_n
        B = x.size(0)
        if hidden is not None and hidden.size(1) == B:
            hidden = hidden.detach()
        else:
            hidden = None
        out,_ = self.gru(x)
        y_seq = self.fc(out)         # (B, seq_len, 1)
        y_seq = y_seq.squeeze(-1)    # (B, seq_len)
        return y_seq, hidden



class MPC:
    def __init__(self, model, N):
        self.N = N
        self.model = model

    @property
    def simulator(self):
        return AcadosSimSolver(self.sim())

    @property
    def solver(self):
        return AcadosOcpSolver(self.ocp())

    def sim(self):
        model = self.model

        t_horizon = 1.
        N = self.N

        # Get model
        model_ac = self.acados_model(model=model)
        model_ac.p = model.p

        # Dimensions
        nu = 1
        ny = 1

        # Create OCP object to formulate the optimization
        sim = AcadosSim()
        sim.model = model_ac
        sim.dims.N = N
        sim.dims.nu = nu
        sim.dims.ny = ny
        sim.solver_options.tf = t_horizon

        # Solver options
        sim.solver_options.Tsim = 1./ 10.
        sim.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        sim.solver_options.hessian_approx = 'GAUSS_NEWTON'
        sim.solver_options.integrator_type = 'ERK'
        # ocp.solver_options.print_level = 0
        sim.solver_options.nlp_solver_type = 'SQP_RTI'

        return sim

    def ocp(self):
        model = self.model

        t_horizon = 1.
        N = self.N

        # Get model
        model_ac = self.acados_model(model=model)
        model_ac.p = model.p

        # Dimensions
        nx = 2
        nu = 1
        ny = 1

        # Create OCP object to formulate the optimization
        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.dims.N = N
        ocp.dims.nu = nu
        ocp.dims.ny = ny
        ocp.solver_options.tf = t_horizon

        # Initialize cost function
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        ocp.cost.W = np.array([[1.]])

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[0, 0] = 1.
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vz = np.array([[]])
        ocp.cost.Vx_e = np.zeros((ny, nx))
        ocp.cost.W_e = np.array([[0.]])
        ocp.cost.yref_e = np.array([0.])

        # Initial reference trajectory (will be overwritten)
        ocp.cost.yref = np.zeros(1)

        # Initial state (will be overwritten)
        ocp.constraints.x0 = model.x_start

        # Set constraints
        a_max = 10
        ocp.constraints.lbu = np.array([-a_max])
        ocp.constraints.ubu = np.array([a_max])
        ocp.constraints.idxbu = np.array([0])

        # Solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        ocp.parameter_values = model.parameter_values

        return ocp

    def acados_model(self, model):
        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.xdot - model.f_expl
        model_ac.f_expl_expr = model.f_expl
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.name = model.name
        return model_ac


def run():

    # -------- Load the previously trained GRU model --------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = os.path.join("save", "gru_regressor_0527.pth")
    model_gru = DeepGRURegressor0527()                                      # 1) re-create the model with the same architecture
    model_gru.load_state_dict(torch.load(save_path, map_location=device))   # 2) load weights   
    model_gru.to(device).eval()                                             # 3) move to device & eval
    print("Model reloaded and ready for inference on", device)

    MPC_Horizon = 10

    model_CasADi = l4c.realtime.RealTimeL4CasADi(model_gru, approximation_order=2)

    solver = MPC(model=model_CasADi.model(), MPC_Horizon = MPC_Horizon).solver

    print('Warming up model...')
    x_l = []
    for i in range(N):
        x_l.append(solver.get(i, "x"))
    for i in range(20):
        model.get_params(np.stack(x_l, axis=0))
    print('Warmed up!')

    x = []
    x_ref = []
    ts = 1. / N
    xt = np.array([1., 0.])
    opt_times = []

    for i in range(50):
        now = time.time()
        t = np.linspace(i * ts, i * ts + 1., 10)
        yref = np.sin(0.5 * t + np.pi / 2)
        x_ref.append(yref[0])
        for t, ref in enumerate(yref):
            solver.set(t, "yref", ref)
        solver.set(0, "lbx", xt)
        solver.set(0, "ubx", xt)
        solver.solve()
        xt = solver.get(1, "x")
        x.append(xt)

        x_l = []
        for i in range(N):
            x_l.append(solver.get(i, "x"))
        params = model.get_params(np.stack(x_l, axis=0))
        for i in range(N):
            solver.set(i, "p", params[i])

        elapsed = time.time() - now
        opt_times.append(elapsed)

    print(f'Mean iteration time: {1000*np.mean(opt_times):.1f}ms -- {1/np.mean(opt_times):.0f}Hz)')


if __name__ == '__main__':
    run()
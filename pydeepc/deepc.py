import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver
from pydeepc.utils import Data, split_data


class DeePCAcados:
    def __init__(self, data: Data, Tini: int, horizon: int, explained_variance: float = None):
        assert explained_variance is None or 0 < explained_variance <= 1, \
            "explained_variance should be in (0,1] or be None"
        self.Tini = Tini
        self.horizon = horizon
        self.explained_variance = explained_variance
        self.update_data(data)
        self.solver = None

    def update_data(self, data: Data):
        # Build Hankel matrices
        Up, Uf, Yp, Yf = split_data(data, self.Tini, self.horizon, self.explained_variance)
        self.Up = Up
        self.Uf = Uf
        self.Yp = Yp
        self.Yf = Yf
        self.M = data.u.shape[1]
        self.P = data.y.shape[1]
        self.T = data.u.shape[0]
        # number of columns in Hankel = Nd
        self.Nd = self.T - self.Tini - self.horizon + 1
        self.solver = None

    def build_ocp(self,
                  build_cost: Callable[[ca.MX, ca.MX], ca.MX],
                  build_constraints: Optional[Callable[[ca.MX, ca.MX], List[ca.MX]]] = None,
                  lambda_g: float = 0.,
                  lambda_y: float = 0.,
                  lambda_u: float = 0.,
                  lambda_proj: float = 0.) -> AcadosOcpSolver:
        # Symbolic decision variables
        g = ca.MX.sym('g', self.Nd)
        slack_y = ca.MX.sym('slack_y', self.P * self.Tini)
        slack_u = ca.MX.sym('slack_u', self.M * self.Tini)
        # inputs and outputs over horizon
        u_seq = ca.reshape(self.Uf @ g, self.horizon, self.M)
        y_seq = ca.reshape(self.Yf @ g, self.horizon, self.P)

        # initial condition parameters
        uini_p = ca.MX.sym('uini', self.M * self.Tini)
        yini_p = ca.MX.sym('yini', self.P * self.Tini)

        # Build equality constraint: A*g = b
        A = ca.vertcat(self.Up, self.Yp, self.Uf, self.Yf)
        b = ca.vertcat(uini_p + slack_u, yini_p + slack_y, ca.reshape(u_seq, -1, 1), ca.reshape(y_seq, -1, 1))
        constr_eq = A @ g - b

        # Build OCP
        ocp = AcadosOcp()
        model = ocp.model
        model.name = 'deepc'
        # states: g, slack_y, slack_u stacked
        model.x = ca.vertcat(g, slack_y, slack_u)
        # controls: none
        model.u = ca.vertcat()
        # parameters: uini, yini
        model.p = ca.vertcat(uini_p, yini_p)
        # no dynamics (static optimization): xdot = 0
        model.xdot = ca.DM.zeros(model.x.shape[0], 1)

        # dimensions
        ocp.dims.N = 1
        ocp.dims.nx = model.x.shape[0]
        ocp.dims.nbu = 0
        ocp.dims.nbx = 0
        ocp.dims.nbx_0 = 0

        # constraints: equality
        ocp.constr_expr = constr_eq
        ocp.cost.N = 1

        # cost
        cost_expr = build_cost(u_seq, y_seq)
        # add regularizers
        if lambda_g > 0:
            cost_expr += lambda_g * ca.norm_1(g)
        if lambda_u > 0:
            cost_expr += lambda_u * ca.norm_1(slack_u)
        if lambda_y > 0:
            cost_expr += lambda_y * ca.norm_1(slack_y)
        if lambda_proj > 0:
            # build projection term if desired (not shown here)
            pass
        ocp.cost.expr_ext_cost = cost_expr

        # solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        ocp.parameter_values = []

        # create solver
        self.solver = AcadosOcpSolver(ocp, json_file='acados_ocp.json')
        return self.solver

    def solve(self, data_ini: Data) -> Tuple[np.ndarray, Dict]:
        # prepare initial cond
        uini = data_ini.u[:self.Tini].flatten()
        yini = data_ini.y[:self.Tini].flatten()
        # set parameters
        self.solver.set({'p': np.concatenate([uini, yini])})
        # initial guess can be left zero or provided
        status = self.solver.solve()
        if status != 0:
            raise Exception(f'acados solve returned status {status}')
        # retrieve optimal g
        g_opt = self.solver.get('x')[:self.Nd]
        u_opt = (self.Uf @ g_opt).reshape(self.horizon, self.M)
        info = {'g': g_opt, 'status': status}
        return u_opt, info
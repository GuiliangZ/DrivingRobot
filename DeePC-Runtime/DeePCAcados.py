"""
Name: DeePCAcados.py
Author: Guiliang Zheng
Date:at 24/06/2025
version: 1.0.0
Description: Toolbox to formulate the real-time DeePC problem using Acados solver
"""
import time
import warnings
import numpy as np
from scipy.linalg import block_diag
import casadi as cs
from casadi import vcat
import casadi.tools as ctools
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

# packages from deepctools
from deepctools import util

def timer(f):
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        print('Time elapsed: {}'.format(time.time() - start))
        return ret

    return wrapper


class deepctools():
    """
         ----------------------------------------------------------------------------------------------------------
                Standard DeePC design:                    |            Equivalent expression
         min J  =  || y - yref ||_Q^2 + || uloss ||_R^2   |   min J  =  || Uf*g - yref ||_Q^2 + || uloss ||_R^2
             s.t.   [Up]       [uini]                     |      s.t.   Up * g = uini
                    [Yp] * g = [yini]                     |             Yp * g = yini
                    [Uf]       [ u  ]                     |             ulb <= u <= uub
                    [Yf]       [ y  ]                     |             ylb <= y <= yub
                    ulb <= u <= uub                       |
                    ylb <= y <= yub                       |  uloss = (u)   or    (u - uref)    or    (du)
         -------------------------------------------------|----------------------------------------------------------
                  Robust DeePC design:                    |            Equivalent expression
         min J  =  || y - yref ||_Q^2 + || uloss ||_R^2   |   min J  =  || Uf*g - ys ||_Q^2 + || uloss ||_R^2
                     + lambda_y||sigma_y||_2^2            |             + lambda_y||Yp*g-yini||_2^2
                     + lambda_g||g||_2^2                  |             + lambda_g||g||_2^2
             s.t.   [Up]       [uini]     [   0   ]       |      s.t.   Up * g = uini
                    [Yp] * g = [yini]  +  [sigma_y]       |             ulb <= u <= uub
                    [Uf]       [ u  ]     [   0   ]       |             ylb <= y <= yub
                    [Yf]       [ y  ]     [   0   ]       |
                    ulb <= u <= uub                       |
                    ylb <= y <= yub                       |  uloss = (u)   or    (u - uref)    or    (du)
         ----------------------------------------------------------------------------------------------------------
                        Functions                |                            Usage
            initialize_RTDeePCsolver(uloss, opts)|  construct real-time DeePC solver based on acados

            initialize_DeePCsolver(uloss, opts)  |  construct DeePC solver
            initialize_RDeePCsolver(uloss, opts) |  construct Robust DeePC solver
            solver_step(uini, yini)              |  solve the optimization problem one step
         ----------------------------------------------------------------------------------------------------------
    """

    def __init__(self, u_dim, y_dim, T, Tini, Np, ineqconidx=None, ineqconbd=None):
        """
            ------Initialize the system parameters and DeePC config------                                                       | ----- Drive Robot --------
                 u_dim: [int]             |  the dimension of control inputs                                                    |           1
                 y_dim: [int]             |  the dimension of controlled outputs                                                |           1
                     T: [int]             |  the length of offline collected data                                               |   collected data length    
                  Tini: [int]             |  the initialization length of the online loop                                       |   Hyperparameter to tune
                    Np: [int]             |  the length of predicted future trajectories                                        |   Hyperparameter to tun            
            ineqconidx: [dict|[str,list]] |  specify the wanted constraints for u and y, if None, no constraints                |   
                                          |      e.g., only have constraints on u2, u3, {'u': [1,2]}; 'y' as well
             ineqconbd: [dict|[str,list]] |  specify the bounds for u and y, should be consistent with "ineqconidx"
                                          |      e.g., bound on u2, u3, {'lbu': [1,0], 'ubu': [10,5]}; lby, uby as well

            # Put those parameters as part of optimization parameter to avoid large matrix size handling
                     Q: [array]           |  the weighting matrix of controlled outputs y
                     R: [array]           |  the weighting matrix of control inputs u
              lambda_y: [array]           |  the weighting matrix of mismatch of controlled output y
              lambda_u: [array]           |  the weighting matrix of mismatch of controlled output u
              lambda_g: [array]           |  the weighting matrix of norm of operator g
      """

        self.u_dim = u_dim
        self.y_dim = y_dim
        self.T = T
        self.Tini = Tini
        self.Np = Np
        self.g_dim = T - Tini - Np + 1 

        # init the casadi variables
        self._init_variables()

        self.solver = None
        self.lbc = None
        self.ubc = None
        self.lb_du = None
        self.ub_du = None
    
    def _init_variables(self):
        """
            Initialize variables of DeePC and RDeePC design
                   parameters: uini, yini, yref, Up, Yp, Uf, Yf,    |   updated each iteration
                                lambda_g, lambda_y, lambda_u, Q, R  |   stay the same at each iteration, put them as parameters to avoid large matrix handling
            optimizing_target: g            |   decision variable
        """
        ## define casadi variables
        self.optimizing_target = ctools.struct_symSX([
            (
                ctools.entry('g', shape=tuple([self.g_dim, 1]))
            )
        ])
        
        parameters = [
                ctools.entry('uini', shape=tuple([self.u_dim * self.Tini, 1])),
                ctools.entry('yini', shape=tuple([self.y_dim * self.Tini, 1])),
                ctools.entry('yref', shape=tuple([self.y_dim * self.Np, 1])),
                ctools.entry('Up',   shape=tuple([self.u_dim * self.Tini, self.g_dim])),
                ctools.entry('Yp',   shape=tuple([self.y_dim * self.Tini, self.g_dim])),
                ctools.entry('Uf',   shape=tuple([self.u_dim * self.Np, self.g_dim])),
                ctools.entry('Yf',   shape=tuple([self.y_dim * self.Np, self.g_dim])),
                ctools.entry('lambda_g', shape=tuple([self.g_dim,       self.g_dim ])),
                ctools.entry('lambda_y', shape=tuple([self.y_dim*self.Tini,       self.y_dim*self.Tini ])),
                ctools.entry('lambda_u', shape=tuple([self.u_dim*self.Tini,       self.u_dim*self.Tini ])),
                # tracking‐error weight and regularization scalar
                ctools.entry('Q',        shape=tuple([self.y_dim * self.Np,    self.y_dim * self.Np])),  # diagonal entries of Q
                ctools.entry('R',        shape=tuple([self.u_dim * self.Np,    self.u_dim * self.Np])),  # scalar weight on g-regularization
        ]

        self.parameters = ctools.struct_symSX(parameters)

    def _init_ineq_cons(self, ineqconidx, ineqconbd, Uf, Yf, du):
        """
            Obtain Hankel matrix that used for the inequality constrained variables
                           lbc <= Hc * g <= ubc
            return  Hc, lbc, ubc
        """
        if ineqconidx is None:
            print(">> DeePC design have no constraints on 'u' and 'y'.")
            Hc, lbc, ubc = [], [], []
            ineq_flag = False
        else:
            Hc_list = []
            lbc_list = []
            ubc_list = []
            for varname, idx in ineqconidx.items():
                if varname == 'u':
                    H_all = Uf
                    dim = self.u_dim
                    lb = ineqconbd['lbu']
                    ub = ineqconbd['ubu']
                elif varname == 'y':
                    H_all = Yf
                    dim = self.y_dim
                    lb = ineqconbd['lby']
                    ub = ineqconbd['uby']
                elif varname == 'du':
                    continue  # Handle 'du' separately in the caller, as it is affine
                # elif varname == 'du':
                #     # print(f'H_all is{H_all}')
                #     dim = 1
                #     lb = ineqconbd['lbdu']
                #     ub = ineqconbd['ubdu']
                #     idx_H = [] 
                else:
                    raise ValueError("%s variable not exist, should be 'u' or/and 'y'!" % varname)

                idx_H = [v + i * dim for i in range(self.Np) for v in idx]
                # if varname != 'du':
                Hc_list.append(H_all[idx_H, :])
                lbc_list.append(np.tile(lb, self.Np))
                ubc_list.append(np.tile(ub, self.Np))

            Hc = cs.vertcat(*Hc_list)
            lbc = np.concatenate(lbc_list).flatten().tolist()
            ubc = np.concatenate(ubc_list).flatten().tolist()
            ineq_flag = True if Hc_list else False
        return Hc, lbc, ubc, ineq_flag

    @timer
    def init_DeePCAcadosSolver(self, recompile_solver=True, ineqconidx=None, ineqconbd=None):
        """
        Build an acados QP-OC solver for DeePC:
        min  ½ gᵀ H g + qᵀ g
        s.t. Aeq g = beq
            lbc <= Hc g <= ubc
        where H, Aeq, Hc come from your precomputed Up, Uf, Yp, Yf blocks.
        """
        print('>> Acados based Real-time DeePC design formulating.. This may take a while...')
        # if uloss not in ["u", "du"]:
        #     raise ValueError("uloss should be one of: 'u', 'du'!")
        if self.g_dim <= (self.u_dim + self.y_dim) * self.Tini:
            raise ValueError(f'OCP do not have enough degrees of freedom | Should: g_dim > (u_dim + y_dim) * Tini, but got: {self.g_dim} <= {(self.u_dim + self.y_dim) * self.Tini}!')

        # define parameters and decision variable
        uini, yini, yref, Up_cur, Yp_cur, Uf_cur, Yf_cur, lambda_g, lambda_y, lambda_u, Q, R = self.parameters[...]
        # uini, yini, yref, Up_cur, Yp_cur, Uf_cur, Yf_cur, lambda_g, lambda_y, Q, R = self.parameters[...]
        g, = self.optimizing_target[...]  # data are stored in list [], notice that ',' cannot be missed

        # To get du
        u_cur = cs.mtimes(Uf_cur, g)
        u_prev = cs.vertcat(uini[-self.u_dim:], cs.mtimes(Uf_cur, g)[:-self.u_dim])
        du = u_cur - u_prev
        D_du   = cs.jacobian(du, g)   # (Np*u_dim × g_dim)

        # To get constrains
        Hc, lbc_ineq, ubc_ineq, ineq_flag = self._init_ineq_cons(ineqconidx, ineqconbd, Uf_cur, Yf_cur, du)
    
        # Handle 'du' constraints separately if present
        du_flag = False
        if ineqconidx is not None and 'du' in ineqconidx:
            idx = ineqconidx['du']
            if idx:  # Check if there are indices to constrain
                dim = self.u_dim
                idx_H = [v + i * dim for i in range(self.Np) for v in idx]
                h_du = du[idx_H]
                lbdu = ineqconbd['lbdu']
                ubdu = ineqconbd['ubdu']
                lbc_du = np.tile(lbdu, self.Np).flatten().tolist()
                ubc_du = np.tile(ubdu, self.Np).flatten().tolist()
                du_flag = True

        # Start the Acados formulation
        model = AcadosModel()
        model.name = 'deepc'
        model.x = g                         # x is the decision variable
        model.p = cs.vertcat(
            uini, yini, yref,
            cs.reshape(Up_cur, -1, 1),
            cs.reshape(Yp_cur, -1, 1),
            cs.reshape(Uf_cur, -1, 1),
            cs.reshape(Yf_cur, -1, 1),
            cs.reshape(Q, -1, 1), cs.reshape(R, -1, 1),
            cs.reshape(lambda_y, -1, 1), cs.reshape(lambda_g, -1, 1), cs.reshape(lambda_u, -1, 1), 
        )
        model.disc_dyn_expr = model.x          # shape (g_dim, 1) # zero‐dynamics: ẋ = 0. # Using purely hankel-based styatic DeePC - In this setup you’re treating your entire decision vector g as a “state” with zero dynamics so that Acados turns your one‐step OCP into a pure static QP

        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = 1                      # single shooting step means only have one interval btw initial state and terminal state
        ocp.dims.nx = self.g_dim            # state size
        ocp.dims.np = model.p.size()[0]     # parameter size

        # Define the objective function from CasADi parameters
        r1 = Yf_cur @ g - yref              # (n_y*Np x 1): tracking error
        r2 = Uf_cur @ g                     # (n_u*Np x 1): control effort
        r3 = Yp_cur @ g - yini              # (n_y*Tini x 1): init‐condition slack - Y
        r4 = Up_cur @ g - uini              # (n_u*Tini x 1): init‐condition slack - U
        r5 = g                              # (n_g    x 1): regularization
        print(f'The shape of r1:{r1.shape}, r2:{r2.shape},r3:{r3.shape},r4:{r4.shape},r5:{r5.shape}')
        # ocp.cost.cost_type = 'LINEAR_LS'        # For the LINEAR_LS, the weight matrix should be np array instead of casadi matrix, so should use "EXTERNAL" instead
        # H = Yf_cur.T @ Q @ Yf_cur + Uf_cur.T @ R @ Uf_cur + Yp_cur.T @ lambda_y @ Yp_cur + Up_cur.T @ lambda_u @ Up_cur + lambda_g
        # f = - Yp_cur.T @ lambda_y @ yini - Yf_cur.T @ Q @ yref - Up_cur.T @ lambda_u @ uini  
        # obj = 0.5 * cs.mtimes(cs.mtimes(g.T, H), g) + cs.mtimes(f.T, g)
        
        # minimize the value of u
        obj = r1.T @ Q @ r1 + r2.T @ R @ r2 + r3.T @ lambda_y @ r3 + r4.T @ lambda_u @ r4 + r5.T @ lambda_g @ r5

        # minimize the change of u (du)
        # obj = r1.T @ Q @ r1 + du.T @ R @ du + r3.T @ lambda_y @ r3 + r4.T @ lambda_u @ r4 + r5.T @ lambda_g @ r5
        ocp.cost.cost_type = 'EXTERNAL'
        ocp.cost.cost_type_e = 'EXTERNAL'
        ocp.cost.cost_type = 'AUTO'
        ocp.cost.cost_type_e = 'AUTO'
        model.cost_expr_ext_cost   = obj
        model.cost_expr_ext_cost_e = obj

        # Add equality constraint: Up g - uini == 0
        h_eq = cs.reshape(cs.mtimes(Up_cur, g) - uini, (-1, 1))  # Shape: (u_dim * Tini, 1)

        # Build the constraint expression h and bounds
        has_constraints = ineq_flag or du_flag
        if has_constraints:
            ocp.constraints.constr_type = 'BGH'
            if ineq_flag:
                h1 = cs.reshape(Hc @ g, (-1, 1))
                lh_ineq = np.array(lbc_ineq)
                uh_ineq = np.array(ubc_ineq)
                if du_flag:
                    h = cs.vertcat(h1, cs.reshape(h_du, (-1, 1)))
                    lh_arr = np.concatenate((lh_ineq, np.array(lbc_du)))
                    uh_arr = np.concatenate((uh_ineq, np.array(ubc_du)))
                else:
                    h = h1
                    lh_arr = lh_ineq
                    uh_arr = uh_ineq
            else:  # Only du constraints
                h = cs.reshape(h_du, (-1, 1))
                lh_arr = np.array(lbc_du)
                uh_arr = np.array(ubc_du)
            # print(f"h_sx size: ({h.size1()}×{h.size2()})")
            # print(f"  → expr_h has {h.size1()} rows and {h.size2()} cols")
            # print(f"  → lh   shape = {lh_arr.shape}")
            # print(f"  → uh   shape = {uh_arr.shape}")
            ocp.dims.nh                 = h.shape[0]
            ocp.model.con_h_expr        = h             # SX of shape (n_rows, 1)
            # — set matching numeric bounds: lh ≤ h(x,u) ≤ uh
            ocp.constraints.lh          = lh_arr        # shape = (n_rows,)
            ocp.constraints.uh          = uh_arr        # shape = (n_rows,)
        else:
            # — no constraints: empty h, empty bounds
            ocp.constraints.constr_type = 'BGH'
            ocp.constraints.con_h_expr  = cs.SX.zeros(0, 1)            # zero-row SX
            ocp.constraints.lh          = np.zeros(0)
            ocp.constraints.uh          = np.zeros(0)
        # if ineq_flag:
        #     # — build a single SX vector h = [Hc@g; du]
        #     h1 = cs.reshape(Hc @ g, (-1, 1))
        #     # h2 = cs.reshape(du,     (-1, 1))
        #     # h  = cs.vertcat(h1,     h2)      # now definitely (n_rows, 1)
        #     # h = cs.vertcat(h1, h2, h_eq)
        #     h = h1
        #     lh_arr = np.array(lbc_ineq)
        #     uh_arr = np.array(ubc_ineq)
        #     # print(f"h_sx size: ({h.size1()}×{h.size2()})")
        #     # print(f"  → expr_h has {h.size1()} rows and {h.size2()} cols")
        #     # print(f"  → lh   shape = {lh_arr.shape}")
        #     # print(f"  → uh   shape = {uh_arr.shape}")
        #     ocp.constraints.constr_type = 'BGH'
        #     ocp.dims.nh                 = h.shape[0]
        #     ocp.model.con_h_expr        = h             # SX of shape (n_rows, 1)
        #     # — set matching numeric bounds: lh ≤ h(x,u) ≤ uh
        #     ocp.constraints.lh          = lh_arr        # shape = (n_rows,)
        #     ocp.constraints.uh          = uh_arr        # shape = (n_rows,)
        # else:
        #     # — no constraints: empty h, empty bounds
        #     ocp.constraints.constr_type = 'BGH'
        #     ocp.constraints.con_h_expr  = cs.SX.zeros(0, 1)            # zero-row SX
        #     ocp.constraints.lh          = np.zeros(0)
        #     ocp.constraints.uh          = np.zeros(0)

        ocp.solver_options.qp_solver          = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx       = 'EXACT'                    # 'GAUSS_NEWTON' for "LINEAR_LS" cost_type
        ocp.solver_options.integrator_type      = 'DISCRETE'                 # Using purely hankel-based styatic DeePC, best choice-'DISCRETE'. actual dynamic in OCP: 'ERK'-classic Runge-Kutta 4. Options: 'IRK', 'GNSF', 'LIFTED_IRK', 'DISCRETE
        ocp.solver_options.nlp_solver_type      = 'SQP_RTI'                  # Need g_opt warm start! Real‐Time Iteration SQP: performs exactly one SQP step per control cycle. Ultra‐low latency, but may require more frequent warm starts or robustification
        ocp.solver_options.levenberg_marquardt = 1e-4
        ocp.solver_options.tf                   = 1.0                        # For s discrete dynamics, static QP in g, tf is unused - can leave it at the default(1.0)
        ocp.solver_options.qp_solver_warm_start = 1                         # Enable primal-dual warm-starting for faster subsequent solves.
        ocp.solver_options.print_level = 0                                  # Minimize output for real-time speed; set to 1-2 for debugging.
        # ocp.solver_options.nlp_solver_max_iter = 1                          # Enforce RTI behavior: exactly one iteration per call.
        
        # ocp.solver_options.qp_solver          = 'PARTIAL_CONDENSING_HPIPM' # also could try 'FULL_CONDENSING_QPOASES'        
        # ocp.solver_options.qp_solver          = 'FULL_CONDENSING_QPOASES'
        # ocp.solver_options.ext_cost_num_hess  = True                       # turn on numeric external cost Hessians
        # ocp.solver_options.nlp_solver_type      = "SQP_WITH_FEASIBLE_QP"    # 'SQP', 'DDP'
        # ocp.solver_options.qp_solver_cond_N = self.Np                       # For condensing; match N for efficiency.
        # ocp.dims.N                              = self.Np                    # number of shooting intervals = prediction steps
  
        # Initilize g and all the parameters with zero for acados initial build, will update those parameters at run time
        n_p = ocp.model.p.numel()                                           # total number of entries in the SX parameter vector
        ocp.parameter_values = np.zeros(n_p)                                # or fill with your actual parameter data
        
        # === Create or load solver ===
        if recompile_solver:
            # regenerate C code + make
            self.solver = AcadosOcpSolver(
                ocp,
                json_file='DeePC_acados_ocp.json',
                generate=True,
                build=True,
                verbose=True,
            )
        else:
            # load existing compiled solver from c_generated_code
            # skips code_export() & make()
            self.solver = AcadosOcpSolver(
                None,
                json_file='DeePC_acados_ocp.json',
                generate=False,
                build=False,
                verbose=True,
                # acados_lib_path='c_generated_code/lib',
                # acados_include_path='c_generated_code/include',
            )
        print('>> Acados solver ready (recompile=' + str(recompile_solver) + ')')
    
    def acados_solver_step(self, uini, yini, yref, Up_cur, Uf_cur, Yp_cur, Yf_cur, Q_val, R_val, lambda_g_val, lambda_y_val, lambda_u_val, g_prev=None):
    # def acados_solver_step(self, uini, yini, yref, Up_cur, Uf_cur, Yp_cur, Yf_cur, Q_val, R_val, lambda_g_val, lambda_y_val, g_prev=None):
        """
            solver solve the nlp for one time
            uini, yini:  [array]   | (dim*Tini, 1)
            uref, yref:  [array]   | (dim*Horizon, 1) if sp_change=True
              g0_guess:  [array]   | (T-L+1, 1)
            return:
                u_opt:  the optimized control input for the next Np steps
                 g_op:  the optimized operator g
                  t_s:  solving time
        """
        if yref is None:
            raise ValueError("Did not give value of 'yref', but required in objective function!")
        
        parameters = np.concatenate([
            uini.ravel(), yini.ravel(), yref.ravel(),
            Up_cur.ravel(), Yp_cur.ravel(),
            Uf_cur.ravel(), Yf_cur.ravel(),
            Q_val.ravel(),    R_val.ravel(),
            lambda_g_val.ravel(),   lambda_y_val.ravel(), lambda_u_val.ravel(),
        ])

        # # choose hot-start if available
        # # if g_prev is not None:
        # #     # ensure the shape matches
        # #     g0 = g_prev.ravel()
        # # else:
        # # give acados initial guess
        # sqrt_lambda_u = np.sqrt(lambda_u_val)
        # sqrt_lambda_y = np.sqrt(lambda_y_val)
        # sqrt_lambda_g = np.sqrt(lambda_g_val)
        # # Weighted stack for least-squares
        # A_weighted = np.vstack([sqrt_lambda_u @ Up_cur, sqrt_lambda_y @ Yp_cur])
        # b_weighted = np.vstack([sqrt_lambda_u @ uini, sqrt_lambda_y @ yini])
        # if lambda_g_val[0,0] > 0:
        #     I_g = np.eye(self.g_dim)
        #     A_aug = np.vstack([A_weighted, sqrt_lambda_g @ I_g])
        #     b_aug = np.vstack([b_weighted, np.zeros((self.g_dim, 1))])
        #     g_default = np.linalg.lstsq(A_aug, b_aug, rcond=None)[0]
        # else:
        #     g_default = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)[0]

        # g_default = np.linalg.pinv(np.concatenate((Up_cur, Yp_cur), axis=0)) @ np.concatenate((uini, yini))
        
        # choose hot-start if available
        if g_prev is not None:
            # ensure the shape matches
            g0 = g_prev.ravel()
        else:
        # give acados initial guess
            sqrt_lambda_u = np.sqrt(lambda_u_val)
            sqrt_lambda_y = np.sqrt(lambda_y_val)
            sqrt_lambda_g = np.sqrt(lambda_g_val)
            # Weighted stack for least-squares
            A_weighted = np.vstack([sqrt_lambda_u @ Up_cur, sqrt_lambda_y @ Yp_cur])
            b_weighted = np.vstack([sqrt_lambda_u @ uini, sqrt_lambda_y @ yini])
            if lambda_g_val[0,0] > 0:
                I_g = np.eye(self.g_dim)
                A_aug = np.vstack([A_weighted, sqrt_lambda_g @ I_g])
                b_aug = np.vstack([b_weighted, np.zeros((self.g_dim, 1))])
                g_default = np.linalg.lstsq(A_aug, b_aug, rcond=None)[0]
            else:
                g_default = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)[0]
            g_default = g_default.ravel()
            g0 = g_default

        self.solver.set( 0, "x", g0)
        self.solver.set( 0, "p", parameters )

        t0 = time.time()
        status = self.solver.solve()
        exist_feasible_sol = (status == 0)
        t_s = round((time.time() - t0) * 1_000, 3)

        g_opt = self.solver.get(0,"x")
        u_opt = Uf_cur @ g_opt              # which is same as np.matmul(Uf_cur, g_opt)
        cost = self.solver.get_cost()
        return u_opt, g_opt, t_s, exist_feasible_sol, cost
    
    @timer
    def init_DeePCsolver(self, uloss='u', ineqconidx=None, ineqconbd=None, opts={}):
        """
                              Formulate NLP solver for: DeePC design
            Initialize CasADi nlp solver, !!! only need to formulate the nlp problem at the first time !!!
            treat the updated variables as parameters of the nlp problem
            At each time, update the initial guess of the decision variables and the required parameters
            ----------------------------------------------------------------------------------------------
            nlp_prob = {'f': obj, 'x': optimizing_target, 'p': parameters, 'g': cs.vertcat(*C)}
            self.solver = cs.nlpsol('solver', 'ipopt', nlp_prob, opts)
            sol = self.solver(x0=g_guess, p=parameters, lbg=self.lbc, ubg=self.ubc)
            uloss: the loss of u in objective function
                   "u"  :  ||u||_R^2
                   "uus":  ||u - us||_R^2
                   "du" :  || du ||_R^2
            opts: the config of the solver; max iteration, print level, etc.
                   e.g.:         opts = {
                                            'ipopt.max_iter': 100,  # 50
                                            'ipopt.tol': 1e-5,
                                            'ipopt.print_level': 1,
                                            'print_time': 0,
                                            # 'ipopt.acceptable_tol': 1e-8,
                                            # 'ipopt.acceptable_obj_change_tol': 1e-6,
                                        }
            -----------------------------------------------------------------------------------------------
            g_dim:
                if DeePC:
                    g_dim >= (u_dim + y_dim) * Tini
                if Robust DeePC:
                    g_dim >= u_dim * Tini
                to ensure have enough degree of freedom of nlp problem for g
                this is the equality constraints should be less than decision variables
            -----------------------------------------------------------------------------------------------
        """
        print('>> DeePC design formulating.. This may take a while...')
        if uloss not in ["u", "du"]:
            raise ValueError("uloss should be one of: 'u', 'du'!")
        if self.g_dim <= (self.u_dim + self.y_dim) * self.Tini:
            raise ValueError(f'NLP do not have enough degrees of freedom | Should: g_dim > (u_dim + y_dim) * Tini, but got: {self.g_dim} <= {(self.u_dim + self.y_dim) * self.Tini}!')

        # define parameters and decision variable
        # uini, yini, yref, Up_cur, Yp_cur, Uf_cur, Yf_cur, lambda_g, lambda_y, lambda_u, Q, R = self.parameters[...]
        uini, yini, yref, Up_cur, Yp_cur, Uf_cur, Yf_cur, lambda_g, lambda_y, Q, R = self.parameters[...]
        g, = self.optimizing_target[...]  # data are stored in list [], notice that ',' cannot be missed

        # To get du
        u_cur = cs.mtimes(Uf_cur, g)
        u_prev = cs.vertcat(uini[-self.u_dim:], cs.mtimes(Uf_cur, g)[:-self.u_dim])
        du = u_cur - u_prev
        
        ## J  =  || Uf * g - ys ||_Q^2 + || uloss ||_R^2
        ## s.t.   Up * g = uini
        ##        Yp * g = yini
        ##        ulb <= u <= uub

        ## objective function in QP form
        if uloss == 'u':
            ## QP problem
            H = Yf_cur.T @ Q @ Yf_cur + Uf_cur.T @ R @ Uf_cur
            f = - Yf_cur.T @ Q @ yref  # - Uf_cur.T @ R @ uref - doesn't have uref
            obj = 0.5 * cs.mtimes(cs.mtimes(g.T, H), g) + cs.mtimes(f.T, g)

        if uloss == 'du':
            ## Not a QP problem
            y = cs.mtimes(Yf_cur, g)
            y_loss = y - yref
            obj = cs.mtimes(cs.mtimes(y_loss.T, Q), y_loss) + cs.mtimes(cs.mtimes(du.T, R), du)

        #### constrains
        # init inequality constrains
        Hc, lbc_ineq, ubc_ineq, ineq_flag = self._init_ineq_cons(ineqconidx, ineqconbd, Up_cur, Yp_cur, du)

        C = []
        lbc, ubc = [], []
        # equal constrains:  Up * g = uini, Yp * g = yini - hard constrain on matching both 'Up' and 'Yp' with no slack variables
        C += [cs.mtimes(Up_cur, g) - uini]
        for i in range(uini.shape[0]):
            lbc += [0]
            ubc += [0]
        C += [cs.mtimes(Yp_cur, g) - yini]
        for i in range(yini.shape[0]):
            lbc += [0]
            ubc += [0]

        # inequality constrains:    ulb <= Uf_u * g <= uub 
        if ineq_flag:
            C += [cs.mtimes(Hc, g)]
            C += [du]
            lbc.extend(lbc_ineq)
            ubc.extend(ubc_ineq)

        # formulate the nlp prolbem
        nlp_prob = {'f': obj, 'x': self.optimizing_target, 'p': self.parameters, 'g': cs.vertcat(*C)}

        self.solver = cs.nlpsol('solver', 'ipopt', nlp_prob, opts)
        self.lbc = lbc
        self.ubc = ubc

    @timer
    def init_RDeePCsolver(self, uloss='u', ineqconidx=None, ineqconbd=None, opts={}):
        """
                              Formulate NLP solver for: Robust DeePC design
            Initialize CasADi nlp solver, !!! only need to formulate the nlp problem at the first time !!!
            treat the updated variables as parameters of the nlp problem
            At each time, update the initial guess of the decision variables and the required parameters
            ----------------------------------------------------------------------------------------------
            nlp_prob = {'f': obj, 'x': optimizing_target, 'p': parameters, 'g': cs.vertcat(*C)}
            self.solver = cs.nlpsol('solver', 'ipopt', nlp_prob, opts)
            sol = self.solver(x0=g_guess, p=parameters, lbg=self.lbc, ubg=self.ubc)
            uloss: the loss of u in objective function
                   "u"  :  ||u||_R^2
                   "uus":  ||u - us||_R^2
                   "du" :  || du ||_R^2
            opts: the config of the solver; max iteration, print level, etc.
                   e.g.:         opts = {
                                            'ipopt.max_iter': 100,  # 50
                                            'ipopt.tol': 1e-5,
                                            'ipopt.print_level': 1,
                                            'print_time': 0,
                                            # 'ipopt.acceptable_tol': 1e-8,
                                            # 'ipopt.acceptable_obj_change_tol': 1e-6,
                                        }
            ----------------------------------------------------------------------------------------------
            g_dim:
                if DeePC:
                    g_dim >= (u_dim + y_dim) * Tini
                if Robust DeePC:
                    g_dim >= u_dim * Tini
                to ensure have enough degree of freedom of nlp problem for g
                this is the equality constraints should be less than decision variables
            -----------------------------------------------------------------------------------------------
        """
        print('>> Robust DeePC design formulating.. This may take a while...')
        if uloss not in ["u", "du"]:
            raise ValueError("uloss should be one of: 'u', 'du'!")
        if self.g_dim <= self.u_dim * self.Tini:
            raise ValueError(f'NLP do not have enough degrees of freedom | Should: g_dim >= u_dim * Tini, but got: {self.g_dim} <= {self.u_dim * self.Tini}!')

        # uini, yini, yref, Up_cur, Yp_cur, Uf_cur, Yf_cur, lambda_g, lambda_y, lambda_u, Q, R = self.parameters[...]      # define parameters and decision variable
        uini, yini, yref, Up_cur, Yp_cur, Uf_cur, Yf_cur, lambda_g, lambda_y, Q, R = self.parameters[...]
        g, = self.optimizing_target[...]  # data are stored in list [], notice that ',' cannot be missed

        if lambda_g is None or lambda_y is None:
            raise ValueError(
                "Do not give value of 'lambda_g' or 'lambda_y', but required in objective function for Robust DeePC!")

        u_cur = cs.mtimes(Uf_cur, g)
        u_prev = cs.vertcat(uini[-self.u_dim:], cs.mtimes(Uf_cur, g)[:-self.u_dim])
        du = u_cur - u_prev
        ## J  =  || Uf * g - ys ||_Q^2 + || uloss ||_R^2 + lambda_y || Yp * g - yini||_2^2 + lambda_g || g ||_2^2
        ## s.t.   Up * g = uini - only hard constrain on Up
        ##        ulb <= u <= uub

        ## objective function
        if uloss == 'u':
            ## QP problem
            H = Yf_cur.T @ Q @ Yf_cur + Uf_cur.T @ R @ Uf_cur + Yp_cur.T @ lambda_y @ Yp_cur + lambda_g
            f = - Yp_cur.T @ lambda_y @ yini - Yf_cur.T @ Q @ yref  # - self.Uf.T @ self.R @ uref
            obj = 0.5 * cs.mtimes(cs.mtimes(g.T, H), g) + cs.mtimes(f.T, g)

        if uloss == 'du':
            ## Not a QP problem
            y = cs.mtimes(Yf_cur, g)
            y_loss = y - yref

            sigma_y = cs.mtimes(Yp_cur, g) - yini
            obj = cs.mtimes(cs.mtimes(y_loss.T, Q), y_loss) + cs.mtimes(cs.mtimes(du.T, R), du) + cs.mtimes(
                cs.mtimes(g.T, lambda_g), g) + cs.mtimes(cs.mtimes(sigma_y.T, lambda_y), sigma_y)

        #### constrains
        # init inequality constrains
        Hc, lbc_ineq, ubc_ineq, ineq_flag = self._init_ineq_cons(ineqconidx, ineqconbd, Up_cur, Yp_cur, du)
        C = []
        lbc, ubc = [], []
        # equal constrains:  Up * g = uini - hard constrain on Up with slack variable on Y
        C += [cs.mtimes(Up_cur, g) - uini]
        for i in range(uini.shape[0]):
            lbc += [0]
            ubc += [0]

        # inequality constrains:    ulb <= Uf_u * g <= uub 
        if ineq_flag:
            C += [cs.mtimes(Hc, g)]
            C += [du]
            lbc.extend(lbc_ineq)
            ubc.extend(ubc_ineq)

        # formulate the nlp prolbem
        nlp_prob = {'f': obj, 'x': self.optimizing_target, 'p': self.parameters, 'g': cs.vertcat(*C)}

        self.solver = cs.nlpsol('solver', 'ipopt', nlp_prob, opts)
        self.lbc = lbc
        self.ubc = ubc

    # def solver_step(self, uini, yini, yref, Up_cur, Uf_cur, Yp_cur, Yf_cur, Q_val, R_val, lambda_g_val, lambda_y_val, lambda_u_val):
    def solver_step(self, uini, yini, yref, Up_cur, Uf_cur, Yp_cur, Yf_cur, Q_val, R_val, lambda_g_val, lambda_y_val):
        """
            solver solve the nlp for one time
            uini, yini:  [array]   | (dim*Tini, 1)
            uref, yref:  [array]   | (dim*Horizon, 1) if sp_change=True
              g0_guess:  [array]   | (T-L+1, 1)
            return:
                u_opt:  the optimized control input for the next Np steps
                 g_op:  the optimized operator g
                  t_s:  solving time
        """

        if yref is None:
            raise ValueError("Did not give value of 'yref', but required in objective function!")
        # parameters = np.concatenate((uini, yini, yref, Up_cur, Uf_cur, Yp_cur, Yf_cur, Q_val, R_val, lambda_g_val, lambda_y_val, lambda_u_val))
        parameters = np.concatenate((uini, yini, yref, Up_cur, Uf_cur, Yp_cur, Yf_cur, Q_val, R_val, lambda_g_val, lambda_y_val))
        g0_guess = np.linalg.pinv(np.concatenate((Up_cur, Yp_cur), axis=0)) @ np.concatenate((uini, yini))

        t0 = time.time()
        sol = self.solver(x0=g0_guess, p=parameters, lbg=self.lbc, ubg=self.ubc)
        t_s = round((time.time() - t0) * 1_000, 3)

        g_opt = sol['x'].full().ravel()
        u_opt = np.matmul(Uf_cur, g_opt)
        return u_opt, g_opt, t_s
    

# Adding lambda_u feature for dealing noisy u input
    # @timer
    # def init_FullRDeePCsolver(self, uloss='u', ineqconidx=None, ineqconbd=None, opts={}):
    #     """
    #         Add both Yp and Up slack variables in RDeePCsolver, where RDeePCsolver only have Yp as slack variable
    #     """
    #     print('>>Full Robust DeePC design formulating.. This may take a while...')
    #     if uloss not in ["u", "du"]:
    #         raise ValueError("uloss should be one of: 'u', 'du'!")
    #     if self.g_dim <= self.u_dim * self.Tini:
    #         raise ValueError(f'NLP do not have enough degrees of freedom | Should: g_dim >= u_dim * Tini, but got: {self.g_dim} <= {self.u_dim * self.Tini}!')

    #     uini, yini, yref, Up_cur, Yp_cur, Uf_cur, Yf_cur, lambda_g, lambda_y, lambda_u, Q, R = self.parameters[...]      # define parameters and decision variable
    #     g, = self.optimizing_target[...]  # data are stored in list [], notice that ',' cannot be missed

    #     if lambda_g is None or lambda_y is None:
    #         raise ValueError(
    #             "Do not give value of 'lambda_g' or 'lambda_y', but required in objective function for Robust DeePC!")

    #     u_cur = cs.mtimes(Uf_cur, g)
    #     u_prev = cs.vertcat(uini[-self.u_dim:], cs.mtimes(Uf_cur, g)[:-self.u_dim])
    #     du = u_cur - u_prev
    #     ## J  =  || Uf * g - ys ||_Q^2 + || uloss ||_R^2 + lambda_y || Yp * g - yini||_2^2 + lambda_g || g ||_2^2
    #     ## s.t.   Up * g = uini - only hard constrain on Up
    #     ##        ulb <= u <= uub

    #     ## objective function
    #     if uloss == 'u':
            
    #         ## QP problem
    #         H = Yf_cur.T @ Q @ Yf_cur + Uf_cur.T @ R @ Uf_cur + Yp_cur.T @ lambda_y @ Yp_cur + Up_cur.T @ lambda_u @ Up_cur + lambda_g
    #         f = - Yp_cur.T @ lambda_y @ yini - Yf_cur.T @ Q @ yref  # - self.Uf.T @ self.R @ uref
    #         obj = 0.5 * cs.mtimes(cs.mtimes(g.T, H), g) + cs.mtimes(f.T, g)

    #     if uloss == 'du':

    #         ## Not a QP problem
    #         y = cs.mtimes(Yf_cur, g)
    #         y_loss = y - yref

    #         sigma_y = cs.mtimes(Yp_cur, g) - yini
    #         sigma_u = cs.mtimes(Up_cur, g) - uini
    #         obj = cs.mtimes(cs.mtimes(y_loss.T, Q), y_loss) + cs.mtimes(cs.mtimes(du.T, R), du) + cs.mtimes(
    #             cs.mtimes(g.T, lambda_g), g) + cs.mtimes(cs.mtimes(sigma_y.T, lambda_y), sigma_y) + cs.mtimes(cs.mtimes(sigma_u.T, lambda_u), sigma_u)
        
    #     #### constrains
    #     # init inequality constrains
    #     Hc, lbc_ineq, ubc_ineq, ineq_flag = self._init_ineq_cons(ineqconidx, ineqconbd, Up_cur, Yp_cur, du)
    #     C = []
    #     lbc, ubc = [], []
    #     # equal constrains:  No hard equality constrains
    #     # inequality constrains:    ulb <= Uf_u * g <= uub 
    #     if ineq_flag:
    #         C += [cs.mtimes(Hc, g)]
    #         C += [du]
    #         lbc.extend(lbc_ineq)
    #         ubc.extend(ubc_ineq)
        
    #     # formulate the nlp prolbem
    #     nlp_prob = {'f': obj, 'x': self.optimizing_target, 'p': self.parameters, 'g': cs.vertcat(*C)}

    #     self.solver = cs.nlpsol('solver', 'ipopt', nlp_prob, opts)
    #     self.lbc = lbc
    #     self.ubc = ubc
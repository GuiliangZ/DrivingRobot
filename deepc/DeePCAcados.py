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
import casadi as cs
import casadi.tools as ctools
from acados_template import AcadosOcp, AcadosOcpSolver

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
                    dim = 1
                    lb = ineqconbd['lbdu']
                    ub = ineqconbd['ubdu']
                else:
                    raise ValueError("%s variable not exist, should be 'u' or/and 'y'!" % varname)

                idx_H = [v + i * dim for i in range(self.Np) for v in idx]
                Hc_list.append(H_all[idx_H, :])
                lbc_list.append(np.tile(lb, self.Np))
                ubc_list.append(np.tile(ub, self.Np))

            Hc = cs.vertcat(*Hc_list)
            lbc = np.concatenate(lbc_list).flatten().tolist()
            ubc = np.concatenate(ubc_list).flatten().tolist()
            ineq_flag = True
        return Hc, lbc, ubc, ineq_flag

    @timer
    def init_DeePCAcadosSolver(self, ineqconidx=None, ineqconbd=None):
        """
        Build an acados QP-OC solver for DeePC:
        min  ½ gᵀ H g + qᵀ g
        s.t. Aeq g = beq
            lbc <= Hc g <= ubc
        where H, Aeq, Hc come from your precomputed Up, Uf, Yp, Yf blocks.
        """
        # define parameters and decision variable
        uini, yini, yref, Up_cur, Yp_cur, Uf_cur, Yf_cur, lambda_g, lambda_y, lambda_u, Q, R = self.parameters[...]
        g, = self.optimizing_target[...]  # data are stored in list [], notice that ',' cannot be missed

        # 2) Build cost matrices
        H =     self.Uf.T @ self.R @ self.Uf \
            + self.Yf.T @ self.Q @ self.Yf
        H = 0.5*(H + H.T)  # ensure symmetry

        # 3) Equality constraints: Up g = uini, Yp g = yini
        Aeq = np.vstack([self.Up, self.Yp])
        neq = Aeq.shape[0]

        # 4) Inequality constraints, if any
        Hc, lbc, ubc, ineq_flag = self._init_ineq_cons(self.ineq_idx, self.ineq_bd)

        # 5) Create acados OCP object
        ocp = AcadosOcp()
        ocp.model = ocp.create_model("deePC_qp")
        # decision variable is g only
        ocp.model.x = cs.SX.sym("g", self.g_dim)

        # 6) cost
        ocp.cost.Q = H
        # q will be updated every solve, so leave as zeros here
        ocp.cost.yref = np.zeros(self.g_dim)

        # 7) eq constraints
        ocp.constraints.constr_expr = Aeq @ ocp.model.x
        ocp.constraints.idxbx = np.arange(neq)    # equality: lb = ub
        ocp.constraints.lbx = np.zeros(neq)
        ocp.constraints.ubx = np.zeros(neq)

        ocp.constraints.C   = Hc
        ocp.constraints.lc  = lbc
        ocp.constraints.uc  = ubc


        # 8) ineq constraints
        if ineq_flag:
            nc = Hc.shape[0]
            ocp.constraints.constr_expr = cs.vertcat(
                ocp.constraints.constr_expr,
                Hc @ ocp.model.x
            )
            ocp.constraints.idxbx = np.arange(neq + nc)
            ocp.constraints.lbx = np.hstack([np.zeros(neq), lbc])
            ocp.constraints.ubx = np.hstack([np.zeros(neq), ubc])

        # 9) solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
        for k, v in opts.items():
            ocp.solver_options.__dict__[k] = v

        # 10) create solver
        self.acados_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')


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
        print('>> DeePC design formulating')
        if uloss not in ["u", "uus", "du"]:
            raise ValueError("uloss should be one of: 'u', 'uus', 'du'!")
        if self.g_dim <= (self.u_dim + self.y_dim) * self.Tini:
            raise ValueError(f'NLP do not have enough degrees of freedom | Should: g_dim > (u_dim + y_dim) * Tini, but got: {self.g_dim} <= {(self.u_dim + self.y_dim) * self.Tini}!')

        # define parameters and decision variable
        uini, yini, yref, Up_cur, Yp_cur, Uf_cur, Yf_cur, lambda_g, lambda_y, lambda_u, Q, R = self.parameters[...]
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
        print('>> Robust DeePC design formulating')
        if uloss not in ["u", "du"]:
            raise ValueError("uloss should be one of: 'u', 'du'!")
        if self.g_dim <= self.u_dim * self.Tini:
            raise ValueError(f'NLP do not have enough degrees of freedom | Should: g_dim >= u_dim * Tini, but got: {self.g_dim} <= {self.u_dim * self.Tini}!')

        uini, yini, yref, Up_cur, Yp_cur, Uf_cur, Yf_cur, lambda_g, lambda_y, lambda_u, Q, R = self.parameters[...]      # define parameters and decision variable
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
            lbc.extend(lbc_ineq)
            ubc.extend(ubc_ineq)

        # formulate the nlp prolbem
        nlp_prob = {'f': obj, 'x': self.optimizing_target, 'p': self.parameters, 'g': cs.vertcat(*C)}

        self.solver = cs.nlpsol('solver', 'ipopt', nlp_prob, opts)
        self.lbc = lbc
        self.ubc = ubc

    @timer
    def init_FullRDeePCsolver(self, uloss='u', ineqconidx=None, ineqconbd=None, opts={}):
        """
            Add both Yp and Up slack variables in RDeePCsolver, where RDeePCsolver only have Yp as slack variable
        """
        print('>> Robust DeePC design formulating')
        if uloss not in ["u", "du"]:
            raise ValueError("uloss should be one of: 'u', 'du'!")
        if self.g_dim <= self.u_dim * self.Tini:
            raise ValueError(f'NLP do not have enough degrees of freedom | Should: g_dim >= u_dim * Tini, but got: {self.g_dim} <= {self.u_dim * self.Tini}!')

        uini, yini, yref, Up_cur, Yp_cur, Uf_cur, Yf_cur, lambda_g, lambda_y, lambda_u, Q, R = self.parameters[...]      # define parameters and decision variable
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
            H = Yf_cur.T @ Q @ Yf_cur + Uf_cur.T @ R @ Uf_cur + Yp_cur.T @ lambda_y @ Yp_cur + + Up_cur.T @ lambda_u @ Up_cur + lambda_g
            f = - Yp_cur.T @ lambda_y @ yini - Yf_cur.T @ Q @ yref  # - self.Uf.T @ self.R @ uref
            obj = 0.5 * cs.mtimes(cs.mtimes(g.T, H), g) + cs.mtimes(f.T, g)

        if uloss == 'du':

            ## Not a QP problem
            y = cs.mtimes(Yf_cur, g)
            y_loss = y - yref

            sigma_y = cs.mtimes(Yp_cur, g) - yini
            sigma_u = cs.mtimes(Up_cur, g) - uini
            obj = cs.mtimes(cs.mtimes(y_loss.T, Q), y_loss) + cs.mtimes(cs.mtimes(du.T, R), du) + cs.mtimes(
                cs.mtimes(g.T, lambda_g), g) + cs.mtimes(cs.mtimes(sigma_y.T, lambda_y), sigma_y) + cs.mtimes(cs.mtimes(sigma_u.T, lambda_u), sigma_u)
        
        #### constrains
        # init inequality constrains
        Hc, lbc_ineq, ubc_ineq, ineq_flag = self._init_ineq_cons(ineqconidx, ineqconbd, Up_cur, Yp_cur, du)
        C = []
        lbc, ubc = [], []
        # equal constrains:  No hard equality constrains
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

    def solver_step(self, uini, yini, yref, Up_cur, Uf_cur, Yp_cur, Yf_cur, Q_val, R_val, lambda_g_val, lambda_y_val, lambda_u_val):
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
            raise ValueError("Do not give value of 'uref' or 'yref', but required in objective function!")
        parameters = np.concatenate((uini, yini, yref, Up_cur, Uf_cur, Yp_cur, Yf_cur, Q_val, R_val, lambda_g_val, lambda_y_val, lambda_u_val))
        g0_guess = np.linalg.pinv(np.concatenate((Up_cur, Yp_cur), axis=0)) @ np.concatenate((uini, yini))

        t_ = time.time()
        sol = self.solver(x0=g0_guess, p=parameters, lbg=self.lbc, ubg=self.ubc)
        t_s = time.time() - t_

        g_opt = sol['x'].full().ravel()
        u_opt = np.matmul(Uf_cur, g_opt)
        return u_opt, g_opt, t_s


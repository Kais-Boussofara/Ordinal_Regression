import warnings

import jax.numpy as jnp
import jax
import jaxopt
from jax import grad, hessian, jit, vmap, config
from functools import partial
from jaxopt import ScipyBoundedMinimize
import jaxlib as jaxlib
import numpy as np
import copy
import scipy
import warnings

do_JIT = False
if not do_JIT:
    print("JIT is disabled")
    config.update("jax_disable_jit", True)

# class RegularizedRegression():
#     def __init__(self, X=None, y=None, u=None):
#         self._check_inputs(X, y)
#
#     ## verify the categories of the outputs
#     def _check_inputs(self, X, y, u=None):
#         # X : input/independent variables
#         # y : output/dependent
#         # u : auxiliary input variables
#         self.M, self.T = X.shape
#         if len(y) != self.T:
#             raise Exception("X should have as many rows as there are elements in y")
#         self.X = X
#         self.y = y
#         self.categories, counts = np.unique(y, return_counts=True)
#         self.L = len(self.categories)
#         self.categories_freq_ = counts/self.T
#         if self.L == 1:
#             raise Exception("nothing to fit: only one category in the data")
#
#         if u is not None:
#             if len(u) != self.T:
#                 raise Exception("u should have T elements")
#         else:  # define artificial classes = 0
#             u = np.zeros(self.T).astype(int)
#         self.u = u
#         self.categories_aux, counts = np.unique(self.u, return_counts=True)
#         self.categories_aux_freq_ = counts/self.T
#     def fit_params(self, params):
#         self.fit_parameters = params
#     def do_fit(self, model="AC+L2"):
#         data = (self.X, self.y, self.u)
#         ## if "weights" are not defined set them to 1
#         if "weight" not in self.fit_parameters:
#             self.fit_parameters.update({"weight": np.ones_like(self.categories_aux, float)})
#         pp = self.fit_parameters
#         if model == "AC+L2":
#             self.theta_hat = theta_hat(pp, data)
#         else:
#             raise Exception("model not defined")

def optimize_ALO_over_gamma(pp_init, data, opt_scheduling, gamma_vec):

    params = copy.deepcopy(pp_init)
    params_in = []
    for k, gg in enumerate(gamma_vec):
        params["gamma"] = gg
        params_in.append(copy.deepcopy(params))
    results = optimize_ALO(params_in, data, opt_scheduling)
    return results

def optimize_ALO(pp_init, data, opt_scheduling, maxrounds=10):
    @jit
    def ALO_partial(pp_var, pp_const, data):
        pp_combined = dict(**pp_var, **pp_const)
        return ALO(pp_combined, data)

    def dict2num(sch, gg, HH):
        # take the elements from the dictionaries gg and HH and construct vector and matrix
        # do not use the elements which are masked as defined in the dictionary sch
        KK = 0  # size of the vector to be optimized
        for elem in sch:
            KK += np.sum(sch[elem])
        grad_ALO = np.zeros((KK,))
        Hess_ALO = np.zeros((KK, KK))
        k_start = 0
        for elem in sch:
            bb = (sch[elem] == np.ones_like(sch[elem]))  ## for boolean indexing
            # k_end = k_start + sch[elem].sum()
            k_end = k_start + (bb * 1).sum()
            # extract gradient elements
            grad_ALO[k_start: k_end] = gg[elem].reshape(len(bb),)[bb]   ## take only elements index with True
            k_start2 = 0
            for elem2 in sch:
                bb2 = (sch[elem2] == np.ones_like(sch[elem2]))  ## for boolean indexing
                # k_end2 = k_start2 + sch[elem2].sum()
                k_end2 = k_start2 + (bb2 * 1).sum()
                # extract Hessian elements
                Hess_ALO[k_start: k_end, k_start2: k_end2] = HH[elem][elem2].reshape((len(bb), len(bb2)))[bb][:, bb2]
                k_start2 = k_end2
            k_start = k_end

        return grad_ALO, Hess_ALO

    input_is_LIST = isinstance(pp_init, list)
    if not input_is_LIST:
        pp_init = [pp_init]  # make it a list

    step = 0.5
    pp_out_list = []
    for pp_init_k in pp_init:
        pp_out = copy.deepcopy(pp_init_k)
        # replace scalars by 1-D np.arrays
        for key in pp_out:
            if isinstance(pp_out[key], float):
                pp_out[key] = np.array(pp_out[key]).reshape(1,)
        J_ref = np.inf
        eps_precision = 1e-6
        for i_round in range(maxrounds):
            # verification if there is improvement
            J_ref_old = J_ref
            J_ref = ALO(pp_out, data)
            HAS_AUX = isinstance(J_ref, tuple)
            if HAS_AUX:
                J_ref = J_ref[0]
            if np.abs(J_ref - J_ref_old) < eps_precision * J_ref:
                break
            # optimize using the order defined in the list opt_scheduling
            for sch in opt_scheduling:
                ## split the dictionaries
                pp_var = {key: pp_out[key] for key in sch}
                pp_const = {key: pp_out[key] for key in set(pp_out.keys()).difference(pp_var.keys())}
                ## calculate gradient and Hessian, the function hessian() does not work here
                gg = grad(ALO_partial, has_aux=HAS_AUX)(pp_var, pp_const, data)
                HH = jax.jacrev(jax.jacrev(ALO_partial, has_aux=HAS_AUX), has_aux=HAS_AUX)(pp_var, pp_const, data)
                if HAS_AUX:
                    gg = gg[0]
                    HH = HH[0]

                ### Here we transform the gradient and the Hessian jax variables into the vector/matrix forms
                grad_ALO, Hess_ALO = dict2num(sch=sch, gg=gg, HH=HH)

                ####  Now, we can execute one step of the Newton (or steepest descent) method
                det_Hess = np.linalg.det(Hess_ALO)
                if det_Hess > 0:
                    ## solve linear equation (for a Hermitian matrix)
                    d_pp = jax.scipy.linalg.solve(Hess_ALO, grad_ALO, assume_a="her")
                else:
                    ## steepest descent
                    warnings.warn("Hessian not invertible: using gradient")
                    d_pp = grad_ALO * step
                ## update the parameters in pp_out
                k_start = 0
                for elem in sch:
                    bb = (sch[elem] == np.ones_like(sch[elem]))  ## for boolean indexing
                    k_end = k_start + (bb * 1).sum()
                    #pp_out[elem][bb] -= d_pp[k_start: k_end]    ## update elements
                    pp_out[elem]=pp_out[elem].at[bb].add(-d_pp[k_start: k_end])
                    k_start = k_end

        # find the Hessian over variables which were used in optimization
        ## split the dictionaries
        # sch_tot = {}
        # for sch in opt_scheduling:
        #     sch_tot.update(sch)     # will contain information about all the elements which were optimized
        # pp_var = {key: pp_out[key] for key in sch_tot}
        # pp_const = {key: pp_out[key] for key in set(pp_out.keys()).difference(pp_var.keys())}
        #
        # gg = grad(ALO_partial, has_aux=HAS_AUX)(pp_var, pp_const, data)
        # HH = jax.jacrev(jax.jacrev(ALO_partial, has_aux=HAS_AUX), has_aux=HAS_AUX)(pp_var, pp_const, data)
        # if HAS_AUX:
        #     gg = gg[0]
        #     HH = HH[0]
        # grad_out, Hess_out = dict2num(sch=sch_tot, gg=gg, HH=HH)
        pp_out_list.append(copy.deepcopy(pp_out))
    if input_is_LIST:
        pp_out = pp_out_list
    else:
        pp_out = pp_out_list[0]
    return pp_out

def optimize_ALO_jax(pp_init, data, opt_scheduling, maxrounds=10):
    # use Jax optimization routine
    @jit
    def ALO_partial(pp_var):
        pp_combined = dict(**pp_var, **pp_const)
        return ALO(pp_combined, data)

    pp_out = copy.deepcopy(pp_init)
    for i_round in range(maxrounds):
        for sch in opt_scheduling:
            ## split the dictionaries: pp_var will be optimized, pp_const does not change
            pp_var = {}
            pp_const = {}
            ## these are the lower and upper bounds on the variables in pp_var
            pp_var_lower = {}
            pp_var_upper = {}

            for key, value in pp_out.items():
                if key in sch:
                    pp_var[key] = value
                    ## this will enable unconstrained optimization
                    inf_value = jnp.zeros_like(pp_var[key]).astype(float)
                    inf_value  = inf_value.at[sch[key] == 1].set(jnp.inf)       # boolean indexing
                    ### define (-inf, inf) box constraint for all values which are not to be fixed
                    pp_var_lower[key] = pp_var[key]-inf_value
                    pp_var_upper[key] = pp_var[key]+inf_value
                else:
                    pp_const[key] = value

            ### Here we optimize with box-constraints
            pp_var_bounds = (pp_var_lower, pp_var_upper)
            lbfgsb = jaxopt.LBFGSB(fun=ALO_partial)
            solver_state = lbfgsb.init_state(pp_var, bounds=pp_var_bounds)
            pp_out_opt = lbfgsb.update(pp_var, state=solver_state, bounds=pp_var_bounds).params
            pp_out.update(pp_out_opt)

    return pp_out

@jit
def ALO(pp, data):
    X, y, u = data
    xi = jnp.ones_like(y)
    hfa = u["hfa"] if "hfa" in u else 0
    M, T = X.shape

    theta_hat_out = theta_hat(pp, data)
    H = hessian(J_obj)(theta_hat_out, data, pp)
    H_inv = jax.scipy.linalg.inv(H)
    a = jnp.sum(jnp.multiply(H_inv @ X, X), axis=0)

    z_hat = X.T @ theta_hat_out + pp["eta"] * hfa
    ######

    ell_dot = vmap(grad(loss_fun), in_axes=[0, 0, 0, None])(z_hat, y, xi, pp)
    ell_ddot = vmap(hessian(loss_fun), in_axes=[0, 0, 0, None])(z_hat, y, xi, pp)

    # ell_dot2 = jnp.array([grad(loss_fun)(z_hat[i], y[i], xi[i], pp) for i in range(T)])
    # ell_ddot2 = jnp.array([hessian(loss_fun)(z_hat[i], y[i], xi[i], pp) for i in range(T)])

    z_hat_approx = z_hat + ell_dot * a /(1 - ell_ddot * a)
    pred = vmap(validation_fun, in_axes=[0, 0, 0, None])(z_hat_approx, y, xi, pp)

    return jnp.sum(pred)/len(y), pred


def GALO(pp, data, leave_out):
    ## generalized ALO, uses set-of-sets ii to indicate which elements are left-out
    ## becomes ALO when ii=[[0], [1], [2], ... , [T-1]]
    X, y, u = data
    xi = pp["weight"][u["category"]]
    theta_hat_out = theta_hat(pp, data)
    H = hessian(J_obj)(theta_hat_out, X, y, u, pp)
    H_inv = jax.scipy.linalg.inv(H)
    # A = [X[:, ii].T @ H_inv @ X[:, ii]  for ii in leave_out]

    z_hat = X.T @ theta_hat_out + pp["eta"] * u["hfa"]
    ell_dot = vmap(grad(loss_fun), in_axes=[0, 0, 0, None])(z_hat, y, xi, pp)
    ell_ddot = vmap(hessian(loss_fun), in_axes=[0, 0, 0, None])(z_hat, y, xi, pp)

    pred = 0.0
    for ii in leave_out:
        A = X[:, ii].T @ H_inv @ X[:, ii]
        z_hat_approx = z_hat[ii] + A @ jnp.linalg.inv(jnp.eye(len(ii)) - jnp.diag(ell_ddot[ii]) @ A ) @ ell_dot[ii]
        pred += vmap(validation_fun, in_axes=[0, 0, 0, None])(z_hat_approx, y[ii], xi[ii], pp).mean()
    return jnp.sum(pred)/len(leave_out)

def LOO(pp, data):
    X, y, u = data
    hfa = u["hfa"]
    M, T = X.shape

    pred = jnp.zeros(T)
    tt_full = jnp.arange(T)
    t_keep = tt_full
    for t in range(T):
        x_test = X[:, t]
        t_keep = jnp.delete(tt_full, t)

        theta_hat_out = theta_hat(pp, data, t_keep)
        z_hat = jnp.dot(x_test, theta_hat_out) + pp["eta"] * hfa[t]
        pred_result = validation_fun(z_hat, y[t], 1.0, pp)
        pred = pred.at[t].set(pred_result)

    return jnp.mean(pred), pred

def LOO_explicit(pp, data):
    X, y, u = data
    hfa = u["hfa"]
    M, T = X.shape

    pred = jnp.zeros(T)
    u_train = {f : 0 for f in u}  # initialize the dictionary
    for t in range(T):
        X_train = jnp.delete(X, t, axis=1)
        y_train = jnp.delete(y, t)
        for f in u_train:
            u_train[f] = jnp.delete(u[f], t)

        x_test = X[:, t]
        theta_hat_out = theta_hat(pp, (X_train, y_train, u_train))
        z_hat = jnp.dot(x_test, theta_hat_out) + pp["eta"] * hfa[t]
        pred_result = validation_fun(z_hat, y[t], 1.0, pp)
        pred = pred.at[t].set(pred_result)

    return jnp.mean(pred), pred


@jit
def theta_hat(pp, data, t_keep=None):
    maxiter = 10
    X, _, _ = data
    M, T = X.shape

    solver = jaxopt.BFGS(fun=J_obj, maxiter=maxiter, implicit_diff=True)
    theta_init = jnp.zeros(M)
    res = solver.run(theta_init, data, pp, t_keep)
    return res.params

@jit
def J_obj(theta, data, pp, t_keep=None):
    X, y, u = data
    xi = jnp.ones_like(y)
    hfa = u["hfa"] if "hfa" in u else 0

    z = X.T @ theta
    z += pp["eta"] * hfa
    loss = vmap(loss_fun, in_axes=[0, 0, 0, None])(z, y, xi, pp)      # vectorization
    return jnp.sum(loss[t_keep]) + regularization_fun(theta, pp)

@jit
def regularization_fun(theta, params):
    REGULARIZATION_FUN = "L2"
    if REGULARIZATION_FUN == "L2":
        # this is the ridge regularization function
        gamma = params["gamma"]
        return jnp.sum(0.5 * gamma * theta**2)
    elif REGULARIZATION_FUN == "XXXX":
        None
    else:
        raise Exception("regularization function undefined")

@jit
def validation_fun(z, y, xi, params):
    return logarithmic_loss_CL(z, y, 1.0, params)

@jit
def loss_fun(z, y, xi, params):
    # this is the scalar ordinal model (should not be vectorized after being called)
    # functions on the list loss_functions_list are selected using the value of fun_switch["LOSS_FUN"]

    loss_functions_list = [logarithmic_loss_CL, fivb_loss]
    return jax.lax.switch(params["LOSS_FUN"], loss_functions_list,z, y, xi, params)

@jit
def fivb_loss(z, y, xi, params):
    r = params["r"]
    Ar = params["Ar"]
    rr = Ar @ r
    c = params["c"]
    Ac = params["Ac"]
    cc = Ac @ c
    # Nc_tot = Ac.shape[0]
    dr = - jnp.diff(rr)
    zz = z + cc
    psi = jax.scipy.stats.norm.cdf(zz) * zz + jax.scipy.stats.norm.pdf(zz)

    #return jnp.dot(psi[:-1], dr) + jnp.dot(rr[-1] - rr[y], z)
    return jnp.dot(psi, dr) + jnp.dot(rr[-1] - rr[y], z)

@jit
def logarithmic_loss_AC(z, y, xi, params):
    ## this is the scalar ordinal model (should not be vectorized after being called)
    alpha = params["alpha"]
    delta = params["delta"]
    v = alpha + z * delta
    # negated log-loss
    ell = jax.scipy.special.logsumexp(v) - v[y]
    return jnp.dot(ell, xi)

@jit
def logarithmic_loss_CL(z, y, xi, params):
    ## this is the scalar ordinal model (should not be vectorized after being called)
    c = params["c"]
    Ac = params["Ac"]
    cc = Ac @ c

    Nc_tot = Ac.shape[0]

    # v = z + Ac @ c
    # ell = ell_tmp_log[y]

    is_positive = (z > 0) * 1
    v = z * (1 - 2 * is_positive) + cc
    ell_tmp_log = ell_CL_log(v)
    ## we are exploiting the fact that ell_y(z) = ell_{L-1-y}(-z)
    ## this should be generalized for arbitrary c !!!!
    ell = ell_tmp_log[y] * (1 - is_positive) + ell_tmp_log[Nc_tot - y] * is_positive

    return jnp.dot(ell, xi)

@jit
def ell_CL_log(zz):
    ## implement ell_CL using logarithms
    ## log(cdf(a)-cdf(b)) = logcdf(a) + log[1-exp(logcdf(a)-logcdf(b))]
    ## since z[-1]=-inf, and z[L] = inf, we have: logcdf(z[-1]) = -inf and logcdf(z[L]) = 0

    CDF = "Gauss"
    # CDF = "logistic"
    if CDF == "Gauss":
        uu = jax.scipy.stats.norm.logcdf(zz)
    elif CDF == "logistic":
        uu = jax.scipy.stats.logistic.logcdf(zz)
    else:
        raise Exception("CDF undefined")

    Nc_tot = len(uu)
    A1 = jnp.eye(Nc_tot, Nc_tot + 1, 0)
    a = uu @ A1

    A2 = -jnp.eye(Nc_tot, Nc_tot , 0) + jnp.eye(Nc_tot, Nc_tot, -1)
    expd = jnp.exp(-uu @ A2)

    A3 = jnp.eye(Nc_tot, Nc_tot + 1, 1)
    b = jnp.log(1 - expd @ A3)

    return - (a + b)

# @jit
# def ell_CL(zz):
#     Nc_tot = 5
#     Acdf = jnp.eye(Nc_tot, Nc_tot+1, 0) - jnp.eye(Nc_tot, Nc_tot+1, 1)
#     bb = jnp.eye(1, Nc_tot+1, Nc_tot)
#
#     CDF = "Gauss"
#     # CDF = "logistic"
#     if CDF == "Gauss":
#         uu = jax.scipy.stats.norm.cdf(zz)
#     elif CDF == "logistic":
#         uu = jax.scipy.stats.logistic.cdf(zz)
#     else:
#         raise Exception("CDF undefined")
#
#     # cdf_diff = uu[None, :] @ A + bb[None, :]
#     cdf_diff = uu @ Acdf + bb
#     val = -jnp.log(cdf_diff)
#
#     return val.ravel()

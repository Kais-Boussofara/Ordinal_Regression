import numpy as np
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import scipy
import jax
import RegularizedRegression
from joblib import Parallel, delayed
import datetime
import time

from sklearn.datasets import load_breast_cancer

import manage_data

########################################################################
c_CL = np.array([-np.inf, -1.06, -0.394, 0, 0.394, 1.06, np.inf])
alpha_AC = np.array([0.0, -0.2, -0.5, -0.5, -0.2, 0.0])
delta_AC = np.array([0.0, 0.3, 0.45, 0.55, 0.7, 1.0])

########################################################################
def AC_model(z, alpha, delta):
    v = alpha[None, :] + z[:, None] * delta[None, :]
    # negated log-loss
    ell = scipy.special.logsumexp(v, axis=1) - v
    val = np.exp(-ell)
    return val

########################################################################
def CLM_model(z, c):
    bb = z[:, None] - c[None, :]
    val = -np.diff(stats.norm.cdf(bb), axis=1)
    return val


########################################################################
def synthetic_data(M=10, T=1000, var_theta=1):
    theta = np.random.normal(loc=0, scale=np.sqrt(var_theta), size=M)
    teams = np.arange(M)
    L = len(c_CL) - 1
    outputs = np.arange(L)
    X = np.zeros((M, T)).astype(int)
    y = np.zeros(T).astype(int)
    for t in range(T):
        selected_teams = np.random.choice(teams, size=2, replace=False)
        X[selected_teams, t] = [1, -1]
        z = np.array([(X[:,t] * theta).sum()])
        PMF = CLM_model(z, c_CL)
        y[t] = np.random.choice(outputs, p=PMF.ravel())

    return theta, X, y


########################################################################
def main():
    # data_type = "sport"
    data_type = "synthetic"
    if data_type == "sport":
        sport = "EPL"
        season_list = list(range(2001, 2001 + 1))  # testing
        # season_list = list(range(1995, 2018 + 1))
        if not isinstance(season_list, list):
            season_list = [season_list]
        XX_list = list()
        yy_list = list()
        for season in season_list:
            # import data
            my_season = manage_data.get_season(sport, season, print_info=True)
            # define the results
            my_season.define_results(result_cat="league")
            # define the regression format
            my_season.regression_format()

            XX_list.append(my_season.X)
            yy_list.append(my_season.y)
    elif data_type == "synthetic":
        M = 4
        T = 50
        var_theta = 0.1
        theta, XX, yy = synthetic_data(M=M, T=T, var_theta=var_theta)

    ## Ordinal regression
    L_lambd = 10
    lambd_list = np.logspace(-2, 2, L_lambd)
    # lambd_list = [1.0]
    alo = np.zeros(L_lambd)
    alo_opt = np.zeros(L_lambd)
    for iL, lambd_ord in enumerate(lambd_list):
        # the model is defined inside the function loss_fun, regularization_fun, and prediction_fun
        # which are inside the module RegularizedRegression
        params = {}
        ## define data
        data = (XX, yy, np.zeros_like(yy, int))
        L = len(np.unique(yy))  # number of possibile outcomes
        ## define model's parameters
        params["alpha"] = np.zeros(L)
        params["delta"] = np.linspace(0, 1, L)
        params["gamma"] = np.array([lambd_ord])
        params["weight"] = np.array([1.0])
        params["eta"] = 0
        params["LOSS_FUN"] = 0
        params["c"]=np.array([-1.06,-0.394,0,0.394,1.06])
        params["Ac"]=np.eye(L-1)
        params["r"] = jnp.array([2.0,1.5,1.0,-1.0,-1.5,-2.0])
        params["Ar"] = jnp.eye(L)

        theta_hat = RegularizedRegression.theta_hat(params, data)
        print(theta_hat)
        alo[iL], pred = RegularizedRegression.ALO(params, data)

        opt_scheduling = []       # list of dict
        # alpha_mask = np.ones(L, int)
        # alpha_mask[0] = 0   # this element will not be optimized
        # opt_scheduling.append({"alpha": alpha_mask})
        # delta_mask = np.ones(L, int)
        #delta_mask[[0,L-1]] = 0  # this element will not be optimized
        #opt_scheduling.append({"delta": delta_mask})



        r_mask=np.ones(L,int)
        opt_scheduling.append({"r": r_mask})
        # xi_mask = np.ones(GLIR.L, int)
        # xi_mask[0] = 0 # this element will not be optimized
        # opt_scheduling.append({"weight": xi_mask})

        pp_opt = RegularizedRegression.optimize_ALO(params, data, opt_scheduling)
        alo_opt[iL], _ = RegularizedRegression.ALO(pp_opt, data)

    print("alo:", alo)
    print("alo_opt:", alo_opt)

if __name__ == "__main__":
    main()

# Ordinal Regularized Regression with ALO (Synthetic Sports Outcomes for Soccer)

## Overview
This project simulates **ordinal match outcomes** using a **Cumulative Link Model (CLM)** and fits an **ordinal regularized regression** model. Hyperparameters (e.g., regularization strength `lambda`) are selected using **ALO** (Approximate Leave-One-Out) and optionally refined by **optimizing ALO** over selected model parameters.

Main features:
- Synthetic match data generation (team skill latent vector `theta`)
- Ordinal outcome generation via CLM cutpoints
- Regularized ordinal regression estimation (`RegularizedRegression.theta_hat`)
- Model selection with ALO (`RegularizedRegression.ALO`)
- Optional parameter tuning by ALO optimization (`RegularizedRegression.optimize_ALO`)


# Code provided by Pauli-Isosomppi on github: https://github.com/Pauli-Isosomppi/Heston-model.git


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import exp, log, sqrt
from statistics import NormalDist
from py_vollib_vectorized import vectorized_implied_volatility as implied_vol

# ============================================================================
# HESTON MODEL - STOCHASTIC VOLATILITY SIMULATION
# ============================================================================

# Parameters
# simulation dependent
S0 = 100.0             # asset price
T = 1.0                # time in years
r = 0.02               # risk-free rate
N = 250                # number of time steps in simulation
M = 10000               # number of simulations

# Heston dependent parameters
kappa = 2              # rate of mean reversion of variance under risk-neutral dynamics
theta = 0.04        # long-term mean of variance under risk-neutral dynamics
v0 = 0.04           # initial variance under risk-neutral dynamics
rho = -0.7              # correlation between returns and variances under risk-neutral dynamics
sigma = 0.5            # volatility of volatility

print(f"Long-term variance (theta): {theta}")
print(f"Initial variance (v0): {v0}")


def heston_model_sim(S0, v0, rho, kappa, theta, sigma, T, N, M):
    """
    Heston Model Simulation under Risk-Neutral Measure
    
    Inputs:
     - S0, v0: initial parameters for asset and variance
     - rho   : correlation between asset returns and variance
     - kappa : rate of mean reversion in variance process
     - theta : long-term mean of variance process
     - sigma : vol of vol / volatility of variance process
     - T     : time of simulation
     - N     : number of time steps
     - M     : number of scenarios / simulations
    
    Outputs:
    - asset prices over time (numpy array)
    - variance over time (numpy array)
    """
    # initialise other parameters
    dt = T / N
    mu = np.array([0, 0])
    cov = np.array([[1, rho],
                    [rho, 1]])

    # arrays for storing prices and variances
    S = np.full(shape=(N+1, M), fill_value=S0)
    v = np.full(shape=(N+1, M), fill_value=v0)

    # sampling correlated brownian motions under risk-neutral measure
    Z = np.random.multivariate_normal(mu, cov, (N, M))

    for i in range(1, N+1):
        S[i] = S[i-1] * np.exp((r - 0.5*v[i-1])*dt + np.sqrt(v[i-1] * dt) * Z[i-1, :, 0])

        # Enforce non-negative variance (active)
        v[i] = np.maximum(v[i-1] + kappa*(theta - v[i-1])*dt + sigma*np.sqrt(v[i-1]*dt)*Z[i-1, :, 1], 0)

        # No non-negativity enforcement (inactive)
        # v[i] = v[i-1] + kappa*(theta - v[i-1])*dt + sigma*np.sqrt(v[i-1]*dt)*Z[i-1, :, 1]
    
    return S, v


def heston_model_sim_stable(S0, v0, rho, kappa, theta, sigma, T, N, M):
    """Heston simulation with variance floored at zero for clean comparisons."""
    dt = T / N
    mu = np.array([0, 0])
    cov = np.array([[1, rho],
                    [rho, 1]])

    S = np.full(shape=(N+1, M), fill_value=S0)
    v = np.full(shape=(N+1, M), fill_value=v0)
    Z = np.random.multivariate_normal(mu, cov, (N, M))

    for i in range(1, N+1):
        v_prev = np.maximum(v[i-1], 0)
        S[i] = S[i-1] * np.exp((r - 0.5*v_prev)*dt + np.sqrt(v_prev * dt) * Z[i-1, :, 0])
        v[i] = np.maximum(v_prev + kappa*(theta - v_prev)*dt + sigma*np.sqrt(v_prev*dt)*Z[i-1, :, 1], 0)

    return S, v


def black_scholes_sim(S0, sigma_bs, T, N, M):
    """Black-Scholes simulation with constant volatility."""
    dt = T / N
    Z = np.random.normal(size=(N, M))

    S = np.full(shape=(N+1, M), fill_value=S0)

    for i in range(1, N+1):
        S[i] = S[i-1] * np.exp((r - 0.5*sigma_bs**2)*dt + sigma_bs*np.sqrt(dt) * Z[i-1])

    return S


def black_scholes_call_price(S0, K, T, r, sigma):
    """Closed-form Black-Scholes price for a European call."""
    normal_dist = NormalDist()
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S0 * normal_dist.cdf(d1) - K * exp(-r * T) * normal_dist.cdf(d2)


# ============================================================================
# SECTION 1: SIMULATE WITH DIFFERENT CORRELATIONS (rho = 0.98 and rho = -0.98)
# ============================================================================

rho_p = 0.98
rho_n = -0.98
S_p, v_p = heston_model_sim(S0, v0, rho_p, kappa, theta, sigma, T, N, M)
S_n, v_n = heston_model_sim(S0, v0, rho_n, kappa, theta, sigma, T, N, M)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
time = np.linspace(0, T, N+1)

# Plot only 10 sample paths
n_paths = 10
ax1.plot(time, S_p[:, :n_paths], alpha=0.7)
ax1.set_title(f'Heston Model Asset Prices (ρ = 0.98) - {n_paths} paths')
ax1.set_xlabel('Time')
ax1.set_ylabel('Asset Prices')

ax2.plot(time, v_p[:, :n_paths], alpha=0.7)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Zero Line')
ax2.set_title(f'Heston Model Variance Process (ρ = 0.98) - {n_paths} paths')
ax2.set_xlabel('Time')
ax2.set_ylabel('Variance')
# Set y-axis limits to show negative values
finite_v = v_p[:, :n_paths][np.isfinite(v_p[:, :n_paths])]
if finite_v.size > 0:
    ax2.set_ylim([np.min(finite_v)*1.1, np.max(finite_v)*1.1])
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()


# ============================================================================
# SECTION 2: COMPARE HESTON MODEL WITH BLACK-SCHOLES
# ============================================================================

# Black-Scholes volatility chosen as sigma = sqrt(theta) = 20%
sigma_bs = np.sqrt(theta)

# Stable Heston simulation for comparison with Black-Scholes
S_heston_cmp, v_heston_cmp = heston_model_sim_stable(S0, v0, -0.7, kappa, theta, sigma, T, N, M)
S_bs = black_scholes_sim(S0, sigma_bs, T, N, M)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
time = np.linspace(0, T, N+1)
n_paths = 10

ax1.plot(time, S_heston_cmp[:, :n_paths], alpha=0.65)
ax1.plot(time, S_bs[:, :n_paths], alpha=0.65, linestyle='--')
ax1.set_title('Heston vs Black-Scholes: sample paths')
ax1.set_xlabel('Time')
ax1.set_ylabel('Asset Price')
ax1.legend(['Heston', 'Black-Scholes'])

ax2.hist(S_heston_cmp[-1], bins=50, alpha=0.55, label='Heston', density=False)
ax2.hist(S_bs[-1], bins=50, alpha=0.55, label='Black-Scholes', density=False)
ax2.set_title('Terminal price distribution (frequencies)')
ax2.set_xlabel('$S_T$')
ax2.set_ylabel('Frequency')
ax2.legend()

plt.tight_layout()


# ============================================================================
# SECTION 3: OPTION PRICING AND IMPLIED VOLATILITY SMILE
# ============================================================================

rho = -0.7
S, v = heston_model_sim(S0, v0, rho, kappa, theta, sigma, T, N, M)

# Set strikes and compute MC option price for different strikes
K = np.arange(70, 130, 2)

# Task 3: compare European call prices (Heston vs Black-Scholes)
K_task3 = np.array([90, 100, 110])
sigma_bs = np.sqrt(theta)
heston_discounted_payoffs_task3 = [
    np.exp(-r * T) * np.maximum(S[-1] - k, 0) for k in K_task3
]
heston_calls_task3 = np.array([
    np.mean(payoff) for payoff in heston_discounted_payoffs_task3
])
heston_std_task3 = np.array([
    np.std(payoff, ddof=1) for payoff in heston_discounted_payoffs_task3
])
heston_se_task3 = heston_std_task3 / np.sqrt(M)
bs_calls_task3 = np.array([
    black_scholes_call_price(S0, k, T, r, sigma_bs) for k in K_task3
])

print("\nTask 3 - European Call Price Comparison (K = 90, 100, 110)")
print("Strike | Heston (MC) | BS closed-form | MC std dev | MC std error")
for k, h_price, bs_price, h_std, h_se in zip(
    K_task3,
    heston_calls_task3,
    bs_calls_task3,
    heston_std_task3,
    heston_se_task3,
):
    print(f"{k:>6} | {h_price:>11.4f} | {bs_price:>14.4f} | {h_std:>10.4f} | {h_se:>12.4f}")

# Monte Carlo option prices
puts = np.array([np.exp(-r*T)*np.mean(np.maximum(k - S[-1], 0)) for k in K])
calls = np.array([np.exp(-r*T)*np.mean(np.maximum(S[-1] - k, 0)) for k in K])

# Compute implied volatilities
put_ivs = implied_vol(puts, S0, K, T, r, flag='p', q=0, return_as='numpy', on_error='ignore')
call_ivs = implied_vol(calls, S0, K, T, r, flag='c', q=0, return_as='numpy', on_error='ignore')

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(K, call_ivs, label=r'IV calls', marker='o', linestyle='-')
ax.plot(K, put_ivs, label=r'IV puts', marker='s', linestyle='-')

ax.set_ylabel('Implied Volatility')
ax.set_xlabel('Strike')
ax.set_title('Implied Volatility Smile from Heston Model (ρ = -0.7)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Simulazione completata!")
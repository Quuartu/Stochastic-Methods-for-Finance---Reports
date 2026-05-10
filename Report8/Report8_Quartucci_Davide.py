# Code provided by Pauli-Isosomppi on github: https://github.com/Pauli-Isosomppi/Heston-model.git

#!pip install py_vollib_vectorized # Uncomment only in google colab 

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
import os
from math import exp, log, sqrt
from statistics import NormalDist
from py_vollib_vectorized import vectorized_implied_volatility as implied_vol
from scipy.stats import skew, kurtosis
# ============================================================================
# RANDOM SEED FOR REPRODUCIBILITY
# ============================================================================
np.random.seed(42) # Any integer seed works; 42 used for reproducibility

# ============================================================================
# OUTPUT DIRECTORY SETUP
# ============================================================================
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}/")

fig_counter = 0  # Counter for figure filenames

# ============================================================================
# HESTON MODEL - STOCHASTIC VOLATILITY SIMULATION
# ============================================================================

# Parameters
# simulation dependent
S0 = 100.0             # asset price
T = 1.0                # time in years
r = 0.02               # risk-free rate
N = 250                # number of time steps in simulation
M = 10000              # number of simulations

# Heston dependent parameters
kappa = 2.0            # rate of mean reversion of variance under risk-neutral dynamics
theta = 0.04           # long-term mean of variance under risk-neutral dynamics
v0 = 0.04              # initial variance under risk-neutral dynamics
rho = -0.7             # correlation between returns and variances under risk-neutral dynamics
sigma = 0.5            # volatility of volatility

print("\n" + "="*70)
print("HESTON MODEL - STOCHASTIC VOLATILITY SIMULATION")
print("="*70)
print(f"\nInitial Parameters:")
print(f"  Asset price (S0): {S0}")
print(f"  Time to maturity (T): {T} year(s)")
print(f"  Risk-free rate (r): {r}")
print(f"  Number of time steps (N): {N}")
print(f"  Number of simulations (M): {M}")
print(f"\nHeston Model Parameters:")
print(f"  Mean reversion rate (kappa): {kappa}")
print(f"  Long-term variance (theta): {theta}")
print(f"  Initial variance (v0): {v0}")
print(f"  Correlation (rho): {rho}")
print(f"  Volatility of volatility (sigma): {sigma}")
print("="*70)


# Based on Paul (Pauli-Isosomppi) Heston simulation code, adapted here.
# Improvement: full truncation is applied before each step to safely handle negative variance states.
def heston_model_sim(S0, v0, rho, kappa, theta, sigma, T, N, M):
    """Heston simulation with full truncation: max(v,0) applied before each step."""
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
# TASK 1: SIMULATE S AND v WITH BASE PARAMETERS — 10 SAMPLE PATHS EACH
# ============================================================================
print("\n" + "="*70)
print("TASK 1: SAMPLE PATHS OF S_t AND v_t (BASE PARAMETERS)")
print("="*70)

S_base, v_base = heston_model_sim(S0, v0, rho, kappa, theta, sigma, T, N, M)
time = np.linspace(0, T, N+1)
n_paths = 10

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

colors = plt.cm.tab10(np.linspace(0, 1, n_paths))
for i in range(n_paths):
    ax1.plot(time, S_base[:, i], alpha=0.8, color=colors[i], linewidth=1.0)
ax1.set_title(r'Asset Price $S_t$ — 10 Sample Paths ($\rho = -0.7$)')
ax1.set_xlabel('Time')
ax1.set_ylabel('$S_t$')
ax1.grid(True, alpha=0.3)

for i in range(n_paths):
    ax2.plot(time, v_base[:, i], alpha=0.8, color=colors[i], linewidth=1.0)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.2, label='Zero threshold')
ax2.set_title(r'Variance Process $v_t$ — 10 Sample Paths ($\rho = -0.7$)')
ax2.set_xlabel('Time')
ax2.set_ylabel('$v_t$')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
fig_counter += 1
plt.savefig(os.path.join(output_dir, f"fig_{fig_counter:02d}_task1_sample_paths.png"), dpi=150, bbox_inches='tight')
print(f"[Figure {fig_counter}] Saved: Task 1 — S_t and v_t sample paths")
plt.close()


# ============================================================================
# TASK 2: COMPARE TERMINAL DISTRIBUTIONS — HESTON VS BLACK-SCHOLES
# ============================================================================
print("\n" + "="*70)
print("TASK 2: COMPARISON OF TERMINAL DISTRIBUTIONS")
print("="*70)

# Black-Scholes volatility: sigma_bs = sqrt(theta) = 20%
sigma_bs = np.sqrt(theta)

S_heston_cmp, v_heston_cmp = heston_model_sim(S0, v0, rho, kappa, theta, sigma, T, N, M)
S_bs = black_scholes_sim(S0, sigma_bs, T, N, M)

# --- Plot: sample paths + terminal distributions ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

heston_colors = ['steelblue'] * n_paths
bs_colors     = ['darkorange'] * n_paths

for i in range(n_paths):
    ax1.plot(time, S_heston_cmp[:, i], alpha=0.7, color='steelblue', linewidth=1.0,
             label='Heston' if i == 0 else "")
    ax1.plot(time, S_bs[:, i], alpha=0.7, color='darkorange', linewidth=1.0,
             linestyle='--', label='Black-Scholes' if i == 0 else "")
ax1.set_title('Heston vs Black-Scholes: Sample Paths')
ax1.set_xlabel('Time')
ax1.set_ylabel('$S_t$')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.hist(S_heston_cmp[-1], bins=60, alpha=0.55, label='Heston', density=True, color='steelblue')
ax2.hist(S_bs[-1], bins=60, alpha=0.55, label='Black-Scholes', density=True, color='darkorange')
ax2.set_title('Terminal Distribution $S_T$ (density)')
ax2.set_xlabel('$S_T$')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig_counter += 1
plt.savefig(os.path.join(output_dir, f"fig_{fig_counter:02d}_task2_heston_vs_bs.png"), dpi=150, bbox_inches='tight')
print(f"[Figure {fig_counter}] Saved: Task 2 — Heston vs Black-Scholes")
plt.close()

# --- Statistical moments ---
ST_h  = S_heston_cmp[-1]
ST_bs = S_bs[-1]
log_ret_h  = np.log(ST_h  / S0)
log_ret_bs = np.log(ST_bs / S0)

print("\n" + "-"*70)
print("Statistics on Terminal Prices S_T")
print("-"*70)
print(f"{'Metric':<20} | {'Heston':<15} | {'Black-Scholes':<15}")
print("-"*70)
print(f"{'Mean':<20} | {np.mean(ST_h):>15.4f} | {np.mean(ST_bs):>15.4f}")
print(f"{'Variance':<20} | {np.var(ST_h,  ddof=1):>15.4f} | {np.var(ST_bs, ddof=1):>15.4f}")
print(f"{'Skewness':<20} | {skew(ST_h):>15.4f} | {skew(ST_bs):>15.4f}")
print(f"{'Kurtosis':<20} | {kurtosis(ST_h):>15.4f} | {kurtosis(ST_bs):>15.4f}")
print("\n" + "-"*70)
print("Statistics on Log-Returns ln(S_T/S0)")
print("-"*70)
print(f"{'Metric':<20} | {'Heston':<15} | {'Black-Scholes':<15}")
print("-"*70)
print(f"{'Mean':<20} | {np.mean(log_ret_h):>15.4f} | {np.mean(log_ret_bs):>15.4f}")
print(f"{'Variance':<20} | {np.var(log_ret_h,  ddof=1):>15.4f} | {np.var(log_ret_bs, ddof=1):>15.4f}")
print(f"{'Skewness':<20} | {skew(log_ret_h):>15.4f} | {skew(log_ret_bs):>15.4f}")
print(f"{'Kurtosis':<20} | {kurtosis(log_ret_h):>15.4f} | {kurtosis(log_ret_bs):>15.4f}")
print("="*70)


# ============================================================================
# TASK 3: EUROPEAN CALL PRICES — HESTON MC VS BLACK-SCHOLES CLOSED-FORM
# ============================================================================
print("\n" + "="*70)
print("TASK 3: EUROPEAN CALL PRICE COMPARISON (K = 90, 100, 110)")
print("="*70)

rho = -0.7
S, v = heston_model_sim(S0, v0, rho, kappa, theta, sigma, T, N, M)

K_task3 = np.array([90, 100, 110])
sigma_bs = np.sqrt(theta)

heston_discounted_payoffs_task3 = [
    np.exp(-r * T) * np.maximum(S[-1] - k, 0) for k in K_task3
]
heston_calls_task3 = np.array([np.mean(p) for p in heston_discounted_payoffs_task3])
heston_std_task3   = np.array([np.std(p, ddof=1) for p in heston_discounted_payoffs_task3])
heston_se_task3    = heston_std_task3 / np.sqrt(M)
bs_calls_task3     = np.array([black_scholes_call_price(S0, k, T, r, sigma_bs) for k in K_task3])

print(f"{'Strike':<8} | {'Heston MC':<12} | {'BS Closed':<12} | {'Std Dev':<10} | {'Std Error':<10}")
print("-"*60)
for k, h, b, sd, se in zip(K_task3, heston_calls_task3, bs_calls_task3, heston_std_task3, heston_se_task3):
    print(f"{k:<8} | {h:>12.4f} | {b:>12.4f} | {sd:>10.4f} | {se:>10.4f}")
print("="*70)


# ============================================================================
# TASK 4: IMPLIED VOLATILITY SMILE — K ∈ [70, 130]
# ============================================================================
print("\n" + "="*70)
print("TASK 4: IMPLIED VOLATILITY SMILE")
print("="*70)

K_smile_t4 = np.arange(70, 131, 2)

calls_t4 = np.array([np.exp(-r*T)*np.mean(np.maximum(S[-1] - k, 0)) for k in K_smile_t4])
puts_t4  = np.array([np.exp(-r*T)*np.mean(np.maximum(k - S[-1], 0)) for k in K_smile_t4])

call_ivs_t4 = implied_vol(calls_t4, S0, K_smile_t4, T, r, flag='c', q=0, return_as='numpy', on_error='ignore')
put_ivs_t4  = implied_vol(puts_t4,  S0, K_smile_t4, T, r, flag='p', q=0, return_as='numpy', on_error='ignore')

# Filter out unreliable IV values (outside 1% – 150%)
valid_c = (call_ivs_t4 > 0.01) & (call_ivs_t4 < 1.5)
valid_p = (put_ivs_t4  > 0.01) & (put_ivs_t4  < 1.5)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(K_smile_t4[valid_c], call_ivs_t4[valid_c], marker='o', markersize=4,
        label='IV Calls', color='steelblue')
ax.plot(K_smile_t4[valid_p], put_ivs_t4[valid_p],  marker='s', markersize=4,
        label='IV Puts',  color='darkorange')
ax.set_title(r'Implied Volatility Smile — Heston ($\rho = -0.7$)')
ax.set_xlabel('Strike $K$')
ax.set_ylabel('Implied Volatility')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig_counter += 1
plt.savefig(os.path.join(output_dir, f"fig_{fig_counter:02d}_task4_iv_smile.png"), dpi=150, bbox_inches='tight')
print(f"[Figure {fig_counter}] Saved: Task 4 — Implied Volatility Smile")
plt.close()


# ============================================================================
# TASK 5: SENSITIVITY OF IV SMILE TO rho AND sigma
# ============================================================================
print("\n" + "="*70)
print("TASK 5: SENSITIVITY OF IMPLIED VOLATILITY SMILE")
print("="*70)
print("Computing sensitivities...")

rhos_to_test   = [-0.9, -0.5, 0.0]
sigmas_to_test = [0.2, 0.5, 1.0]
K_smile = np.arange(70, 131, 2)

fig_sens, (ax_rho, ax_sigma) = plt.subplots(1, 2, figsize=(14, 6))

# --- Sensitivity to rho (sigma fixed at 0.5) ---
for r_val in rhos_to_test:
    S_sim, _ = heston_model_sim(S0, v0, r_val, kappa, theta, sigma, T, N, M)
    calls_sim = np.array([np.exp(-r*T) * np.mean(np.maximum(S_sim[-1] - k, 0)) for k in K_smile])
    ivs_sim = implied_vol(calls_sim, S0, K_smile, T, r, flag='c', q=0, return_as='numpy', on_error='ignore')
    valid = (ivs_sim > 0.01) & (ivs_sim < 1.5)
    ax_rho.plot(K_smile[valid], ivs_sim[valid], marker='o', markersize=4, label=rf'$\rho = {r_val}$')

ax_rho.set_title(r'Sensitivity to $\rho$ ($\sigma = 0.5$)')
ax_rho.set_xlabel('Strike $K$')
ax_rho.set_ylabel('Implied Volatility')
ax_rho.legend()
ax_rho.grid(True, alpha=0.3)

# --- Sensitivity to sigma (rho fixed at -0.7) ---
for s_val in sigmas_to_test:
    S_sim, _ = heston_model_sim(S0, v0, rho, kappa, theta, s_val, T, N, M)
    calls_sim = np.array([np.exp(-r*T) * np.mean(np.maximum(S_sim[-1] - k, 0)) for k in K_smile])
    ivs_sim = implied_vol(calls_sim, S0, K_smile, T, r, flag='c', q=0, return_as='numpy', on_error='ignore')
    valid = (ivs_sim > 0.01) & (ivs_sim < 1.5)
    ax_sigma.plot(K_smile[valid], ivs_sim[valid], marker='s', markersize=4, label=rf'$\sigma = {s_val}$')

ax_sigma.set_title(r'Sensitivity to $\sigma$ ($\rho = -0.7$)')
ax_sigma.set_xlabel('Strike $K$')
ax_sigma.set_ylabel('Implied Volatility')
ax_sigma.legend()
ax_sigma.grid(True, alpha=0.3)

plt.tight_layout()
fig_counter += 1
plt.savefig(os.path.join(output_dir, f"fig_{fig_counter:02d}_task5_sensitivity.png"), dpi=150, bbox_inches='tight')
print(f"[Figure {fig_counter}] Saved: Task 5 — Sensitivity of IV Smile")
plt.close()


# ============================================================================
# TASK 6: FELLER CONDITION AND FULL TRUNCATION
# ============================================================================
print("\n" + "="*70)
print("TASK 6: FELLER CONDITION AND VARIANCE SIMULATION")
print("="*70)

feller_left  = 2 * kappa * theta
feller_right = sigma**2
print(f"Feller condition: 2κθ = {feller_left:.3f},  σ² = {feller_right:.3f}")
if feller_left > feller_right:
    print("Result: Feller condition SATISFIED.")
else:
    print("Result: Feller condition VIOLATED — variance can become negative.")

# Variance simulation WITHOUT truncation (abs trick to avoid NaN)
def variance_untruncated_sim(v0, kappa, theta, sigma, T, N, M):
    """Simulates variance allowing negative values (|v| trick under sqrt to avoid NaN)."""
    dt = T / N
    v_untrunc = np.full(shape=(N+1, M), fill_value=v0)
    Z = np.random.normal(size=(N, M))
    for i in range(1, N+1):
        v_untrunc[i] = (v_untrunc[i-1]
                        + kappa * (theta - v_untrunc[i-1]) * dt
                        + sigma * np.sqrt(np.abs(v_untrunc[i-1]) * dt) * Z[i-1])
    return v_untrunc

# Empirical check on 1000 paths
M_test = 1000
v_unprotected = variance_untruncated_sim(v0, kappa, theta, sigma, T, N, M_test)
negative_paths_count = np.sum(np.any(v_unprotected < 0, axis=0))
print(f"\nEmpirical check ({M_test} paths): {negative_paths_count} paths reached negative variance.")

# Plot 25 paths for each case (more informative than 10)
n_feller = 25
v_unprot_plot = variance_untruncated_sim(v0, kappa, theta, sigma, T, N, n_feller)
_, v_prot_plot = heston_model_sim(S0, v0, rho, kappa, theta, sigma, T, N, n_feller)
time_grid = np.linspace(0, T, N+1)

fig_feller, (ax_untrunc, ax_trunc) = plt.subplots(1, 2, figsize=(14, 5))

ax_untrunc.plot(time_grid, v_unprot_plot, alpha=0.6, linewidth=0.8)
ax_untrunc.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Zero threshold')
ax_untrunc.set_title(f'Variance WITHOUT Truncation ({n_feller} paths)')
ax_untrunc.set_xlabel('Time')
ax_untrunc.set_ylabel('Variance')
ax_untrunc.legend()
ax_untrunc.grid(True, alpha=0.3)

ax_trunc.plot(time_grid, v_prot_plot, alpha=0.6, linewidth=0.8)
ax_trunc.axhline(0, color='red', linestyle='--', linewidth=1.5, label='Zero threshold')
ax_trunc.set_title(f'Variance WITH Full Truncation ({n_feller} paths)')
ax_trunc.set_xlabel('Time')
ax_trunc.set_ylabel('Variance')
ax_trunc.legend()
ax_trunc.grid(True, alpha=0.3)

plt.tight_layout()
fig_counter += 1
plt.savefig(os.path.join(output_dir, f"fig_{fig_counter:02d}_task6_feller.png"), dpi=150, bbox_inches='tight')
print(f"[Figure {fig_counter}] Saved: Task 6 — Feller Condition")
plt.close()


# ============================================================================
# TASK 7: FOURIER PRICING VS MONTE CARLO
# ============================================================================
print("\n" + "="*70)
print("TASK 7: FOURIER PRICING VS MONTE CARLO")
print("="*70)

# --- Characteristic function (Gil-Pelaez / Heston 1993) ---
def heston_char_func(u, S0, v0, r, kappa, theta, sigma, rho, T):
    """Characteristic function of log-price under Heston."""
    xi = kappa - 1j * rho * sigma * u
    d  = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
    g  = (xi - d) / (xi + d)
    C  = (kappa * theta / sigma**2) * ((xi - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
    D  = ((xi - d) / sigma**2) * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
    return np.exp(C + D * v0 + 1j * u * np.log(S0) + 1j * u * r * T)

def integrand_P1(u, S0, K, v0, r, kappa, theta, sigma, rho, T):
    phi_1 = heston_char_func(u - 1j, S0, v0, r, kappa, theta, sigma, rho, T) / (S0 * np.exp(r * T))
    return np.real(np.exp(-1j * u * np.log(K)) * phi_1 / (1j * u))

def integrand_P2(u, S0, K, v0, r, kappa, theta, sigma, rho, T):
    phi_2 = heston_char_func(u, S0, v0, r, kappa, theta, sigma, rho, T)
    return np.real(np.exp(-1j * u * np.log(K)) * phi_2 / (1j * u))

def heston_call_fourier(S0, K, v0, r, kappa, theta, sigma, rho, T):
    """European call price via numerical integration of the characteristic function."""
    int1, _ = spi.quad(integrand_P1, 1e-4, 100, args=(S0, K, v0, r, kappa, theta, sigma, rho, T))
    int2, _ = spi.quad(integrand_P2, 1e-4, 100, args=(S0, K, v0, r, kappa, theta, sigma, rho, T))
    P1 = 0.5 + (1 / np.pi) * int1
    P2 = 0.5 + (1 / np.pi) * int2
    return S0 * P1 - K * np.exp(-r * T) * P2

# --- MC simulation for Task 7 ---
S_mc_t7, _ = heston_model_sim(S0, v0, rho, kappa, theta, sigma, T, N, M)

K_target = np.array([80, 90, 100, 110, 120])
prices_fourier = []
prices_mc      = []
abs_errors     = []

print(f"\n{'Strike':<8} | {'C_Fourier':>12} | {'C_MC':>12} | {'Abs Error':>12}")
print("-"*52)

for k in K_target:
    c_f  = heston_call_fourier(S0, k, v0, r, kappa, theta, sigma, rho, T)
    c_mc = np.exp(-r * T) * np.mean(np.maximum(S_mc_t7[-1] - k, 0))
    err  = abs(c_f - c_mc)
    prices_fourier.append(c_f)
    prices_mc.append(c_mc)
    abs_errors.append(err)
    print(f"{k:<8} | {c_f:>12.4f} | {c_mc:>12.4f} | {err:>12.4f}")

print("="*52)

# --- Plot: Fourier vs MC comparison ---
prices_fourier = np.array(prices_fourier)
prices_mc      = np.array(prices_mc)
abs_errors     = np.array(abs_errors)

x = np.arange(len(K_target))
width = 0.35

fig, (ax_price, ax_err) = plt.subplots(1, 2, figsize=(13, 5))

ax_price.bar(x - width/2, prices_fourier, width, label='Fourier', color='steelblue', alpha=0.85)
ax_price.bar(x + width/2, prices_mc,      width, label='Monte Carlo', color='darkorange', alpha=0.85)
ax_price.set_xticks(x)
ax_price.set_xticklabels(K_target)
ax_price.set_xlabel('Strike $K$')
ax_price.set_ylabel('Call Price')
ax_price.set_title('Task 7 — Call Prices: Fourier vs Monte Carlo')
ax_price.legend()
ax_price.grid(True, alpha=0.3, axis='y')

ax_err.bar(x, abs_errors, color='firebrick', alpha=0.8)
ax_err.set_xticks(x)
ax_err.set_xticklabels(K_target)
ax_err.set_xlabel('Strike $K$')
ax_err.set_ylabel('Absolute Error')
ax_err.set_title('Task 7 — Absolute Error |C_Fourier − C_MC|')
ax_err.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_counter += 1
plt.savefig(os.path.join(output_dir, f"fig_{fig_counter:02d}_task7_fourier_vs_mc.png"), dpi=150, bbox_inches='tight')
print(f"[Figure {fig_counter}] Saved: Task 7 — Fourier vs Monte Carlo")
plt.close()


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("EXECUTION COMPLETED")
print("="*70)
print(f"All figures saved to '{output_dir}/'")
print(f"Total figures generated: {fig_counter}")
print("="*70)
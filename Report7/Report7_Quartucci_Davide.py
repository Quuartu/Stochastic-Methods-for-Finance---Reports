"""
Homework 7: Down-and-Out European Call Option

Author: Davide Quartucci
Date: Start 2026-04-17, Due 2026-04-25

Description:
Closed-form implementation for pricing barrier options (Down-and-Out and Down-and-In).
The code demonstrates the validity of the "Barrier Option Parity"
(Vanilla = Down-and-In + Down-and-Out) and explores sensitivities (Greeks)
and numerical approximations (Monte Carlo).

Main Academic References:
1. Reiner, E., & Rubinstein, M. (1991). "Breaking down the barriers". Risk, 4(8), 28-35. 
    (Original paper for closed-form continuous-barrier option formulas).
2. Haug, E. G. (2007). "The Complete Guide to Option Pricing Formulas" (2nd Ed.). McGraw-Hill.
    (Reference for algebraic standardization of parameters y and lambda).
"""

import numpy as np
import math
import matplotlib.pyplot as plt


def norm_cdf(x):
    """Compute the Cumulative Distribution Function (CDF) of the Standard Normal distribution.
    
    Implementation note:
    To avoid heavy dependencies on external libraries (e.g., scipy.stats.norm)
    and guarantee maximum execution speed, the CDF is derived analytically
    using the Error Function provided by the built-in 'math' module.
    Formula: \\Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def vanilla_call(S0, K, r, sigma, T):
    """Compute the premium of a standard European Call (Black-Scholes-Merton, 1973)."""
    sig_sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / sig_sqrt_T
    d2 = d1 - sig_sqrt_T
    
    return S0 * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)


def down_and_out_call(S0, K, L, r, sigma, T):
    """Compute the closed-form premium of a European Down-and-Out Call.
    
    Model: Reiner and Rubinstein (1991).
    Assumption: K >= L (strike price greater than or equal to the barrier).
    
    Logic:
    The algorithm computes penalty terms (due to knock-out risk)
    and subtracts them from the potential vanilla value."""
    if S0 <= L:
        return 0.0
    
    # Helper variables for optimization
    sig_sqrt_T = sigma * np.sqrt(T)
    discount_factor = np.exp(-r * T)
    lam = (r + 0.5 * sigma**2) / sigma**2
    
    # Directional parameters
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / sig_sqrt_T
    d2 = d1 - sig_sqrt_T
    y = (np.log(L**2 / (S0 * K)) + (r + 0.5 * sigma**2) * T) / sig_sqrt_T
    
    # Equation decomposition (Vanilla - Barrier Penalty)
    vanilla_part = S0 * norm_cdf(d1) - K * discount_factor * norm_cdf(d2)
    penalty_part = S0 * (L / S0)**(2 * lam) * norm_cdf(y) - K * discount_factor * (L / S0)**(2 * lam - 2) * norm_cdf(y - sig_sqrt_T)
    
    return vanilla_part - penalty_part


def down_and_in_call(S0, K, L, r, sigma, T):
    """Compute the closed-form premium of a European Down-and-In Call.
    
    Model: Reiner and Rubinstein (1991).
    Assumption: K >= L.
    
    Logic:
    If S0 <= L, the "In" condition is satisfied at t=0; the option is vanilla."""
    if S0 <= L:
        return vanilla_call(S0, K, r, sigma, T)
        
    sig_sqrt_T = sigma * np.sqrt(T)
    discount_factor = np.exp(-r * T)
    lam = (r + 0.5 * sigma**2) / sigma**2
    y = (np.log(L**2 / (S0 * K)) + (r + 0.5 * sigma**2) * T) / sig_sqrt_T
    
    knock_in_value = S0 * (L / S0)**(2 * lam) * norm_cdf(y) - K * discount_factor * (L / S0)**(2 * lam - 2) * norm_cdf(y - sig_sqrt_T)
    
    return knock_in_value


def monte_carlo_doc(S0, K, L, r, sigma, T, M=10000, N=252, barrier_method='continuous'):
    """Monte Carlo pricing for down-and-out call."""
    dt = T / N
    nudt = (r - 0.5 * sigma**2) * dt
    sigsdt = sigma * np.sqrt(dt)
    
    S = np.full(M, S0)
    active = np.ones(M, dtype=bool)
    
    for _ in range(N):
        Z = np.random.standard_normal(M)
        S = S * np.exp(nudt + sigsdt * Z)
        if barrier_method == 'discrete':
            active = active & (S > L)
        elif barrier_method == 'continuous':
            active = active & (S > L)
    
    payoff = np.zeros(M)
    payoff[active] = np.maximum(S[active] - K, 0)
    
    C0 = np.exp(-r * T) * np.mean(payoff)
    SE = np.std(np.exp(-r * T) * payoff) / np.sqrt(M)
    
    return C0, SE


def compute_delta_fd(pricing_func, S0, K, L, r, sigma, T, epsilon=1e-4, is_barrier=True):
    """Delta via central finite differences."""
    if is_barrier:
        V_up = pricing_func(S0 + epsilon, K, L, r, sigma, T)
        V_down = pricing_func(S0 - epsilon, K, L, r, sigma, T)
    else:
        V_up = pricing_func(S0 + epsilon, K, r, sigma, T)
        V_down = pricing_func(S0 - epsilon, K, r, sigma, T)
    return (V_up - V_down) / (2.0 * epsilon)


def analyze_convergence_to_barrier(K, L, r, sigma, T):
    """Analyze price convergence as S0 approaches barrier L."""
    print("\n=== PART 2: CONVERGENCE TO BARRIER ===")
    S0_values = [100, 95, 92, 91, 90.5, 90.1, 90.05, 90.01]
    print("\nDOC Price Convergence as S0 → L (Barrier = 90):")
    print(f"{'S0':<8} | {'DOC Price':<12} | {'Distance to L':<15} | {'% of DOC at S0=100':<20}")
    print("-" * 65)
    
    baseline = down_and_out_call(100.0, K, L, r, sigma, T)
    for s in S0_values:
        doc = down_and_out_call(s, K, L, r, sigma, T)
        dist = s - L
        pct = (doc / baseline) * 100 if baseline != 0 else 0
        print(f"{s:<8.2f} | {doc:<12.6f} | {dist:<15.2f} | {pct:<20.2f}%")
    
def analyze_delta_discontinuity(K, L, r, sigma, T):
    """Analyze Delta instability near barrier."""
    print("\n=== PART 2: DELTA DISCONTINUITIES & INSTABILITIES ===")
    
    S0_values = [100, 95, 92, 91, 90.5, 90.2, 90.1, 90.05, 90.01]
    print("\nDelta Comparison: Vanilla vs DOC (Very Close to Barrier):")
    print(f"{'S0':<8} | {'Distance to L':<15} | {'Vanilla Δ':<12} | {'DOC Δ':<12} | {'Δ Difference':<15}")
    print("-" * 70)
    
    for s in S0_values:
        d_v = compute_delta_fd(vanilla_call, s, K, L, r, sigma, T, epsilon=1e-4, is_barrier=False)
        d_d = compute_delta_fd(down_and_out_call, s, K, L, r, sigma, T, epsilon=1e-4, is_barrier=True)
        dist = s - L
        diff = abs(d_v - d_d)
        print(f"{s:<8.2f} | {dist:<15.2f} | {d_v:<12.6f} | {d_d:<12.6f} | {diff:<15.6f}")
    
    print("\nDelta Convergence Rate (Multiple Epsilon Values at S0=90.1):")
    epsilons = [1e-2, 1e-3, 1e-4, 1e-5]
    print(f"{'Epsilon':<10} | {'Vanilla Δ':<15} | {'DOC Δ':<15}")
    print("-" * 45)
    for eps in epsilons:
        d_v = compute_delta_fd(vanilla_call, 90.1, K, L, r, sigma, T, epsilon=eps, is_barrier=False)
        d_d = compute_delta_fd(down_and_out_call, 90.1, K, L, r, sigma, T, epsilon=eps, is_barrier=True)
        print(f"{eps:<10.0e} | {d_v:<15.6f} | {d_d:<15.6f}")
    
def analyze_price_sensitivity(K, L, r, sigma, T):
    """Analyze price sensitivity near barrier."""
    print("\n=== PART 3: PRICE SENSITIVITY NEAR BARRIER ===")
    
    S0_values = np.linspace(100, 90.01, 20)
    print("\nPrice Sensitivity: DOC and Vanilla Call Near Barrier")
    print(f"{'S0':<8} | {'Distance':<10} | {'DOC Price':<12} | {'Vanilla Price':<15} | {'DOC Elasticity':<15}")
    print("-" * 75)
    
    doc_prev = down_and_out_call(100.0, K, L, r, sigma, T)
    s_prev = 100.0
    
    for s in S0_values:
        doc = down_and_out_call(s, K, L, r, sigma, T)
        vanilla = vanilla_call(s, K, r, sigma, T)
        dist = s - L
        
        if s < s_prev:
            ds = s_prev - s
            ddoc = doc_prev - doc
            elasticity = (ddoc / doc_prev) / (ds / s_prev) if doc_prev != 0 else 0
        else:
            elasticity = 0.0
        
        print(f"{s:<8.2f} | {dist:<10.4f} | {doc:<12.6f} | {vanilla:<15.6f} | {elasticity:<15.4f}")
        doc_prev = doc
        s_prev = s
    
    print("\nSensitivity Metrics:")
    print(f"{'S0 Range':<20} | {'Avg Delta (Vanilla)':<20} | {'Avg Delta (DOC)':<20}")
    print("-" * 62)
    
    ranges = [(100, 95), (95, 91), (91, 90.5), (90.5, 90.1)]
    for s_high, s_low in ranges:
        d_v_high = compute_delta_fd(vanilla_call, s_high, K, L, r, sigma, T, epsilon=1e-4, is_barrier=False)
        d_v_low = compute_delta_fd(vanilla_call, s_low, K, L, r, sigma, T, epsilon=1e-4, is_barrier=False)
        d_d_high = compute_delta_fd(down_and_out_call, s_high, K, L, r, sigma, T, epsilon=1e-4, is_barrier=True)
        d_d_low = compute_delta_fd(down_and_out_call, s_low, K, L, r, sigma, T, epsilon=1e-4, is_barrier=True)
        
        avg_d_v = (d_v_high + d_v_low) / 2
        avg_d_d = (d_d_high + d_d_low) / 2
        
        print(f"[{s_high}, {s_low}]       | {avg_d_v:<20.6f} | {avg_d_d:<20.6f}")


def analyze_monte_carlo_convergence(S0, K, L, r, sigma, T):
    """Analyze Monte Carlo convergence and discretization bias."""
    print("\n=== PART 4: MONTE CARLO SIMULATION & DISCRETIZATION BIAS ===")
    
    analytical_price = down_and_out_call(S0, K, L, r, sigma, T)
    
    print("\n1. Convergence with Paths (M) - Fixed N=252:")
    print(f"{'M (Paths)':<12} | {'MC Price':<12} | {'Std Error':<12} | {'Bias':<12} | {'Rel. Error %':<12}")
    print("-" * 65)
    
    M_values = [100, 500, 1000, 5000, 10000, 50000, 100000]
    mc_prices_fixed_n = []
    
    for M in M_values:
        mc_price, se = monte_carlo_doc(S0, K, L, r, sigma, T, M=M, N=252)
        bias = mc_price - analytical_price
        rel_error = abs(bias) / analytical_price * 100 if analytical_price != 0 else 0
        mc_prices_fixed_n.append((M, mc_price, se, bias))
        print(f"{M:<12} | {mc_price:<12.6f} | {se:<12.6f} | {bias:<12.6f} | {rel_error:<12.4f}")
    
    print("\n2. Discretization Bias with Time Steps (N) - Fixed M=5000:")
    print(f"{'N (Steps)':<12} | {'MC Price':<12} | {'Std Error':<12} | {'Bias':<12} | {'Rel. Error %':<12}")
    print("-" * 65)
    
    N_values = [12, 26, 52, 104, 252, 504, 1000]
    mc_prices_fixed_m = []
    
    for N in N_values:
        mc_price, se = monte_carlo_doc(S0, K, L, r, sigma, T, M=5000, N=N)
        bias = mc_price - analytical_price
        rel_error = abs(bias) / analytical_price * 100 if analytical_price != 0 else 0
        mc_prices_fixed_m.append((N, mc_price, se, bias))
        print(f"{N:<12} | {mc_price:<12.6f} | {se:<12.6f} | {bias:<12.6f} | {rel_error:<12.4f}")
    
    print("\n3. Joint Convergence Surface (M × N):")
    print(f"{'M \\ N':<8} | N=52    | N=104   | N=252   | N=504   | N=1000")
    print("-" * 60)
    
    M_surface = [100, 1000, 5000, 10000, 50000]
    N_surface = [52, 104, 252, 504, 1000]
    
    for M in M_surface:
        row = f"{M:<8} |"
        for N in N_surface:
            mc_price, _ = monte_carlo_doc(S0, K, L, r, sigma, T, M=M, N=N)
            bias = mc_price - analytical_price
            row += f" {bias:7.4f} |"
        print(row)
    
    print(f"\nAnalytical Price (Closed-Form): {analytical_price:.6f}")
    print("\nNote: All values are BIAS (MC Price - Analytical Price)")


def implied_vol_bisection(market_price, pricing_func, sigma_low=1e-6, sigma_high=3.0, tol=1e-8, max_iter=100):
    """Estimate implied volatility by bisection."""
    price_low = pricing_func(sigma_low)
    price_high = pricing_func(sigma_high)

    if market_price < price_low or market_price > price_high:
        raise ValueError("Market price is outside the bisection bracket.")

    low = sigma_low
    high = sigma_high

    for iteration in range(1, max_iter + 1):
        mid = 0.5 * (low + high)
        price_mid = pricing_func(mid)
        error = price_mid - market_price

        if abs(error) < tol or 0.5 * (high - low) < tol:
            return mid, iteration

        if price_mid < market_price:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high), max_iter


def analyze_implied_volatility(S0, K, L, r, sigma, T):
    """Estimate and compare implied volatility for DOC and vanilla calls."""
    print("\n=== PART 5: IMPLIED VOLATILITY ESTIMATION ===")

    market_doc = down_and_out_call(S0, K, L, r, sigma, T)
    market_vanilla = vanilla_call(S0, K, r, sigma, T)

    doc_iv, doc_iter = implied_vol_bisection(
        market_doc,
        lambda vol: down_and_out_call(S0, K, L, r, vol, T),
        sigma_low=1e-6,
        sigma_high=3.0,
        tol=1e-10,
        max_iter=200,
    )

    vanilla_iv, vanilla_iter = implied_vol_bisection(
        market_vanilla,
        lambda vol: vanilla_call(S0, K, r, vol, T),
        sigma_low=1e-6,
        sigma_high=3.0,
        tol=1e-10,
        max_iter=200,
    )

    print("\nImplied Volatility Recovery (Synthetic Market Prices):")
    print(f"{'Option':<14} | {'Market Price':<14} | {'Implied Vol':<14} | {'Iterations':<10} | {'Abs. Error':<12}")
    print("-" * 80)
    print(f"{'DOC':<14} | {market_doc:<14.6f} | {doc_iv:<14.8f} | {doc_iter:<10d} | {abs(doc_iv - sigma):<12.2e}")
    print(f"{'Vanilla':<14} | {market_vanilla:<14.6f} | {vanilla_iv:<14.8f} | {vanilla_iter:<10d} | {abs(vanilla_iv - sigma):<12.2e}")

    print("\nBisection Convergence Check for DOC Implied Volatility:")
    print(f"{'Tolerance':<12} | {'Implied Vol':<14} | {'Abs. Error':<12} | {'Iterations':<10} | {'Residual':<12}")
    print("-" * 78)
    tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    for tol in tolerances:
        vol_est, iterations = implied_vol_bisection(
            market_doc,
            lambda vol: down_and_out_call(S0, K, L, r, vol, T),
            sigma_low=1e-6,
            sigma_high=3.0,
            tol=tol,
            max_iter=200,
        )
        residual = abs(down_and_out_call(S0, K, L, r, vol_est, T) - market_doc)
        print(f"{tol:<12.0e} | {vol_est:<14.8f} | {abs(vol_est - sigma):<12.2e} | {iterations:<10d} | {residual:<12.2e}")


def main():
    # Input parameters
    S0 = 100.0
    K = 100.0
    L = 90.0
    r = 0.01
    sigma = 0.2
    T = 1.0
    
    doc_price = down_and_out_call(S0, K, L, r, sigma, T)
    vanilla_price = vanilla_call(S0, K, r, sigma, T)
    dic_price = down_and_in_call(S0, K, L, r, sigma, T)
    
    print("=== PART 1: BASIC PRICING & PARITY ===")
    print(f"Vanilla: {vanilla_price:.6f}")
    print(f"DOC:     {doc_price:.6f}")
    print(f"DIC:     {dic_price:.6f}")
    print(f"DIC+DOC: {dic_price + doc_price:.6f}")
    print(f"Error:   {abs(vanilla_price - (dic_price + doc_price)):.2e}")
    
    analyze_convergence_to_barrier(K, L, r, sigma, T)
    analyze_delta_discontinuity(K, L, r, sigma, T)
    
    analyze_price_sensitivity(K, L, r, sigma, T)
    
    analyze_monte_carlo_convergence(S0, K, L, r, sigma, T)

    analyze_implied_volatility(S0, K, L, r, sigma, T)
    
    print("\n=== DELTA COMPARISON (STANDARD RANGE) ===")
    S0_values = [110, 105, 100, 95, 92, 91, 90.5]
    print("\nDelta (Vanilla vs DOC):")
    for s in S0_values:
        d_v = compute_delta_fd(vanilla_call, s, K, L, r, sigma, T, epsilon=1e-4, is_barrier=False)
        d_d = compute_delta_fd(down_and_out_call, s, K, L, r, sigma, T, epsilon=1e-4, is_barrier=True)
        print(f"S0={s:6.1f}: Vanilla={d_v:.6f}, DOC={d_d:.6f}")
    
    print("\n=== GENERATING PLOT ===")
    S_range = np.linspace(85, 120, 300)
    vanilla_prices = [vanilla_call(s, K, r, sigma, T) for s in S_range]
    doc_prices = [down_and_out_call(s, K, L, r, sigma, T) for s in S_range]
    
    plt.figure(figsize=(10, 6))
    plt.plot(S_range, vanilla_prices, label='Vanilla Call', linestyle='--')
    plt.plot(S_range, doc_prices, label='Down-and-Out Call', color='red')
    plt.axvline(x=L, color='gray', linestyle=':', label=f'Barrier L={L}')
    plt.title('Option Price vs Underlying Asset Price')
    plt.xlabel('Spot Price (S0)')
    plt.ylabel('Option Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('barrier_price_sensitivity.png')
    print("Plot saved as barrier_price_sensitivity.png")


if __name__ == "__main__":
    main()
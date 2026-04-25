# Homework 7 Report: Down-and-Out European Call

## 1. Objective of the Assignment

The goal of the homework is to implement and analyze pricing formulas for barrier options under Black-Scholes assumptions, with focus on a down-and-out European call (DOC).

Based on the requested tasks, the script is expected to:

1. Implement closed-form pricing for Vanilla Call, Down-and-In Call (DIC), and Down-and-Out Call (DOC).
2. Verify the financial inequality `DOC < Vanilla`.
3. Verify the In-Out parity `Vanilla = DIC + DOC`.
4. Quantify and comment on numerical error from floating-point arithmetic.
5. Provide a sensitivity analysis for Delta, especially near the barrier.
6. Produce a visualization of option value versus spot price around the barrier region.
7. Add a Monte Carlo estimate for DOC as a numerical cross-check.

## 2. What Was Implemented in Python

Implementation file: [homework7.py](homework7.py)

### 2.1 Standard Normal CDF

The function `norm_cdf(x)` computes $\Phi(x)$ using `math.erf`, avoiding external dependency on `scipy`.

### 2.2 Closed-Form Vanilla Call

Function: `vanilla_call(S0, K, r, sigma, T)`

Implemented Black-Scholes formula:

$$
C = S_0\Phi(d_1) - Ke^{-rT}\Phi(d_2)
$$

with

$$
d_1 = \frac{\ln(S_0/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}},
\quad d_2 = d_1 - \sigma\sqrt{T}.
$$

### 2.3 Closed-Form Barrier Prices

Functions:

1. `down_and_out_call(S0, K, L, r, sigma, T)`
2. `down_and_in_call(S0, K, L, r, sigma, T)`

Both formulas use the standard barrier-option terms:

$$
\lambda = \frac{r + \frac{1}{2}\sigma^2}{\sigma^2},
\quad
y = \frac{\ln\!\left(\frac{L^2}{S_0K}\right) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}.
$$

Edge behavior is explicitly handled:

1. If $S_0 \le L$, DOC is immediately zero (already knocked out).
2. If $S_0 \le L$, DIC behaves like Vanilla (already knocked in).

### 2.4 Monte Carlo DOC Cross-Check

Function: `monte_carlo_doc(S0, K, L, r, sigma, T, M, N)`

The simulation:

1. Generates geometric Brownian motion paths.
2. Flags each path as active/inactive depending on whether the barrier was hit.
3. Computes discounted payoff only on active paths.
4. Returns both the estimated price and standard error.

## 3. Parameters Used in the Current Script

Main block values in [homework7.py](homework7.py):

1. $S_0 = 100.0$
2. $K = 100.0$
3. $L = 90.0$
4. $r = 0.01$
5. $\sigma = 0.2$
6. $T = 1.0$

Additional Monte Carlo settings:

1. `M_sims = 1000`
2. `N_steps = 360000`

## 4. Requested Verifications and How They Are Addressed

### 4.1 Check: DOC Must Be Lower Than Vanilla

In `main()`, the script computes both prices and prints:

1. `Is DOC < Vanilla?`

Financial interpretation: DOC has extra knockout risk, so its premium should be lower than a vanilla call with the same strike/maturity.

### 4.2 Check: In-Out Parity

In `main()`, the script computes:

$$
\text{sum\_in\_out} = DOC + DIC
$$

and compares it to Vanilla. This is the core identity required by the assignment:

$$
\text{Vanilla} = \text{DIC} + \text{DOC}.
$$

### 4.3 Numerical Error Comment

The script reports:

$$
\left|\text{Vanilla} - (\text{DIC} + \text{DOC})\right|
$$

Expected behavior: tiny non-zero value due to double-precision floating-point limits, not due to formula inconsistency.

## 5. Delta and Barrier-Sensitivity Analysis

### 5.1 Finite-Difference Delta

Function `compute_delta_fd(...)` implements central finite differences:

$$
\Delta \approx \frac{V(S_0 + \varepsilon) - V(S_0 - \varepsilon)}{2\varepsilon}.
$$

Function `analyze_delta_and_barrier(...)` evaluates convergence across multiple $\varepsilon$ values and compares Vanilla vs DOC Delta.

### 5.2 Behavior Near the Barrier

The script evaluates Delta for spot values approaching $L$ from above. The reportable result is that DOC Delta becomes highly unstable/asymmetric near the barrier, consistent with barrier discontinuity risk.

### 5.3 Plot Generation

`analyze_delta_and_barrier(...)` generates and saves:

1. `barrier_price_sensitivity.png`

The plot compares Vanilla and DOC prices over a spot grid and marks the barrier with a vertical line.

## 6. Consistency With Sources

The formulas and decomposition are consistent with standard references for barrier options (Haug) and common open-source implementations (including the repository cited in [references/source.txt](references/source.txt)).

## 7. Final Summary

The Python script in [homework7.py](homework7.py) satisfies the assignment requirements by:

1. Implementing closed-form Vanilla, DIC, and DOC pricing.
2. Verifying the key financial inequality and In-Out parity.
3. Explaining numerical residuals correctly.
4. Extending the analysis with Monte Carlo validation.
5. Performing Delta convergence and near-barrier sensitivity analysis.
6. Producing a plot that documents barrier effects visually.

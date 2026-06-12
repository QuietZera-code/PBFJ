"""
Module 4: Dynamic Hedging Strategy and Backtesting
Description: Constructs the skewness-based market-timing strategy net of transaction costs,
and generates Figure 6 and Table 4 for the manuscript.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['pdf.fonttype'] = 42

df = pd.read_csv('data/sieve_rnd_factors_5years.csv', index_col=0)
df.index = pd.to_datetime(df.index)

# Strategy parameters mimicking real-world ETF frictions
TRANSACTION_COST = 0.0003  # 3 bps cost
QUANTILE_THRESHOLD = 0.05  # Empirical 5% extreme panic threshold

# Rolling risk threshold and signal generation
df['skew_roll_quantile'] = df['implied_skew_c3'].rolling(window=60).quantile(QUANTILE_THRESHOLD)
df['signal'] = np.where(df['implied_skew_c3'] < df['skew_roll_quantile'], 0, 1) # 0: Cash-out hedging, 1: Full long

# Turnover calculation (Absolute difference indicates trade execution)
df['turnover'] = df['signal'].diff().abs()

# Gross and Net return calculations
df['strategy_return_gross'] = df['signal'].shift(1) * df['future_10d_return'] / 10.0
df['benchmark_return'] = df['future_10d_return'] / 10.0
df['strategy_return_net'] = df['strategy_return_gross'] - (df['turnover'].shift(1).fillna(0) * TRANSACTION_COST)

df_backtest = df[['strategy_return_net', 'benchmark_return', 'signal']].dropna().copy()
df_backtest['Strategy_NetValue'] = (1 + df_backtest['strategy_return_net']).cumprod()
df_backtest['Benchmark_NetValue'] = (1 + df_backtest['benchmark_return']).cumprod()

# =====================================================================
# Fig. 6: Cumulative Returns of the Skew-Based Strategy
# =====================================================================
fig6 = plt.figure(figsize=(10, 5))
plt.plot(df_backtest.index, df_backtest['Strategy_NetValue'], color='tab:red', linewidth=2, label='Option-Skew Dynamic Timing (Net of 3 bps Costs)')
plt.plot(df_backtest.index, df_backtest['Benchmark_NetValue'], color='tab:gray', linewidth=1.5, alpha=0.8, label='Buy and Hold 50ETF (Benchmark)')

# Shade the hedging periods
for d in df_backtest[df_backtest['signal'] == 0].index:
    plt.axvline(x=d, color='blue', alpha=0.05)

plt.title('Fig. 6 Cumulative Returns of the Skew-Based Strategy', pad=15)
plt.ylabel('Cumulative Net Value (Base=1.0)')
plt.legend(loc='upper left')
plt.savefig('Fig6_Strategy_Backtest_NetOfCosts.pdf', bbox_inches='tight')
plt.show()

# =====================================================================
# Table 4: Multidimensional Performance Metrics
# =====================================================================
def calculate_metrics(returns):
    nav = (1 + returns).cumprod()
    ann_return = (nav.iloc[-1]) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    max_drawdown = ((nav - nav.cummax()) / nav.cummax()).min()
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
    return ann_return, ann_vol, max_drawdown, calmar

# Full Sample (2021-2026)
r_strat, v_strat, d_strat, c_strat = calculate_metrics(df_backtest['strategy_return_net'])
r_bench, v_bench, d_bench, c_bench = calculate_metrics(df_backtest['benchmark_return'])

# Crisis Sub-Sample (2022-2024)
df_crisis = df_backtest.loc['2022-01-01':'2024-06-30']
cr_strat, cv_strat, cd_strat, cc_strat = calculate_metrics(df_crisis['strategy_return_net'])
cr_bench, cv_bench, cd_bench, cc_bench = calculate_metrics(df_crisis['benchmark_return'])

print("\n=== Table 4: Multidimensional Performance Metrics of the Skew-Based Strategy (Net of 3 bps Costs) ===")
print(f"{'Performance Metrics':<25} | {'Full Sample (2021-2026)':<26} | {'Crisis Period (2022-2024)':<26}")
print(f"{'':<25} | {'Strategy':<10} vs {'Benchmark':<10} | {'Strategy':<10} vs {'Benchmark':<10}")
print("-" * 85)
print(f"{'Annualized Return':<25} | {r_strat*100:.2f}% {'':<3} {r_bench*100:.2f}% | {cr_strat*100:.2f}% {'':<3} {cr_bench*100:.2f}%")
print(f"{'Annualized Volatility':<25} | {v_strat*100:.2f}% {'':<3} {v_bench*100:.2f}% | {cv_strat*100:.2f}% {'':<3} {cv_bench*100:.2f}%")
print(f"{'Maximum Drawdown':<25} | {d_strat*100:.2f}% {'':<2} {d_bench*100:.2f}% | {cd_strat*100:.2f}% {'':<2} {cd_bench*100:.2f}%")
print(f"{'Calmar Ratio':<25} | {c_strat:.3f} {'':<4} {c_bench:.3f} | {cc_strat:.3f} {'':<4} {cc_bench:.3f}")
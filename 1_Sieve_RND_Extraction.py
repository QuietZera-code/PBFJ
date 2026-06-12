"""
Module 1: Sieve Semi-Nonparametric RND Extraction
Description: Extracts option-implied higher moments (skewness and kurtosis)
from the SSE 50 ETF options market using the Gauss-Hermite polynomial expansion.
Environment: JoinQuant Quantitative Platform (Requires JQData API)
"""

from jqdata import *
import pandas as pd
import numpy as np
import scipy.optimize as opt_solver
import scipy.stats as stats
from math import exp, log, sqrt
import warnings

warnings.filterwarnings('ignore')


class SieveRNDExtractor:
    def __init__(self, S, r, T):
        self.S = S
        self.r = r
        self.T = T

    def _hermite_polynomials(self, z):
        # First four standardized orthogonal Gauss-Hermite polynomials
        H0 = 1.0
        H1 = z
        H2 = (z ** 2 - 1) / sqrt(2)
        H3 = (z ** 3 - 3 * z) / sqrt(6)
        H4 = (z ** 4 - 6 * z ** 2 + 3) / sqrt(24)
        return np.array([H0, H1, H2, H3, H4])

    def _snp_density(self, ST, params):
        mu, sigma, c3, c4 = params
        if sigma <= 0.001: return 1e-10
        z = (log(ST / self.S) - mu) / sigma
        phi_z = stats.norm.pdf(z)
        H = self._hermite_polynomials(z)

        # Sieve higher-moment correction term
        correction = 1.0 + c3 * H[3] + c4 * H[4]
        f_ST = (1 / (ST * sigma)) * phi_z * correction
        return max(f_ST, 1e-10)

    def _pricing_error(self, params, strikes, mkt_calls, mkt_puts):
        error = 0.0
        ST_grid = np.linspace(self.S * 0.1, self.S * 2.0, 300)
        dST = ST_grid[1] - ST_grid[0]
        rnd_curve = np.array([self._snp_density(st, params) for st in ST_grid])

        # Penalty for integration to unity (Martingale condition)
        area = np.sum(rnd_curve) * dST
        penalty = 1000 * (area - 1.0) ** 2

        for K, C_mkt, P_mkt in zip(strikes, mkt_calls, mkt_puts):
            payoff_C = np.maximum(ST_grid - K, 0)
            C_model = exp(-self.r * self.T) * np.sum(payoff_C * rnd_curve) * dST
            payoff_P = np.maximum(K - ST_grid, 0)
            P_model = exp(-self.r * self.T) * np.sum(payoff_P * rnd_curve) * dST
            # Mean Squared Error (MSE) objective function
            error += (C_model - C_mkt) ** 2 + (P_model - P_mkt) ** 2

        return error + penalty

    def fit_rnd(self, strikes, mkt_calls, mkt_puts):
        # Sequential Least Squares Programming (SLSQP) constrained optimization
        initial_guess = [(self.r - 0.5 * 0.04) * self.T, 0.2 * sqrt(self.T), 0.0, 0.0]
        bounds = ((-2.0, 2.0), (0.01, 2.0), (-1.5, 1.5), (-1.5, 1.5))
        result = opt_solver.minimize(
            self._pricing_error,
            initial_guess,
            args=(strikes, mkt_calls, mkt_puts),
            method='SLSQP', bounds=bounds,
            options={'disp': False, 'maxiter': 500}
        )
        return result.x


# Data Pipeline (Sample period: 2021-01-01 to 2026-05-31)
start_date = '2021-01-01'
end_date = '2026-05-31'
underlying = '510050.XSHG'

trade_days = get_trade_days(start_date, end_date)
all_prices = get_price(underlying, start_date=start_date, end_date='2026-06-30', frequency='daily', fields=['close'])[
    'close']
factor_results = []

for date in trade_days[::2]:  # Rolling extraction every alternate day
    date_str = str(date)
    try:
        if date not in all_prices.index: continue
        spot_price = all_prices[date]

        # Filter option contracts
        q = query(opt.OPT_CONTRACT_INFO).filter(
            opt.OPT_CONTRACT_INFO.underlying_symbol == underlying,
            opt.OPT_CONTRACT_INFO.list_date <= date_str,
            opt.OPT_CONTRACT_INFO.delist_date > date_str
        )
        df_contracts = opt.run_query(q)
        if df_contracts.empty: continue

        df_contracts['days_to_expire'] = (
                    pd.to_datetime(df_contracts['delist_date']) - pd.to_datetime(date_str)).dt.days
        valid_contracts = df_contracts[(df_contracts['days_to_expire'] > 15) & (df_contracts['days_to_expire'] < 60)]
        if valid_contracts.empty: continue

        target_days = valid_contracts['days_to_expire'].min()
        df_target = valid_contracts[valid_contracts['days_to_expire'] == target_days]

        q_price = query(opt.OPT_DAILY_PRICE).filter(
            opt.OPT_DAILY_PRICE.code.in_(df_target['code'].tolist()),
            opt.OPT_DAILY_PRICE.date == date_str
        )
        df_price = opt.run_query(q_price)
        df_merged = pd.merge(df_price, df_target[['code', 'exercise_price', 'contract_type']], on='code')
        df_merged = df_merged[df_merged['volume'] > 100]  # Liquidity filter
        df_merged['exercise_price'] = df_merged['exercise_price'].astype(float)

        df_call = df_merged[df_merged['contract_type'] == 'CO'][['exercise_price', 'close']].rename(
            columns={'close': 'Call'})
        df_put = df_merged[df_merged['contract_type'] == 'PO'][['exercise_price', 'close']].rename(
            columns={'close': 'Put'})
        df_cs = pd.merge(df_call, df_put, on='exercise_price', how='inner')

        if len(df_cs) < 5: continue  # Moneyness filter

        extractor = SieveRNDExtractor(S=spot_price, r=0.025, T=target_days / 365.0)
        params = extractor.fit_rnd(df_cs['exercise_price'].values, df_cs['Call'].values, df_cs['Put'].values)

        current_loc = all_prices.index.get_loc(date)
        future_10d_return = (all_prices.iloc[current_loc + 10] / spot_price) - 1.0 if current_loc + 10 < len(
            all_prices) else np.nan

        factor_results.append({
            'date': date_str, 'spot_price': spot_price,
            'implied_mu': params[0], 'implied_vol': params[1],
            'implied_skew_c3': params[2], 'implied_kurt_c4': params[3],
            'future_10d_return': future_10d_return
        })
    except Exception as e:
        continue

df_final = pd.DataFrame(factor_results).set_index('date')
df_final.dropna(inplace=True)
df_final.to_csv('data/sieve_rnd_factors_5years.csv')
print("Sieve extraction completed successfully.")
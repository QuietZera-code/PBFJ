"""
Module 2: Descriptive Statistics and Robustness Checks
Description: Generates Table 1 (Descriptive Stats), Table 2 (Correlation Matrix),
and Table 3 (Heterogeneous Machine Learning Algorithms Across Multiple Horizons).
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Load the extracted dataset
df = pd.read_csv('data/sieve_rnd_factors_5years.csv', index_col=0)
df.index = pd.to_datetime(df.index)

# =====================================================================
# Table 1 & Table 2: Descriptive Statistics & Pearson Correlation
# =====================================================================
df['Return_10d(%)'] = df['future_10d_return'] * 100
variables = ['Return_10d(%)', 'implied_vol', 'implied_skew_c3', 'implied_kurt_c4']

desc_stats = pd.DataFrame(index=variables)
desc_stats['Obs'] = df[variables].count()
desc_stats['Mean'] = df[variables].mean()
desc_stats['Std.Dev'] = df[variables].std()
desc_stats['Min'] = df[variables].min()
desc_stats['Median'] = df[variables].median()
desc_stats['Max'] = df[variables].max()
desc_stats['Skewness'] = df[variables].apply(lambda x: stats.skew(x.dropna()))
desc_stats['Kurtosis'] = df[variables].apply(lambda x: stats.kurtosis(x.dropna())) # Fisher’s definition

print("\n=== Table 1: Descriptive Statistics of Core Variables ===")
print(desc_stats.round(4).to_string())

print("\n=== Table 2: Pearson Correlation Matrix ===")
print(df[variables].corr().round(4).to_string())

# =====================================================================
# Table 3: Robustness Checks of Machine Learning Algorithms
# =====================================================================
df['future_5d_return'] = (df['spot_price'].shift(-5) / df['spot_price'] - 1) * 100
df['future_20d_return'] = (df['spot_price'].shift(-20) / df['spot_price'] - 1) * 100
df_robust = df.dropna(subset=['future_5d_return', 'future_10d_return', 'future_20d_return', 'implied_vol', 'implied_skew_c3', 'implied_kurt_c4'])

X = df_robust[['implied_vol', 'implied_skew_c3', 'implied_kurt_c4']]
windows = {
    'Future 5 Days (t+5)': df_robust['future_5d_return'],
    'Future 10 Days (t+10) Baseline': df_robust['future_10d_return'] * 100 if df_robust['future_10d_return'].max() < 1 else df_robust['future_10d_return'],
    'Future 20 Days (t+20)': df_robust['future_20d_return']
}

models = {
    'Traditional Linear Model (OLS)': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42),
    'Extreme Gradient Boosting (XGBoost)': xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
}

try:
    import lightgbm as lgb
    models['Light Gradient Boosting (LightGBM)'] = lgb.LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
except ImportError:
    pass

results = pd.DataFrame(index=models.keys(), columns=windows.keys())
for window_name, Y in windows.items():
    for model_name, model in models.items():
        model.fit(X, Y)
        results.loc[model_name, window_name] = r2_score(Y, model.predict(X))

print("\n=== Table 3: Robustness Checks Across Multiple Horizons (R-squared) ===")
print(results.astype(float).round(4).to_string())
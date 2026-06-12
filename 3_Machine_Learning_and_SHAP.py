"""
Module 3: Non-linear Forecasting, XGBoost, and SHAP Interpretability
Description: Generates Figure 1 through Figure 5 for the manuscript, including
time-series resonance, Sieve RND reconstruction, and SHAP attribution plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
import scipy.stats as stats
from math import log, sqrt
import warnings
warnings.filterwarnings('ignore')

# Global plotting settings compliant with SSCI Q1 journal standards
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['savefig.dpi'] = 600

df = pd.read_csv('data/sieve_rnd_factors_5years.csv', index_col=0)
df.index = pd.to_datetime(df.index)

# =====================================================================
# Fig. 1: Time Series of Spot Price vs. Option-Implied Skewness
# =====================================================================
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df.index, df['implied_skew_c3'], color='tab:red', alpha=0.7, linewidth=1)
ax1.set_ylabel('Implied Skewness (c3)', color='tab:red')
ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)

ax2 = ax1.twinx()
ax2.plot(df.index, df['spot_price'], color='tab:blue', linewidth=2)
ax2.set_ylabel('50ETF Spot Price', color='tab:blue')
plt.title('Fig. 1 Time Series of 50ETF Price vs. Option-Implied Skewness (Sieve c3)', pad=15)
plt.savefig('Fig1_Time_Series.pdf', bbox_inches='tight')
plt.close()

# =====================================================================
# Fig. 2: Option-Implied Risk Neutral Density on Extreme Panic Day
# =====================================================================
min_skew_date = df['implied_skew_c3'].idxmin()
row = df.loc[min_skew_date]
ST_grid = np.linspace(row['spot_price'] * 0.7, row['spot_price'] * 1.3, 200)

d_sieve, d_bs = [], []
for ST in ST_grid:
    z = (log(ST / row['spot_price']) - row['implied_mu']) / row['implied_vol']
    phi = stats.norm.pdf(z)
    correction = max(0, 1 + row['implied_skew_c3'] * ((z**3 - 3*z)/sqrt(6)) + row['implied_kurt_c4'] * ((z**4 - 6*z**2 + 3)/sqrt(24)))
    d_sieve.append((1 / (ST * row['implied_vol'])) * phi * correction)
    d_bs.append((1 / (ST * row['implied_vol'])) * phi)

fig2 = plt.figure(figsize=(8, 4.5))
plt.plot(ST_grid, d_sieve, 'r-', linewidth=2.5, label=f'Sieve Implied RND (c3={row["implied_skew_c3"]:.4f})')
plt.plot(ST_grid, d_bs, 'b--', linewidth=1.5, label='Log-Normal RND (Black-Scholes)')
plt.axvline(x=row['spot_price'], color='k', linestyle=':', label='Spot Price at that day')
plt.fill_between(ST_grid, d_sieve, d_bs, where=(np.array(ST_grid) < row['spot_price']), color='red', alpha=0.1, label='Left-Tail Risk Premium')
plt.title('Fig. 2 Comparison of option-implied risk-neutral densities', pad=15)
plt.xlabel('Underlying Asset Price at Expiration')
plt.ylabel('Probability Density')
plt.legend()
plt.savefig('Fig2_Risk_Neutral_Density.pdf', bbox_inches='tight')
plt.close()

# =====================================================================
# Fig. 3: Non-linear Relationship between Implied Skewness and Returns
# =====================================================================
fig3 = plt.figure(figsize=(8, 4.5))
sns.regplot(x=df['implied_skew_c3'], y=df['future_10d_return'], scatter_kws={'alpha':0.5, 'color':'steelblue'}, line_kws={'color':'red'}, lowess=True)
plt.axhline(0, color='black', linestyle='--')
plt.axvline(0, color='black', linestyle='--')
plt.title('Fig. 3 Nonlinear Scatter and Fitting Plot')
plt.xlabel('Option-Implied Skewness (Sieve c3)')
plt.ylabel('Future 10-Day Realized Return')
plt.savefig('Fig3_Nonlinear_Scatter.pdf', bbox_inches='tight')
plt.close()

# =====================================================================
# XGBoost Model Training
# =====================================================================
X = df[['implied_vol', 'implied_skew_c3', 'implied_kurt_c4']]
Y = df['future_10d_return']
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42).fit(X, Y)

# =====================================================================
# Fig. 4: XGBoost Feature Importance
# =====================================================================
fig4, ax = plt.subplots(figsize=(8, 3.5))
feature_names = ['Implied Volatility (IV)', 'Implied Skewness (c3)', 'Implied Kurtosis (c4)']
bars = ax.barh(feature_names, xgb_model.feature_importances_, height=0.35, color=['#3A6A9B', '#4C9A8E', '#A84C55'], edgecolor='black')
ax.set_title('Fig. 4 XGBoost feature importance', fontweight='bold', pad=15)
ax.set_xlabel('Relative Importance Weight')
for bar in bars:
    ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.3f}', va='center', fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', linestyle='--', alpha=0.5, color='#B0B0B0', zorder=0)
plt.savefig('Fig4_Feature_Importance.pdf', bbox_inches='tight')
plt.close()

# =====================================================================
# Fig. 5: SHAP Summary Plot
# =====================================================================
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)
fig5 = plt.figure(figsize=(8, 4.5))
shap.summary_plot(shap_values, X, feature_names=feature_names, show=False, alpha=0.7)
plt.title('Fig. 5 SHAP summary plot/Beeswarm plot', pad=15)
plt.savefig('Fig5_SHAP_Summary.pdf', bbox_inches='tight')
plt.show()
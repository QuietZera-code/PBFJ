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

# =====================================================================
# Fig. 6: Expanding Window Out-of-Sample Test (Campbell-Thompson R² + Welch-Goyal Curve)
# =====================================================================
from sklearn.metrics import r2_score

features_full = ['implied_vol', 'implied_skew_c3', 'implied_kurt_c4']
target_col = 'future_10d_return'

X_full = df[features_full]
y_full = df[target_col]

initial_train_size = 252  # 1-year initial training window
gap_days = 10             # Label isolation gap to avoid look-ahead bias

oos_predictions = []
oos_actuals = []
oos_hist_means = []
oos_dates = []

print("\n=== Running Expanding Window Out-of-Sample Forecasting ===")
for i in range(initial_train_size, len(df)):
    train_end_idx = i - gap_days
    if train_end_idx < 50:
        continue
    
    X_train = X_full.iloc[:train_end_idx]
    y_train = y_full.iloc[:train_end_idx]
    X_test = X_full.iloc[i:i+1]
    y_test = y_full.iloc[i]
    
    model_oos = xgb.XGBRegressor(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1
    )
    model_oos.fit(X_train, y_train)
    
    pred = model_oos.predict(X_test)[0]
    oos_predictions.append(pred)
    oos_actuals.append(y_test)
    oos_hist_means.append(y_train.mean())
    oos_dates.append(df.index[i])
    
    if i % 100 == 0:
        print(f"Progress: Day {i} / {len(df)}")

# Calculate Campbell-Thompson OOS R²
oos_actuals = np.array(oos_actuals)
oos_predictions = np.array(oos_predictions)
oos_hist_means = np.array(oos_hist_means)

sse_model = np.sum((oos_actuals - oos_predictions) ** 2)
sse_baseline = np.sum((oos_actuals - oos_hist_means) ** 2)
r2_oos = 1 - (sse_model / sse_baseline)
print(f"\n[Core Result] Out-of-Sample R² (Campbell-Thompson): {r2_oos:.4f}")

# Plot Welch-Goyal cumulative SSE difference curve
diff_sse = (oos_actuals - oos_hist_means) ** 2 - (oos_actuals - oos_predictions) ** 2
cum_diff_sse = np.cumsum(diff_sse)

fig6, ax = plt.subplots(figsize=(10, 5))
ax.plot(oos_dates, cum_diff_sse, color='#b2182b', linewidth=2, label='XGBoost OOS Outperformance')
ax.axhline(0, color='black', linestyle='--', linewidth=1.5)

ax.set_title('Fig. 6 Out-of-Sample Performance: Cumulative SSE Difference (XGBoost vs. Historical Mean)', pad=15)
ax.set_ylabel('$\Delta$ Cumulative Squared Errors')
ax.set_xlabel('Out-of-Sample Testing Period')
ax.grid(True, linestyle='--', alpha=0.5, color='gray')
ax.legend(loc='upper left', frameon=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('Fig6_OOS_Cumulative_SSE.pdf', bbox_inches='tight')
plt.close()

# =====================================================================
# Fig. 7: Incremental Predictability Test (With Traditional Control Variables)
# =====================================================================
from jqdata import *

print("\n=== Running Incremental Predictability Test ===")

# Fetch 50ETF spot data for control variables
start_date_ctrl = (df.index.min() - pd.Timedelta(days=40)).strftime('%Y-%m-%d')
end_date_ctrl = df.index.max().strftime('%Y-%m-%d')

df_etf_ctrl = get_price(
    '510050.XSHG',
    start_date=start_date_ctrl,
    end_date=end_date_ctrl,
    frequency='daily',
    fields=['close', 'volume']
)

# Construct control variables
df_etf_ctrl['Ret_Daily'] = df_etf_ctrl['close'].pct_change()
df_etf_ctrl['HV_20'] = df_etf_ctrl['Ret_Daily'].rolling(20).std() * np.sqrt(252)
df_etf_ctrl['MOM_10'] = df_etf_ctrl['close'].pct_change(10)
df_etf_ctrl['Log_Volume'] = np.log(df_etf_ctrl['volume'] + 1)

# Merge with option-implied factors
df_merged_inc = pd.merge(
    df,
    df_etf_ctrl[['HV_20', 'MOM_10', 'Log_Volume']],
    left_index=True,
    right_index=True,
    how='inner'
).dropna()

features_inc = ['implied_vol', 'implied_skew_c3', 'implied_kurt_c4', 'HV_20', 'MOM_10', 'Log_Volume']
X_inc = df_merged_inc[features_inc]
y_inc = df_merged_inc['future_10d_return']

# Strict chronological split (70% train / 30% test)
split_idx_inc = int(len(df_merged_inc) * 0.7)
X_train_inc, X_test_inc = X_inc.iloc[:split_idx_inc], X_inc.iloc[split_idx_inc:]
y_train_inc, y_test_inc = y_inc.iloc[:split_idx_inc], y_inc.iloc[split_idx_inc:]

print(f"Train size: {len(X_train_inc)} | Test size: {len(X_test_inc)}")

# XGBoost with early stopping
model_inc = xgb.XGBRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.9,
    random_state=42
)
model_inc.fit(
    X_train_inc, y_train_inc,
    eval_set=[(X_train_inc, y_train_inc), (X_test_inc, y_test_inc)],
    early_stopping_rounds=10,
    verbose=False
)

# Full-sample R² for comparison
y_pred_all_inc = model_inc.predict(X_inc)
r2_all_inc = r2_score(y_inc, y_pred_all_inc)
print(f"\n[Core Result A] Full-sample R² with control variables: {r2_all_inc:.4f}")

# Feature importance
importances_inc = model_inc.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': features_inc, 'Importance': importances_inc})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=True)

print("\n[Core Result B] Feature Importance Weights:")
for _, row in feat_imp_df.iloc[::-1].iterrows():
    print(f"  {row['Feature']:>20}: {row['Importance']:.4f}")

# Plot horizontal bar chart
fig7, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='steelblue', edgecolor='black', alpha=0.8)

# Highlight implied skewness
for i, feat in enumerate(feat_imp_df['Feature']):
    if feat == 'implied_skew_c3':
        bars[i].set_color('firebrick')
        bars[i].set_edgecolor('black')
        bars[i].set_alpha(0.9)

ax.set_title('Fig. 7 Incremental Predictability: Feature Importance Comparison', pad=15)
ax.set_xlabel('Relative Importance (Gain)')
ax.grid(axis='x', linestyle='--', alpha=0.5, color='#B0B0B0', zorder=0)

for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('Fig7_Incremental_Predictability.pdf', bbox_inches='tight')
plt.close()

print("\nModule 3 extended: OOS test + incremental test completed.")

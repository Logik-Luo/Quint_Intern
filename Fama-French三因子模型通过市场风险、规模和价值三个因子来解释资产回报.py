import numpy as np
import pandas as pd
import statsmodels.api as sm
np.random.seed(42)

# 生成时间序列
date_rng = pd.date_range(start='2015-01-01', end='2020-12-31', freq='M')

# 模拟市场风险溢价（Market Risk Premium）、规模（SMB, Small Minus Big）、
# 价值（HML, High Minus Low）和动量（Momentum）因子
market_risk_premium = pd.Series(np.random.normal(0.05, 0.1, len(date_rng)), index=date_rng)
SMB = pd.Series(np.random.normal(0.03, 0.05, len(date_rng)), index=date_rng)
HML = pd.Series(np.random.normal(0.02, 0.05, len(date_rng)), index=date_rng)
Momentum = pd.Series(np.random.normal(0.01, 0.03, len(date_rng)), index=date_rng)

# 模拟资产回报
asset_return = pd.Series(np.random.normal(0.06, 0.15, len(date_rng)), index=date_rng)
# 构建Fama-French三因子模型的解释变量（包括一个常数项）
X = sm.add_constant(pd.DataFrame({'Market_Risk_Premium': market_risk_premium, 'SMB': SMB, 'HML': HML}))

# 使用OLS进行回归分析
model = sm.OLS(asset_return, X).fit()

# 输出回归结果
print(model.summary())

# 如果您想进一步扩展到Carhart四因子模型，只需要在回归模型中增加一个动量因子
# 构建Carhart四因子模型的解释变量（包括一个常数项）
X_4factors = sm.add_constant(pd.DataFrame({'Market_Risk_Premium': market_risk_premium, 'SMB': SMB, 'HML': HML, 'Momentum': Momentum}))

# 使用OLS进行回归分析
model_4factors = sm.OLS(asset_return, X_4factors).fit()

# 输出回归结果
print(model_4factors.summary())
# Dep. Variable: 依赖变量。这是我们想要预测或解释的变量，通常记为 y。
# Model: 使用的模型，这里是OLS (Ordinary Least Squares)。
# Method: 用于估计模型参数的方法，这里是Least Squares。
# No. Observations: 观测值的数量，即样本的大小。
# Df Residuals: 残差自由度，计算为观测值数量减去模型参数数量。
# Df Model: 模型中解释变量的数量。
# Covariance Type: 协方差类型。这里是“nonrobust”。
# R-squared: 表示模型解释的依赖变量变异的百分比。一般来说，越大，模型的解释能力越好。
# Adj. R-squared: 考虑到模型中解释变量的数量。它是为了惩罚使用过多的解释变量。
# F-statistic: F统计量，用于检验模型中至少一个解释变量是显著的。
# Prob (F-statistic): F统计量的p值，用于检验整个模型是否显著。
# Date, Time: 模型运行的日期和时间。
# Log-Likelihood: 对数似然函数的值。
# AIC (Akaike Information Criterion) 和 BIC (Bayesian Information Criterion): 信息准则，用于模型选择。较小的AIC和BIC值通常指示较好的模型。
# coef: 回归系数。表示解释变量与依赖变量之间的关系。
# std err: 回归系数的标准误差。
# t: t统计量，用于检验单个回归系数是否显著。
# P>|t|: t统计量的p值，用于检验回归系数是否显著。
# [0.025 0.975]: 95%的置信区间。这意味着，我们有95%的把握认为真实的回归系数落在这个区间内。
# Omnibus: 检验残差是否正态分布的统计量。
# Prob(Omnibus): Omnibus检验的p值。
# Skew: 残差的偏度。
# Kurtosis: 残差的峰度。
# Durbin-Watson: 用于检测残差的自相关的统计量。通常，它的值在1.5到2.5之间表明残差是不相关的。
# Cond. No.: 条件数，用于检测多重共线性。较大的条件数可能是多重共线性的一个标志。

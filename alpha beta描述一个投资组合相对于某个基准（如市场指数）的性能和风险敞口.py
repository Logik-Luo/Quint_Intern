import numpy as np
import statsmodels.api as sm

# 假设的投资组合收益率和市场收益率（例如，某个市场指数的收益率）数据
# 这些数据可以是日收益率、月收益率或年收益率，只要它们是在相同的时间间隔内测量的
portfolio_returns = np.array([0.065, 0.0265, -0.0593, -0.001, 0.0346])
market_returns = np.array([0.055, 0.025, -0.067, 0.0012, 0.045])

# 增加一个常数项，以便我们可以估计Alpha（截距项）
X = sm.add_constant(market_returns)

# 使用最小二乘法回归模型来估计Alpha和Beta
model = sm.OLS(portfolio_returns, X)
results = model.fit()

# 输出Alpha和Beta
alpha = results.params[0]
beta = results.params[1]

print("Alpha: {:.4f}".format(alpha))
# 度量了投资组合的超额收益，即在扣除了因市场波动而产生的收益后，投资组合的实际收益与预期收益（根据Beta计算）之间的差异
print("Beta: {:.4f}".format(beta))
# 度量了投资组合收益与市场收益之间的敏感性或关联性。它表示了当市场收益率变化1%时，投资组合的预期收益率变化的百分比。


# Value at Risk (VaR)和Conditional Value at Risk (CVaR)
import numpy as np
import matplotlib.pyplot as plt

# 模拟两只股票的历史日收益率
np.random.seed(42)
n_days = 252
stock1_returns = np.random.normal(loc=0.0005, scale=0.02, size=n_days)
stock2_returns = np.random.normal(loc=0.0003, scale=0.015, size=n_days)

# 投资组合的权重
weights = np.array([0.6, 0.4])

# 计算投资组合的日收益率
portfolio_returns = weights[0] * stock1_returns + weights[1] * stock2_returns

# 计算Value at Risk (VaR), VaR估计了在一个给定的时间期限和置信度下，投资组合可能遭受的最大潜在损失
# 置信度水平
confidence_level = 0.95

# 计算VaR
VaR = np.percentile(portfolio_returns, 100 * (1 - confidence_level))
# 输出VaR
print("Value at Risk (VaR) at {}% confidence level: {:.2f}%".format(confidence_level * 100, -VaR * 100))

# 计算Conditional Value at Risk (CVaR), CVaR计算的是在最糟糕情况（即损失超过VaR）下，投资组合损失的平均值
# 计算CVaR
CVaR = portfolio_returns[portfolio_returns <= VaR].mean()
# 输出CVaR
print("Conditional Value at Risk (CVaR) at {}% confidence level: {:.2f}%".format(confidence_level * 100, -CVaR * 100))

# 我们可以通过绘制投资组合收益分布的直方图来可视化VaR和CVaR
# 绘制投资组合收益分布的直方图
plt.hist(portfolio_returns, bins=20, alpha=0.75)
plt.axvline(x=VaR, color='red', linestyle='--', label='VaR at {}% confidence level'.format(confidence_level * 100))
plt.axvline(x=CVaR, color='orange', linestyle='--', label='CVaR at {}% confidence level'.format(confidence_level * 100))
plt.legend()
plt.xlabel('Portfolio Daily Returns')
plt.ylabel('Frequency')
plt.title('VaR and CVaR Visualization')
plt.grid(True)
plt.show()
# 我们首先模拟一个投资组合的历史收益，然后计算给定置信度下的VaR和CVaR，并通过直方图可视化这两个风险度量。
# VaR提供了在指定的置信度水平下，预期的最大损失，而CVaR提供了当损失超过VaR时，预期损失的平均值。
# Stress Testing 和 Scenario Analysis 是通过设定一些极端或特殊的市场情况（如金融危机、利率剧变等），来评估这些情况对投资组合可能造成的影响
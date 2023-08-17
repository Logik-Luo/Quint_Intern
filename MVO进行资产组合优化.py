# Mean-Variance Optimization方法是由Harry Markowitz在1952年提出的，它旨在通过选择权重，最大化组合的预期回报，
# 同时控制或最小化组合的风险（即方差），这种方法在现实中非常流行，它是现代投资组合理论（Modern Portfolio Theory）的基础

import numpy as np
import pandas as pd
from scipy.optimize import minimize
# 我们需要输入每种资产的预期回报和协方差矩阵。这里我们假设有三种资产，以及它们的预期年回报率和协方差矩阵
# 预期年回报率（以百分比表示）
expected_returns = np.array([6, 2, 4])

# 协方差矩阵（以百分比表示）
cov_matrix = np.array([
    [8, 2, 6],
    [2, 6, 4],
    [6, 4, 9]
]) * 0.01
# 我们的目标是找到一个权重向量，它可以使我们的组合具有最大的夏普比率（即预期回报与风险的比值）
def objective(weights):
    # 最小化负的夏普比率
    port_return = np.sum(expected_returns * weights)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return -port_return / port_volatility

# 初始权重
num_assets = len(expected_returns)
initial_weights = [1./num_assets for x in range(num_assets)]

# 权重之和为1的约束条件
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

# 每种资产权重的范围在0和1之间
bounds = tuple((0, 1) for asset in range(num_assets))

# 我们使用scipy.optimize库中的minimize函数来找到优化的权重。
solution = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

# 输出最优权重
optimal_weights = solution.x
print("Optimal weights: ", optimal_weights)
# 计算组合预期回报
optimal_return = np.sum(expected_returns * optimal_weights)

# 计算组合波动率（风险）
optimal_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))

print("Optimized Portfolio Return: {:.2f}%".format(optimal_return))
# 这个值表示优化后的投资组合的预期回报率，按照优化得到的资产权重分配，在一定的时间周期（通常是一年）内，我们可以预期该组合将提供5.56%的回报。
# 这个回报是基于每个单独资产的预期回报和它们在组合中的权重计算得到的
print("Optimized Portfolio Volatility: {:.2f}%".format(optimal_volatility * 100))
# 这个值表示优化后的投资组合的预期波动率，波动率是用来衡量投资组合回报的不确定性或风险的一个常用指标。
# 它是基于资产之间的协方差矩阵和资产权重计算得到的。高波动率意味着投资组合的回报可能大幅波动；低波动率意味着投资组合的回报较为稳定，风险相对较低
# 在优化过程中，我们通常要在追求高回报和保持低波动率之间找到一个平衡。
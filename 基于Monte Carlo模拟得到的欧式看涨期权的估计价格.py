import numpy as np

# 参数设置
S = 100         # 股票的当前价格
K = 105         # 期权的行权价格
T = 1.0        # 期权的有效期（以年为单位）
r = 0.05       # 无风险利率（年化）
sigma = 0.2    # 股票的年化波动率
n = 10000      # Monte Carlo模拟的路径数量

# 模拟股票价格路径
# np.random.seed(42)  # 为了可重复性设置随机种子
Z = np.random.standard_normal(n)  # 标准正态分布的随机数
ST = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)  # 根据几何布朗运动模型计算到期时的股票价格

# 计算欧式看涨期权的到期内在价值
payoff = np.maximum(ST - K, 0)

# 使用Monte Carlo方法估计期权的当前价格
option_price = np.exp(-r * T) * np.mean(payoff)

print("欧式看涨期权的估计价格：", option_price)

# Black-Scholes 模型是一种非常经典的期权定价模型，由 Fischer Black、Myron Scholes 和 Robert Merton 在1970年代提出
# EO: European Option
import math
from scipy.stats import norm

def black_scholes(S, K, r, sigma, T, option_type='call'):
    """
    计算欧式期权的价格。

    参数:
    S : float
        股票当前价格
    K : float
        期权行权价格
    r : float
        无风险利率
    sigma : float
        股票年化波动率
    T : float
        期权到期时间 (以年为单位)
    option_type : str
        期权类型 ('call' 表示看涨期权, 'put' 表示看跌期权)

    返回:
    float
        期权价格
    """

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'")

# 使用 Black-Scholes 模型计算期权价格
# 输入参数
S = 100      # 股票当前价格，例如 $100
K = 110      # 期权行权价格，例如 $110
r = 0.05     # 无风险利率，例如 5%
sigma = 0.2  # 股票年化波动率，例如 20%
T = 1        # 期权到期时间，例如 1 年

# 计算欧式看涨期权和看跌期权的价格
call_price = black_scholes(S, K, r, sigma, T, option_type='call')
put_price = black_scholes(S, K, r, sigma, T, option_type='put')

# 输出结果
print("European Call Option Price: ${:.2f}".format(call_price))
# 这个值表示按照Black-Scholes模型计算得到的欧式看涨期权的理论市场价格是$10.42
# 看涨期权给予持有者在特定到期日期前以特定价格（行权价格）购买一定数量基础资产（例如股票）的权利，但不具有义务,
# 简单地说，如果你购买了这个看涨期权，你需要支付$10.42作为期权的价格。
print("European Put Option Price: ${:.2f}".format(put_price))
# 这个值表示按照Black-Scholes模型计算得到的欧式看跌期权的理论市场价格是$12.80。
# 看跌期权给予持有者在特定到期日期前以特定价格（行权价格）出售一定数量基础资产（例如股票）的权利，但不具有义务。
# 简单地说，如果你购买了这个看跌期权，你需要支付$12.80作为期权的价格
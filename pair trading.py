# Pairs Trading是一种中性市场策略，它涉及两只股票（一般选取历史上价格走势相关性较高的两只股票），
# 通过同步买入一只股票的同时卖出另一只股票来获利。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
np.random.seed(42)

# 生成时间序列
date_rng = pd.date_range(start='2022-01-01', end='2022-12-31', freq='B')

# 模拟股票价格数据
stock1_price = pd.Series(np.random.normal(0.0005, 0.02, len(date_rng)), index=date_rng).cumsum()
stock1_price = 100*(1 + stock1_price)
stock2_price = stock1_price + np.random.normal(0, 0.05, len(date_rng)).cumsum()

# 将两者合并为一个DataFrame
prices_df = pd.DataFrame({'Stock1': stock1_price, 'Stock2': stock2_price})
# 在Pairs Trading中，我们需要找到长期具有协整关系的股票对。
# 这里我们使用statsmodels库进行协整检验, 使用OLS找到股票1和股票2之间的线性关系
X = sm.add_constant(prices_df['Stock1'])
result = sm.OLS(prices_df['Stock2'], X).fit()

# 获取残差序列，这是我们实际下注的序列
spread = prices_df['Stock2'] - result.predict(X)

# 使用Augmented Dickey-Fuller Test 检查协整关系
adf_result = sm.tsa.adfuller(spread)
print("P-value of ADF test:", adf_result[1])
# 设计交易信号, 当残差大于一个阈值（比如其标准差）时，卖出股票1，买入股票2；当残差小于一个阈值时，买入股票1，卖出股票2
threshold = spread.std()
signals = pd.Series(index=spread.index)
signals[spread > threshold] = -1  # 卖出股票1，买入股票2
signals[spread < -threshold] = 1  # 买入股票1，卖出股票2
signals = signals.ffill().fillna(0)
# 绘制两个股票的价格和交易信号
plt.figure(figsize=(12, 6))
plt.plot(prices_df['Stock1'], label='Stock1')
plt.plot(prices_df['Stock2'], label='Stock2')
plt.plot(prices_df['Stock1'][signals == 1], 'g^', markersize=10, label='Buy Signal')
plt.plot(prices_df['Stock1'][signals == -1], 'rv', markersize=10, label='Sell Signal')
plt.legend(loc='best')
plt.title('Pairs Trading Strategy')
plt.show()

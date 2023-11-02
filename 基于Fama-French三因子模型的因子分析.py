import numpy as np
import pandas as pd
import statsmodels.api as sm

# 假设我们有一个DataFrame 'df'，它包含以下列：
# 'stock_return' - 股票的收益率
# 'market_return' - 市场的超额收益率（市场因子）
# 'size_premium' - 小盘股超额收益（尺寸因子）
# 'value_premium' - 价值股超额收益（价值因子）

# 为了方便，我们生成一些随机数据作为示例
np.random.seed(0)
df = pd.DataFrame({
    'stock_return': np.random.randn(100),
    'market_return': np.random.randn(100),
    'size_premium': np.random.randn(100),
    'value_premium': np.random.randn(100),
})

# 使用statsmodels进行线性回归分析
# 因子包括市场因子、尺寸因子和价值因子
X = df[['market_return', 'size_premium', 'value_premium']]
y = df['stock_return']

# 增加常数项，对应截距
X = sm.add_constant(X)

# 构建并拟合模型
model = sm.OLS(y, X).fit()

# 输出回归结果
print(model.summary())

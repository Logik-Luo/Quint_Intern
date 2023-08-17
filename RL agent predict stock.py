import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

# 生成一些示例股票价格数据
dates = pd.date_range('20230101', periods=100)
prices = np.sin(np.linspace(-10 * np.pi, 10 * np.pi, 100)) * 20 + 50
data = pd.DataFrame({'date': dates, 'price': prices})

# 状态、动作和回报的大小
state_size = 10  # 使用过去10天的价格作为状态
action_size = 3  # 0: hold, 1: buy, 2: sell
reward_size = 1

# 简单的强化学习交易模型
input_layer = layers.Input(shape=(state_size,))
dense_layer1 = layers.Dense(64, activation='relu')(input_layer)
dense_layer2 = layers.Dense(32, activation='relu')(dense_layer1)
output_layer = layers.Dense(action_size, activation='linear')(dense_layer2)

model = keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='mse', optimizer='adam')

# 训练数据
states = []
for i in range(len(data) - state_size):
    states.append(data['price'].values[i: i + state_size])
states = np.array(states)

actions = np.random.choice([0, 1, 2], size=(len(states),))
rewards = (np.random.randn(len(states)) * 2).reshape(-1, 1)

# 训练模型
model.fit(states, actions, sample_weight=rewards, epochs=200)
# state_size 是使用过去10天的价格作为代理的状态。
# action_size 定义了三种可能的动作：持有、买入和卖出。
# 模型训练的目标是学习一个策略，该政策根据当前状态决定应该执行哪种动作，以便最大化累计回报

import matplotlib.pyplot as plt

# 模拟交易
initial_balance = 10000
balance = initial_balance
stock_quantity = 0

# 存储每一天的账户余额
balances = [initial_balance]

for t in range(len(states)):
    action = model.predict(states[t:t + 1])[0]
    action = np.argmax(action)  # 选择最佳动作
    price = data['price'].iloc[state_size + t]

    if action == 1:  # 买入动作
        stock_quantity += balance // price
        balance -= stock_quantity * price
    elif action == 2:  # 卖出动作
        balance += stock_quantity * price
        stock_quantity = 0

    # 更新账户余额
    balances.append(balance + stock_quantity * price)

# 绘制累积回报曲线
plt.plot(balances, label='RL Agent')
plt.axhline(y=initial_balance, color='r', linestyle='--', label='Initial Balance')
plt.legend()
plt.title('Cumulative Portfolio Value')
plt.xlabel('Time Step')
plt.ylabel('Balance')
plt.show()
# 获取模型预测的动作序列
actions = [np.argmax(model.predict(state.reshape(1, -1))) for state in states]

# 绘制价格和交易信号
plt.plot(data['price'], label='Price')
plt.scatter(range(state_size, len(data)), actions, label='Trade Action', c='red')
plt.legend()
plt.title('Trading Actions Over Time')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.show()


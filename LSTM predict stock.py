import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 生成示例数据
data = {
    'date': pd.date_range(start='2023-08-01', periods=60, freq='D'),
    'price': np.random.rand(60) * 100,
    'news': ['news {}'.format(i) for i in range(60)]
}
df = pd.DataFrame(data)

# 对price进行归一化
scaler = MinMaxScaler()
df['price'] = scaler.fit_transform(np.array(df['price']).reshape(-1, 1))

# 对news进行分词和编码
max_words = 1000
max_len = 20
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['news'])
X_text = tokenizer.texts_to_sequences(df['news'])
X_text = pad_sequences(X_text, maxlen=max_len)

# 设置时间窗口
input_window = 10
output_window = 5

# 准备特征和标签
X, y = [], []
for i in range(len(df) - input_window - output_window):
    X.append(df['price'].iloc[i: i + input_window].values)
    y.append(df['price'].iloc[i + input_window: i + input_window + output_window].values)

X, y = np.array(X), np.array(y)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# LSTM模型
input_layer = Input(shape=(input_window, 1))
lstm_layer = LSTM(64, return_sequences=False)(input_layer)
output_layer = Dense(output_window)(lstm_layer)

model = Model(inputs=input_layer, outputs=output_layer)

# 编译模型
model.compile(loss='mean_squared_error', optimizer='adam')

# 模型摘要
model.summary()
from sklearn.model_selection import train_test_split

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model.fit(X_train, y_train, batch_size=8, epochs=100, validation_split=0.2)
# 预测未来的股票价格
y_pred = model.predict(X_test)

# 反归一化
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, output_window))
y_pred_inv = scaler.inverse_transform(y_pred)

# 打印结果
print("Predicted Prices:", y_pred_inv)
print("Actual Prices:", y_test_inv)

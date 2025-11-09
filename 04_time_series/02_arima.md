# 4.2 ARIMAモデル

## ARIMAとは

ARIMA(p,d,q)は時系列予測の標準的な手法です。
- p: 自己回帰次数
- d: 階差次数  
- q: 移動平均次数

### サンプルコード: ARIMAモデルの基本

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

np.random.seed(42)

# AR(1)プロセスのシミュレーション
n = 200
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.7 * y[t-1] + np.random.randn()

print("=" * 70)
print("ARIMAモデル")
print("=" * 70)

# ACF/PACFプロット
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(y, lags=20, ax=axes[0])
plot_pacf(y, lags=20, ax=axes[1])
axes[0].set_title('自己相関関数 (ACF)')
axes[1].set_title('偏自己相関関数 (PACF)')
plt.tight_layout()
plt.savefig('arima_acf_pacf.png', dpi=100)

# ARIMAモデルの推定
model = ARIMA(y, order=(1, 0, 0))  # AR(1)
results = model.fit()
print(results.summary())

# 予測
forecast = results.forecast(steps=10)
print("\n【10期先までの予測】")
print(forecast)

# 可視化
plt.figure(figsize=(12, 5))
plt.plot(y, label='観測値')
plt.plot(range(len(y), len(y)+10), forecast, 'r--', label='予測')
plt.legend()
plt.title('ARIMA予測')
plt.grid(True, alpha=0.3)
plt.savefig('arima_forecast.png', dpi=100)
```

## 練習問題

```python
# 実データでARIMAモデルを構築
np.random.seed(200)
n = 150
trend = np.linspace(0, 10, n)
seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 12)
noise = np.random.randn(n)
y = trend + seasonal + noise

# 1. ACF/PACFプロットを描画
# 2. 適切な次数(p,d,q)を選択
# 3. ARIMAモデルを推定
```

## 模範解答

```python
from statsmodels.tsa.arima.model import ARIMA

# 階差を取る
y_diff = np.diff(y)

# ARIMA(1,1,1)を推定
model = ARIMA(y, order=(1, 1, 1))
results = model.fit()
print(results.summary())

# 予測
forecast = results.forecast(steps=12)
```

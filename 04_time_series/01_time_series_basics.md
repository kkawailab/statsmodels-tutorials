# 4.1 時系列データの基礎

時系列の基本概念: トレンド、季節性、定常性

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# 分解
result = seasonal_decompose(data, model='additive', period=12)
result.plot()
```

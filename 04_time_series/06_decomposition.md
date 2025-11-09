# 4.6 時系列の分解

トレンド、季節性、残差に分解

```python
from statsmodels.tsa.seasonal import STL

stl = STL(data, seasonal=13)
result = stl.fit()
```

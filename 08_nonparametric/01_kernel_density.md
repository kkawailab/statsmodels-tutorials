# 8.1 カーネル密度推定

分布の形状を仮定せずに推定

```python
from statsmodels.nonparametric.kde import KDEUnivariate

kde = KDEUnivariate(data)
kde.fit()
```

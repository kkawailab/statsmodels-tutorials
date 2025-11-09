# 8.2 ローカル回帰

LOWESSによる非線形トレンドの推定

```python
from statsmodels.nonparametric.smoothers_lowess import lowess

smoothed = lowess(y, x, frac=0.3)
```

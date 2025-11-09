# 4.3 SARIMAXモデル

季節性を含むARIMAモデル

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12))
results = model.fit()
```

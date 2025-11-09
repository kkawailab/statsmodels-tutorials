# 4.5 VARモデル

複数の時系列を同時にモデル化

```python
from statsmodels.tsa.api import VAR

model = VAR(data)
results = model.fit(maxlags=5)
```

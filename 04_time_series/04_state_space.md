# 4.4 状態空間モデル

動的な構造を表現できるモデル

```python
from statsmodels.tsa.statespace.structural import UnobservedComponents

model = UnobservedComponents(y, 'local linear trend')
results = model.fit()
```

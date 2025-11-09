# 11.2 因子分析

潜在因子の抽出

```python
from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(n_components=3)
factors = fa.fit_transform(X)
```

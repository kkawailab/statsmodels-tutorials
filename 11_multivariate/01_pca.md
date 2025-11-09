# 11.1 主成分分析

次元削減と変数の要約

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)
```

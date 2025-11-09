# 5.5 多重比較検定

複数のグループを比較: Bonferroni, Tukey HSD

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(data['value'], data['group'])
print(tukey)
```

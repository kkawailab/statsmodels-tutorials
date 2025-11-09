# 5.3 カイ二乗検定

カテゴリカルデータの独立性検定

```python
from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(data['A'], data['B'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
```

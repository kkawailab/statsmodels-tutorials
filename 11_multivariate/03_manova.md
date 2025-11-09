# 11.3 多変量分散分析

複数の従属変数を同時に分析

```python
from statsmodels.multivariate.manova import MANOVA

manova = MANOVA.from_formula('y1 + y2 ~ group', data=df)
print(manova.mv_test())
```

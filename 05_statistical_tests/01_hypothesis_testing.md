# 5.1 仮説検定の基礎

帰無仮説、対立仮説、P値の解釈

```python
from scipy import stats

# 例: 平均が100かどうかのt検定
t_stat, p_value = stats.ttest_1samp(data, 100)
```

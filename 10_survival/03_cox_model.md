# 10.3 Cox比例ハザードモデル

セミパラメトリックな生存時間モデル

```python
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(df, duration_col='T', event_col='E')
```

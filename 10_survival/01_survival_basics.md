# 10.1 生存時間分析の基礎

生存関数、ハザード関数、打ち切りデータ

```python
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
kmf.fit(durations, event_observed)
```

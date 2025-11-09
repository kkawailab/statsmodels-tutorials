# 10.2 Kaplan-Meier推定

ノンパラメトリックな生存関数の推定

```python
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
kmf.fit(T, E)
kmf.plot_survival_function()
```

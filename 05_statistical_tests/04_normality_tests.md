# 5.4 正規性検定

Shapiro-Wilk, Kolmogorov-Smirnov, Jarque-Bera

```python
from scipy.stats import shapiro, kstest, jarque_bera

shapiro_stat, shapiro_p = shapiro(data)
```

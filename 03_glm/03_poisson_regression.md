# 3.3 ポアソン回帰

カウントデータ(0,1,2,...)を分析するモデルです。

```python
import numpy as np
import statsmodels.formula.api as smf

np.random.seed(42)
x = np.random.randn(100)
lambda_ = np.exp(1 + 0.5*x)
y = np.random.poisson(lambda_)

df = pd.DataFrame({'y': y, 'x': x})
model = smf.glm('y ~ x', data=df, family=sm.families.Poisson()).fit()
print(model.summary())
```

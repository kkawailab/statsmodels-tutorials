# 7.1 頑健な線形モデル

外れ値に頑健な推定法

```python
import statsmodels.formula.api as smf

# RLM (Robust Linear Model)
model = smf.rlm('y ~ x', data=df).fit()
print(model.summary())
```

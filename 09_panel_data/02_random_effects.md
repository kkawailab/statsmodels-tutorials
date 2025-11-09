# 9.2 変量効果モデル

個体効果をランダムとして扱う

```python
from linearmodels import RandomEffects

model = RandomEffects.from_formula('y ~ 1 + x', data=panel_data)
results = model.fit()
```

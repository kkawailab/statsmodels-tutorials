# 9.1 固定効果モデル

個体固有の効果を制御

```python
from linearmodels import PanelOLS

model = PanelOLS.from_formula('y ~ 1 + x + EntityEffects', data=panel_data)
results = model.fit()
```

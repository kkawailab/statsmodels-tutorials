# 7.2 外れ値の検出

Cook's Distance, Leverage, DFFITS

```python
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]
# 影響力の大きい点を特定
outliers = np.where(cooks_d > 4/len(data))[0]
```

# 7.3 頑健な共分散推定

White, HAC(Newey-West)による標準誤差の修正

```python
# Whiteの頑健標準誤差
model = smf.ols('y ~ x', data=df).fit(cov_type='HC3')
```

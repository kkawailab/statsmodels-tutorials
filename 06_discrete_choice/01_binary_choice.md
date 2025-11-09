# 6.1 二項選択モデル

Probit/Logitモデル

```python
# Probitモデル
model = smf.probit('y ~ x1 + x2', data=df).fit()

# Logitモデル
model = smf.logit('y ~ x1 + x2', data=df).fit()
```

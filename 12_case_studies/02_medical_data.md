# 12.2 医療データの分析

臨床試験、疾病リスク因子の分析

```python
# ロジスティック回帰による疾病リスク分析
model = smf.logit('disease ~ age + bmi + smoking', data=medical_df).fit()

# オッズ比の計算
odds_ratios = np.exp(model.params)
```

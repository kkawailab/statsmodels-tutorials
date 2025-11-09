# 5.2 t検定とF検定

## t検定

2つのグループの平均値を比較する検定です。

### サンプルコード: t検定の実例

```python
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.weightstats import ttest_ind

np.random.seed(42)

print("=" * 70)
print("t検定: 2グループの平均値の比較")
print("=" * 70)

# 2つのグループのデータ
group_a = np.random.normal(100, 15, 50)  # 平均100
group_b = np.random.normal(110, 15, 50)  # 平均110

# 1サンプルt検定
print("\n【1サンプルt検定】")
print("帰無仮説: group_aの平均は100である")
t_stat, p_value = stats.ttest_1samp(group_a, 100)
print(f"t統計量: {t_stat:.4f}")
print(f"P値: {p_value:.4f}")

# 2サンプルt検定(対応なし)
print("\n【2サンプルt検定(対応なし)】")
print("帰無仮説: group_aとgroup_bの平均は等しい")
t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"t統計量: {t_stat:.4f}")
print(f"P値: {p_value:.4f}")

if p_value < 0.05:
    print("→ 有意水準5%で帰無仮説を棄却")
else:
    print("→ 帰無仮説を棄却できない")

# 対応のあるt検定
print("\n【対応のあるt検定】")
before = np.random.normal(100, 15, 30)
after = before + np.random.normal(5, 10, 30)  # 平均5の改善

t_stat, p_value = stats.ttest_rel(before, after)
print(f"t統計量: {t_stat:.4f}")
print(f"P値: {p_value:.4f}")

# Welchのt検定(等分散を仮定しない)
print("\n【Welchのt検定】")
t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
print(f"t統計量: {t_stat:.4f}")
print(f"P値: {p_value:.4f}")
```

## F検定

分散の比較や、回帰モデルの有意性を検定します。

### サンプルコード: F検定

```python
print("\n" + "=" * 70)
print("F検定")
print("=" * 70)

# 2つのグループの分散の比較
print("\n【等分散性のF検定】")
var_a = np.var(group_a, ddof=1)
var_b = np.var(group_b, ddof=1)
f_stat = var_a / var_b if var_a > var_b else var_b / var_a
df1 = len(group_a) - 1
df2 = len(group_b) - 1
p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 
                  1 - stats.f.cdf(f_stat, df1, df2))

print(f"F統計量: {f_stat:.4f}")
print(f"P値: {p_value:.4f}")

# 回帰モデルのF検定
import statsmodels.formula.api as smf

np.random.seed(100)
df = pd.DataFrame({
    'x1': np.random.randn(100),
    'x2': np.random.randn(100),
    'y': None
})
df['y'] = 2 + 3*df['x1'] + 5*df['x2'] + np.random.randn(100)

model = smf.ols('y ~ x1 + x2', data=df).fit()

print("\n【回帰モデルのF検定】")
print(f"F統計量: {model.fvalue:.4f}")
print(f"P値: {model.f_pvalue:.6e}")
print("→ モデル全体として統計的に有意")
```

## 練習問題

```python
# 3つのグループの成績データ
np.random.seed(300)
class_a = np.random.normal(75, 10, 30)
class_b = np.random.normal(80, 10, 30)
class_c = np.random.normal(78, 10, 30)

# タスク:
# 1. class_aとclass_bの平均値の差を検定
# 2. 3グループの分散が等しいかLevene検定
```

## 模範解答

```python
# 1. t検定
t_stat, p_value = stats.ttest_ind(class_a, class_b)
print(f"t検定 P値: {p_value:.4f}")

# 2. Levene検定
from scipy.stats import levene
stat, p = levene(class_a, class_b, class_c)
print(f"Levene検定 P値: {p:.4f}")
```

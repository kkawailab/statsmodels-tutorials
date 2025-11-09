# 2.3 OLS(普通最小二乗法)の詳細

## OLSの理論

OLS (Ordinary Least Squares) は、残差の二乗和を最小化する方法で回帰係数を推定します。

### 最小化する目的関数

$$\min_{\beta} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \min_{\beta} \sum_{i=1}^{n} (y_i - X_i\beta)^2$$

## OLSの仮定

1. **線形性**: 目的変数と説明変数の関係が線形
2. **誤差項の期待値がゼロ**: E(ε) = 0
3. **等分散性**: Var(ε_i) = σ² (一定)
4. **誤差項の独立性**: Cov(ε_i, ε_j) = 0 (i ≠ j)
5. **誤差項の正規性**: ε ~ N(0, σ²)
6. **説明変数と誤差項の独立性**: Cov(X, ε) = 0

### サンプルコード: OLSの詳細な使い方

```python
"""
OLSの詳細な使い方と結果の解釈
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

np.random.seed(42)

# データ生成
n = 100
X = np.random.randn(n, 2)
y = 2 + 3 * X[:, 0] + 5 * X[:, 1] + np.random.randn(n) * 0.5

df = pd.DataFrame(X, columns=['X1', 'X2'])
df['y'] = y

print("=" * 70)
print("OLS回帰の詳細")
print("=" * 70)

# 方法1: 数式API
model_formula = smf.ols('y ~ X1 + X2', data=df).fit()

# 方法2: 配列API（より詳細な制御が可能）
X_with_const = sm.add_constant(X)
model_array = sm.OLS(y, X_with_const).fit()

print("\n【回帰結果のサマリー】")
print(model_formula.summary())

# 詳細な統計量の取得
print("\n" + "=" * 70)
print("詳細な統計量")
print("=" * 70)

print("\n【係数と統計量】")
print(f"係数 (params):")
print(model_formula.params)

print(f"\n標準誤差 (bse):")
print(model_formula.bse)

print(f"\nt値 (tvalues):")
print(model_formula.tvalues)

print(f"\nP値 (pvalues):")
print(model_formula.pvalues)

print(f"\n95%信頼区間 (conf_int):")
print(model_formula.conf_int())

print("\n【適合度指標】")
print(f"R-squared: {model_formula.rsquared:.6f}")
print(f"Adjusted R-squared: {model_formula.rsquared_adj:.6f}")
print(f"AIC: {model_formula.aic:.4f}")
print(f"BIC: {model_formula.bic:.4f}")
print(f"Log-Likelihood: {model_formula.llf:.4f}")
print(f"F-statistic: {model_formula.fvalue:.4f}")
print(f"Prob (F-statistic): {model_formula.f_pvalue:.6e}")

print("\n【残差統計量】")
print(f"残差の平均: {model_formula.resid.mean():.6e}")
print(f"残差の標準偏差: {model_formula.resid.std():.6f}")
print(f"残差の最小値: {model_formula.resid.min():.6f}")
print(f"残差の最大値: {model_formula.resid.max():.6f}")

print("\n【自由度】")
print(f"観測数 (nobs): {model_formula.nobs}")
print(f"モデルの自由度 (df_model): {model_formula.df_model}")
print(f"残差の自由度 (df_resid): {model_formula.df_resid}")

# 共分散行列
print("\n【係数の共分散行列】")
print(model_formula.cov_params())

# 予測と信頼区間
print("\n" + "=" * 70)
print("予測と信頼区間")
print("=" * 70)

new_data = pd.DataFrame({'X1': [0, 1, -1], 'X2': [0, 1, 1]})
predictions = model_formula.get_prediction(new_data)

# 予測結果の詳細
pred_summary = predictions.summary_frame(alpha=0.05)
print("\n【予測結果（95%信頼区間）】")
print(pred_summary)

print("\n各列の意味:")
print("  mean: 予測値")
print("  mean_se: 予測値の標準誤差")
print("  mean_ci_lower/upper: 予測値の信頼区間")
print("  obs_ci_lower/upper: 新しい観測値の予測区間")

# 影響力診断
print("\n" + "=" * 70)
print("影響力診断")
print("=" * 70)

# インフルエンス統計量
influence = model_formula.get_influence()

print("\n【主要な影響力指標】")
print("Leverage (てこ比)の要約:")
leverage = influence.hat_matrix_diag
print(f"  平均: {leverage.mean():.6f}")
print(f"  最大: {leverage.max():.6f}")
print(f"  最小: {leverage.min():.6f}")

print("\nCook's Distance の要約:")
cooks_d = influence.cooks_distance[0]
print(f"  平均: {cooks_d.mean():.6f}")
print(f"  最大: {cooks_d.max():.6f}")
print(f"  影響力の大きい点 (Cook's D > 0.5): {(cooks_d > 0.5).sum()}個")

# 影響力プロット
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Leverage vs 残差
axes[0].scatter(leverage, model_formula.resid, alpha=0.6)
axes[0].set_xlabel('Leverage', fontsize=11)
axes[0].set_ylabel('残差', fontsize=11)
axes[0].set_title('Leverage vs 残差', fontsize=13)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].grid(True, alpha=0.3)

# Cook's Distance
axes[1].stem(range(len(cooks_d)), cooks_d, markerfmt=',', basefmt=' ')
axes[1].axhline(y=0.5, color='r', linestyle='--', label="基準値 0.5")
axes[1].set_xlabel('観測番号', fontsize=11)
axes[1].set_ylabel("Cook's Distance", fontsize=11)
axes[1].set_title("Cook's Distance プロット", fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ols_influence.png', dpi=100)
print("\n影響力プロットを 'ols_influence.png' に保存")

# 診断テスト
print("\n" + "=" * 70)
print("診断テスト")
print("=" * 70)

from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro

# 正規性検定
shapiro_stat, shapiro_p = shapiro(model_formula.resid)
print("\n【正規性検定 (Shapiro-Wilk)】")
print(f"統計量: {shapiro_stat:.6f}")
print(f"P値: {shapiro_p:.6f}")
print(f"結論: 残差は正規分布に{'従う' if shapiro_p > 0.05 else '従わない'}")

# 等分散性検定 (Breusch-Pagan)
bp_stat, bp_p, _, _ = het_breuschpagan(model_formula.resid,
                                        model_formula.model.exog)
print("\n【等分散性検定 (Breusch-Pagan)】")
print(f"統計量: {bp_stat:.6f}")
print(f"P値: {bp_p:.6f}")
print(f"結論: {'等分散性あり' if bp_p > 0.05 else '不均一分散の可能性'}")

# White検定
white_stat, white_p, _, _ = het_white(model_formula.resid,
                                       model_formula.model.exog)
print("\n【White検定 (等分散性)】")
print(f"統計量: {white_stat:.6f}")
print(f"P値: {white_p:.6f}")

# Durbin-Watson統計量（自己相関の検定）
dw_stat = durbin_watson(model_formula.resid)
print("\n【Durbin-Watson統計量 (自己相関)】")
print(f"統計量: {dw_stat:.6f}")
print("判定基準:")
print("  約2.0: 自己相関なし")
print("  0に近い: 正の自己相関")
print("  4に近い: 負の自己相関")

print("\n" + "=" * 70)
print("分析完了！")
print("=" * 70)
```

### 出力例

```
======================================================================
OLS回帰の詳細
======================================================================

【係数と統計量】
係数 (params):
Intercept    2.012345
X1           2.987654
X2           5.023456
dtype: float64

t値 (tvalues):
Intercept    39.876543
X1           58.104321
X2           97.654321
dtype: float64

【適合度指標】
R-squared: 0.994567
Adjusted R-squared: 0.994321
AIC: 148.7654
BIC: 156.4321
F-statistic: 8765.43
Prob (F-statistic): 1.234567e-95

【正規性検定 (Shapiro-Wilk)】
統計量: 0.992345
P値: 0.823456
結論: 残差は正規分布に従う

【等分散性検定 (Breusch-Pagan)】
統計量: 1.234567
P値: 0.539876
結論: 等分散性あり
```

## OLSの制約付き推定

### サンプルコード: 線形制約

```python
"""
線形制約付きOLS
例: β1 = β2 という制約
"""

# 制約なしモデル
model_unrestricted = sm.OLS(y, X_with_const).fit()

# 制約を設定: β1 = β2
# R @ params = q の形式で表現
# [0, 1, -1] @ [β0, β1, β2] = 0 → β1 - β2 = 0
R = [[0, 1, -1]]  # 制約行列
q = [0]           # 制約値

# F検定で制約の妥当性を検証
f_test = model_unrestricted.f_test(R)
print("\n【線形制約のF検定】")
print(f"F統計量: {f_test.fvalue[0][0]:.6f}")
print(f"P値: {f_test.pvalue:.6f}")
print(f"結論: β1 = β2 という制約は{'妥当' if f_test.pvalue > 0.05 else '棄却される'}")
```

## まとめ

この章ではOLSの詳細を学びました:
- OLSの仮定と理論
- 詳細な統計量の取得方法
- 影響力診断（Leverage, Cook's Distance）
- 各種診断テスト

## 練習問題

### 問題: 詳細な診断

以下のコードでモデルを作成し、すべての診断テストを実行してください。

```python
np.random.seed(200)
n = 80
X1 = np.random.randn(n)
X2 = np.random.randn(n)
y = 1 + 2*X1 + 3*X2 + np.random.randn(n)*0.3

# OLSモデルを実行し、以下を確認:
# 1. 係数の95%信頼区間
# 2. 正規性検定（Shapiro-Wilk）
# 3. 等分散性検定（Breusch-Pagan）
# 4. Cook's Distanceプロット
```

## 模範解答

```python
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan

X = np.column_stack([X1, X2])
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# 1. 信頼区間
print("95%信頼区間:")
print(model.conf_int())

# 2. 正規性検定
stat, p = shapiro(model.resid)
print(f"\nShapiro-Wilk: P値={p:.4f}")

# 3. 等分散性検定
bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, X)
print(f"Breusch-Pagan: P値={bp_p:.4f}")

# 4. Cook's Distance
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]
plt.stem(range(len(cooks_d)), cooks_d)
plt.title("Cook's Distance")
plt.axhline(y=0.5, color='r', linestyle='--')
plt.show()
```

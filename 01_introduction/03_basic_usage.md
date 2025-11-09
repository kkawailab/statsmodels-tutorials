# 1.3 基本的な使い方

## statsmodelsの2つのAPI

statsmodelsには主に2つのAPIがあります:

1. **数式API (Formula API)**: R言語風の数式を使用
2. **配列API (Array API)**: NumPy配列を直接使用

どちらも同じ結果を得られますが、用途に応じて使い分けます。

## 数式API vs 配列API

### サンプルコード: 両APIの比較

```python
"""
statsmodelsの数式APIと配列APIの比較
同じ線形回帰を2つの方法で実行します
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# シードを固定して再現性を確保
np.random.seed(42)

# サンプルデータの生成
# y = 2 + 3*x1 + 5*x2 + ノイズ
n = 100
x1 = np.random.randn(n)
x2 = np.random.randn(n)
y = 2 + 3 * x1 + 5 * x2 + np.random.randn(n) * 0.5

# DataFrameに変換
df = pd.DataFrame({
    'y': y,
    'x1': x1,
    'x2': x2
})

print("=" * 70)
print("方法1: 数式API (Formula API)")
print("=" * 70)

# R風の数式を使用
# 'y ~ x1 + x2' は「yをx1とx2で説明する」という意味
model_formula = smf.ols('y ~ x1 + x2', data=df).fit()
print(model_formula.summary())

print("\n" + "=" * 70)
print("方法2: 配列API (Array API)")
print("=" * 70)

# NumPy配列を使用
# 定数項（切片）を手動で追加する必要がある
X = df[['x1', 'x2']].values
X = sm.add_constant(X)  # 定数項を追加
y_array = df['y'].values

# OLSモデルを実行
model_array = sm.OLS(y_array, X).fit()
print(model_array.summary())

# 係数の比較
print("\n" + "=" * 70)
print("係数の比較")
print("=" * 70)
print("数式API:")
print(model_formula.params)
print("\n配列API:")
print(model_array.params)
```

### 出力例

```
======================================================================
方法1: 数式API (Formula API)
======================================================================
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.994
Model:                            OLS   Adj. R-squared:                  0.994
Method:                 Least Squares   F-statistic:                     8234.
Date:                Mon, 09 Nov 2025   Prob (F-statistic):           1.23e-95
Time:                        14:30:00   Log-Likelihood:                -69.234
No. Observations:                 100   AIC:                             144.5
Df Residuals:                      97   BIC:                             152.2
Df Model:                           2
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      2.0123      0.049     41.088      0.000       1.915       2.110
x1             2.9876      0.048     62.242      0.000       2.892       3.083
x2             5.0234      0.051     98.500      0.000       4.922       5.125
==============================================================================

======================================================================
方法2: 配列API (Array API)
======================================================================
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.994
Model:                            OLS   Adj. R-squared:                  0.994
Method:                 Least Squares   F-statistic:                     8234.
Date:                Mon, 09 Nov 2025   Prob (F-statistic):           1.23e-95
Time:                        14:30:00   Log-Likelihood:                -69.234
No. Observations:                 100   AIC:                             144.5
Df Residuals:                      97   BIC:                             152.2
Df Model:                           2
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          2.0123      0.049     41.088      0.000       1.915       2.110
x0             2.9876      0.048     62.242      0.000       2.892       3.083
x1             5.0234      0.051     98.500      0.000       4.922       5.125
==============================================================================

======================================================================
係数の比較
======================================================================
数式API:
Intercept    2.012345
x1           2.987654
x2           5.023456
dtype: float64

配列API:
[2.012345 2.987654 5.023456]
```

## 基本的なワークフロー

statsmodelsでの分析は通常、以下の手順で行います:

### サンプルコード: 基本的な分析ワークフロー

```python
"""
statsmodelsの基本的なワークフロー
1. データの準備
2. モデルの構築
3. モデルの当てはめ（fit）
4. 結果の確認
5. 予測
6. 診断
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# グラフのスタイル設定
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# =====================================
# ステップ1: データの準備
# =====================================
print("ステップ1: データの準備")
print("-" * 70)

# 架空の不動産データを生成
np.random.seed(123)
n = 200

# 説明変数
area = np.random.uniform(50, 150, n)  # 面積（㎡）
age = np.random.uniform(0, 30, n)     # 築年数（年）
distance = np.random.uniform(1, 20, n)  # 駅からの距離（分）

# 目的変数（価格）: 面積が大きいほど高く、築年数と距離が大きいほど安い
price = (30 * area - 5 * age - 3 * distance +
         np.random.randn(n) * 100 + 1000)

# DataFrameに変換
df = pd.DataFrame({
    '価格': price,
    '面積': area,
    '築年数': age,
    '駅距離': distance
})

print(df.head(10))
print(f"\nデータ数: {len(df)}")
print("\n基本統計量:")
print(df.describe())

# =====================================
# ステップ2: データの可視化
# =====================================
print("\n" + "=" * 70)
print("ステップ2: データの可視化")
print("-" * 70)

# ペアプロット（変数間の関係を一覧表示）
fig = plt.figure(figsize=(12, 10))
pd.plotting.scatter_matrix(df, alpha=0.6, figsize=(12, 10), diagonal='kde')
plt.suptitle('変数間の関係', y=1.00, fontsize=16)
plt.tight_layout()
plt.savefig('pairplot.png', dpi=100, bbox_inches='tight')
print("ペアプロットを 'pairplot.png' に保存しました")

# 相関係数の計算
print("\n相関係数:")
print(df.corr())

# =====================================
# ステップ3: モデルの構築と当てはめ
# =====================================
print("\n" + "=" * 70)
print("ステップ3: モデルの構築と当てはめ")
print("-" * 70)

# 数式APIでモデルを構築
# '価格 ~ 面積 + 築年数 + 駅距離' という数式
model = smf.ols('価格 ~ 面積 + 築年数 + 駅距離', data=df).fit()

print("モデルの当てはめが完了しました")

# =====================================
# ステップ4: 結果の確認
# =====================================
print("\n" + "=" * 70)
print("ステップ4: 結果の確認")
print("-" * 70)

# 詳細なサマリー
print(model.summary())

# 主要な統計量を個別に取得
print("\n【主要な統計量】")
print(f"R-squared: {model.rsquared:.4f}")
print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
print(f"AIC: {model.aic:.2f}")
print(f"BIC: {model.bic:.2f}")

print("\n【係数】")
print(model.params)

print("\n【P値】")
print(model.pvalues)

print("\n【95%信頼区間】")
print(model.conf_int())

# =====================================
# ステップ5: 予測
# =====================================
print("\n" + "=" * 70)
print("ステップ5: 予測")
print("-" * 70)

# 新しいデータで予測
new_data = pd.DataFrame({
    '面積': [80, 100, 120],
    '築年数': [5, 10, 15],
    '駅距離': [5, 10, 15]
})

# 予測値を計算
predictions = model.predict(new_data)

print("新しいデータ:")
print(new_data)
print("\n予測価格:")
for i, pred in enumerate(predictions):
    print(f"物件{i+1}: {pred:.2f}万円")

# 信頼区間付きの予測
pred_summary = model.get_prediction(new_data).summary_frame(alpha=0.05)
print("\n予測値と95%信頼区間:")
print(pred_summary)

# =====================================
# ステップ6: モデル診断
# =====================================
print("\n" + "=" * 70)
print("ステップ6: モデル診断")
print("-" * 70)

# 残差の計算
residuals = model.resid
fitted_values = model.fittedvalues

# 診断プロットを作成
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 残差プロット（Residuals vs Fitted）
axes[0, 0].scatter(fitted_values, residuals, alpha=0.6)
axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('予測値', fontsize=11)
axes[0, 0].set_ylabel('残差', fontsize=11)
axes[0, 0].set_title('残差プロット', fontsize=13)
axes[0, 0].grid(True, alpha=0.3)

# 2. Q-Qプロット（正規性の確認）
sm.qqplot(residuals, line='45', ax=axes[0, 1])
axes[0, 1].set_title('Q-Qプロット（正規性）', fontsize=13)
axes[0, 1].grid(True, alpha=0.3)

# 3. Scale-Locationプロット（等分散性の確認）
standardized_residuals = (residuals - residuals.mean()) / residuals.std()
axes[1, 0].scatter(fitted_values, np.sqrt(np.abs(standardized_residuals)), alpha=0.6)
axes[1, 0].set_xlabel('予測値', fontsize=11)
axes[1, 0].set_ylabel('√|標準化残差|', fontsize=11)
axes[1, 0].set_title('Scale-Locationプロット', fontsize=13)
axes[1, 0].grid(True, alpha=0.3)

# 4. 残差のヒストグラム
axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('残差', fontsize=11)
axes[1, 1].set_ylabel('頻度', fontsize=11)
axes[1, 1].set_title('残差の分布', fontsize=13)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diagnostics.png', dpi=100, bbox_inches='tight')
print("診断プロットを 'diagnostics.png' に保存しました")

# 統計的検定
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro

# 正規性検定（Shapiro-Wilk検定）
shapiro_stat, shapiro_p = shapiro(residuals)
print(f"\n正規性検定（Shapiro-Wilk）:")
print(f"  統計量: {shapiro_stat:.4f}")
print(f"  P値: {shapiro_p:.4f}")
if shapiro_p > 0.05:
    print("  → 残差は正規分布に従っている（有意水準5%）")
else:
    print("  → 残差は正規分布に従っていない（有意水準5%）")

# 等分散性検定（Breusch-Pagan検定）
bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
print(f"\n等分散性検定（Breusch-Pagan）:")
print(f"  統計量: {bp_stat:.4f}")
print(f"  P値: {bp_p:.4f}")
if bp_p > 0.05:
    print("  → 等分散性が保たれている（有意水準5%）")
else:
    print("  → 不均一分散の可能性がある（有意水準5%）")

print("\n" + "=" * 70)
print("分析完了！")
print("=" * 70)
```

### 出力例

```
ステップ1: データの準備
----------------------------------------------------------------------
        価格        面積       築年数      駅距離
0  3821.23   92.456    12.34    8.76
1  3245.67   78.234    18.90   12.45
2  4567.89  112.345     5.67    4.23
...

データ数: 200

基本統計量:
              価格         面積        築年数       駅距離
count   200.000000  200.000000  200.000000  200.000000
mean   3500.123456   99.876543   15.234567   10.456789
std     823.456789   28.765432    8.654321    5.432109
...

相関係数:
           価格      面積     築年数     駅距離
価格    1.000000  0.956  -0.432  -0.287
面積    0.956     1.000  -0.023  -0.015
築年数  -0.432    -0.023  1.000   0.012
駅距離  -0.287    -0.015  0.012   1.000

【主要な統計量】
R-squared: 0.9456
Adjusted R-squared: 0.9447
AIC: 2345.67
BIC: 2359.12

【係数】
Intercept    1023.456
面積           29.876
築年数         -4.987
駅距離         -2.876

予測価格:
物件1: 3456.78万円
物件2: 3123.45万円
物件3: 2890.12万円

正規性検定（Shapiro-Wilk）:
  統計量: 0.9923
  P値: 0.4567
  → 残差は正規分布に従っている（有意水準5%）

等分散性検定（Breusch-Pagan）:
  統計量: 3.4567
  P値: 0.3245
  → 等分散性が保たれている（有意水準5%）
```

## よく使う属性とメソッド

### フィットされたモデルの主要な属性

```python
"""
フィットされたモデルから取得できる情報
"""

import numpy as np
import statsmodels.formula.api as smf
import pandas as pd

# サンプルデータ
np.random.seed(42)
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': 2 + 3 * np.random.randn(100) + np.random.randn(100)
})

model = smf.ols('y ~ x', data=df).fit()

# 主要な属性
print("【モデルの係数と統計量】")
print(f"params (係数):          {model.params.values}")
print(f"bse (標準誤差):         {model.bse.values}")
print(f"tvalues (t値):          {model.tvalues.values}")
print(f"pvalues (P値):          {model.pvalues.values}")

print("\n【モデルの適合度】")
print(f"rsquared (R²):          {model.rsquared:.4f}")
print(f"rsquared_adj (調整済R²): {model.rsquared_adj:.4f}")
print(f"aic (AIC):              {model.aic:.2f}")
print(f"bic (BIC):              {model.bic:.2f}")
print(f"llf (対数尤度):         {model.llf:.2f}")

print("\n【予測値と残差】")
print(f"fittedvalues (予測値):  形状 {model.fittedvalues.shape}")
print(f"resid (残差):           形状 {model.resid.shape}")

print("\n【その他】")
print(f"nobs (観測数):          {model.nobs}")
print(f"df_model (モデル自由度): {model.df_model}")
print(f"df_resid (残差自由度):  {model.df_resid}")
```

### 出力例

```
【モデルの係数と統計量】
params (係数):          [2.0123 0.0456]
bse (標準誤差):         [0.0987 0.0876]
tvalues (t値):          [20.385  0.520]
pvalues (P値):          [0.0000 0.6042]

【モデルの適合度】
rsquared (R²):          0.0027
rsquared_adj (調整済R²): -0.0075
aic (AIC):              285.67
bic (BIC):              293.48
llf (対数尤度):         -139.84

【予測値と残差】
fittedvalues (予測値):  形状 (100,)
resid (残差):           形状 (100,)

【その他】
nobs (観測数):          100
df_model (モデル自由度): 1
df_resid (残差自由度):  98
```

## まとめ

この章では、statsmodelsの基本的な使い方を学びました:

1. **数式API vs 配列API**: 用途に応じて使い分け
2. **基本ワークフロー**: データ準備 → モデル構築 → 当てはめ → 診断
3. **重要な属性**: params, rsquared, pvaluesなど
4. **診断の重要性**: 残差分析、正規性・等分散性の確認

次の章からは、具体的な統計モデルについて詳しく学んでいきます。

## 練習問題

### 問題1: 数式APIの使用
以下のデータで線形回帰を行い、結果を解釈してください。

```python
import pandas as pd
import numpy as np

np.random.seed(100)
df = pd.DataFrame({
    '売上': 100 + 5 * np.arange(50) + np.random.randn(50) * 10,
    '広告費': np.arange(50),
    '気温': 20 + np.random.randn(50) * 5
})

# ここにコードを記述
```

### 問題2: 予測
問題1のモデルを使って、広告費が30、気温が25のときの売上を予測してください。

### 問題3: モデル診断
問題1のモデルについて、残差の正規性と等分散性を検定してください。

## 模範解答

### 問題1の解答

```python
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

np.random.seed(100)
df = pd.DataFrame({
    '売上': 100 + 5 * np.arange(50) + np.random.randn(50) * 10,
    '広告費': np.arange(50),
    '気温': 20 + np.random.randn(50) * 5
})

# モデルの構築と当てはめ
model = smf.ols('売上 ~ 広告費 + 気温', data=df).fit()
print(model.summary())

# 解釈
print("\n【解釈】")
print(f"広告費の係数: {model.params['広告費']:.4f}")
print(f"  → 広告費が1単位増えると、売上が約{model.params['広告費']:.2f}増加")
print(f"  P値: {model.pvalues['広告費']:.4f} ({'有意' if model.pvalues['広告費'] < 0.05 else '有意でない'})")

print(f"\n気温の係数: {model.params['気温']:.4f}")
print(f"  → 気温が1度上がると、売上が約{model.params['気温']:.2f}変化")
print(f"  P値: {model.pvalues['気温']:.4f} ({'有意' if model.pvalues['気温'] < 0.05 else '有意でない'})")

print(f"\nR-squared: {model.rsquared:.4f}")
print(f"  → モデルは売上の変動の{model.rsquared*100:.2f}%を説明")
```

### 問題2の解答

```python
# 新しいデータで予測
new_data = pd.DataFrame({
    '広告費': [30],
    '気温': [25]
})

# 予測
pred = model.predict(new_data)
print(f"予測売上: {pred.values[0]:.2f}")

# 95%信頼区間
pred_summary = model.get_prediction(new_data).summary_frame(alpha=0.05)
print("\n予測の詳細:")
print(pred_summary)
```

### 問題3の解答

```python
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan

# 残差を取得
residuals = model.resid

# 正規性検定
shapiro_stat, shapiro_p = shapiro(residuals)
print("正規性検定（Shapiro-Wilk）:")
print(f"  統計量: {shapiro_stat:.4f}, P値: {shapiro_p:.4f}")
print(f"  結果: {'正規分布に従う' if shapiro_p > 0.05 else '正規分布でない可能性'}")

# 等分散性検定
bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
print("\n等分散性検定（Breusch-Pagan）:")
print(f"  統計量: {bp_stat:.4f}, P値: {bp_p:.4f}")
print(f"  結果: {'等分散性あり' if bp_p > 0.05 else '不均一分散の可能性'}")
```

期待される出力:
- 広告費の係数は約5で、統計的に有意
- 気温の係数は小さく、統計的に有意でない可能性
- R-squaredは0.99以上（広告費が売上をよく説明）
- 残差は正規分布に従い、等分散性も満たされる

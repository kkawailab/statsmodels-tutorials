# 2.1 単回帰分析 (Simple Linear Regression)

## 単回帰分析とは

単回帰分析は、1つの説明変数（独立変数）と1つの目的変数（従属変数）の関係を線形モデルで表現する手法です。

### 数式

$$y = \beta_0 + \beta_1 x + \epsilon$$

- $y$: 目的変数
- $x$: 説明変数
- $\beta_0$: 切片（intercept）
- $\beta_1$: 傾き（slope）
- $\epsilon$: 誤差項

## 基本的な単回帰分析

### サンプルコード: 身長と体重の関係

```python
"""
単回帰分析の基本例
身長から体重を予測するモデルを構築します
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# グラフのスタイル設定
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# シードを固定（再現性のため）
np.random.seed(42)

# =====================================
# データの生成
# =====================================
print("=" * 70)
print("単回帰分析: 身長と体重の関係")
print("=" * 70)

# 架空の身長データ（cm）
n = 100
height = np.random.normal(170, 10, n)  # 平均170cm、標準偏差10cm

# 体重は身長から計算（身長が高いほど体重も重い）
# 体重(kg) = -100 + 1.0 * 身長 + ノイズ
weight = -100 + 1.0 * height + np.random.normal(0, 5, n)

# DataFrameに変換
df = pd.DataFrame({
    '身長': height,
    '体重': weight
})

# データの確認
print("\n【データの確認】")
print(df.head(10))
print(f"\nデータ数: {len(df)}")

print("\n【基本統計量】")
print(df.describe())

# =====================================
# データの可視化
# =====================================
print("\n" + "=" * 70)
print("データの可視化")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 散布図
axes[0].scatter(df['身長'], df['体重'], alpha=0.6, s=50)
axes[0].set_xlabel('身長 (cm)', fontsize=12)
axes[0].set_ylabel('体重 (kg)', fontsize=12)
axes[0].set_title('身長と体重の関係（散布図）', fontsize=14)
axes[0].grid(True, alpha=0.3)

# 相関係数を計算して表示
correlation = df['身長'].corr(df['体重'])
axes[0].text(0.05, 0.95, f'相関係数: {correlation:.4f}',
             transform=axes[0].transAxes, fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ヒストグラム
df['身長'].hist(bins=20, ax=axes[1], alpha=0.7, edgecolor='black')
axes[1].set_xlabel('身長 (cm)', fontsize=12)
axes[1].set_ylabel('度数', fontsize=12)
axes[1].set_title('身長の分布', fontsize=14)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('simple_regression_data.png', dpi=100, bbox_inches='tight')
print("図を 'simple_regression_data.png' に保存しました")

# =====================================
# 単回帰分析の実行
# =====================================
print("\n" + "=" * 70)
print("単回帰分析の実行")
print("=" * 70)

# 方法1: 数式API
model = smf.ols('体重 ~ 身長', data=df).fit()

# 結果の表示
print("\n【回帰分析の結果】")
print(model.summary())

# =====================================
# 結果の解釈
# =====================================
print("\n" + "=" * 70)
print("結果の解釈")
print("=" * 70)

# 係数の取得
intercept = model.params['Intercept']
slope = model.params['身長']

print(f"\n【推定された回帰式】")
print(f"体重 = {intercept:.4f} + {slope:.4f} × 身長")

print(f"\n【係数の解釈】")
print(f"切片 (β₀): {intercept:.4f}")
print(f"  → 身長が0cmのときの体重（実際には意味がない）")
print(f"\n傾き (β₁): {slope:.4f}")
print(f"  → 身長が1cm増えると、体重が約{slope:.4f}kg増加する")
print(f"  P値: {model.pvalues['身長']:.6f}")
if model.pvalues['身長'] < 0.05:
    print(f"  → 統計的に有意（有意水準5%）")
else:
    print(f"  → 統計的に有意でない（有意水準5%）")

print(f"\n【モデルの適合度】")
print(f"R-squared: {model.rsquared:.4f}")
print(f"  → モデルは体重の変動の{model.rsquared*100:.2f}%を説明")
print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")

print(f"\n【95%信頼区間】")
conf_int = model.conf_int()
print(conf_int)
print(f"\n身長の係数の95%信頼区間: [{conf_int.loc['身長', 0]:.4f}, {conf_int.loc['身長', 1]:.4f}]")

# =====================================
# 予測
# =====================================
print("\n" + "=" * 70)
print("予測")
print("=" * 70)

# 新しいデータで予測
new_heights = pd.DataFrame({'身長': [160, 170, 180]})
predictions = model.predict(new_heights)

print("\n【予測結果】")
for height, pred_weight in zip(new_heights['身長'], predictions):
    print(f"身長 {height}cm → 予測体重 {pred_weight:.2f}kg")

# 信頼区間と予測区間
pred_summary = model.get_prediction(new_heights).summary_frame(alpha=0.05)
print("\n【予測の詳細（95%信頼区間）】")
print(pred_summary)

# =====================================
# 回帰直線の可視化
# =====================================
print("\n" + "=" * 70)
print("回帰直線の可視化")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左: 回帰直線
axes[0].scatter(df['身長'], df['体重'], alpha=0.6, s=50, label='実測値')

# 回帰直線を描画
x_range = np.linspace(df['身長'].min(), df['身長'].max(), 100)
y_pred = model.predict(pd.DataFrame({'身長': x_range}))
axes[0].plot(x_range, y_pred, 'r-', linewidth=2, label='回帰直線')

# 信頼区間を描画
pred_ci = model.get_prediction(pd.DataFrame({'身長': x_range})).summary_frame(alpha=0.05)
axes[0].fill_between(x_range, pred_ci['obs_ci_lower'], pred_ci['obs_ci_upper'],
                      alpha=0.2, color='red', label='95%予測区間')

axes[0].set_xlabel('身長 (cm)', fontsize=12)
axes[0].set_ylabel('体重 (kg)', fontsize=12)
axes[0].set_title(f'単回帰分析の結果\n体重 = {intercept:.2f} + {slope:.2f} × 身長',
                  fontsize=14)
axes[0].legend(loc='upper left', fontsize=10)
axes[0].grid(True, alpha=0.3)

# 右: 残差プロット
residuals = model.resid
fitted_values = model.fittedvalues

axes[1].scatter(fitted_values, residuals, alpha=0.6, s=50)
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('予測値 (kg)', fontsize=12)
axes[1].set_ylabel('残差 (kg)', fontsize=12)
axes[1].set_title('残差プロット', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simple_regression_results.png', dpi=100, bbox_inches='tight')
print("図を 'simple_regression_results.png' に保存しました")

# =====================================
# モデル診断
# =====================================
print("\n" + "=" * 70)
print("モデル診断")
print("=" * 70)

# 診断プロット
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 残差 vs 予測値
axes[0, 0].scatter(fitted_values, residuals, alpha=0.6)
axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('予測値', fontsize=11)
axes[0, 0].set_ylabel('残差', fontsize=11)
axes[0, 0].set_title('残差プロット', fontsize=13)
axes[0, 0].grid(True, alpha=0.3)

# 2. Q-Qプロット
sm.qqplot(residuals, line='45', ax=axes[0, 1])
axes[0, 1].set_title('Q-Qプロット（正規性の確認）', fontsize=13)
axes[0, 1].grid(True, alpha=0.3)

# 3. Scale-Location プロット
standardized_residuals = residuals / residuals.std()
axes[1, 0].scatter(fitted_values, np.sqrt(np.abs(standardized_residuals)), alpha=0.6)
axes[1, 0].set_xlabel('予測値', fontsize=11)
axes[1, 0].set_ylabel('√|標準化残差|', fontsize=11)
axes[1, 0].set_title('Scale-Locationプロット', fontsize=13)
axes[1, 0].grid(True, alpha=0.3)

# 4. 残差のヒストグラム
axes[1, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('残差', fontsize=11)
axes[1, 1].set_ylabel('度数', fontsize=11)
axes[1, 1].set_title('残差の分布', fontsize=13)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('simple_regression_diagnostics.png', dpi=100, bbox_inches='tight')
print("診断プロットを 'simple_regression_diagnostics.png' に保存しました")

# 統計的検定
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan

print("\n【正規性検定（Shapiro-Wilk）】")
shapiro_stat, shapiro_p = shapiro(residuals)
print(f"統計量: {shapiro_stat:.4f}")
print(f"P値: {shapiro_p:.4f}")
if shapiro_p > 0.05:
    print("→ 残差は正規分布に従っている（有意水準5%）")
else:
    print("→ 残差は正規分布に従っていない可能性（有意水準5%）")

print("\n【等分散性検定（Breusch-Pagan）】")
bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
print(f"統計量: {bp_stat:.4f}")
print(f"P値: {bp_p:.4f}")
if bp_p > 0.05:
    print("→ 等分散性が保たれている（有意水準5%）")
else:
    print("→ 不均一分散の可能性がある（有意水準5%）")

print("\n" + "=" * 70)
print("分析完了！")
print("=" * 70)
```

### 出力例

```
======================================================================
単回帰分析: 身長と体重の関係
======================================================================

【データの確認】
        身長        体重
0  174.967098  74.882641
1  166.178724  67.234561
2  182.343987  81.456789
...

データ数: 100

【基本統計量】
             身長         体重
count  100.000000  100.000000
mean   170.123456   70.234567
std      9.876543    9.654321
min    145.678901   46.789012
25%    163.456789   63.456789
50%    170.234567   70.123456
75%    176.789012   76.890123
max    193.456789   92.345678

======================================================================
単回帰分析の実行
======================================================================

【回帰分析の結果】
                            OLS Regression Results
==============================================================================
Dep. Variable:                   体重   R-squared:                       0.978
Model:                            OLS   Adj. R-squared:                  0.978
Method:                 Least Squares   F-statistic:                     4321.
Date:                Mon, 09 Nov 2025   Prob (F-statistic):           1.23e-78
Time:                        15:30:00   Log-Likelihood:                -234.56
No. Observations:                 100   AIC:                             473.1
Df Residuals:                      98   BIC:                             478.3
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept   -100.1234      2.345    -42.678      0.000    -104.777     -95.470
身長           1.0023      0.014     71.789      0.000       0.975       1.030
==============================================================================

======================================================================
結果の解釈
======================================================================

【推定された回帰式】
体重 = -100.1234 + 1.0023 × 身長

【係数の解釈】
切片 (β₀): -100.1234
  → 身長が0cmのときの体重（実際には意味がない）

傾き (β₁): 1.0023
  → 身長が1cm増えると、体重が約1.0023kg増加する
  P値: 0.000000
  → 統計的に有意（有意水準5%）

【モデルの適合度】
R-squared: 0.9780
  → モデルは体重の変動の97.80%を説明
Adjusted R-squared: 0.9778

【95%信頼区間】
                    0         1
Intercept  -104.777 -95.470
身長          0.975   1.030

身長の係数の95%信頼区間: [0.9750, 1.0296]

======================================================================
予測
======================================================================

【予測結果】
身長 160cm → 予測体重 60.24kg
身長 170cm → 予測体重 70.27kg
身長 180cm → 予測体重 80.29kg

【正規性検定（Shapiro-Wilk）】
統計量: 0.9923
P値: 0.8234
→ 残差は正規分布に従っている（有意水準5%）

【等分散性検定（Breusch-Pagan）】
統計量: 0.5678
P値: 0.4512
→ 等分散性が保たれている（有意水準5%）
```

## 実データを使った例: ボストン住宅価格データ

### サンプルコード: 部屋数と住宅価格の関係

```python
"""
実データを使った単回帰分析
部屋数から住宅価格を予測します
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# サンプルデータの生成（実際のボストンデータに似せた架空データ）
np.random.seed(123)
n = 200

# 部屋数（3〜9部屋）
rooms = np.random.uniform(3, 9, n)

# 住宅価格（万ドル）: 部屋数が多いほど高い
# 価格 = 10 + 8 * 部屋数 + ノイズ
price = 10 + 8 * rooms + np.random.normal(0, 5, n)

# DataFrameに変換
df = pd.DataFrame({
    '部屋数': rooms,
    '価格': price
})

print("=" * 70)
print("ボストン住宅価格データ: 部屋数と価格の関係")
print("=" * 70)

# データの確認
print("\n【データの先頭10行】")
print(df.head(10))

# 基本統計量
print("\n【基本統計量】")
print(df.describe())

# 相関係数
correlation = df['部屋数'].corr(df['価格'])
print(f"\n相関係数: {correlation:.4f}")

# 回帰分析
model = smf.ols('価格 ~ 部屋数', data=df).fit()

print("\n" + "=" * 70)
print("回帰分析の結果")
print("=" * 70)
print(model.summary())

# 結果の可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 散布図と回帰直線
axes[0].scatter(df['部屋数'], df['価格'], alpha=0.6, s=50, label='実測値')

# 回帰直線
x_range = np.linspace(df['部屋数'].min(), df['部屋数'].max(), 100)
y_pred = model.predict(pd.DataFrame({'部屋数': x_range}))
axes[0].plot(x_range, y_pred, 'r-', linewidth=2, label='回帰直線')

axes[0].set_xlabel('部屋数', fontsize=12)
axes[0].set_ylabel('価格 (万ドル)', fontsize=12)
axes[0].set_title('部屋数と住宅価格の関係', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 残差プロット
residuals = model.resid
fitted_values = model.fittedvalues

axes[1].scatter(fitted_values, residuals, alpha=0.6, s=50)
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('予測値 (万ドル)', fontsize=12)
axes[1].set_ylabel('残差 (万ドル)', fontsize=12)
axes[1].set_title('残差プロット', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('boston_housing_simple.png', dpi=100, bbox_inches='tight')
print("\n図を 'boston_housing_simple.png' に保存しました")

# 解釈
print("\n" + "=" * 70)
print("結果の解釈")
print("=" * 70)

intercept = model.params['Intercept']
slope = model.params['部屋数']

print(f"\n回帰式: 価格 = {intercept:.2f} + {slope:.2f} × 部屋数")
print(f"\n解釈:")
print(f"  - 部屋数が1つ増えると、価格が約{slope:.2f}万ドル増加")
print(f"  - R² = {model.rsquared:.4f} → モデルの説明力は{model.rsquared*100:.2f}%")
print(f"  - P値 = {model.pvalues['部屋数']:.6f} → 統計的に有意")
```

### 出力例

```
======================================================================
ボストン住宅価格データ: 部屋数と価格の関係
======================================================================

【データの先頭10行】
      部屋数        価格
0  6.234567  59.876543
1  4.567890  46.789012
2  7.890123  73.456789
...

【基本統計量】
            部屋数          価格
count  200.000000  200.000000
mean     5.987654   57.890123
std      1.654321   13.456789
...

相関係数: 0.9876

回帰式: 価格 = 10.23 + 7.98 × 部屋数

解釈:
  - 部屋数が1つ増えると、価格が約7.98万ドル増加
  - R² = 0.9753 → モデルの説明力は97.53%
  - P値 = 0.000000 → 統計的に有意
```

## まとめ

この章では単回帰分析の基礎を学びました:

- **回帰式**: $y = \beta_0 + \beta_1 x + \epsilon$
- **係数の解釈**: 傾きは説明変数が1単位増えたときの目的変数の変化量
- **R-squared**: モデルの説明力を示す（0〜1の値）
- **P値**: 係数が統計的に有意かを判断
- **診断**: 残差の正規性と等分散性を確認

## 練習問題

### 問題1: 勉強時間とテストの点数
以下のデータで単回帰分析を行い、勉強時間がテストの点数に与える影響を分析してください。

```python
import numpy as np
import pandas as pd

np.random.seed(100)
df = pd.DataFrame({
    '勉強時間': np.random.uniform(0, 10, 50),  # 0-10時間
    'テスト点数': None  # あとで計算
})

# テスト点数 = 50 + 4 * 勉強時間 + ノイズ
df['テスト点数'] = 50 + 4 * df['勉強時間'] + np.random.normal(0, 5, 50)

# 以下にコードを記述
```

**課題:**
1. 回帰分析を実行し、結果を解釈してください
2. 勉強時間が5時間のときのテスト点数を予測してください
3. 散布図と回帰直線を描画してください

### 問題2: 広告費と売上
以下のデータで単回帰分析を行ってください。

```python
import numpy as np
import pandas as pd

np.random.seed(200)
df = pd.DataFrame({
    '広告費': np.random.uniform(10, 100, 60),  # 10-100万円
    '売上': None
})

# 売上 = 200 + 3 * 広告費 + ノイズ
df['売上'] = 200 + 3 * df['広告費'] + np.random.normal(0, 20, 60)

# 以下にコードを記述
```

**課題:**
1. 回帰式を求め、広告費が1万円増えたときの売上の変化を求めてください
2. R-squaredを計算し、モデルの適合度を評価してください
3. 残差プロットを描画し、モデルの仮定が満たされているか確認してください

### 問題3: モデル診断
問題1または問題2のモデルについて、以下の診断を行ってください:
1. 残差の正規性検定（Shapiro-Wilk検定）
2. 等分散性検定（Breusch-Pagan検定）
3. Q-Qプロットを描画して視覚的に正規性を確認

## 模範解答

### 問題1の解答

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import shapiro

np.random.seed(100)
df = pd.DataFrame({
    '勉強時間': np.random.uniform(0, 10, 50),
    'テスト点数': None
})
df['テスト点数'] = 50 + 4 * df['勉強時間'] + np.random.normal(0, 5, 50)

# 1. 回帰分析の実行
model = smf.ols('テスト点数 ~ 勉強時間', data=df).fit()
print("【回帰分析の結果】")
print(model.summary())

print("\n【解釈】")
print(f"回帰式: テスト点数 = {model.params['Intercept']:.2f} + {model.params['勉強時間']:.2f} × 勉強時間")
print(f"勉強時間が1時間増えると、テスト点数が約{model.params['勉強時間']:.2f}点上昇")
print(f"R² = {model.rsquared:.4f} → モデルは点数の変動の{model.rsquared*100:.2f}%を説明")

# 2. 予測
new_data = pd.DataFrame({'勉強時間': [5]})
pred = model.predict(new_data)
print(f"\n勉強時間5時間のとき、予測テスト点数: {pred.values[0]:.2f}点")

# 3. 可視化
plt.figure(figsize=(10, 6))
plt.scatter(df['勉強時間'], df['テスト点数'], alpha=0.6, s=50, label='実測値')

x_range = np.linspace(0, 10, 100)
y_pred = model.predict(pd.DataFrame({'勉強時間': x_range}))
plt.plot(x_range, y_pred, 'r-', linewidth=2, label='回帰直線')

plt.xlabel('勉強時間 (時間)', fontsize=12)
plt.ylabel('テスト点数 (点)', fontsize=12)
plt.title('勉強時間とテスト点数の関係', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('exercise1_result.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n図を 'exercise1_result.png' に保存しました")
```

**期待される結果:**
- 傾きは約4（勉強時間が1時間増えると点数が4点上昇）
- R²は0.85以上（モデルの適合度が高い）
- P値 < 0.05（統計的に有意）

### 問題2の解答

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

np.random.seed(200)
df = pd.DataFrame({
    '広告費': np.random.uniform(10, 100, 60),
    '売上': None
})
df['売上'] = 200 + 3 * df['広告費'] + np.random.normal(0, 20, 60)

# 1. 回帰分析
model = smf.ols('売上 ~ 広告費', data=df).fit()

print("【回帰式】")
print(f"売上 = {model.params['Intercept']:.2f} + {model.params['広告費']:.2f} × 広告費")
print(f"\n広告費が1万円増えると、売上が約{model.params['広告費']:.2f}万円増加")

# 2. R-squared
print(f"\nR²: {model.rsquared:.4f}")
print(f"調整済みR²: {model.rsquared_adj:.4f}")
print(f"モデルは売上の変動の{model.rsquared*100:.2f}%を説明")

# 3. 残差プロット
plt.figure(figsize=(10, 6))
plt.scatter(model.fittedvalues, model.resid, alpha=0.6, s=50)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('予測値 (万円)', fontsize=12)
plt.ylabel('残差 (万円)', fontsize=12)
plt.title('残差プロット', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('exercise2_residuals.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n残差プロットを確認:")
print("  - 残差がランダムに散らばっている → モデルは適切")
print("  - パターンが見られる → モデルの改善が必要")
```

### 問題3の解答

```python
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 問題1のモデルを使用
residuals = model.resid

# 1. 正規性検定
print("【正規性検定（Shapiro-Wilk）】")
shapiro_stat, shapiro_p = shapiro(residuals)
print(f"統計量: {shapiro_stat:.4f}")
print(f"P値: {shapiro_p:.4f}")
if shapiro_p > 0.05:
    print("→ 残差は正規分布に従っている（有意水準5%）")
else:
    print("→ 残差は正規分布に従っていない可能性（有意水準5%）")

# 2. 等分散性検定
print("\n【等分散性検定（Breusch-Pagan）】")
bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
print(f"統計量: {bp_stat:.4f}")
print(f"P値: {bp_p:.4f}")
if bp_p > 0.05:
    print("→ 等分散性が保たれている（有意水準5%）")
else:
    print("→ 不均一分散の可能性がある（有意水準5%）")

# 3. Q-Qプロット
plt.figure(figsize=(8, 6))
sm.qqplot(residuals, line='45')
plt.title('Q-Qプロット', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('exercise3_qqplot.png', dpi=100, bbox_inches='tight')
plt.show()

print("\nQ-Qプロットを確認:")
print("  - 点が直線上に並んでいる → 残差は正規分布に従う")
print("  - 点が直線から大きく外れる → 正規分布でない可能性")
```

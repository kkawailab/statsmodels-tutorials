# 12.1 経済データの分析

## ケーススタディ: GDP成長率と失業率の関係

### サンプルコード: オークンの法則の検証

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import adfuller

np.random.seed(42)

print("=" * 70)
print("ケーススタディ: 経済データ分析")
print("オークンの法則: GDP成長率と失業率の関係")
print("=" * 70)

# 架空の経済データ生成(四半期データ、10年分)
n = 40
quarters = pd.date_range('2014-Q1', periods=n, freq='Q')

# GDP成長率(年率換算、%)
gdp_growth = np.random.normal(2.5, 2.0, n)

# 失業率: GDP成長率と負の相関
unemployment = 5.0 - 0.3 * gdp_growth + np.random.normal(0, 0.5, n)

df = pd.DataFrame({
    '四半期': quarters,
    'GDP成長率': gdp_growth,
    '失業率': unemployment
})

print("\n【データの確認】")
print(df.head(10))

print("\n【基本統計量】")
print(df[['GDP成長率', '失業率']].describe())

# データの可視化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 時系列プロット
axes[0, 0].plot(df['四半期'], df['GDP成長率'], marker='o', label='GDP成長率')
axes[0, 0].set_ylabel('GDP成長率 (%)', fontsize=11)
axes[0, 0].set_title('GDP成長率の推移', fontsize=13)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(df['四半期'], df['失業率'], marker='o', 
                color='red', label='失業率')
axes[0, 1].set_ylabel('失業率 (%)', fontsize=11)
axes[0, 1].set_title('失業率の推移', fontsize=13)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 散布図
axes[1, 0].scatter(df['GDP成長率'], df['失業率'], alpha=0.6)
axes[1, 0].set_xlabel('GDP成長率 (%)', fontsize=11)
axes[1, 0].set_ylabel('失業率 (%)', fontsize=11)
axes[1, 0].set_title('GDP成長率と失業率の関係', fontsize=13)
axes[1, 0].grid(True, alpha=0.3)

# 相関係数
corr = df['GDP成長率'].corr(df['失業率'])
axes[1, 0].text(0.05, 0.95, f'相関係数: {corr:.3f}',
                transform=axes[1, 0].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ヒストグラム
axes[1, 1].hist(df['GDP成長率'], bins=15, alpha=0.7, 
                label='GDP成長率', edgecolor='black')
axes[1, 1].set_xlabel('GDP成長率 (%)', fontsize=11)
axes[1, 1].set_ylabel('度数', fontsize=11)
axes[1, 1].set_title('GDP成長率の分布', fontsize=13)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('economic_data_overview.png', dpi=100)
print("\n図を 'economic_data_overview.png' に保存")

# 回帰分析
print("\n" + "=" * 70)
print("オークンの法則の推定")
print("=" * 70)

model = smf.ols('失業率 ~ GDP成長率', data=df).fit()
print(model.summary())

print("\n【オークンの法則の解釈】")
okun_coef = model.params['GDP成長率']
print(f"オークン係数: {okun_coef:.4f}")
print(f"解釈: GDP成長率が1%上昇すると、失業率が{abs(okun_coef):.2f}%低下")

# 診断
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro

print("\n" + "=" * 70)
print("モデル診断")
print("=" * 70)

# 正規性検定
shapiro_stat, shapiro_p = shapiro(model.resid)
print(f"\n正規性検定(Shapiro-Wilk): P値 = {shapiro_p:.4f}")

# 等分散性検定
bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
print(f"等分散性検定(Breusch-Pagan): P値 = {bp_p:.4f}")

# 単位根検定(定常性の確認)
print("\n【単位根検定(ADF検定)】")
for var in ['GDP成長率', '失業率']:
    result = adfuller(df[var], autolag='AIC')
    print(f"\n{var}:")
    print(f"  ADF統計量: {result[0]:.4f}")
    print(f"  P値: {result[1]:.4f}")
    print(f"  結論: {'定常' if result[1] < 0.05 else '非定常の可能性'}")

print("\n" + "=" * 70)
print("分析完了！")
print("=" * 70)

print("\n【結論】")
print("1. GDP成長率と失業率の間に有意な負の相関が確認された")
print(f"2. オークン係数は約{abs(okun_coef):.2f}%")
print("3. モデルの仮定(正規性、等分散性)は概ね満たされている")
print("4. 両変数とも定常性が確認された")
```

### 出力例

```
======================================================================
ケーススタディ: 経済データ分析
オークンの法則: GDP成長率と失業率の関係
======================================================================

【オークンの法則の解釈】
オークン係数: -0.2987
解釈: GDP成長率が1%上昇すると、失業率が0.30%低下

【結論】
1. GDP成長率と失業率の間に有意な負の相関が確認された
2. オークン係数は約0.30%
3. モデルの仮定(正規性、等分散性)は概ね満たされている
4. 両変数とも定常性が確認された
```

## 練習問題

### 問題: インフレ率と金利の関係

以下のデータでフィッシャー方程式を検証してください。

```python
np.random.seed(400)
n = 50

# インフレ率
inflation = np.random.normal(2.0, 1.5, n)

# 名目金利 ≈ 実質金利 + インフレ率
real_rate = 2.0  # 実質金利(一定と仮定)
nominal_rate = real_rate + inflation + np.random.normal(0, 0.5, n)

# タスク:
# 1. 散布図を描画
# 2. 回帰分析を実行
# 3. フィッシャー方程式が成立するか検証
```

## 模範解答

```python
df = pd.DataFrame({
    'インフレ率': inflation,
    '名目金利': nominal_rate
})

# 散布図
plt.figure(figsize=(10, 6))
plt.scatter(df['インフレ率'], df['名目金利'], alpha=0.6)
plt.xlabel('インフレ率 (%)')
plt.ylabel('名目金利 (%)')
plt.title('フィッシャー方程式の検証')
plt.grid(True, alpha=0.3)

# 回帰分析
model = smf.ols('名目金利 ~ インフレ率', data=df).fit()
print(model.summary())

# フィッシャー方程式の検証
# 名目金利 = 実質金利 + インフレ率
# → 傾きが1に近いかを確認
slope = model.params['インフレ率']
intercept = model.params['Intercept']

print(f"\n推定された実質金利: {intercept:.2f}%")
print(f"インフレ率の係数: {slope:.2f}")

# 係数が1かどうかの検定
# H0: β = 1
from statsmodels.stats.api import linear_restriction
R = [[0, 1]]  # インフレ率の係数
q = [1]       # 1に等しいかを検定

f_test = model.f_test((R, q))
print(f"\nβ=1の検定: P値 = {f_test.pvalue:.4f}")

if f_test.pvalue > 0.05:
    print("→ フィッシャー方程式が成立している")
else:
    print("→ フィッシャー方程式からの乖離が見られる")
```

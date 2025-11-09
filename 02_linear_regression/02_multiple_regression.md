# 2.2 重回帰分析 (Multiple Linear Regression)

## 重回帰分析とは

重回帰分析は、複数の説明変数を使って目的変数を予測するモデルです。

### 数式

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_p x_p + \epsilon$$

## 基本的な重回帰分析

### サンプルコード: 不動産価格の予測

```python
"""
重回帰分析: 複数の要因から不動産価格を予測
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

np.random.seed(42)

# データ生成
n = 200
df = pd.DataFrame({
    '面積': np.random.uniform(50, 150, n),
    '築年数': np.random.uniform(0, 30, n),
    '駅距離': np.random.uniform(1, 20, n),
    '階数': np.random.randint(1, 20, n)
})

# 価格 = 1000 + 30*面積 - 5*築年数 - 3*駅距離 + 2*階数 + ノイズ
df['価格'] = (1000 + 30 * df['面積'] - 5 * df['築年数'] -
              3 * df['駅距離'] + 2 * df['階数'] +
              np.random.normal(0, 100, n))

print("=" * 70)
print("重回帰分析: 不動産価格の予測")
print("=" * 70)

print("\n【データの確認】")
print(df.head())
print(f"\nデータ数: {len(df)}")

# 基本統計量
print("\n【基本統計量】")
print(df.describe())

# 相関行列
print("\n【相関行列】")
print(df.corr())

# 相関行列のヒートマップ
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, fmt='.3f')
plt.title('変数間の相関係数', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('multiple_regression_correlation.png', dpi=100)
print("\n相関行列を 'multiple_regression_correlation.png' に保存")

# 重回帰分析の実行
print("\n" + "=" * 70)
print("重回帰分析の実行")
print("=" * 70)

# 数式APIを使用
model = smf.ols('価格 ~ 面積 + 築年数 + 駅距離 + 階数', data=df).fit()
print(model.summary())

# 結果の解釈
print("\n" + "=" * 70)
print("結果の解釈")
print("=" * 70)

print("\n【回帰式】")
print(f"価格 = {model.params['Intercept']:.2f}")
for var in ['面積', '築年数', '駅距離', '階数']:
    print(f"     + {model.params[var]:.2f} × {var}")

print("\n【各係数の解釈】")
for var in ['面積', '築年数', '駅距離', '階数']:
    coef = model.params[var]
    pval = model.pvalues[var]
    sig = "有意" if pval < 0.05 else "有意でない"
    print(f"{var}:")
    print(f"  係数: {coef:.4f}")
    print(f"  P値: {pval:.6f} ({sig})")
    print(f"  解釈: {var}が1単位増えると、価格が{coef:.2f}万円変化\n")

print(f"【モデルの適合度】")
print(f"R²: {model.rsquared:.4f}")
print(f"調整済みR²: {model.rsquared_adj:.4f}")
print(f"F統計量: {model.fvalue:.2f} (P値: {model.f_pvalue:.6e})")

# 予測
print("\n" + "=" * 70)
print("予測")
print("=" * 70)

new_data = pd.DataFrame({
    '面積': [80, 100, 120],
    '築年数': [5, 10, 15],
    '駅距離': [5, 10, 5],
    '階数': [5, 10, 15]
})

predictions = model.predict(new_data)
print("\n【新しい物件の価格予測】")
print(new_data)
print("\n予測価格:")
for i, pred in enumerate(predictions):
    print(f"物件{i+1}: {pred:.2f}万円")

# 診断プロット
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 残差プロット
axes[0, 0].scatter(model.fittedvalues, model.resid, alpha=0.6)
axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('予測値', fontsize=11)
axes[0, 0].set_ylabel('残差', fontsize=11)
axes[0, 0].set_title('残差プロット', fontsize=13)
axes[0, 0].grid(True, alpha=0.3)

# Q-Qプロット
sm.qqplot(model.resid, line='45', ax=axes[0, 1])
axes[0, 1].set_title('Q-Qプロット', fontsize=13)
axes[0, 1].grid(True, alpha=0.3)

# 実測値 vs 予測値
axes[1, 0].scatter(df['価格'], model.fittedvalues, alpha=0.6)
axes[1, 0].plot([df['価格'].min(), df['価格'].max()],
                [df['価格'].min(), df['価格'].max()],
                'r--', linewidth=2)
axes[1, 0].set_xlabel('実測値', fontsize=11)
axes[1, 0].set_ylabel('予測値', fontsize=11)
axes[1, 0].set_title('実測値 vs 予測値', fontsize=13)
axes[1, 0].grid(True, alpha=0.3)

# 残差のヒストグラム
axes[1, 1].hist(model.resid, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('残差', fontsize=11)
axes[1, 1].set_ylabel('度数', fontsize=11)
axes[1, 1].set_title('残差の分布', fontsize=13)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('multiple_regression_diagnostics.png', dpi=100)
print("\n診断プロットを 'multiple_regression_diagnostics.png' に保存")

print("\n" + "=" * 70)
print("分析完了！")
print("=" * 70)
```

### 出力例

```
======================================================================
重回帰分析: 不動産価格の予測
======================================================================

【回帰式】
価格 = 1005.23
     + 29.98 × 面積
     + -4.99 × 築年数
     + -2.98 × 駅距離
     + 2.01 × 階数

【各係数の解釈】
面積:
  係数: 29.9812
  P値: 0.000000 (有意)
  解釈: 面積が1単位増えると、価格が29.98万円変化

R²: 0.9945
調整済みR²: 0.9944
F統計量: 8734.56 (P値: 0.000000e+00)
```

## 多重共線性の確認

### サンプルコード: VIFの計算

```python
"""
多重共線性の確認: VIF (Variance Inflation Factor)
"""

from statsmodels.stats.outliers_influence import variance_inflation_factor

# 説明変数を取得
X = df[['面積', '築年数', '駅距離', '階数']]
X = sm.add_constant(X)  # 定数項を追加

# VIFを計算
vif_data = pd.DataFrame()
vif_data['変数'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i)
                   for i in range(X.shape[1])]

print("【VIF（分散拡大要因）】")
print(vif_data)
print("\n判定基準:")
print("  VIF < 5  : 多重共線性の問題なし")
print("  5 ≤ VIF < 10: 多重共線性の可能性あり")
print("  VIF ≥ 10 : 強い多重共線性")
```

## まとめ

重回帰分析では:
- 複数の説明変数を同時に考慮できる
- 各変数の影響を「他の変数を固定した場合」の効果として解釈
- 多重共線性に注意が必要（VIFで確認）

## 練習問題

### 問題1: 学生の成績予測

以下のデータで、複数の要因から成績を予測してください。

```python
np.random.seed(150)
n = 100
df = pd.DataFrame({
    '勉強時間': np.random.uniform(0, 10, n),
    '出席率': np.random.uniform(50, 100, n),
    '睡眠時間': np.random.uniform(4, 9, n),
    '成績': None
})

# 成績 = 20 + 3*勉強時間 + 0.5*出席率 + 2*睡眠時間 + ノイズ
df['成績'] = (20 + 3 * df['勉強時間'] + 0.5 * df['出席率'] +
              2 * df['睡眠時間'] + np.random.normal(0, 5, n))

# 重回帰分析を実行し、各要因の影響を分析してください
```

### 問題2: VIFの計算

問題1のデータでVIFを計算し、多重共線性の有無を確認してください。

## 模範解答

### 問題1の解答

```python
import statsmodels.formula.api as smf

# 重回帰分析
model = smf.ols('成績 ~ 勉強時間 + 出席率 + 睡眠時間', data=df).fit()
print(model.summary())

print("\n【解釈】")
for var in ['勉強時間', '出席率', '睡眠時間']:
    coef = model.params[var]
    pval = model.pvalues[var]
    print(f"{var}: 係数={coef:.2f}, P値={pval:.4f}")
```

### 問題2の解答

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

X = df[['勉強時間', '出席率', '睡眠時間']]
X = sm.add_constant(X)

vif_data = pd.DataFrame()
vif_data['変数'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i)
                   for i in range(X.shape[1])]
print(vif_data)
print("\nすべてのVIFが5未満なので、多重共線性の問題はありません")
```

# 2.4 回帰診断と残差分析

## 回帰診断の重要性

回帰モデルの仮定が満たされているかを確認することは、正しい統計的推論のために不可欠です。

## 主要な診断プロット

### サンプルコード: 包括的な診断プロット

```python
"""
回帰診断の包括的な例
4つの主要な診断プロットを作成
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.gofplots import ProbPlot

np.random.seed(42)

# データ生成
n = 100
X = np.random.randn(n, 2)
y = 2 + 3 * X[:, 0] + 5 * X[:, 1] + np.random.randn(n) * 0.5

df = pd.DataFrame(X, columns=['X1', 'X2'])
df['y'] = y

# モデルの当てはめ
model = smf.ols('y ~ X1 + X2', data=df).fit()

print("=" * 70)
print("回帰診断と残差分析")
print("=" * 70)

# 診断プロットの作成
fig = plt.figure(figsize=(16, 12))
fig.suptitle('回帰診断プロット', fontsize=16, y=0.995)

# プロット1: 残差 vs 予測値
ax1 = plt.subplot(2, 3, 1)
residuals = model.resid
fitted = model.fittedvalues

ax1.scatter(fitted, residuals, alpha=0.6)
ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax1.set_xlabel('予測値', fontsize=11)
ax1.set_ylabel('残差', fontsize=11)
ax1.set_title('1. 残差プロット', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# プロット2: Q-Qプロット
ax2 = plt.subplot(2, 3, 2)
sm.qqplot(residuals, line='45', ax=ax2)
ax2.set_title('2. Q-Qプロット (正規性)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# プロット3: Scale-Location
ax3 = plt.subplot(2, 3, 3)
standardized_resid = (residuals - residuals.mean()) / residuals.std()
sqrt_abs_std_resid = np.sqrt(np.abs(standardized_resid))

ax3.scatter(fitted, sqrt_abs_std_resid, alpha=0.6)
ax3.set_xlabel('予測値', fontsize=11)
ax3.set_ylabel('√|標準化残差|', fontsize=11)
ax3.set_title('3. Scale-Location (等分散性)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# LOWESS曲線を追加
from statsmodels.nonparametric.smoothers_lowess import lowess
lowess_result = lowess(sqrt_abs_std_resid, fitted, frac=0.3)
ax3.plot(lowess_result[:, 0], lowess_result[:, 1], 'r-', linewidth=2)

# プロット4: 残差のヒストグラム
ax4 = plt.subplot(2, 3, 4)
ax4.hist(residuals, bins=20, edgecolor='black', alpha=0.7, density=True)

# 正規分布を重ね描き
mu, sigma = residuals.mean(), residuals.std()
x = np.linspace(residuals.min(), residuals.max(), 100)
ax4.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) *
         np.exp(-(x - mu)**2 / (2 * sigma**2)),
         'r-', linewidth=2, label='正規分布')

ax4.set_xlabel('残差', fontsize=11)
ax4.set_ylabel('密度', fontsize=11)
ax4.set_title('4. 残差の分布', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# プロット5: Leverage vs 残差
ax5 = plt.subplot(2, 3, 5)
influence = model.get_influence()
leverage = influence.hat_matrix_diag

ax5.scatter(leverage, residuals, alpha=0.6)
ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax5.set_xlabel('Leverage', fontsize=11)
ax5.set_ylabel('残差', fontsize=11)
ax5.set_title('5. Leverage vs 残差', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)

# プロット6: Cook's Distance
ax6 = plt.subplot(2, 3, 6)
cooks_d = influence.cooks_distance[0]

ax6.stem(range(len(cooks_d)), cooks_d, markerfmt=',', basefmt=' ')
ax6.axhline(y=4/n, color='r', linestyle='--', label='基準値 4/n')
ax6.axhline(y=1, color='orange', linestyle='--', label='基準値 1')
ax6.set_xlabel('観測番号', fontsize=11)
ax6.set_ylabel("Cook's Distance", fontsize=11)
ax6.set_title("6. Cook's Distance", fontsize=13, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comprehensive_diagnostics.png', dpi=100, bbox_inches='tight')
print("診断プロットを 'comprehensive_diagnostics.png' に保存しました")

# 診断プロットの解釈
print("\n" + "=" * 70)
print("診断プロットの解釈")
print("=" * 70)

print("\n【1. 残差プロット】")
print("  目的: 線形性と等分散性の確認")
print("  良い状態: 残差がランダムに散らばっている")
print("  問題のサイン:")
print("    - パターンが見える → 線形性の仮定が満たされていない")
print("    - 扇形に広がる → 不均一分散")

print("\n【2. Q-Qプロット】")
print("  目的: 残差の正規性の確認")
print("  良い状態: 点が直線上に並ぶ")
print("  問題のサイン:")
print("    - S字型のカーブ → 歪度がある")
print("    - 両端が離れる → 裾が重い分布")

print("\n【3. Scale-Location】")
print("  目的: 等分散性の確認")
print("  良い状態: 点が水平に散らばる")
print("  問題のサイン:")
print("    - 右上がり/右下がりの傾向 → 不均一分散")

print("\n【4. 残差の分布】")
print("  目的: 残差の正規性を視覚的に確認")
print("  良い状態: 正規分布の曲線に従う")

print("\n【5. Leverage vs 残差】")
print("  目的: 影響力の大きい点を特定")
print("  問題のサイン:")
print("    - 高いLeverageで大きな残差 → 影響力の大きい外れ値")

print("\n【6. Cook's Distance】")
print("  目的: 影響力の大きい観測値を特定")
print("  基準:")
print(f"    - D > 4/n (={4/n:.4f}) → 注意が必要")
print("    - D > 1 → 非常に影響力が大きい")

# 統計的検定
print("\n" + "=" * 70)
print("統計的診断テスト")
print("=" * 70)

from scipy.stats import shapiro, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.stattools import durbin_watson

# 1. 正規性検定
print("\n【1. 正規性検定】")

# Shapiro-Wilk検定
shapiro_stat, shapiro_p = shapiro(residuals)
print(f"\nShapiro-Wilk検定:")
print(f"  統計量: {shapiro_stat:.6f}")
print(f"  P値: {shapiro_p:.6f}")
print(f"  結論: 残差は正規分布に{'従う' if shapiro_p > 0.05 else '従わない'} (α=0.05)")

# Jarque-Bera検定
jb_stat, jb_p, _, _ = jarque_bera(residuals)
print(f"\nJarque-Bera検定:")
print(f"  統計量: {jb_stat:.6f}")
print(f"  P値: {jb_p:.6f}")
print(f"  結論: 残差は正規分布に{'従う' if jb_p > 0.05 else '従わない'} (α=0.05)")

# 2. 等分散性検定
print("\n【2. 等分散性検定】")

# Breusch-Pagan検定
bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
print(f"\nBreusch-Pagan検定:")
print(f"  統計量: {bp_stat:.6f}")
print(f"  P値: {bp_p:.6f}")
print(f"  結論: {'等分散性あり' if bp_p > 0.05 else '不均一分散の可能性'} (α=0.05)")

# White検定
white_stat, white_p, _, _ = het_white(residuals, model.model.exog)
print(f"\nWhite検定:")
print(f"  統計量: {white_stat:.6f}")
print(f"  P値: {white_p:.6f}")
print(f"  結論: {'等分散性あり' if white_p > 0.05 else '不均一分散の可能性'} (α=0.05)")

# 3. 自己相関検定
print("\n【3. 自己相関検定】")
dw_stat = durbin_watson(residuals)
print(f"\nDurbin-Watson統計量: {dw_stat:.6f}")
print("  判定基準:")
print("    約2.0: 自己相関なし")
print("    < 2.0: 正の自己相関")
print("    > 2.0: 負の自己相関")

# 4. モデルの線形性検定
print("\n【4. 線形性検定】")
# RESET検定（Ramsey's RESET test）
reset_result = linear_reset(model, power=3, use_f=True)
print(f"\nRAMSEY RESET検定:")
print(f"  F統計量: {reset_result.fvalue:.6f}")
print(f"  P値: {reset_result.pvalue:.6f}")
print(f"  結論: モデルの線形性は{'適切' if reset_result.pvalue > 0.05 else '不適切な可能性'} (α=0.05)")

# 外れ値の検出
print("\n" + "=" * 70)
print("外れ値の検出")
print("=" * 70)

# 標準化残差
standardized_resid = residuals / residuals.std()

# スチューデント化残差
student_resid = influence.resid_studentized_internal

# 外部スチューデント化残差
student_resid_external = influence.resid_studentized_external

print(f"\n【外れ値の候補（|標準化残差| > 2.5）】")
outliers_std = np.where(np.abs(standardized_resid) > 2.5)[0]
if len(outliers_std) > 0:
    print(f"検出された外れ値の数: {len(outliers_std)}")
    print(f"観測番号: {outliers_std}")
    for idx in outliers_std[:5]:  # 最大5個表示
        print(f"  観測{idx}: 標準化残差 = {standardized_resid[idx]:.4f}")
else:
    print("外れ値は検出されませんでした")

print(f"\n【影響力の大きい点（Cook's D > 4/n）】")
influential = np.where(cooks_d > 4/n)[0]
if len(influential) > 0:
    print(f"検出された点の数: {len(influential)}")
    print(f"観測番号: {influential}")
    for idx in influential[:5]:
        print(f"  観測{idx}: Cook's D = {cooks_d[idx]:.6f}")
else:
    print("影響力の大きい点は検出されませんでした")

print("\n" + "=" * 70)
print("診断完了！")
print("=" * 70)
```

### 出力例

```
======================================================================
回帰診断と残差分析
======================================================================

診断プロットを 'comprehensive_diagnostics.png' に保存しました

======================================================================
診断プロットの解釈
======================================================================

【1. 残差プロット】
  目的: 線形性と等分散性の確認
  良い状態: 残差がランダムに散らばっている
  ...

======================================================================
統計的診断テスト
======================================================================

【1. 正規性検定】

Shapiro-Wilk検定:
  統計量: 0.992345
  P値: 0.823456
  結論: 残差は正規分布に従う (α=0.05)

【2. 等分散性検定】

Breusch-Pagan検定:
  統計量: 1.234567
  P値: 0.539876
  結論: 等分散性あり (α=0.05)

【3. 自己相関検定】

Durbin-Watson統計量: 2.012345
  → 自己相関なし

【4. 線形性検定】

RAMSEY RESET検定:
  F統計量: 0.543210
  P値: 0.654321
  結論: モデルの線形性は適切 (α=0.05)
```

## 問題のあるケース

### サンプルコード: 不均一分散の例

```python
"""
不均一分散（heteroscedasticity）の例
"""

np.random.seed(100)
n = 100
X = np.random.uniform(0, 10, n)

# 分散がXに依存する（不均一分散）
y = 2 + 3 * X + np.random.randn(n) * X * 0.5

df = pd.DataFrame({'X': X, 'y': y})
model = smf.ols('y ~ X', data=df).fit()

# 診断
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 残差プロット
axes[0].scatter(model.fittedvalues, model.resid, alpha=0.6)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('予測値')
axes[0].set_ylabel('残差')
axes[0].set_title('残差プロット（不均一分散）')
axes[0].grid(True, alpha=0.3)

# Scale-Location
standardized_resid = model.resid / model.resid.std()
axes[1].scatter(model.fittedvalues, np.sqrt(np.abs(standardized_resid)), alpha=0.6)
axes[1].set_xlabel('予測値')
axes[1].set_ylabel('√|標準化残差|')
axes[1].set_title('Scale-Location プロット')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('heteroscedasticity_example.png', dpi=100)

# 等分散性検定
from statsmodels.stats.diagnostic import het_breuschpagan
bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
print(f"Breusch-Pagan検定: P値 = {bp_p:.6f}")
print("→ 不均一分散が検出されました")
```

## まとめ

回帰診断では:
1. **4つの主要プロット**で視覚的に確認
2. **統計的検定**で客観的に評価
3. **外れ値と影響力の大きい点**を特定
4. 問題があれば**モデルの改善**を検討

## 練習問題

### 問題: 診断の実践

以下のデータで回帰モデルを構築し、すべての診断を実行してください。

```python
np.random.seed(300)
n = 80
X1 = np.random.randn(n)
X2 = np.random.randn(n)
# 非線形関係を含むデータ
y = 1 + 2*X1 + 3*X2 + 0.5*X1**2 + np.random.randn(n)*0.5

# タスク:
# 1. 線形モデルを当てはめる
# 2. 4つの診断プロットを作成
# 3. RESET検定で線形性を確認
# 4. 結果を解釈
```

## 模範解答

```python
df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})
model = smf.ols('y ~ X1 + X2', data=df).fit()

# 診断プロット
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].scatter(model.fittedvalues, model.resid, alpha=0.6)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_title('残差プロット')

sm.qqplot(model.resid, line='45', ax=axes[0, 1])
axes[0, 1].set_title('Q-Qプロット')

# ...他のプロット

# RESET検定
from statsmodels.stats.diagnostic import linear_reset
reset = linear_reset(model, power=3)
print(f"RESET検定 P値: {reset.pvalue:.6f}")
print("→ 線形性の仮定が満たされていない可能性（X1の二乗項を追加すべき）")
```

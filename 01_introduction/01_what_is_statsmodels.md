# 1.1 statsmodelsとは

## 概要

statsmodelsは、Pythonで統計モデリング、仮説検定、データ探索を行うための包括的なライブラリです。科学計算の基盤であるNumPy、SciPy、pandasと統合されており、Rのような統計専用言語に匹敵する機能を提供します。

## 主な特徴

### 1. 豊富な統計モデル
- **線形モデル**: OLS、WLS、GLS、Ridge回帰など
- **一般化線形モデル (GLM)**: ロジスティック回帰、ポアソン回帰など
- **時系列モデル**: ARIMA、SARIMAX、VAR、状態空間モデルなど
- **離散選択モデル**: Logit、Probit、多項選択モデルなど
- **パネルデータモデル**: 固定効果、変量効果モデルなど

### 2. 充実した統計検定
- t検定、F検定、カイ二乗検定
- 正規性検定（Shapiro-Wilk、Kolmogorov-Smirnovなど）
- 多重比較検定
- 単位根検定、共和分検定（時系列）

### 3. データ探索とビジュアライゼーション
- 記述統計量の計算
- 残差診断
- QQプロット、影響力プロットなど

### 4. R形式の数式サポート
pandasのDataFrameとPatsy記法を使って、Rのようにモデルを指定できます。

```python
# R形式の数式
model = smf.ols('y ~ x1 + x2 + x3', data=df).fit()
```

## statsmodelsの位置づけ

### scikit-learnとの違い

| 特徴 | statsmodels | scikit-learn |
|------|-------------|--------------|
| **目的** | 統計的推論と仮説検定 | 機械学習と予測 |
| **出力** | 詳細な統計量（p値、信頼区間など） | 予測精度の指標 |
| **モデル解釈** | 重視（係数の解釈が重要） | 予測性能重視 |
| **数式API** | あり（R形式） | なし |
| **用途** | 統計分析、計量経済学、研究 | 予測、分類、クラスタリング |

### 使い分けの例

**statsmodelsを使う場合:**
- 変数間の関係を統計的に検証したい
- モデルの係数に統計的有意性が必要
- 研究論文や報告書に統計量を記載する
- 因果関係の推定を行いたい

**scikit-learnを使う場合:**
- 予測精度が最優先
- 複雑な非線形モデル（ランダムフォレスト、XGBoostなど）を使いたい
- クロスバリデーションで汎化性能を評価したい
- 大規模データの分類・クラスタリング

## 簡単な例

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# サンプルデータの生成
np.random.seed(42)
n = 100
x = np.random.randn(n)
y = 2 + 3 * x + np.random.randn(n) * 0.5

# DataFrameに変換
df = pd.DataFrame({'x': x, 'y': y})

# R形式の数式でモデルを構築
model = smf.ols('y ~ x', data=df).fit()

# 結果のサマリーを表示
print(model.summary())
```

### 出力例

```
                            OLS Regression Results
==============================================================================
Dep. Variable:                      y   R-squared:                       0.972
Model:                            OLS   Adj. R-squared:                  0.972
Method:                 Least Squares   F-statistic:                     3376.
Date:                Mon, 09 Nov 2025   Prob (F-statistic):           3.21e-73
Time:                        12:00:00   Log-Likelihood:                -72.345
No. Observations:                 100   AIC:                             148.7
Df Residuals:                      98   BIC:                             153.9
Df Model:                           1
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      2.0234      0.051     39.876      0.000       1.923       2.124
x              2.9876      0.051     58.104      0.000       2.886       3.089
==============================================================================
Omnibus:                        1.234   Durbin-Watson:                   2.045
Prob(Omnibus):                  0.539   Jarque-Bera (JB):                1.156
Skew:                          -0.234   Prob(JB):                        0.561
Kurtosis:                       2.876   Cond. No.                         1.02
==============================================================================
```

この出力から以下のことがわかります:
- **係数**: 切片が約2.02、xの係数が約2.99（真値は2と3）
- **R-squared**: 0.972（モデルの説明力が高い）
- **P値**: どちらの係数も統計的に有意（P>|t| < 0.05）
- **信頼区間**: [0.025, 0.975]の範囲で真値を含んでいる

## statsmodelsで何ができるか

1. **因果推論**: 変数間の関係を統計的に検証
2. **予測と信頼区間**: 点推定だけでなく不確実性も評価
3. **モデル診断**: 残差分析、多重共線性の検出など
4. **時系列予測**: ARIMAモデルなどで将来を予測
5. **実験計画**: A/Bテストの統計的検証
6. **医学統計**: 生存時間分析、臨床試験の解析
7. **計量経済学**: 需要予測、政策効果の測定

## まとめ

statsmodelsは、統計的推論を重視した分析に最適なライブラリです。研究、データサイエンス、ビジネス分析など、幅広い分野で活用されています。次のセクションでは、実際にstatsmodelsをインストールして環境を整えていきます。

## 参考資料

- [statsmodels公式サイト](https://www.statsmodels.org/)
- [statsmodels GitHubリポジトリ](https://github.com/statsmodels/statsmodels)
- [statsmodels Documentation](https://www.statsmodels.org/stable/index.html)

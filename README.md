# statsmodels チュートリアル

statsmodelsは、Pythonで統計モデリング、仮説検定、データ探索を行うための包括的なライブラリです。このリポジトリでは、statsmodelsの主要な機能について日本語で解説します。

## 目次

### 1. はじめに
- [1.1 statsmodelsとは](./01_introduction/01_what_is_statsmodels.md)
- [1.2 インストールとセットアップ](./01_introduction/02_installation.md)
- [1.3 基本的な使い方](./01_introduction/03_basic_usage.md)

### 2. 線形回帰モデル
- [2.1 単回帰分析 (Simple Linear Regression)](./02_linear_regression/01_simple_regression.md)
- [2.2 重回帰分析 (Multiple Linear Regression)](./02_linear_regression/02_multiple_regression.md)
- [2.3 OLS（普通最小二乗法）の詳細](./02_linear_regression/03_ols_details.md)
- [2.4 回帰診断と残差分析](./02_linear_regression/04_diagnostics.md)
- [2.5 WLS（加重最小二乗法）とGLS（一般化最小二乗法）](./02_linear_regression/05_wls_gls.md)

### 3. 一般化線形モデル (GLM)
- [3.1 GLMの基礎](./03_glm/01_glm_basics.md)
- [3.2 ロジスティック回帰](./03_glm/02_logistic_regression.md)
- [3.3 ポアソン回帰](./03_glm/03_poisson_regression.md)
- [3.4 その他のGLMファミリー](./03_glm/04_other_families.md)

### 4. 時系列分析
- [4.1 時系列データの基礎](./04_time_series/01_time_series_basics.md)
- [4.2 ARIMA モデル](./04_time_series/02_arima.md)
- [4.3 SARIMAX モデル（季節性ARIMA）](./04_time_series/03_sarimax.md)
- [4.4 状態空間モデル](./04_time_series/04_state_space.md)
- [4.5 VAR（ベクトル自己回帰）モデル](./04_time_series/05_var.md)
- [4.6 時系列の分解とトレンド分析](./04_time_series/06_decomposition.md)

### 5. 統計検定
- [5.1 仮説検定の基礎](./05_statistical_tests/01_hypothesis_testing.md)
- [5.2 t検定とF検定](./05_statistical_tests/02_t_test_f_test.md)
- [5.3 カイ二乗検定](./05_statistical_tests/03_chi_square_test.md)
- [5.4 正規性検定](./05_statistical_tests/04_normality_tests.md)
- [5.5 多重比較検定](./05_statistical_tests/05_multiple_comparisons.md)

### 6. 離散選択モデル
- [6.1 二項選択モデル（Probit/Logit）](./06_discrete_choice/01_binary_choice.md)
- [6.2 多項選択モデル](./06_discrete_choice/02_multinomial_choice.md)
- [6.3 順序選択モデル](./06_discrete_choice/03_ordered_choice.md)

### 7. 頑健な統計手法
- [7.1 頑健な線形モデル (RLM)](./07_robust/01_robust_linear_models.md)
- [7.2 外れ値の検出と対処](./07_robust/02_outlier_detection.md)
- [7.3 頑健な共分散推定](./07_robust/03_robust_covariance.md)

### 8. ノンパラメトリック手法
- [8.1 カーネル密度推定](./08_nonparametric/01_kernel_density.md)
- [8.2 ローカル回帰（LOWESS/LOESS）](./08_nonparametric/02_lowess.md)
- [8.3 ノンパラメトリック検定](./08_nonparametric/03_nonparametric_tests.md)

### 9. パネルデータ分析
- [9.1 固定効果モデル](./09_panel_data/01_fixed_effects.md)
- [9.2 変量効果モデル](./09_panel_data/02_random_effects.md)
- [9.3 動的パネルモデル](./09_panel_data/03_dynamic_panels.md)

### 10. 生存時間分析
- [10.1 生存関数とハザード関数](./10_survival/01_survival_basics.md)
- [10.2 Kaplan-Meier推定](./10_survival/02_kaplan_meier.md)
- [10.3 Cox比例ハザードモデル](./10_survival/03_cox_model.md)

### 11. 多変量解析
- [11.1 主成分分析 (PCA)](./11_multivariate/01_pca.md)
- [11.2 因子分析](./11_multivariate/02_factor_analysis.md)
- [11.3 多変量分散分析 (MANOVA)](./11_multivariate/03_manova.md)

### 12. 実践例とケーススタディ
- [12.1 経済データの分析](./12_case_studies/01_economic_data.md)
- [12.2 医療データの分析](./12_case_studies/02_medical_data.md)
- [12.3 マーケティングデータの分析](./12_case_studies/03_marketing_data.md)
- [12.4 金融データの分析](./12_case_studies/04_financial_data.md)

## 必要な環境

```bash
pip install statsmodels pandas numpy matplotlib seaborn jupyter
```

## 推奨される前提知識

- Pythonの基本的な文法
- NumPyとpandasの基礎
- 基本的な統計学の知識

## 貢献

このチュートリアルへの貢献を歓迎します。誤りの修正、内容の改善、新しいトピックの追加など、プルリクエストをお待ちしています。

## ライセンス

MIT License

## 参考資料

- [statsmodels 公式ドキュメント](https://www.statsmodels.org/)
- [statsmodels GitHub](https://github.com/statsmodels/statsmodels)

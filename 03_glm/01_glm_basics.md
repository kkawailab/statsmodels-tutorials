# 3.1 GLM(一般化線形モデル)の基礎

## GLMとは

一般化線形モデル(Generalized Linear Models)は、目的変数が正規分布以外の分布に従う場合に使用します。

### GLMの3要素

1. **確率分布**: 指数型分布族(正規、二項、ポアソンなど)
2. **線形予測子**: η = Xβ
3. **リンク関数**: g(μ) = η

### サンプルコード: GLMの基本

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 二項分布データの例
np.random.seed(42)
n = 200
x1 = np.random.randn(n)
x2 = np.random.randn(n)

# ロジット関数で確率を生成
linear_pred = -1 + 2*x1 + 3*x2
prob = 1 / (1 + np.exp(-linear_pred))
y = np.random.binomial(1, prob)

df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

# GLM with binomial family and logit link
model = smf.glm('y ~ x1 + x2', data=df, 
                family=sm.families.Binomial()).fit()
print(model.summary())

# 予測
new_data = pd.DataFrame({'x1': [0, 1, -1], 'x2': [0, 1, 1]})
predictions = model.predict(new_data)
print("\n予測確率:")
print(predictions)
```

## 主なリンク関数

| 分布 | リンク関数 | 用途 |
|------|------------|------|
| 二項分布 | logit | 二値分類 |
| ポアソン分布 | log | カウントデータ |
| ガンマ分布 | log | 正の連続値 |
| 正規分布 | identity | 線形回帰 |

## 練習問題

```python
# カウントデータの例(ポアソン回帰)
np.random.seed(100)
x = np.random.randn(100)
lambda_ = np.exp(0.5 + 1.5*x)
y = np.random.poisson(lambda_)

# ポアソンGLMを実行してください
```

## 模範解答

```python
df = pd.DataFrame({'y': y, 'x': x})
model = smf.glm('y ~ x', data=df, 
                family=sm.families.Poisson()).fit()
print(model.summary())
```

# 3.2 ロジスティック回帰

## ロジスティック回帰とは

二値の目的変数(0/1)を予測するモデルです。

### サンプルコード: ロジスティック回帰の実例

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

np.random.seed(42)
n = 300

# 試験の点数と合格/不合格
study_hours = np.random.uniform(0, 10, n)
prev_score = np.random.uniform(40, 100, n)

# ロジット: log(p/(1-p)) = -5 + 0.3*study_hours + 0.05*prev_score
logit = -5 + 0.3*study_hours + 0.05*prev_score
prob = 1 / (1 + np.exp(-logit))
passed = np.random.binomial(1, prob)

df = pd.DataFrame({
    '勉強時間': study_hours,
    '前回点数': prev_score,
    '合格': passed
})

print("=" * 70)
print("ロジスティック回帰: 合格予測モデル")
print("=" * 70)

# ロジスティック回帰
model = smf.logit('合格 ~ 勉強時間 + 前回点数', data=df).fit()
print(model.summary())

# オッズ比の計算
print("\n【オッズ比】")
odds_ratios = np.exp(model.params)
print(odds_ratios)

print("\n【解釈】")
print(f"勉強時間が1時間増えると、合格オッズが{odds_ratios['勉強時間']:.2f}倍になる")
print(f"前回点数が1点増えると、合格オッズが{odds_ratios['前回点数']:.2f}倍になる")

# 予測
new_students = pd.DataFrame({
    '勉強時間': [2, 5, 8],
    '前回点数': [50, 70, 85]
})

pred_prob = model.predict(new_students)
print("\n【新しい学生の合格確率】")
for i, (hours, score, prob) in enumerate(zip(
    new_students['勉強時間'],
    new_students['前回点数'],
    pred_prob
)):
    print(f"学生{i+1}: 勉強{hours}時間, 前回{score}点 → 合格確率{prob:.1%}")

# 可視化: 決定境界
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 散布図
colors = ['red' if p == 0 else 'blue' for p in passed]
axes[0].scatter(df['勉強時間'], df['前回点数'], c=colors, alpha=0.5)
axes[0].set_xlabel('勉強時間', fontsize=11)
axes[0].set_ylabel('前回点数', fontsize=11)
axes[0].set_title('合格/不合格の分布', fontsize=13)
axes[0].legend(['不合格', '合格'])

# ROC曲線
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(df['合格'], model.predict(df))
roc_auc = auc(fpr, tpr)

axes[1].plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
axes[1].plot([0, 1], [0, 1], 'k--', linewidth=2)
axes[1].set_xlabel('偽陽性率', fontsize=11)
axes[1].set_ylabel('真陽性率', fontsize=11)
axes[1].set_title('ROC曲線', fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logistic_regression.png', dpi=100)
print("\n図を保存しました")

# モデル評価
print("\n" + "=" * 70)
print("モデル評価")
print("=" * 70)

# 混同行列
threshold = 0.5
predicted_class = (model.predict(df) > threshold).astype(int)
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(df['合格'], predicted_class)
print("\n【混同行列】")
print(cm)

print("\n【分類レポート】")
print(classification_report(df['合格'], predicted_class, 
                           target_names=['不合格', '合格']))
```

## 練習問題

```python
# スパムメール判定
np.random.seed(150)
n = 200
word_count = np.random.poisson(50, n)
link_count = np.random.poisson(3, n)

# スパム確率
logit = -2 + 0.03*word_count + 0.5*link_count
prob = 1 / (1 + np.exp(-logit))
is_spam = np.random.binomial(1, prob)

# ロジスティック回帰を実行し、オッズ比を計算してください
```

## 模範解答

```python
df = pd.DataFrame({
    'word_count': word_count,
    'link_count': link_count,
    'is_spam': is_spam
})

model = smf.logit('is_spam ~ word_count + link_count', data=df).fit()
print(model.summary())

odds_ratios = np.exp(model.params)
print("\nオッズ比:")
print(odds_ratios)
```

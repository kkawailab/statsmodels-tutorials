# 6.2 多項選択モデル

3つ以上のカテゴリから選択

```python
from statsmodels.discrete.discrete_model import MNLogit

model = MNLogit(y, X).fit()
```

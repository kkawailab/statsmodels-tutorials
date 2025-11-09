# 3.4 その他のGLMファミリー

ガンマ分布、負の二項分布など、様々な分布族に対応できます。

```python
# ガンマ回帰の例
model = smf.glm('y ~ x', data=df, family=sm.families.Gamma()).fit()
```

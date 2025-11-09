# 12.3 マーケティングデータの分析

顧客セグメンテーション、購買予測

```python
# RFM分析とクラスタリング
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4)
segments = kmeans.fit_predict(rfm_data)
```

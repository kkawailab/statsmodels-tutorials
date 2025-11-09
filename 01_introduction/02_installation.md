# 1.2 インストールとセットアップ

## 必要な環境

- Python 3.8以上
- pip または conda

## インストール方法

### 方法1: pipを使用（推奨）

```bash
# statsmodelsのみをインストール
pip install statsmodels

# 推奨: 関連パッケージも一緒にインストール
pip install statsmodels pandas numpy matplotlib seaborn scipy jupyter
```

### 方法2: condaを使用

```bash
# conda環境でインストール
conda install -c conda-forge statsmodels

# 関連パッケージも一緒にインストール
conda install -c conda-forge statsmodels pandas numpy matplotlib seaborn scipy jupyter
```

### 方法3: 仮想環境を作成してインストール（推奨）

```bash
# 仮想環境を作成
python -m venv statsmodels_env

# 仮想環境を有効化（Windows）
statsmodels_env\Scripts\activate

# 仮想環境を有効化（Mac/Linux）
source statsmodels_env/bin/activate

# パッケージをインストール
pip install statsmodels pandas numpy matplotlib seaborn scipy jupyter
```

## インストールの確認

以下のPythonスクリプトを実行して、インストールが正常に完了したか確認します。

### サンプルコード: installation_check.py

```python
"""
statsmodelsのインストール確認スクリプト
各パッケージのバージョンと動作を確認します
"""

import sys

def check_installation():
    """必要なパッケージのインストール状況を確認"""

    packages = {
        'numpy': 'NumPy',
        'pandas': 'pandas',
        'statsmodels': 'statsmodels',
        'matplotlib': 'Matplotlib',
        'scipy': 'SciPy',
        'seaborn': 'Seaborn'
    }

    print("=" * 60)
    print("statsmodels環境の確認")
    print("=" * 60)
    print(f"Python version: {sys.version}\n")

    # 各パッケージのインポートとバージョン確認
    for module_name, display_name in packages.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'バージョン不明')
            print(f"✓ {display_name:15s}: {version}")
        except ImportError:
            print(f"✗ {display_name:15s}: インストールされていません")

    print("\n" + "=" * 60)

    # 簡単な動作確認
    try:
        import numpy as np
        import pandas as pd
        import statsmodels.api as sm

        print("\n動作確認テスト:")
        print("-" * 60)

        # サンプルデータで線形回帰
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = 1 + 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(50) * 0.1

        # 定数項を追加
        X_with_const = sm.add_constant(X)

        # OLSモデルの実行
        model = sm.OLS(y, X_with_const).fit()

        print("✓ 線形回帰モデルの実行: 成功")
        print(f"  R-squared: {model.rsquared:.4f}")
        print(f"  係数: {model.params}")

        print("\n" + "=" * 60)
        print("すべてのテストが正常に完了しました！")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ エラーが発生しました: {e}")
        return False

if __name__ == "__main__":
    check_installation()
```

### 出力例

```
============================================================
statsmodels環境の確認
============================================================
Python version: 3.10.12 (main, Nov  9 2024, 10:00:00) [GCC 11.4.0]

✓ NumPy          : 1.26.4
✓ pandas         : 2.2.1
✓ statsmodels    : 0.14.1
✓ Matplotlib     : 3.8.3
✓ SciPy          : 1.12.0
✓ Seaborn        : 0.13.2

============================================================

動作確認テスト:
------------------------------------------------------------
✓ 線形回帰モデルの実行: 成功
  R-squared: 0.9994
  係数: [0.99876543 2.00123456 2.99876543]

============================================================
すべてのテストが正常に完了しました！
============================================================
```

## Jupyter Notebookのセットアップ

statsmodelsの学習には、Jupyter Notebookの使用を推奨します。

### Jupyter Notebookのインストールと起動

```bash
# Jupyter Notebookをインストール
pip install jupyter

# Jupyter Notebookを起動
jupyter notebook
```

### サンプル: Jupyter Notebookでの基本的な使い方

```python
# セル1: パッケージのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

# グラフの表示設定
%matplotlib inline
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("インポート完了！")
```

```python
# セル2: サンプルデータで動作確認
# 身長(cm)と体重(kg)の架空データ
np.random.seed(123)
height = np.random.normal(170, 10, 100)  # 平均170cm、標準偏差10cm
weight = 0.5 * height - 20 + np.random.normal(0, 5, 100)  # 身長と体重に相関

# DataFrameに変換
df = pd.DataFrame({'身長': height, '体重': weight})

# データの確認
print(df.head())
print(f"\nデータ数: {len(df)}")
print(f"\n基本統計量:")
print(df.describe())
```

```python
# セル3: 散布図の表示
plt.figure(figsize=(10, 6))
plt.scatter(df['身長'], df['体重'], alpha=0.6)
plt.xlabel('身長 (cm)', fontsize=12)
plt.ylabel('体重 (kg)', fontsize=12)
plt.title('身長と体重の関係', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()
```

```python
# セル4: 線形回帰の実行
model = smf.ols('体重 ~ 身長', data=df).fit()
print(model.summary())

# 予測値を追加して可視化
df['予測体重'] = model.predict(df['身長'])

plt.figure(figsize=(10, 6))
plt.scatter(df['身長'], df['体重'], alpha=0.6, label='実測値')
plt.plot(df['身長'], df['予測体重'], 'r-', linewidth=2, label='回帰直線')
plt.xlabel('身長 (cm)', fontsize=12)
plt.ylabel('体重 (kg)', fontsize=12)
plt.title('線形回帰の結果', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## よくあるインストールエラーと対処法

### エラー1: "No module named 'statsmodels'"

**原因**: statsmodelsがインストールされていない

**対処法**:
```bash
pip install statsmodels
```

### エラー2: "Microsoft Visual C++ 14.0 is required" (Windows)

**原因**: C++コンパイラが必要

**対処法**:
1. Anacondaを使用する（推奨）
2. または [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) をインストール

### エラー3: バージョンの不整合

**原因**: NumPyやpandasのバージョンが古い

**対処法**:
```bash
# パッケージを最新版にアップデート
pip install --upgrade numpy pandas statsmodels
```

### エラー4: Jupyter Notebookでインポートエラー

**原因**: カーネルが仮想環境を認識していない

**対処法**:
```bash
# 仮想環境にipykernelをインストール
pip install ipykernel

# カーネルを登録
python -m ipykernel install --user --name=statsmodels_env
```

## 開発環境の推奨設定

### VS Codeを使用する場合

1. Python拡張機能をインストール
2. Jupyter拡張機能をインストール
3. `.vscode/settings.json` を設定:

```json
{
    "python.defaultInterpreterPath": "./statsmodels_env/bin/python",
    "jupyter.notebookFileRoot": "${workspaceFolder}",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true
}
```

### PyCharmを使用する場合

1. File > Settings > Project > Python Interpreter
2. 仮想環境を選択または新規作成
3. Jupyter Notebookサポートを有効化

## まとめ

これでstatsmodelsの開発環境が整いました。次のセクションでは、statsmodelsの基本的な使い方を学んでいきます。

## 練習問題

### 問題1: 環境確認
上記の `installation_check.py` を実行して、すべてのパッケージが正常にインストールされているか確認してください。

### 問題2: Jupyter Notebookでのテスト
Jupyter Notebookを起動して、以下のコードを実行し、正常に動作することを確認してください。

```python
import statsmodels.api as sm
import numpy as np

# サンプルデータ
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# 線形回帰
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
```

### 問題3: データの可視化
以下のコードを実行して、グラフが正常に表示されることを確認してください。

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True)
plt.show()
```

## 模範解答

### 問題1の解答
スクリプトを実行すると、以下のような出力が得られるはずです:
- すべてのパッケージに ✓ マークがつく
- バージョン番号が表示される
- 動作確認テストが成功する

### 問題2の解答
出力されるサマリーテーブルに以下が含まれていれば正常です:
- 係数の推定値
- R-squared値
- P値
- 信頼区間

### 問題3の解答
正弦波のグラフが表示されれば成功です。グラフには:
- x軸: 0から10まで
- y軸: -1から1まで
- 滑らかな正弦波の曲線

が表示されます。

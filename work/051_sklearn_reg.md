# 5 scikit-learnによる機械学習プログラミング

scikit-learnはデータ解析やデータマイニングをシンプルに、かつ効果的に行えるライブラリです。線形回帰、決定木、SVMなど多くの機械学習手法が実装されているため、手軽に機械学習にチャレンジすることができます。

http://scikit-learn.org/stable/

scikit-learnのサイトには以下のキーワードがあります。

+ Classification（分類）
+ Regression（回帰）
+ Clustering（クラスタリング）
+ Dimensionality reduction（次元削減）
+ Model selection（モデル選定）
+ Preprocessing（前処理）

分類や回帰は教師あり学習、クラスタリングや次元削減は教師なし学習の範囲になります。加えて、機械学習を行う上でのデータの前処理や、作成したモデルの検証など、機械学習に必要な様々な機能がライブラリとして用意されています。

<div style="page-break-before:always"></div>


## 演習1 - 教師あり学習（回帰問題）

ここではPythonの機械学習ライブラリscikit-learnを使って、教師あり学習プログラミングに取り組みます。まずは回帰問題を取り上げます。


### 線形回帰

まずはシンプルなプログラムを作成してみましょう。次のプログラムは線形回帰を行うものです。

> ここでは y = a * x + b という数式モデルの線形回帰問題を取り上げます。ここで a は係数、b は切片と呼びます。また y は 目的変数、 x は説明変数などと呼びます。与えられたデータから 係数 a と切片 b を見つけるのがプログラムの目的になります。

```python
import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([5, 7, 9, 11, 13, 15]).reshape(-1, 1)

reg = ???()
reg.???(x, y)

print(reg.???([[6], [7], [8]]))

print(reg.coef_)
print(reg.intercept_)
```

実行結果は次のように表示されるでしょう。

```
[[ 17.]
 [ 19.]
 [ 21.]]
[[ 2.]]
[ 5.]
```

実行結果から x が 6, 7, 8 のとき、yの値はそれぞれ 17, 19, 21と予測しているのがわかります。また算出したモデルの係数、切片をそれぞれ 係数：2.0、切片 5.0と算出しています。

<div style="page-break-before:always"></div>


プログラムの詳細を見てみましょう。scikit-learnには様々な機械学習アルゴリズムが実装されているので、importによって利用することができます。

```python
from sklearn.linear_model import LinearRegression
```

ここでは線形回帰（LinearRegression）クラスをimportとしています。

次にサンプルデータを準備しています。

```python
x = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([5, 7, 9, 11, 13, 15]).reshape(-1, 1)
```

ここではreshapeメソッドを使って、変数 x と y を2次元配列に変換しています。

> reshapeメソッドによって x は 二次元配列 [[0], [1], [2], [3], [4], [5]] となります。y も同様に[[5], [7], [9], [11], [13], [15]] となります。

次にLinearRegressionオブジェクトを生成し、reg.fitメソッドによってデータを学習させています。

```python
reg = LinearRegression()
reg.fit(x, y)
```

学習はすぐに終わるので、未知のデータを投入して、予測（回帰）を確認してみましょう。

```python
print(reg.predict([[6], [7], [8]])) #=>[[ 17.] [ 19.] [ 21.]]
```

また求めた係数や切片は以下のように確認できます。

```python
print(reg.coef_)
print(reg.intercept_)
```

変数regのcoef_は coefficient（係数）、intercept_ はintercept（切片）を意味します。scikit-learnでは学習済みのモデルから係数や切片の値を取得することができます。

> このサンプルではノイズのないデータを扱いましたが、実際には訓練データにノイズが含まれるのが普通です。

<div style="page-break-before:always"></div>

### 演習課題（教師あり学習-回帰問題）

次のデータは数学のテスト結果（math）と物理のテスト結果（physics）の結果をまとめたものです。

|math|physics|
|:--|:--|
|80|90|
|82|95|
|65|71|
|45|42|
|72|88|
|66|72|
|68|76|
|90|94|
|83|83|
|77|82|

上記のテスト結果を線形回帰で分析します。ここでは数学のテスト結果から物理のテスト結果を推論するものとします。回帰直線の回帰係数（coef_）と切片（intercept_）を求めてください。

```python
import numpy as np
from sklearn.linear_model import LinearRegression

math = np.array([80, 82, 65, 45, 72, 66, 68, 90, 83, 77])
physics = np.array([90, 95, 71, 42, 88, 72, 76, 94, 83, 82])

???
```

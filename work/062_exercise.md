## 総合演習2 - 手書き数字データの認識

scikit-learnに付属されている手書き数字データセットを使って機械学習に取り組みます。

> 課題のノートブック名は「yourname_digits」としてください。

http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits

### 課題2.1 - ニューラルネットワークによる分類

scikit-learnに付属するニューラルネットワークモジュール（Multi Layer Perceptron）を使って、手書き数字データを分類してください。

#### 実行結果

次のように分類結果の正答率を表示します。

```
0.984444444444
```

> 訓練データとテストデータをランダムに分割しているため、上記とは異なる結果が表示されることがあります。


#### ヒント

1. 手書き数字データをロードする
2. 訓練データとテストデータを分割する
3. Multi Layer Perceptronインスタンスを生成し、訓練データを使って学習する
4. テストデータを使って学習済みのモデルを評価する

```python
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

???
```


<!--
#### 解答例


```python
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target)

model = MLPClassifier()
model.fit(x_train, y_train)

print(model.score(x_test, y_test))
```

-->

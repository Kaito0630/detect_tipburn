再現実験をする場合は，まずdataset/set_directory.pyを実行して
アノテーションのデータのパスを現在のものに書き換えてください。
その後，train.pyを実行してください。
実験結果のcsvファイルはresultディレクトリに，
モデルはmodelsディレクトリに保存されます。


各ファイルの役割は以下のようになっています。

train.py
BackboneごとにFaster R-CNNの学習と評価を行い，モデルと実験結果を保存する。

result.py
train.pyで保存された実験結果を見やすい形で表示する。

export_graph.py
train.pyで保存された実験結果からグラフを作成する。

output.py
train.pyで保存されたモデルを用いて物体検出を行い，画像に予測領域を描画し，保存する。

dataset.py
データセットを読み込んでpytorchで扱えるように処理するためのファイル。

functions.py
他のファイルで使う関数を定義したファイル。

transforms.py
データの前処理を行うための関数を定義したファイル。

dataset/set_directory.py
アノテーションのデータ内のパスを現在のものに書き換える。

dataset/train.csv, dataset/test.csv
訓練データとテストデータのパスを記録したファイル。

dataset/images/*.jpg
レタスの群落画像

dataset/annotations/*.xml
各画像に対するアノテーションのデータ
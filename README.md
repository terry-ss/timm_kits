# Timmのキット

## 環境

python==3.9.5

* torch==1.9.0
* pytorch-lightning==1.5.0
* timm==0.4.12

他はrequirements.txtをご参考してください。



## データのフォルダー

```bash
cd classify # multi_label, regression
mkdir data
mkdir data/train data/val data/test
```






## 分類


* データを準備：
指定のラベルの種類は`data/labels_name.csv`の一列目に順に書きます。二列目は対応の整数、人が見やすいためにだけ、コードに読まられません。
画像はImageNetようにフォルダーの名がラベルこそです。訓練セットなどは`data/`中に入ります。フォルダーの名が指定のらべるにいないのフォルダーは抜かられります。




* 訓練
``` Bash
cd classify
python train.py env.gpu=0 common.epoch=100
```
env.gpuは使うGPUのIDです。
詳細情報はconfig.yamlを参考してください。
* テスト
``` Bash
python test.py env.gpu=0 common.mpath= <model path> 
```
テストの最後に、ptモデルはonnxフォーマットに転換られます。`common.is_feature==1`するなら、特徴マップはlogフィエルの同じパイスに保存られます。ブーリアンにかわって、0と1を使われます。

mpathは必要な入力です。



## 多ラベル分類
* データを準備：
仮定に全てラベルは`data/all_labels.yaml`に記録しており、ラベルの名は`data/labels_name.csv`にいます。

* 訓練

``` Bash
cd multi_label
make train
```
詳細情報はconfig.yamlをご参考してください。

* テスト
``` Bash
make test mpath= <model path> 
```
mpathは必要な入力です。


## 回帰

* データを準備：
仮定に全てラベルは`data/all_labels.csv`に記録しており、ラベルの名は`data/labels_name.csv`にいます。
`python csv2yaml.py`で、`data/all_labels.yaml`を生成します。各々ファイルの名は一つリストに対応しています。

* 訓練

``` Bash
cd regression
python train.py env.gpu_num=2 common.epoch=100
```
env.gpu_numは使うGPUの数です。詳細情報はconfig.yamlをご参考してください。

* テスト
``` Bash
python test.py env.gpu=0 common.mpath= <model path> 
```
mpathは必要な入力です。

## ウェブサービス

onnxモデルは`model/`に入ります。モデルのパイスは`config.ini`に書きます。
GPUがあるなら、`pip uninstall onnxruntime -y`て、`pip install onnxruntime-gpu`

``` Bash
cd flask
python meter.py
```
「ctrl+C」でシャットダウンできます。
別個な端末を開き、以下のコマンドを入力します：

```
python post.py
```
`img/`の画像は検知されます。

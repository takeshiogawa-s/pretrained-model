# PIDNet環境構築

セマンティックセグメンテーションのネットワークであるPIDNetを使うための環境構築説明

# 仮想環境の構築

まず仮想環境を作成する(torch_env01)

```bash
python3 -m venv torch_env01
cd torch_env01
```

仮想環境をactivateして仮想環境に入る

```python
source bin/activate
```

pip installでいろいろ入れていく

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install tensorboardX
pip install yacs
pip install tqdm
```

# PIDNetのダウンロード

PIDNetのコードはGithubで公開されているのでgit cloneで持ってくる

```bash
git clone https://github.com/XuJiacong/PIDNet.git
```

# pretrained modelの準備

PIDNet公式では、ImageNetでトレーニングされたpretrainedモデルを利用してCytyscapes or CamVidでtrain→evalする手順が公開されている。

pretrainモデルは現在リンク切れのため、ImageNetで1から学習するかKaggleに落ちてるモデルを持ってくる必要がある。（ここでは時間がかかるのでKaggleを使う）

[https://www.kaggle.com/datasets/artlaran/pidnet-pretrained](https://www.kaggle.com/datasets/artlaran/pidnet-pretrained)

ダウンロードしたpretrainedモデルは下記フォルダに格納する

```bash
PIDNet/pretrained_models/imagenet
```

# datasetの準備

PIDNetには例題データセットとしてCityscapesとCamVidが使用されている。

自分が使うデータセットで学習するのが良いのだけど、とりあえず動かすならこれに従うのが楽

ここでどっちを使うかという話だがCityscapesは研究目的のみで商用利用は不可なのでCamvidを使うことになる

ダウンロード先は下記のどちらか

[https://www.kaggle.com/datasets/carlolepelaars/camvid](https://www.kaggle.com/datasets/carlolepelaars/camvid)

[https://datasets.cms.waikato.ac.nz/ufdl/camvid/](https://datasets.cms.waikato.ac.nz/ufdl/camvid/)

ダウンロードしたdatasetは下記フォルダに格納する

```bash
PIDNet/data/camvid/images
PIDNet/data/camvid/labels
```

# ソースの修正

PIDNetは2021年に作成されており、npのバージョン間互換性でエラーが発生する。

特にintキャストしている部分がnp.int()になっておりこれをint()に直す必要がある。

下記ソース内でCtrl＋Fで該当箇所を探して修正する

```bash
datasets/base_dataset.py
tools/eval.py
utils/utils.py
tools/train.py
```

# train

下記コマンドで学習を実行できる

```bash
python tools/train.py --cfg configs/camvid/pidnet_small_camvid.yaml GPUS [0,] TRAIN.BATCH_SIZE_PER_GPU 6

```

# Evaluation

下記コマンドで推論の実行ができる

```bash
python tools/eval.py --cfg configs/camvid/pidnet_small_camvid.yaml \
                          TEST.MODEL_FILE pretrained_models/camvid/best.pt \
                          DATASET.TEST_SET list/camvid/test.lst
```
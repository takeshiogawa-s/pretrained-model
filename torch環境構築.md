# UbuntuでCUDA＋Pytorch環境構築

# ローカル実行環境

マザーボード：ASUS TUF GAMING B860-PLUS

メモリ : 32GB

CPU :Ryzen5 8500G  

GPU : Geforce RTX5070 Ti 16GB

OS : Ubuntu 24.04

# NVIDIAドライバの導入

まずはNVIDIAグラボを使うための環境を準備する。下記の方法でうまくいかない場合もあるので、Ubuntu CUDA導入とかでググってください。

最近のUbuntuの場合、最初からNvidiaドライバが入っていることがあるのでnvidia-smiコマンドで確認する

```bash
nvidia-smi
```

見つからない場合は、ubuntu-drivers devicesコマンドで対応するドライバを探す

このコマンドでリストが返ってくるのでrecommendedと書かれているものをインストールする

```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-580-open
reboot
```

再起動後、nvidia-smiコマンドで確認する。ここでCUDAのバージョンも出てくるが既にあるわけではなく、今さっている対GPUに対応しているCUDAバージョンが表示されている。

# CUDA Toolkitの導入

次にCUDA Toolkitのインストールに進む。

https://developer.nvidia.com/cuda-downloads

上記サイトから項目を選択していってdeb(network)でコマンドを取得する

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-1
reboot
```

最後にCUDAのパスを通す

```bash
export PATH=/usr/local/cuda:/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

nvccでインストールの確認

```bash
nvcc -V
```

# cuDNNの導入

最近だとpip install でPytorchを入れてる場合、勝手に入るので学習は問題なくできる。

ONNXとか使う場合は必要。

NVIDIAのHPに行って手順通り進めればコマンドが出てくる

[https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

# Pytorch導入

pytorchの公式サイトに行って環境を選ぶとインストールコマンドが出てくる

[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

ここではCUDA13.0を使うので下記コマンドになると思う(インストールは仮想環境推奨)

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

# PytorchでCUDA認識の確認

とりあえず下記Pythonコードを実行してみてそれっぽい内容が返ってくれば認識している

```python
import torch

# CUDAの確認
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())
torch.set_default_device("cuda")
print(torch.get_default_device())

# cuDNNの確認
print(torch.backends.cudnn.is_available())
print(torch.backends.cudnn.version())
```

CUDA認識後はPytorch公式のMNISTを動かしてみると良い
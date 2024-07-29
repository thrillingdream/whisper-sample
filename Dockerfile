# ベースイメージとしてPython 3.12を使用
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# 作業ディレクトリを作成
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# 必要なパッケージのインストール
RUN apt update && apt upgrade -y
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update
RUN apt install python3.12 python3.12-distutils wget -y

RUN apt-get install libcudnn8
RUN apt-get install libcudnn8-dev

# get-pip.pyを使用してpipをインストール
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.12 get-pip.py

# Poetryのインストール
RUN pip install poetry

# Poetryを使ってrequirements.txtを生成
COPY pyproject.toml poetry.lock /app/
RUN poetry export -f requirements.txt --output requirements.txt

# 依存関係をインストール
RUN pip install -r requirements.txt

# CUDA関連パッケージのインストール
RUN pip install nvidia-pyindex
RUN pip install nvidia-cublas

# LD_LIBRARY_PATHの設定
ENV LD_LIBRARY_PATH /usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib

# アプリケーションのソースコードをコピー
COPY . /app

# メインスクリプトを実行
CMD ["python3.12", "main.py"]

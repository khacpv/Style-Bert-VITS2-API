#!/bin/bash
USER="xxxx"
APP_NAME="runpod-style-bert-vits2-api"
VERSION=1.0.0

# VERSIONを目視で確認するのでy/Nで確認
echo "バージョンは ${VERSION} でよろしいですか？"
read -p "y/N: " yn
case "$yn" in [yY]*) ;; *) echo "中止します" ; exit ;; esac

# buildコマンド
sudo DOCKER_BUILDKIT=1 docker build --progress=plain . -f Dockerfile.runpod -t $USER/$APP_NAME:$VERSION

# pushコマンド
sudo docker push $USER/$APP_NAME:$VERSION

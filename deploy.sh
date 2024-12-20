#!/bin/bash
USER="khacpv"
APP_NAME="runpod-style-bert-vits2-api-ken"
VERSION=1.0.1

# Check the VERSION visually, so enter y/N
echo "Is the version ${VERSION} correct?"
read -p "y/N: " yn
case "$yn" in [yY]*) ;; *) echo "Cancel" ; exit ;; esac

sudo docker pull pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime || { echo "Failed to pull base image"; exit 1; }

# Build Command
sudo DOCKER_BUILDKIT=1 docker build --progress=plain . -f Dockerfile.runpod -t $USER/$APP_NAME:$VERSION

# Push Command
sudo docker push $USER/$APP_NAME:$VERSION

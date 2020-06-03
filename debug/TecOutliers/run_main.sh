#!/usr/bin/env bash

docker build -t skin_cancer_run .
docker run -v $PWD/training:/training -v $PWD/models:/models -v $PWD/data:/data --rm  --gpus all -it tec_outlier_detection --batch_size=1 --epochs=20 --version=1 --crop_size=10

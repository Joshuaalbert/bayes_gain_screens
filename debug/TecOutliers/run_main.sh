#!/usr/bin/env bash

docker build -t tec_outlier_detection .
docker run -v $PWD/training:/training -v $PWD/models:/models -v $PWD/outlier_data:/data --rm -it tec_outlier_detection --batch_size=1 --epochs=20 --version=1 --crop_size=10

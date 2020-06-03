FROM tensorflow/tensorflow:1.15.2-gpu-py3

RUN pip install "dm-sonnet<2" "tensorflow_probability<0.9" graph_nets

ADD ./src /tec_outlier_detection
WORKDIR /tec_outlier_detection

VOLUME /models
VOLUME /training
VOLUME /data


ENTRYPOINT ["python3", "-m", "main","--model_dir=/models","--training_dir=/training", "--data_dir=/data"]

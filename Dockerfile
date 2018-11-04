FROM tensorflow/tensorflow:1.4.0-gpu
MAINTAINER Sven Koitka<sven.koitka@uk-essen.de>

# Install additional software
RUN apt-get update && apt-get install -y \
  nano git protobuf-compiler python-pil python-lxml wget python-tk

RUN pip install \
  matplotlib \
  opencv-python==3.2.0.8 \
  pandas==0.20.3 \
  progressbar2

RUN git clone https://github.com/tensorflow/models/ /tf-models && \
    cd /tf-models && \
    git checkout 69e1c50433c6cf7843a7cd337558efbb85656f07 && \
    cd /

#COPY patches/eval.proto /tf-models/research/object_detection/protos/
#COPY patches/evaluator.py /tf-models/research/object_detection/

RUN cd /tf-models/research && \
    protoc object_detection/protos/*.proto --python_out=. && \
    touch object_detection/metrics/__init__.py && \
    touch object_detection/inference/__init__.py && \
    cd /

#RUN chmod -R a+wr /tf-models

ENV PYTHONPATH $PYTHONPATH:/tf-models/research:/tf-models/research/slim

COPY source /source

COPY annotations /annotations

WORKDIR /source

ENTRYPOINT /bin/bash

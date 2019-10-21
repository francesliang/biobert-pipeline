#!/bin/bash

set -u

#LOCAL_MODEL_DIR=$1
LOCAL_MODEL_DIR=/Users/xinliang/airflow/biobert-pipeline/Trainer/output/40/serving_model_dir/export/biobert-pipeline

DOCKER_IMAGE_NAME=tensorflow/serving

echo Pulling the Docker image: $DOCKER_IMAGE_NAME

docker pull $DOCKER_IMAGE_NAME


echo Starting the Model Server to serve from: $LOCAL_MODEL_DIR


CONTAINER_MODEL_DIR=/models/biobert
HOST_PORT=9000
CONTAINER_PORT=8501


echo Model directory: $LOCAL_MODEL_DIR

docker run -it\
  -p 127.0.0.1:$HOST_PORT:$CONTAINER_PORT \
  -v $LOCAL_MODEL_DIR:$CONTAINER_MODEL_DIR \
  -e MODEL_NAME=biobert \
  --rm $DOCKER_IMAGE_NAME


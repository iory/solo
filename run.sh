#!/bin/bash

xhost +local:root
name=solo-gpu0
docker stop $name
docker rm $name
docker run --gpus all --rm \
       --name=$name \
       --privileged \
       -p=8061:8080 \
       --volume="/dev:/dev" \
       --shm-size=8g \
       --env="DISPLAY" \
       --env="QT_X11_NO_MITSHM=1" \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
       --mount type=bind,source=$(pwd)/main.py,target=/SOLO/main.py \
       -it \
       solo /bin/bash
xhost +local:docker

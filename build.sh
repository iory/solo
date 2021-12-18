#!/bin/bash

docker build -f ./Dockerfile \
       --rm \
       -t solo \
       .

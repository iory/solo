ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 \
    git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg wget curl vim libturbojpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install SOLO
RUN conda install cython -y && conda clean --all
RUN git clone https://github.com/WXinlong/SOLO /SOLO
WORKDIR /SOLO
RUN pip install -U pip
RUN pip install --no-cache-dir -e .
RUN pip install pycocotools youtube-dl fastapi gdown fastapi uvicorn python-multipart aiofiles termcolor ipdb
RUN gdown --id 1sF9zmbiKz4l0S7HqqqePyETXhjXoxmcW \
    && tar xvzf solo-foreground-2021-07-26.tar.gz
RUN pip install pybsc==0.1.23

COPY ./entrypoint.sh /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["bash"]

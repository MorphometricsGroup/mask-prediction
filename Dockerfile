FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

USER root

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y python3 python3-pip libopencv-dev  git

COPY ./requirements.txt .

RUN pip3 install -r ./requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENV workdir=/home/user/mask-prediction

WORKDIR ${workdir}
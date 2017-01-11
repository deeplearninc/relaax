FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -y \
        libav-tools \
        python-numpy \
        python-scipy \
        python-setuptools \
        python-pip \
        libpq-dev \
        zlib1g-dev \
        libjpeg-dev \
        curl \
        cmake \
        swig \
        python-opengl \
        libboost-all-dev \
        libsdl2-dev \
        xorg-dev \
        wget \
        unzip \
        git \
        xvfb \
        x11vnc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip/

RUN pip install -U pip

# Install the OpenAI Gym
WORKDIR /opt
RUN git clone https://github.com/openai/gym.git && pip install -e './gym[all]'

# Add client's files to the image
ADD __init__.py /opt/relaax/__init__.py
ADD client.py /opt/relaax/client.py
ADD environment.py /opt/relaax/environment.py
ADD game_process.py /opt/relaax/game_process.py
ADD main /opt/relaax/main
ADD socket_protocol.py /opt/relaax/socket_protocol.py
WORKDIR /opt/relaax
#COPY ../../../relaax/ /opt/relaax
#RUN ls -la /opt/relaax/*
#WORKDIR /opt/relaax/environments/OpenAI_Gym

# Xvnc
EXPOSE 5900
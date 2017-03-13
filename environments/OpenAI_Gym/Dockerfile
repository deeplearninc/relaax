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
        # additional things
        libvncserver-dev \
    && pip install -U pip \
    # Install OpenAI Gym
    && pip install gym[all] \
    # Add RELAAX to the image
    && cd /opt \
    && git clone https://github.com/deeplearninc/relaax.git \
    && cd relaax \
    && git checkout v2_dockers \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip/

# Xvnc
EXPOSE 5900

WORKDIR /opt/relaax/environments/OpenAI_Gym
RUN chmod +x "start.sh" \
    && chmod +x "x.sh"
ENTRYPOINT ["/opt/relaax/environments/OpenAI_Gym/start.sh"]
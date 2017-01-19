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
        ratpoison \
        libvncserver-dev \
    && pip install -U pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip/

# Install the OpenAI Gym
WORKDIR /opt
RUN git clone https://github.com/openai/gym.git && pip install -e './gym[all]' \
    && mkdir /opt/relaax

# Add RELAAX to the image
COPY algorithms /opt/relaax/algorithms
COPY bin /opt/relaax/bin
COPY config /opt/relaax/config
COPY environments /opt/relaax/environments
COPY relaax /opt/relaax/relaax
COPY setup.py /opt/relaax/

# Xvnc
EXPOSE 5900

WORKDIR /opt/relaax/environments/OpenAI_Gym
RUN chmod +x "start.sh" \
    && chmod +x "x.sh"
ENTRYPOINT ["/opt/relaax/environments/OpenAI_Gym/start.sh"]
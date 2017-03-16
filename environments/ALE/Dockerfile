FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -y \
        # main dependencies
        python-pip \
        python-numpy \
        # ale dependencies
        libsdl1.2-dev \
        libsdl-gfx1.2-dev \
        libsdl-image1.2-dev \
        cmake \
        # relax dependencies
        libjpeg-dev \
        libblas-dev \
        liblapack-dev \
        gfortran \
        # mine additions
        git \
        xvfb \
        x11vnc \
        libvncserver-dev \
    && pip install -U pip \
    && pip install Cython \
    && pip install scipy \
    # Add RELAAX to the image
    && cd /opt \
    && git clone https://github.com/deeplearninc/relaax.git \
    && cd relaax \
    && git checkout v2_dockers \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip/

# Install the Arcade-Learning-Environment
WORKDIR /opt
RUN git clone https://github.com/4SkyNet/Arcade-Learning-Environment.git \
    && cd /opt/Arcade-Learning-Environment \
    && mkdir build && cd build \
    && cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=OFF .. \
    && make -j 4 && cd .. && pip install .

# Xvnc
EXPOSE 5900

WORKDIR /opt/relaax/environments/ALE
RUN chmod +x "start.sh" \
    && chmod +x "x.sh"
ENTRYPOINT ["/opt/relaax/environments/ALE/start.sh"]
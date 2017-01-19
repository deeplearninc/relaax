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
        ratpoison \
        libvncserver-dev \
    && pip install -U pip && pip install Cython \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip/

# Install the Arcade-Learning-Environment
WORKDIR /opt
RUN git clone https://github.com/4SkyNet/Arcade-Learning-Environment.git \
    && cd /opt/Arcade-Learning-Environment \
    && mkdir build && cd build \
    && cmake -DUSE_SDL=ON -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=OFF .. \
    && make -j 4 && cd .. && pip install . && mkdir /opt/relaax

# Add RELAAX to the image
COPY algorithms /opt/relaax/algorithms
COPY bin /opt/relaax/bin
COPY config /opt/relaax/config
COPY environments /opt/relaax/environments
COPY relaax /opt/relaax/relaax
COPY setup.py /opt/relaax/

# Install RELAAX framework
WORKDIR /opt/relaax
RUN pip install -e . \
    && chmod +x "/opt/relaax/environments/ALE/start.sh" \
    && chmod +x "/opt/relaax/environments/ALE/x.sh"

# Xvnc
EXPOSE 5900

WORKDIR /opt/relaax/environments/ALE
ENTRYPOINT ["/opt/relaax/environments/ALE/start.sh"]
FROM ubuntu:14.04.3

ENV DEBIAN_FRONTEND noninteractive
ENV HOME /home/ubuntu

# built-in packages
RUN apt-get update \
    && apt-get install -y --force-yes --no-install-recommends software-properties-common curl \
    && sudo sh -c "echo 'deb http://download.opensuse.org/repositories/home:/Horst3180/xUbuntu_16.04/ /' >> /etc/apt/sources.list.d/arc-theme.list" \
    && curl -SL http://download.opensuse.org/repositories/home:Horst3180/xUbuntu_16.04/Release.key | sudo apt-key add - \
    && add-apt-repository ppa:fcwu-tw/ppa \
    && apt-get update \
    && apt-get install -y \
        # additional dependencies
        curl \
        zip \
        unzip \
        software-properties-common \
        python-software-properties \
        # relaax dependencies
        git \
        libjpeg-dev \
        libblas-dev \
        liblapack-dev \
        gfortran \
    && apt-get install -y --force-yes --no-install-recommends \
        supervisor \
        openssh-server pwgen sudo vim-tiny \
        net-tools \
        lxde x11vnc xvfb \
        gtk2-engines-murrine ttf-ubuntu-font-family \
        libreoffice firefox \
        fonts-wqy-microhei \
        language-pack-zh-hant language-pack-gnome-zh-hant firefox-locale-zh-hant libreoffice-l10n-zh-tw \
        nginx \
        python-pip python-dev build-essential \
        mesa-utils libgl1-mesa-dri \
        gnome-themes-standard gtk2-engines-pixbuf gtk2-engines-murrine pinta arc-theme \
    && pip install -U pip \
    && pip install Cython

# Install the fcwu (vnc desktop)
WORKDIR /opt
RUN git clone https://github.com/fcwu/docker-ubuntu-vnc-desktop.git

WORKDIR /opt/docker-ubuntu-vnc-desktop
RUN cp -R web/ /web/ \
    && pip install setuptools wheel \
    && pip install -r /web/requirements.txt

# tini for subreap
ENV TINI_VERSION v0.9.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /bin/tini
RUN chmod +x /bin/tini \
    && cp -R noVNC/ /noVNC/ \
    && cp nginx.conf /etc/nginx/sites-enabled/default \
    && cp -R doro-lxde-wallpapers/ /usr/share/doro-lxde-wallpapers/ \
    && mv gtkrc-2.0 /home/ubuntu/.gtkrc-2.0

# install java for bazel
RUN add-apt-repository -y ppa:openjdk-r/ppa && \
    apt-get update && \
    apt-get install -y openjdk-8-jdk openjdk-8-jre-headless && \
    which java && \
    java -version && \
    update-ca-certificates -f

# Install the most recent bazel release
ENV BAZEL_VERSION 0.4.3
RUN mkdir /bazel && \
    cd /bazel && \
    curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE.txt && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

# Install deepmind-lab dependencies
RUN apt-get install -y \
        lua5.1 \
        liblua5.1-0-dev \
        libffi-dev \
        gettext \
        freeglut3-dev \
        libsdl2-dev \
        libosmesa6-dev \
        python-dev \
        python-numpy \
        realpath \
        build-essential \
    && apt-get clean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip/

# Install the DeepMind Lab
WORKDIR /opt
RUN git clone https://github.com/deepmind/lab.git

WORKDIR /opt/lab
# headless=glx | headless=false
RUN bazel build :deepmind_lab.so --define headless=osmesa \
    && bazel run :random_agent --define headless=osmesa

# Install RELAAX
WORKDIR /opt
RUN git clone https://github.com/deeplearninc/relaax.git \
    && cd relaax \
    && git checkout v2_dockers \
    && pip install -e . \
    && cp /opt/relaax/environments/DeepMind_Lab/startup.sh / \
    && cp /opt/relaax/environments/DeepMind_Lab/supervisord.conf /etc/supervisor/conf.d/ \
    && cp /opt/relaax/environments/DeepMind_Lab/run_env / \
    && mkdir -p /home/ubuntu/.config/autostart \
    && cp /opt/relaax/environments/DeepMind_Lab/run_env.desktop /home/ubuntu/.config/autostart/run_env.desktop

EXPOSE 6080
WORKDIR /root
ENTRYPOINT ["/startup.sh"]
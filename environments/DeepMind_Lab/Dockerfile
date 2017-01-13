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
    # lab ======= beg
    && apt-get install -y \
        curl \
        zip \
        unzip \
        software-properties-common \
        python-software-properties \
        # my add
        git \
    # lab ======= end
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
    && apt-get autoclean \
    && apt-get autoremove \
    && rm -rf /var/lib/apt/lists/*

# my add
# Install the fcwu
WORKDIR /opt
RUN git clone https://github.com/fcwu/docker-ubuntu-vnc-desktop.git
# my end

#ADD web /web/
WORKDIR /opt/docker-ubuntu-vnc-desktop
RUN cp -R web/ /web/
RUN pip install setuptools wheel && pip install -r /web/requirements.txt

# tini for subreap
ENV TINI_VERSION v0.9.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /bin/tini
RUN chmod +x /bin/tini

#ADD noVNC /noVNC/
RUN cp -R noVNC/ /noVNC/
#ADD nginx.conf /etc/nginx/sites-enabled/default
RUN cp nginx.conf /etc/nginx/sites-enabled/default
#ADD startup.sh /
RUN cp startup.sh /
#ADD supervisord.conf /etc/supervisor/conf.d/
RUN cp supervisord.conf /etc/supervisor/conf.d/
#ADD doro-lxde-wallpapers /usr/share/doro-lxde-wallpapers/
RUN cp -R doro-lxde-wallpapers/ /usr/share/doro-lxde-wallpapers/
#ADD gtkrc-2.0 /home/ubuntu/.gtkrc-2.0
RUN mkdir /home/ubuntu
RUN mv gtkrc-2.0 /home/ubuntu/.gtkrc-2.0

# lab ======= beg
RUN add-apt-repository -y ppa:openjdk-r/ppa && \
    apt-get update && \
    apt-get install -y openjdk-8-jdk openjdk-8-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
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
RUN apt-get update && apt-get install -y \
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
  build-essential
# lab ======= end

# my add
# Install the DeepMind Lab
WORKDIR /opt
RUN git clone https://github.com/deepmind/lab.git

WORKDIR /opt/lab
RUN bazel build :deepmind_lab.so --define headless=osmesa
# headless=glx
#RUN bazel run :random_agent --define headless=false

RUN mkdir /opt/relaax

# Add RELAAX to the image
COPY algorithms /opt/relaax/algorithms
COPY bin /opt/relaax/bin
COPY config /opt/relaax/config
COPY environments /opt/relaax/environments
COPY relaax /opt/relaax/relaax
COPY resources /opt/relaax/resources
COPY README.md /opt/relaax/
COPY setup.py /opt/relaax/

EXPOSE 6080
WORKDIR /root
ENTRYPOINT ["/startup.sh"]
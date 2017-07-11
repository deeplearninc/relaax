FROM ubuntu:14.04

MAINTAINER fcarey@gmail.com

# Temporarily shut up warnings.
ENV DISPLAY :0
ENV TERM xterm

# Basic Dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    zip \
    unzip \
    software-properties-common \
    python-dev \
    python-setuptools \
    python-software-properties && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* && \
  easy_install pip

# Dependencies for vnc setup.
RUN apt-get update && apt-get install -y \
    xvfb \
    fluxbox \
    x11vnc && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*


# We need to add a custom PPA to pick up JDK8, since trusty doesn't
# have an openjdk8 backport.  openjdk-r is maintained by a reliable contributor:
# Matthias Klose (https://launchpad.net/~doko).  It will do until
# we either update the base image beyond 14.04 or openjdk-8 is
# finally backported to trusty; see e.g.
#   https://bugs.launchpad.net/trusty-backports/+bug/1368094
RUN add-apt-repository -y ppa:openjdk-r/ppa && \
  apt-get update && apt-get install -y \
    openjdk-8-jdk \
    openjdk-8-jre-headless && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* && \
  which java && \
  java -version && \
  update-ca-certificates -f

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
# RUN echo "startup --batch" >>/root/.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
# RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
#    >>/root/.bazelrc
# ENV BAZELRC /root/.bazelrc
# Install the most recent bazel release.
ENV BAZEL_VERSION 0.4.3
RUN mkdir /bazel && \
    cd /bazel && \
    curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -fSsL -o /bazel/LICENSE https://github.com/bazelbuild/bazel/blob/master/LICENSE && \
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
    build-essential && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*


# Set the default X11 Display.
ENV DISPLAY :1
ENV VNC_PASSWORD=password
ENV XVFB_RESOLUTION=800x600x16

# Set up relaax 
RUN git clone https://github.com/deeplearninc/relaax.git && \
    cd /relaax && \
    git checkout pixie && \
    pip install -e .[all] && \
    cd ..

# Set up deepmind-lab
# Run an actual (headless=false) build since this should make subsequent builds much faster.
# Alternative commands based on the Documentation:
# bazel build :deepmind_lab.so --define headless=osmesa && \
# bazel run :python_module_test --define headless=osmesa && \
RUN git clone https://github.com/deepmind/lab && \
  cd /lab && \
  bazel build :random_agent --define headless=false && \
  cd ..

# This port is the default for connecting to VNC display :1
EXPOSE 5901

# Copy VNC script that handles restarts and make it executable.
COPY ./startup.sh /opt/
RUN chmod u+x /opt/startup.sh

ENV RELAAX_DIR /relaax
ENV DM_LAB_DIR /lab
ENV APP_DIR /app
WORKDIR $APP_DIR

# Finally, start application
CMD ["--show-ui", "false"]
ENTRYPOINT ["python", "environment/start-lab.py", "--config", "/app/app.yaml"]

FROM osrf/ros:noetic-desktop-full

RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 20.04 ships pip 20.0.2, whose bundled TOML parser cannot read
# pyproject.toml files used by current packages.  Upgrade before resolving any
# Python dependencies, while keeping versions that still support Python 3.8.
RUN python3 -m pip install --no-cache-dir --upgrade \
    pip==24.3.1 \
    setuptools==75.3.2 \
    wheel==0.45.1

# # ---- PyPI mirror (default files.pythonhosted.org is unstable from CN -> SSL "bad handshake") ----
# RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/

# ---- ROS navigation stack (OPCD-Navigation / ros_motion_planning deps) ----
RUN apt-get update && apt-get install -y \
    python-is-python3 \
    ros-noetic-gazebo-plugins \
    ros-noetic-gazebo-ros \
    ros-noetic-gazebo-ros-control \
    ros-noetic-geometry-msgs \
    ros-noetic-move-base \
    ros-noetic-roscpp \
    ros-noetic-rospy \
    ros-noetic-std-msgs \
    ros-noetic-tf2 \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-tf2-ros \
    ros-noetic-urdf \
    ros-noetic-xacro \
    ros-noetic-amcl \
    ros-noetic-base-local-planner \
    ros-noetic-map-server \
    ros-noetic-navfn \
    libgoogle-glog-dev \
    && rm -rf /var/lib/apt/lists/*


# ---- Python dependencies (the source is bind-mounted at run time) ----
# PyTorch is a huge wheel and the CN mirror stalls/drops on it -> long timeout + many retries,
# and a separate layer so a flaky retry only re-downloads torch, not everything.
RUN pip3 install --no-cache-dir --timeout 120 --retries 25 --progress-bar=raw \
    torch \
    torchaudio \
    torchvision

# Ubuntu 20.04 ships PyYAML 5.3.1 as an apt (distutils) package.  conan 2.x
# requires PyYAML>=6.0, so the big install below tries to replace it and pip
# aborts with "Cannot uninstall PyYAML 5.3.1 / It is a distutils installed
# project".  Installing PyYAML into /usr/local with --ignore-installed first
# leaves the apt copy untouched and satisfies the requirement.
RUN pip3 install --no-cache-dir --ignore-installed "PyYAML>=6.0"

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# NOTE: ir-sim is quoted below.  Unquoted, sh reads the ">" of "ir-sim>=2.4.0"
# as a redirection: pip receives a bare "ir-sim" (constraint silently dropped)
# and a file named "=2.4.0" is created.  Quoting it for real would fail the
# build -- ir-sim 2.4.0+ requires Python >= 3.9, so 2.3.4 is the newest this
# 3.8 image can have.
RUN pip3 install --timeout 120 --retries 25 --progress-bar=raw \
    matplotlib \
    numpy \
    defusedxml \
    scipy \
    pygame \
    conan \
    rospkg \
    catkin_tools \
    squaternion \
    tensorboard \
    cvxpy \
    cvxpylayers \
    ecos \
    rich \
    pyyaml \
    scikit-learn \
    gctl==1.2 \
    "ir-sim"

RUN apt-get update && apt install -y python3-catkin-tools python3-osrf-pycommon
# github.com is unreachable from this network for large transfers (TLS reset
# mid-clone), and the host proxy drops it too ("early EOF").  Go through the
# ghfast.top mirror instead.  --depth 1 because only the model files are used,
# and .git is dropped so the snapshot is not stored twice in the image.
RUN git clone --depth 1 https://ghfast.top/https://github.com/osrf/gazebo_models ~/.gazebo/models \
    && rm -rf ~/.gazebo/models/.git
# python3-tk: gctl does "from tkinter import Scale" at import time.  tkinter is
# stdlib, not a PyPI package, so it must come from apt.  Deliberately a late
# layer -- adding it to the ROS apt step above would invalidate the cached
# torch and pip layers.
RUN apt install -y psmisc python3-tk
RUN pip3 install --upgrade protobuf Pillow

# NeuPAN is mounted at /NeuPAN when the container starts. Its editable install
# is performed by buildContainer.bash after the mount is available.

ENV DEBIAN_FRONTEND=noninteractive

ENV LIBGL_ALWAYS_SOFTWARE=1
ENV GAZEBO_HEADLESS_RENDERING=0
ENV DISPLAY=
ENV GAZEBO_RESOURCE_PATH=/DRL-robot-navigation/catkin_ws/src/multi_robot_scenario/launch
ENV ROS_HOSTNAME=localhost
ENV ROS_MASTER_URI=http://localhost:11311
ENV ROS_PORT_SIM=11311

CMD ["bash"]

FROM osrf/ros:noetic-desktop-full

RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

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


# ---- OPCD-Navigation python deps (inlined from requirements.txt;
#      the source itself is bind-mounted at run time, not cloned) ----
# PyTorch is a huge wheel and the CN mirror stalls/drops on it -> long timeout + many retries,
# and a separate layer so a flaky retry only re-downloads torch, not everything.
RUN pip3 install --no-cache-dir --timeout 120 --retries 25 --progress-bar=on\
    torch \
    torchaudio \
    torchvision

RUN pip3 install --no-cache-dir --timeout 120 --retries 25 --progress-bar=on\
    matplotlib \
    numpy \
    defusedxml \
    scipy \
    pygame \
    conan \
    rospkg \
    catkin_tools \
    squaternion \
    tensorboard

RUN apt-get update && apt install -y python3-catkin-tools python3-osrf-pycommon
RUN git clone https://github.com/osrf/gazebo_models ~/.gazebo/models
RUN apt install -y psmisc
RUN apt remove -y python3-pip
RUN apt-get install -y wget
RUN wget https://bootstrap.pypa.io/pip/3.8/get-pip.py
RUN python3 get-pip.py
RUN pip3 install --upgrade protobuf Pillow

ENV DEBIAN_FRONTEND=noninteractive

ENV LIBGL_ALWAYS_SOFTWARE=1
ENV GAZEBO_HEADLESS_RENDERING=0
ENV DISPLAY=
ENV GAZEBO_RESOURCE_PATH=/DRL-robot-navigation/catkin_ws/src/multi_robot_scenario/launch
ENV ROS_HOSTNAME=localhost
ENV ROS_MASTER_URI=http://localhost:11311
ENV ROS_PORT_SIM=11311

CMD ["bash"]

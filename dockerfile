FROM osrf/ros:noetic-desktop-full

RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch tensorboard squaternion

RUN git clone --branch feature/docker_headless_noetic --single-branch https://github.com/reiniscimurs/DRL-robot-navigation /DRL-robot-navigation
RUN apt-get update && apt install -y python3-catkin-tools python3-osrf-pycommon
RUN git clone https://github.com/osrf/gazebo_models ~/.gazebo/models
RUN apt install -y psmisc
RUN apt remove -y python3-pip
RUN apt-get install -y wget
RUN wget https://bootstrap.pypa.io/pip/3.8/get-pip.py
RUN python3 get-pip.py
RUN pip3 install --upgrade protobuf


ENV DEBIAN_FRONTEND=noninteractive

ENV LIBGL_ALWAYS_SOFTWARE=1
ENV GAZEBO_HEADLESS_RENDERING=0
ENV DISPLAY=
ENV GAZEBO_RESOURCE_PATH=/DRL-robot-navigation/catkin_ws/src/multi_robot_scenario/launch
ENV ROS_HOSTNAME=localhost
ENV ROS_MASTER_URI=http://localhost:11311
ENV ROS_PORT_SIM=11311

WORKDIR /DRL-robot-navigation
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    source /usr/share/gazebo/setup.bash && \
    cd catkin_ws && \
    catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5"

RUN echo "source /DRL-robot-navigation/catkin_ws/devel/setup.bash" >> /root/.bashrc

CMD ["bash"]
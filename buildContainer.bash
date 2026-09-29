xhost +
docker run -it \
  --gpus all \
  --privileged \
  --name=cspace_container \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /etc/localtime:/etc/localtime:ro \
  -v /etc/timezone:/etc/timezone:ro \
  -v /home/qtqqtq/disk1/DRL-robot-navigation:/DRL-robot-navigation \
  -v /home/qtqqtq/disk1/NeuPAN:/NeuPAN \
  cspace_env \
  bash -lc 'pip3 install -e /NeuPAN && exec bash'
  # -v /home/qtqqtq/disk1/BEVNav/terminator/config:/root/.config/terminator/config\
  

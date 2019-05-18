FROM tano297/bonnetal:base

# who am I
MAINTAINER Andres Milioto <amilioto@uni-bonn.de>

CMD ["bash"]

# to use nvidia driver from within
LABEL com.nvidia.volumes.needed="nvidia_driver"

# this is to be able to use graphics from the container
# Replace 1000 with your user / group id (if needed)
RUN export uid=1000 gid=1000 && \
  mkdir -p /home/developer && \
  mkdir -p /etc/sudoers.d && \
  echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
  echo "developer:x:${uid}:" >> /etc/group && \
  echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
  chmod 0440 /etc/sudoers.d/developer && \
  chown ${uid}:${gid} -R /home/developer && \
  adduser developer sudo

# Set the working directory to $HOME/bonnetal
ENV HOME /home/developer
WORKDIR $HOME/bonnetal

# Copy the current directory contents into the container at $HOME/bonnetal
ADD ./ $HOME/bonnetal

# ownership of directory
RUN chown -R developer:developer $HOME/bonnetal
RUN chmod 755 $HOME/bonnetal

# user stuff (and env variables)
USER developer
RUN cp /etc/skel/.bashrc $HOME/ && \
  echo 'source /opt/ros/melodic/setup.bash' >> $HOME/.bashrc && \
  echo 'source $HOME/bonnetal/deploy/devel/setup.bash' >> $HOME/.bashrc && \
  echo 'export PYTHONPATH=/usr/local/lib/python3.5/dist-packages/cv2/:$PYTHONPATH' >> $HOME/.bashrc && \
  echo 'export NO_AT_BRIDGE=1' >> $HOME/.bashrc

ENTRYPOINT ["/bin/bash","-c"]

# for visual output run as:
# nvidia-docker run -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/developer/.Xauthority -v /home/$USER:/home/$USER --net=host --pid=host --ipc=host tano297/bonnetal:runtime /bin/bash

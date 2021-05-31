# gCurve2D

## Docker
Setup nvidia-docker (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
```bash
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
sudo usermod -aG docker $USER
```

Build
```sh
docker build -t gcurve --network host .
```

Run
```sh
docker run --rm -it --gpus all gcurve bash
```

## Build
```bash
mkdir _build && cd _build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..
make
./gCurve2D
```

# TODO
* tune params? ThreadsPerBlock

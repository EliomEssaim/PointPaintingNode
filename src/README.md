1. 安装docker

参考[tuna](https://mirrors.tuna.tsinghua.edu.cn/help/docker-ce/)

2. 安装Nvidia-container-runtime

参考[Nvidia-container-runtime](https://blog.csdn.net/qq_41295081/article/details/119478402)

```bash
curl -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get install nvidia-container-runtime
sudo systemctl stop docker
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
sudo pkill -SIGHUP dockerd
reboot
sudo systemctl daemon-reload
sudo systemctl restart docker
# 确认docker状态
systemctl status docker
```

3. 导入docker容器

```bash
sudo docker import pcd.tar
# 导入后使用 docker images 查看 IMAGE_ID
sudo docker images
sudo docker tag ${IMAGE_ID} openpcd:latest
```

4. 设置ROS多机

```bash
echo "export ROS_HOSTNAME=out-docker" >> ~/.bashrc
echo "export ROS_MASTER_URI=http://out-docker:11311" >> ~/.bashrc
source ~/.bashrc
sudo vim /etc/hosts
```

在`hosts`中添加以下两行：
```
172.17.0.1 out-docker
172.17.0.2 pcd-docker
```

5. 运行镜像

```bash
sudo docker run -itd --name pcd1 \
  --gpus all \
  --add-host out-docker:172.17.0.1 \
  --add-host pcd-docker:172.17.0.2 \
  --runtime nvidia \
  -v /SRC/home/YOURNAME/:/DEST/home/YOURNAME/ \
  openpcd /bin/bash
```

6. 使用bash打开容器

```bash
sudo docker exec -it pcd1 bash
```

7. 安装CUDA/cudnn

```bash
nvcc -V
cat /usr/local/cuda/include/cudnn_version.h |grep CUDNN_MAJOR -A 2
nvidia-smi
```
应与主机显示相同

8. 安装依赖

```bash
git clone https://git.ustc.edu.cn/JohnHe/sonic-lidar-perception.git
cd sonic-lidar-perception/src/pcd/scripts
pip install -r requirements.txt
```

* **需重新安装与CUDA版本对应的[pytorch](https://pytorch.org/get-started/locally/)，且版本不高于1.10**
* [previous versions](https://pytorch.org/get-started/previous-versions/)

9. 编译安装spconv

```bash
cd /root/spconv
sudo apt-get install libboost-all-dev
cmake --version
# cmake version >= 3.13.2
python setup.py bdist_wheel
pip install dist/spconv-1.2.1-cp38-cp38-linux_x86_64.whl --force-reinstall
```

> setup时如遇错误`THC/THCNumerics.cuh:No such file or directory`
> 
> [注释掉该行](https://github.com/traveller59/spconv/issues/464)即可

10. 安装project

```bash
cd sonic-lidar-perception/src/pcd/scripts
python setup.py develop
```

> 如遇错误`THC/THC.h: No such file or directory`
> 
> 确保[pytorch<=1.10](https://github.com/open-mmlab/OpenPCDet/issues/1014)

11. 修改各config文件内路径

`/openpcd_ws/src/pcd/scripts/tools/cfgs/dataset_configs/kitti_dataset.yaml`，

`/openpcd_ws/src/pcd/scripts/tools/cfgs/kitti_models/pointpillar.yaml`

等需要用到的yaml

12. 生成数据集信息

```bash
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

13. 运行节点

```bash
cd sonic-lidar-perception
catkin_make
source ./devel/setup.sh
cd src/pcd/scripts/
python test.py --cfg_file tools/cfgs/kitti_models/pointpillar.yaml --ckpt data/models/pointpillar_7728.pth --batch_size 1 --mode realtime
```

**Note: 各组件需在相同版本的 CUDA + torch 下编译完成，否则运行时会出现 xxx.so: undefined symbol 报错**

14. 重启主机后

```bash
sudo docker start pcd1
sudo docker exec -it pcd1 bash
```
即可直接运行节点

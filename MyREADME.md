# 《NavRL: Learning Safe Flight in Dynamic Environments》论文精读笔记

这篇由卡内基梅隆大学（CMU）团队发表在 IEEE RA-L 的文章，为无人机在复杂动态环境下的自主导航提供了一个基于深度强化学习（DRL）的方案 [cite: 4, 7, 12]。以下是论文的整体框架、数学建模以及核心细节梳理。

## 一、 整体框架原理 (NavRL Architecture)

传统的无人机导航通常采用串行模块，参数难调且依赖精确的数学模型 [cite: 9, 10]。NavRL 的核心思想是使用强化学习策略（PPO算法）进行导航，同时结合基于物理规则的“安全护盾”来保底 [cite: 12, 14]。整个系统的工作流如下：

1. **感知系统解耦处理**：系统将静态障碍物（映射为 3D 体素图）和动态障碍物（提取边界框与速度）分开处理 [cite: 85, 90, 91]。
2. **特征提取 (Feature Extraction)**：通过卷积神经网络（CNN）将静态和动态障碍物的表示分别提取为 1D 的特征向量（Embeddings） [cite: 249]。
3. **PPO 策略网络 (Actor-Critic)**：将上述特征向量与无人机的内部状态拼接，输入到多层感知机中，输出无人机的速度控制指令 [cite: 249, 250]。
4. **安全护盾 (Safety Shield)**：在部署阶段，使用速度障碍法（Velocity Obstacles, VO）对网络输出的动作进行快速校验，修正可能导致碰撞的危险指令 [cite: 229, 231]。

---

## 二、 数学推导与强化学习建模 (MDP Formulation)

文章将导航问题建模为马尔可夫决策过程 (MDP)，即 $(S, A, P, R)$ [cite: 140]。

### 1. 状态空间 (State Space, $S$)
为加速网络收敛，作者定义了**目标坐标系 (Goal Coordinate Frame)**，其原点在无人机起点，x轴始终指向终点，y轴平行于地面 [cite: 152, 153]。状态由三部分组成：
* **内部状态 ($S_{int}$)**：包含指向目标的单位方向向量、距离以及当前速度 [cite: 146, 147]。
* **动态障碍物状态 ($S_{dyn}$)**：选取最近的 $N_d$ 个动态障碍物，记录它们相对无人机的方向、距离、速度以及尺寸（长宽） [cite: 158]。
* **静态障碍物状态 ($S_{stat}$)**：不直接输入点云，而是从无人机中心向四周进行 3D 射线投射（Ray Casting），记录射线打到障碍物上的距离向量 $R_{\theta}$，这种低维表征极大缩小了 Sim-to-Real 的差距 [cite: 162, 163, 166, 170]。

### 2. 动作空间 (Action Space, $A$)
* 动作输出为三维速度指令 $V_{ctrl} \in \mathbb{R}^3$ [cite: 171]。
* **亮点**：策略网络并不直接输出速度值，而是输出 **Beta 分布**的参数 $(\alpha, \beta)$ [cite: 208]，从中采样得到归一化动作 $\hat{V}_{ctrl}^{G} \in [0,1]$，再线性映射到实际的物理限速 $[-v_{lim}, v_{lim}]$ 区间内 [cite: 205]。

### 3. 奖励函数 (Reward Function, $R$)
奖励函数是一个多目标的加权和 [cite: 212]：
* **$r_{vel}$ (速度引导)**：鼓励无人机以更快的速度朝目标方向飞行 [cite: 214]。
* **$r_{ss}$ 和 $r_{ds}$ (安全性)**：基于对数距离，鼓励无人机远离静态和动态障碍物 [cite: 220, 225]。
* **$r_{smooth}$ (平滑度)**：惩罚前后两帧之间速度的剧烈变化 [cite: 241]。
* **$r_{height}$ (高度限制)**：惩罚无人机为了避障而超出设定的高度范围 [cite: 244]。

---

## 三、 核心方法细节 (Method Details)

### 1. 动静分离的感知策略
* **静态障碍物**：使用固定内存大小的 3D Occupancy Voxel Map，这使得系统能在 $\mathcal{O}(1)$ 的常数时间复杂度内访问和更新体素的占用概率 [cite: 120, 122]。
* **动态障碍物**：作者用两个轻量级检测器（U-depth 和 DBSCAN）进行集成检测以消除假阳性噪声 [cite: 126, 130]，利用 YOLO 确认类别 [cite: 131]，最后用卡尔曼滤波（Kalman Filter）平滑估算障碍物速度 [cite: 136]。

### 2. 安全护盾的线性规划 (Safety Shield via LP)
引入了**速度障碍区 (Velocity Obstacle, VO)** 概念来克服神经网络不可靠的缺点 [cite: 229, 231]。
* 如果策略网络输出的速度 $V_{rl}$ 落在了碰撞锥内，系统会构建一个线性规划（LP）问题，寻找距离原网络输出最近，且在安全区域内的一个安全速度 $V_{safe}$ [cite: 240]。

### 3. 大规模并行与课程学习 (Curriculum Learning)
* 训练在 NVIDIA Isaac Sim 中进行，利用 GPU 并行运行上千架（1024架）无人机收集数据，极大地提升了收敛速度 [cite: 251]。
* 采用了**课程学习**：从较低的动态障碍物密度（60个）开始，当无人机导航成功率达到 80% 时逐步增加障碍物（最高增至120个），实验证明这比一开始就在高难度环境中训练效果更好 [cite: 254, 255, 301]。

---

## 四、 整体数据流与系统架构 (Data Flow & System Architecture)

NavRL 的推理不是"原始传感器 → 执行器"的纯端到端架构，而是**模块化分层方案**。以下为完整的数据流与模块交互图：

### 4.1 数据流全景

```
┌─────────────────────────────────────────────────────────────────┐
│                     外部定位与感知层                              │
│  (External Localization & Perception)                           │
│                                                                   │
│  +─────────────┐    +─────────────────┐    +─────────────┐      │
│  │ VIO/LIO/EKF │    │ 3D 地图管理      │    │ 动态障碍检测 │      │
│  │ (里程计)     │    │ (Occupancy Map)  │    │ (YOLO+KF)   │      │
│  └──────┬──────┘    └────────┬────────┘    └──────┬──────┘      │
│         │                     │                     │            │
│         └─────────────────────┼─────────────────────┘            │
└─────────────────────┬─────────────────────────────────────────────┘
                      │
                      ↓
        ┌──────────────────────────────────┐
        │   NavRL 输入构造层                │
        │ (NavRL Observation Construction) │
        ├──────────────────────────────────┤
        │  • State: 目标相对位置、速度      │
        │  • Lidar: Ray-cast 障碍特征      │
        │  • Dynamic Obs: 邻近障碍特征     │
        │  • Direction: 目标方向向量        │
        └──────────────────┬───────────────┘
                           │
                           ↓
        ┌──────────────────────────────────┐
        │   PPO 策略网络推理                 │
        │ (PPO Policy Network Inference)   │
        ├──────────────────────────────────┤
        │  • CNN 特征提取（静态障碍）       │
        │  • MLP 特征提取（动态障碍）       │
        │  • Actor 网络 → Beta 分布参数    │
        │  • 采样/均值 → 归一化动作         │
        └──────────────────┬───────────────┘
                           │
                           ↓ 局部坐标系 → 世界坐标系转换
        ┌──────────────────────────────────┐
        │   安全约束层 (可选)                │
        │ (Safety Shield - Velocity Obstacle)
        ├──────────────────────────────────┤
        │  • 检测网络输出是否在碰撞锥内     │
        │  • 线性规划求解最近安全速度      │
        └──────────────────┬───────────────┘
                           │
                           ↓
        ┌──────────────────────────────────┐
        │   控制执行层                      │
        │ (Control Execution)              │
        ├──────────────────────────────────┤
        │  • 发布 /cmd_vel (ROS)           │
        │  • 发布 PX4 setpoint             │
        │  • 或其他机器人控制接口           │
        └──────────────────┬───────────────┘
                           │
                           ↓
        ┌──────────────────────────────────┐
        │   机器人执行与闭环                 │
        │ (Robot Execution & Feedback Loop)│
        └──────────────────────────────────┘
```

### 4.2 各模块的输入/输出详解

| 模块 | 输入 | 输出 | 说明 |
|------|------|------|------|
| **里程计** | 传感器原始数据 | Odometry (位置、速度、姿态) | VIO/LIO/EKF 等均可，保证稳定性即可 |
| **地图管理** | 环境边界、预建地图 | Ray-cast 服务（距离向量） | 实时射线投射，O(1) 复杂度 |
| **动态检测** | RGB-D/相机图像 | 障碍位置、速度、尺寸 | YOLO+卡尔曼滤波平滑 |
| **观测构造** | Odom、Raycast、Detection | TensorDict (state/lidar/dyn_obs) | 目标坐标系投影，低维特征化 |
| **PPO 推理** | TensorDict 观测 | 速度向量 $(v_x, v_y, v_z)$ | 局部系→世界系，Beta 分布采样 |
| **安全护盾** | 策略速度、障碍列表 | 校正后的安全速度 | 线性规划在线优化 |
| **控制发布** | 速度指令 | ROS 消息或硬件接口 | 使机器人执行指令 |

### 4.3 关键坐标系转换

NavRL 策略在"**目标坐标系**"中输出动作，需转换回世界系才能发送给机器人：

$$V_{world} = R(target\_dir) \cdot V_{local}$$

其中：
- $V_{local}$：策略网络在目标坐标系中输出的速度
- $R(target\_dir)$：以目标方向为 x 轴的旋转矩阵
- $V_{world}$：最终发送给机器人的世界坐标速度

---

## 五、Orin NX + ROS1 实机部署详细步骤

本章提供一份可落地的**逐步部署指南**，针对搭载 Orin NX 的无人机，使用 ROS1 Noetic。

### 5.1 前置条件检查

在开始部署前，请确保以下条件已满足：

- **硬件**：
  - 无人机配备 Jetson Orin NX（或其他 NVIDIA Jetpack 设备）
  - 稳定的里程计（VIO/LIO/惯导+轮速，或 PX4 EKF）
  - 可接收速度命令的控制器（Pixhawk、自定义控制栈等）
  - RGB-D 相机或深度相机（用于动态障碍检测）

- **系统环境**：
  - Ubuntu 20.04 LTS (Jetpack 5.x 通常预装 Ubuntu 20.04)
  - ROS1 Noetic
  - CUDA 11.x 及 cuDNN 8.x（验证：`nvidia-smi`）
  - Python 3.8+

- **Python 依赖**：
  - PyTorch 2.0+ with CUDA support
  - TorchRL
  - TensorDict
  - OpenCV
  - NumPy, SciPy

### 5.2 环境搭建步骤

#### 5.2.1 创建 conda 虚拟环境（推荐）

```bash
# 查看当前 Jetpack 版本
cat /etc/nv_tegra_release

# 创建 NavRL 虚拟环境（建议 Python 3.10）
conda create -n NavRL python=3.10 -y
conda activate NavRL

# 升级 pip
pip install --upgrade pip
```

#### 5.2.2 安装 PyTorch for ARM (Jetson)

在 Jetson 设备上，建议从 NVIDIA 官方源安装预编译的 PyTorch：

```bash
# 方案 A：使用 NVIDIA 提供的轮子（推荐）
wget https://developer.download.nvidia.com/compute/redist/jp51/pytorch/torch-2.0.0+nv23.05-cp310-cp310-linux_aarch64.whl
pip install torch-2.0.0+nv23.05-cp310-cp310-linux_aarch64.whl

# 方案 B：从源码编译（耗时，不推荐新手）
# 参考：https://pytorch.org/get-started/locally/
```

验证安装：
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

#### 5.2.3 安装 TorchRL 与 TensorDict

```bash
# 按照项目推荐版本安装
pip install numpy==1.26.4
pip install torchvision==0.15.2 torchaudio==2.0.2
pip install tensordict tensorrl

# 如果 pip 安装失败，尝试从源码编译
cd ~/workspace
git clone https://github.com/pytorch-rl/tensordict.git
cd tensordict
pip install -e .

git clone https://github.com/pytorch-rl/rl.git
cd rl
pip install -e .
```

#### 5.2.4 安装其他依赖

```bash
pip install hydra-core einops pyyaml rospkg matplotlib
pip install opencv-python  # 可选，用于动态障碍检测
```

### 5.3 ROS 工作空间与包部署

#### 5.3.1 初始化 catkin 工作空间

```bash
# 假设已有 catkin_ws，如没有则创建
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin_make
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

#### 5.3.2 复制 NavRL ROS1 包

```bash
# 从仓库拉取 ROS1 包
cd ~/catkin_ws/src
git clone <NavRL_repo_URL>  # 或 cp -r <local_path>/NavRL/ros1 .
# 或直接复制
cp -r <NavRL_root>/ros1/* .
```

结构应如下：
```
~/catkin_ws/src/
├── map_manager/
├── navigation_runner/
├── onboard_detector/
└── uav_simulator/
```

#### 5.3.3 编译 ROS 包

```bash
cd ~/catkin_ws
catkin_make -j$(nproc)  # 使用所有 CPU 核编译

# 如遇编译错误（常见：缺少头文件），逐一安装
sudo apt-get update
sudo apt-get install -y ros-noetic-mavros ros-noetic-mavros-extras
sudo apt-get install -y ros-noetic-message-generation ros-noetic-std-msgs
```

验证编译成功：
```bash
source ~/catkin_ws/devel/setup.bash
rospack find navigation_runner
```

### 5.4 NavRL 策略权重与配置

#### 5.4.1 下载/拷贝预训练权重

```bash
# 权重文件位置：NavRL/quick-demos/ckpts/navrl_checkpoint.pt
mkdir -p ~/catkin_ws/src/navigation_runner/scripts/ckpts
cp <NavRL_root>/quick-demos/ckpts/navrl_checkpoint.pt \
   ~/catkin_ws/src/navigation_runner/scripts/ckpts/
```

#### 5.4.2 配置参数文件

编辑 `~/catkin_ws/src/navigation_runner/scripts/cfg/train.yaml` 或 `navigation_param.yaml`：

```yaml
# 设备配置
device: "cuda"  # 或 "cpu"（Orin NX 建议 cuda）

# 传感器参数
sensor:
  lidar_range: 4.0          # 射线投射最大距离 (m)
  lidar_hres: 10.0          # 水平分辨率 (度)
  lidar_vfov: [-10, 20]     # 垂直视场 (度)
  lidar_vbeams: 4           # 垂直光束数

# 算法参数
algo:
  actor:
    action_limit: 2.0       # 最大速度 (m/s)
  feature_extractor:
    dyn_obs_num: 5          # 同时处理的动态障碍数
    
# 里程计话题（根据你的机器人调整）
odom_topic: "/mavros/local_position/odom"  # PX4 格式
# 或
odom_topic: "/CERLAB/quadcopter/odom"      # 自定义格式

# 控制输出话题
cmd_topic: "/mavros/setpoint_raw/local"    # PX4 格式
# 或
cmd_topic: "/CERLAB/quadcopter/cmd_vel"    # 自定义格式
```

### 5.5 启动感知与地图模块（核心）

这是整个系统的基础，必须先启动。

#### 5.5.1 启动地图管理节点

```bash
# 终端1：启动 occupancy map
roslaunch map_manager occupancy_map.launch

# 配置文件示例（~/catkin_ws/src/map_manager/launch/occupancy_map.launch）
# 检查是否有预建地图参数
```

#### 5.5.2 启动动态障碍检测（如有相机）

```bash
# 终端2：启动动态检测节点
roslaunch onboard_detector dynamic_detector.launch

# 如果使用 YOLO，额外启动
rosrun onboard_detector yolo_detector_node.py
```

如果没有实时相机，可用 **fake detector**：

```bash
# 启动虚拟障碍发布器（仅用于测试）
rosrun onboard_detector fake_detector_node
```

#### 5.5.3 验证感知模块就绪

```bash
# 测试 raycast 服务
rosservice call /occupancy_map/raycast \
  "position: {x: 0, y: 0, z: 1} start_angle: 0 range: 4 vfov_min: -10 vfov_max: 20 vbeams: 4 hres: 10"

# 测试动态障碍服务
rosservice call /onboard_detector/get_dynamic_obstacles \
  "current_position: {x: 0, y: 0, z: 1} range: 4"
```

### 5.6 启动 NavRL 导航节点

```bash
# 终端3：激活 conda 环境并启动导航
conda activate NavRL
source ~/catkin_ws/devel/setup.bash

rosrun navigation_runner navigation_node.py
```

或使用 launch 文件：

```bash
roslaunch navigation_runner safety_and_perception_real.launch
```

#### 5.6.1 节点启动顺序检查

建议按以下顺序启动（防止 ROS 消息丢失）：

1. **ROS Master**：`roscore`
2. **里程计源**（px4/VIO/LIO 节点，通常已在飞控侧运行）
3. **地图管理**：`map_manager`
4. **动态检测**：`onboard_detector`
5. **安全护盾**（可选）：`safety_shield`
6. **NavRL 导航**：`navigation_node.py`

### 5.7 实机联调与测试

#### 5.7.1 通信链接验证

```bash
# 查看 ROS 话题是否在线
rostopic list | grep -E "(odom|cmd_vel|goal)"

# 监听里程计
rostopic echo /mavros/local_position/odom -n 5

# 发布测试目标（RViz 的 2D Nav Goal 或手动发布）
rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped \
  '{header: {seq: 0, stamp: now, frame_id: "map"}, pose: {position: {x: 5, y: 5, z: 1}, orientation: {x: 0, y: 0, z: 0, w: 1}}}'
```

#### 5.7.2 低速测试飞行

```bash
# 1. 无人机起飞至安全高度（脚本自动或手动）
# 2. 监控以下话题：
rosrun rqt_plot rqt_plot /mavros/local_position/odom/pose/pose/position

# 3. 在 RViz 中点击 "2D Nav Goal" 设置目标
# 4. 观察无人机是否朝目标飞行
# 5. 检查安全护盾是否激活（可选）：
rostopic echo /rl_navigation/cmd -n 10
```

#### 5.7.3 常见问题排查

| 问题 | 症状 | 解决方案 |
|------|------|--------|
| 里程计延迟 | 无人机位置更新缓慢 | 检查 VIO/EKF 计算频率，确保 ≥ 20 Hz |
| 射线投射失败 | "raycast func err" | 验证 occupancy_map 服务是否运行；检查地图参数 |
| 动态检测无响应 | 策略收不到动态障碍信息 | 检查相机/检测器是否启动；尝试 fake_detector |
| 控制指令不执行 | 无人机不动 | 确认 /cmd_vel 话题是否被订阅；检查飞控是否开启自主模式 |
| 策略输出 NaN | 网络推理出错 | 检查 checkpoint 文件完整性；重新加载权重 |

### 5.8 性能监控与调优

#### 5.8.1 实时性能分析

```bash
# 使用 rqt 监控节点运行频率
rosrun rqt_graph rqt_graph

# 查看计算负载
top -p $(pgrep -f navigation_node.py)

# 监控 CUDA 使用率（Jetson）
jtop  # 需要 pip install jetson-stats
```

#### 5.8.2 调速与调参

- **降低推理频率**（如计算压力大）：修改 `navigation.py` 中的 `timer` 周期
- **减小特征维度**：调整 `lidar_hres`（如从 10° 改为 15°）
- **限制速度**：修改 `action_limit` 参数
- **增加安全护盾**：启用 `safety_shield` 模块进一步约束动作

### 5.9 完整启动脚本参考

为便利，提供一份**一键启动脚本** `launch_navrl.sh`：

```bash
#!/bin/bash

# 一键启动 NavRL 完整系统
source ~/.bashrc
source ~/catkin_ws/devel/setup.bash

# 启动 ROS Master
roscore &
ROSCORE_PID=$!
sleep 2

# 启动感知模块（后台）
echo "[NavRL] Starting perception modules..."
roslaunch map_manager occupancy_map.launch &
sleep 3
roslaunch onboard_detector dynamic_detector.launch &
sleep 3

# 启动导航（前台，便于监听日志）
echo "[NavRL] Starting NavRL navigation..."
conda activate NavRL
rosrun navigation_runner navigation_node.py

# 清理
trap "kill $ROSCORE_PID" EXIT
```

使用：
```bash
chmod +x launch_navrl.sh
./launch_navrl.sh
```

### 5.10 进阶：安全护盾（Safety Shield）启用

如果想启用基于线性规划的安全保证，额外启动：

```bash
# 终端4：启动安全护盾
roslaunch navigation_runner safety_shield.launch
```

该模块会：
- 接收 NavRL 的速度指令
- 检测碰撞锥
- 线性规划求解安全速度
- 发布修正后的指令

配置文件：`~/catkin_ws/src/navigation_runner/cfg/safety_shield.yaml`

---

## 总结与建议

- **快速验证**：先用 quick-demos (Python 脚本) 验证策略有效性
- **逐步集成**：先验证感知模块，再启动导航，最后启用安全护盾
- **性能监控**：Jetson 设备性能有限，建议监控 CPU/GPU/内存使用
- **文档化**：记录你的参数配置与硬件特性，便于日后复现与迭代
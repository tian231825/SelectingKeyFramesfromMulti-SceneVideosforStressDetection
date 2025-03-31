# Ada-2
帧选择实验2.0

项目概述
本项目是一个与视频质量选择和强化学习相关的项目，主要包含视频质量评分计算、环境模拟、强化学习算法实现等功能。项目通过多个模块协同工作，实现对视频质量的评估和优化，以及在特定环境下的策略学习。
代码结构与功能

1. select_video_quality.py
功能：定义了 KFS_SVQ 类，用于计算视频质量得分。通过多次前向传播和距离计算，对给定的视频表示进行质量评估。
关键方法：
__init__(self, t)：初始化类，创建线性层数组。
get_score(self, video_rep, t=100, alpha=130.0, r=0.88)：计算视频质量得分。

2. TD3_.py
功能：实现了 TD3（Twin Delayed Deep Deterministic Policy Gradient）算法。包含经验回放缓冲区、演员网络和评论家网络的定义和训练。
关键类和方法：
Replay_buffer：经验回放缓冲区，用于存储和采样经验数据。
Actor：演员网络，用于生成动作。
Critic：评论家网络，用于评估动作价值。
TD3：TD3 算法的主类，包含模型初始化、动作选择、模型更新、保存和加载等方法。

3. SACD.py
功能：实现了 SACD（Soft Actor-Critic with Distributional Reinforcement Learning）算法。包含策略网络、双 Q 网络的定义和训练，支持自适应熵调整。
关键类和方法：
Policy_Net：策略网络，用于生成动作概率分布。
Double_Q_Net：双 Q 网络，用于评估动作价值。
SACD_Agent：SACD 算法的主类，包含模型初始化、动作选择、训练、保存和加载等方法。

4. Ada-main.py
功能：项目的主运行文件，负责整体流程的控制。包括环境初始化、模型构建、数据加载和训练循环等步骤，支持分布式训练。
关键流程：
初始化环境和评估环境。
构建模型和经验回放缓冲区。
加载数据。
进行训练，包括动作选择、环境交互、模型更新等。

5. KFS_Module.py
功能：定义了自定义的 KFS_Environment 环境类，遵循 Gym 接口。处理环境的状态转移、奖励计算和重置等操作。
关键方法：
__init__(self, total_frame_size, initial_state=0, group_video=3)：初始化环境。
step(self, state, action, video_1, video_2, video_3)：执行一步动作，更新环境状态并计算奖励。
reward_calculate(self, state, video)：计算奖励。
reset(self, batch=1, seed=None, input=None)：重置环境状态。

运行环境
Python 3.x
PyTorch
Gym
NumPy
TensorBoard

运行步骤

安装依赖库：
bash
pip install torch gym numpy tensorboard

配置参数：在 Config.py 文件中配置相关参数，如训练轮数、批次大小等。

启动训练：
bash
python -m torch.distributed.launch --nproc_per_node=4 Ada-main.py --distribute True --max_epoch 100 --save_model Tr

# 数据集申请Data Acsess

"The dataset application form is available in the 'data_access_form' directory. "


Submission email: tjr21@mails.tsinghua.edu.cn"

Let me know if you'd like help refining the content or formatting further!
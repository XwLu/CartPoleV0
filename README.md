# Reinforcement Learning Algorithm on CartPole-V0 Game
- ## offical-example
  - Gym官方教程，简单的遗传算法
- ## tf
  - forked from [kvfrans](https://github.com/kvfrans/openai-cartpole)
- ## pytorch
  - kvfrans工作的pytorch版本

# 代码介绍
- 整体采用了A2C的强化学习算法
- PGNet是策略网络
- VNet是评价网络
- 利用蒙特卡洛采样估计回报值

# 使用介绍
- 训练
> python pg-torch.py -m train
- 验证
> python pg-torch.py -m valid
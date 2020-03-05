import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pretty_errors
import random
import gym
from gym import wrappers, logger
import argparse
from os import path

class PGNet(nn.Module):
    def __init__(self):
        super(PGNet, self).__init__()
        self.fc1 = nn.Linear(4,2)
    
    def forward(self, state):
        x = self.fc1(state)
        x = F.softmax(x, dim=1)
        return x

class VNet(nn.Module):
    def __init__(self):
        super(VNet, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 1)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return x

def train(pgnet, optimizer_pgnet, vnet, optimizer_vnet, criterion_vnet, env, device):
    states = []
    transitions = []
    advantages = []
    updates_vals = []
    actions = []
    total_rewward = 0


    # 初始化环境
    observation = env.reset()
    # 按照当前策略，跑一遍
    for _ in range(200):
        # 根据策略选择动作
        probs = pgnet(torch.FloatTensor(observation).reshape(-1, 4).requires_grad_(True).to(device))
        action = 0 if (random.uniform(0,1) < probs.cpu().detach().numpy()[0][0]) else 1
        # 记录状态转移轨迹
        states.append(observation)
        actionblank = np.zeros(2)
        actionblank[action] = 1
        actions.append(actionblank)
        # 执行动作
        old_observation = observation
        observation, reward, done, info = env.step(action)
        transitions.append((old_observation, action, reward))
        total_rewward += reward

        if done: break
    # MC估算回报值
    for index, trans in enumerate(transitions):
        obs, action, reward = trans
        
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for index2 in range(future_transitions):
            future_reward += transitions[index2 + index][2] * decrease
            decrease = decrease * 0.97
        current_val = vnet(torch.FloatTensor(obs).requires_grad_(True).to(device)).cpu().detach().numpy()[0]

        advantages.append(future_reward - current_val)

        updates_vals.append(future_reward)

    updates_vals = torch.FloatTensor(updates_vals).reshape(-1, 1).requires_grad_(True).to(device)
    states = torch.FloatTensor(states).reshape(-1, 4).requires_grad_(True).to(device)
    actions = torch.FloatTensor(np.array(actions)).reshape(-1, 2).requires_grad_(True).to(device)
    advantages = torch.FloatTensor(advantages).reshape(-1, 1).requires_grad_(True).to(device)
    # 更新策略评价
    optimizer_vnet.zero_grad()
    vals = vnet(states)
    loss = criterion_vnet(vals, updates_vals)
    loss.backward()
    optimizer_vnet.step()
    # 更新策略
    optimizer_pgnet.zero_grad()
    probabilities = pgnet(states)
    good_probabilities = torch.sum(probabilities.mul(actions), dim=1, keepdim=True) # mul: element-wise
    eligibility = torch.log(good_probabilities) * advantages
    loss = - eligibility.sum()
    loss.backward()
    optimizer_pgnet.step()

    return total_rewward

def valid(pgnet, vnet, env, device):
    total_rewward = 0
    observation = env.reset()
    # 按照当前策略，跑一遍
    for _ in range(200):
        # 根据策略选择动作
        probs = pgnet(torch.FloatTensor(observation).reshape(-1, 4).requires_grad_(False).to(device))
        action = 0 if (random.uniform(0,1) < probs.cpu().detach().numpy()[0][0]) else 1
        # 记录状态转移轨迹
        # 执行动作
        observation, reward, done, info = env.step(action)
        total_rewward += reward

        if done: break

    return total_rewward

if __name__ == "__main__":
    logger.set_level(logger.WARN)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', '-m', default="train")
    parser.add_argument('--target', default="CartPole-v0")
    args = parser.parse_args()
    # 环境初始化
    print(args.target)
    env = gym.make(args.target)
    env.seed(0)

    # 监测器初始化
    outdir = './results' # 训练过程中的数据存储地址
    env = wrappers.Monitor(env, outdir, force=True)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        print('use cuda')
        device = torch.device("cuda")          # a CUDA device object

    if args.mode == 'train':
        pgnet = PGNet().to(device)
        optimizer_pgnet = torch.optim.Adam(pgnet.parameters(),lr=0.001,betas=(0.9,0.99))

        vnet = VNet().to(device)
        optimizer_vnet = torch.optim.Adam(vnet.parameters(),lr=0.1,betas=(0.9,0.99))
        criterion_vnet = torch.nn.MSELoss()

        # 训练
        cnt = 0
        for i in range(10000):
            reward = train(pgnet, optimizer_pgnet, vnet, optimizer_vnet, criterion_vnet, env, device)
            print(i, 'th episode reward: ', reward)
            if reward == 200:
                cnt += 1
            else:
                cnt = 0
            if cnt > 15:
                break
        torch.save(pgnet.state_dict(), './model/pgnet/net.pth')
        torch.save(vnet.state_dict(), './model/vnet/net.pth')

    # 验证
    if args.mode == 'valid':
        pgnet = PGNet().to(device)
        pgnet.load_state_dict(torch.load('./model/pgnet/net.pth'))
        vnet = VNet().to(device)
        vnet.load_state_dict(torch.load('./model/vnet/net.pth'))

        total_rew = 0
        for i in range(1000):
            reward = valid(pgnet, vnet, env, device)
            total_rew += reward
        print("Final average reward is: ", total_rew/1000)

    env.close()
    

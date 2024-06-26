import argparse
import math
import os
import numpy as np
import csv
from ss import generate_taskset  # 确保 generate_taskset 已经正确定义
from DQN import DQN  # 确保 DQN 已经正确定义
from utils import plot_learning_curve, create_directory  # 确保相应函数可用
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # 注意这可能需要在支持Tkinter的环境中运行
import os
import numpy as np
import copy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ----注意，以下两个解析参数的代码块重复了，应该删除一个----
parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=1000)
parser.add_argument('--ckpt_dir', type=str, default='C:/Users/12165/Desktop/')
parser.add_argument('--reward_path', type=str, default=r'C:/Users/12165/Desktop/avg_reward.png')
parser.add_argument('--epsilon_path', type=str, default=r'C:/Users/12165/Desktop/epsilon.png')
args = parser.parse_args()


# ----这里的代码块多余，应该被删除----

def main():
    total_rewards = []  # 存储每个回合的总能耗

    # 打开文件，准备好写入CSV，路径要正确，并确保权限允许写入
    with open(r'C:\Users\12165\Desktop\greed.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Total Energy Consumption'])  # 写入表头

    for episode in range(args.max_episodes):
        env = generate_taskset(1000)  # 假设 generate_taskset 已经正确定义
        episode_energy_consumption = 0

        for step in range(500):  # 假设每个回合有500步
            task = env[step]
            B, Q, D, AT, Pi, C, C_l, p = task[1], task[2], task[3], task[4], task[6], task[7], task[9], task[10]

            # 计算两种动作的能耗
            energy_action_0 = (p * B / math.log2(1+20*p))
            energy_action_1 = 5 * (C_l ** 2) * Q

            # 贪心选择能耗更低的动作
            chosen_action = 0 if energy_action_0 < energy_action_1 else 1
            episode_energy_consumption += energy_action_0 if chosen_action == 0 else energy_action_1

        # 记录每个回合的总能耗
        total_rewards.append(episode_energy_consumption)
        # 输出每个回合的信息
        print(f'Episode: {episode + 1}, Total Energy Consumption: {episode_energy_consumption}')

        # 每回合结束后重新写入CSV文件
        with open(r'C:\Users\12165\Desktop\greed.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([episode + 1, episode_energy_consumption])  # 写入每回合的能耗数据

    # 画图展示每个回合的总能耗随时间的变化
    plt.figure()
    plt.plot(total_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Energy Consumption")
    plt.savefig(args.reward_path)  # 保存奖励曲线图
    plt.show()


if __name__ == '__main__':
    main()
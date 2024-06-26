import copy
import math

from ss import generate_taskset, insert_task
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=1000)
parser.add_argument('--ckpt_dir', type=str, default='C:/Users/12165/Desktop/demo/')
parser.add_argument('--reward_path', type=str, default=r'C:\Users\12165\Desktop\demo\avg_reward.png')
parser.add_argument('--epsilon_path', type=str, default=r'C:\Users\12165\Desktop\demo\epsilon.png')
parser.add_argument('--csv_path', type=str, default=r'C:\Users\12165\Desktop\Random.csv')  # 添加用于存储输出结果的 CSV 文件路径参数
args = parser.parse_args()


def main():
    # 初始化奖励和能耗列表
    total_rewards, total_energy_consumption = [], []

    with open(args.csv_path, mode='w', newline='') as file:  # 使用 'w' 模式打开 CSV 文件
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Total Energy Consumption'])  # 写入表头

        for episode in range(args.max_episodes):
            env = generate_taskset(1000)
            total_energy = 0  # 统计每一集的总能耗

            for step in range(500):  # 假设每集有500步
                task = copy.deepcopy(env[step])
                B, Q, D, AT, Pi, C, C_l, p = task[1], task[2], task[3], task[4], task[6], task[7], task[9], task[10]

                # 随机地选择一个动作
                action = np.random.choice([0, 1])

                # 如果选择动作1，则进行本地计算并累加能耗
                if action == 1:
                    Energy = 5 * (C_l ** 2) * Q
                    total_energy += Energy
                else:  # 如果选择动作0，则迁移到云层并输出能耗
                    Energy =  (p * B / math.log2(1+20*p))
                    total_energy += Energy

            # 记录每个回合的能耗
            total_energy_consumption.append(total_energy)
            row = [episode + 1, total_energy]  # 当前回合的索引和总能耗
            writer.writerow(row)  # 将每一轮的能耗数据写入 CSV 文件
            print('EP:{}  Total Energy:{}'.format(episode + 1, total_energy))

    # 绘制能耗曲线
    plt.plot(total_energy_consumption)
    plt.xlabel("Episode")
    plt.ylabel("Total Energy Consumption")
    plt.title("Random Strategy Energy Consumption")
    plt.show()

if __name__ == '__main__':
    main()
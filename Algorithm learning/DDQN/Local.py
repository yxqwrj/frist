import copy
import csv
from ss import generate_taskset, insert_task
import numpy as np
import argparse
from DDQN import DDQN
from utils import plot_learning_curve, create_directory
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import os
import numpy as np
import copy
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=1000)
parser.add_argument('--ckpt_dir', type=str, default='C:/Users/12165/Desktop/demo/')
parser.add_argument('--reward_path', type=str, default=r'C:\Users\12165\Desktop\demo\avg_reward.png')
parser.add_argument('--epsilon_path', type=str, default=r'C:\Users\12165\Desktop\demo\epsilon.png')
parser.add_argument('--energy_path', type=str, default=r'C:\Users\12165\Desktop\demo\local_energy.png')
args = parser.parse_args()

# ...（先前的代码导入和参数解析部分）

def main():
    create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
    total_local_energies = []  # 存储每个回合的总本地能耗

    for episode in range(args.max_episodes):
        env = generate_taskset(1000)
        total_local_energy = 0  # 初始化当回合的总本地能耗为0

        # 假设 Tw_dict 保存了C0, C1, C2各自的等待时间
        Tw_dict = {'C0': 0, 'C1': 0, 'C2': 0}

        for step in range(500):  # 假设每个回合的最大步骤数为500
            task = env[step]
            Q = task[2]
            B, Q, D, AT, Pi, C, C_l, p = task[1], task[2], task[3], task[4], task[6], task[7], task[9], task[10]
            # 计算本地计算的能耗（动作1）
            local_energy =  5 * (C_l ** 2) * Q
            total_local_energy += local_energy  # 累加到该回合的总能耗中

        total_local_energies.append(total_local_energy)  # 将该回合的总能耗追加到列表中
        print(f'EP:{episode + 1}  Total Local Energy:{total_local_energy}')

    # 画出总本地能耗随着回合变化的图
    plt.figure()
    plt.plot(total_local_energies)
    plt.xlabel("Episode")
    plt.ylabel("Total Local Energy Consumption")
    plt.title("Local Energy Consumption Over Episodes")
    plt.savefig(args.energy_path)
    plt.show()
    with open(r'C:\Users\12165\Desktop\FullLoc.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'Total Local Energy'])  # 写入表头
        for i, energy in enumerate(total_local_energies):
            writer.writerow([i + 1, energy])
if __name__ == '__main__':
    main()

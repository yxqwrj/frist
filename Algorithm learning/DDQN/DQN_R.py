import csv
import math
from ss import generate_taskset, insert_task,tasks
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
parser.add_argument('--reward_path', type=str, default=r'C:\Users\12165\Desktop\avg_reward.png')
parser.add_argument('--epsilon_path', type=str, default=r'C:\Users\12165\Desktop\\epsilon.png')
args = parser.parse_args()

def main():
    #alpha_values = [0.1, 0.01, 0.001]
    batch_size_values = [256]
    reward_plots = []
    for batch_size in batch_size_values:
   # for alpha in alpha_values:
        agent = DDQN(alpha=0.001, state_dim=5, action_dim=2,
                    fc1_dim=256, fc2_dim=256, ckpt_dir=args.ckpt_dir, gamma=0.99, tau=0.005, epsilon=1.0,
                    eps_end=0.01, eps_dec=5e-7, max_size=10000, batch_size=batch_size)
        create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
        total_rewards, avg_rewards, eps_history = [], [], []
        episode_rewards = []

        for episode in range(args.max_episodes):
            env = generate_taskset(1000)
            initial_state_index = np.random.randint(len(env))
            state = env[initial_state_index][1:6]
            total_reward = 0
            max_steps = 500

            Tw_dict = {'C0': 0, 'C1': 0, 'C2': 0}
            for step in range(max_steps):
            # 获取任务信息和计算奖励
                task = env[step]
                state = env[step][1:6]
                B, Q, D, AT, Pi, C, C_l, p = task[1], task[2], task[3], task[4], task[6], task[7], task[9], task[10]
                T_w = Tw_dict[f'C{C}']
                SC = (0.5 * (Q / C_l+T_w) + 5 * (C_l ** 2) * Q)
                T_t = (B / math.log2(1+20*p))
                E_t = (p * B / math.log2(1+20*p))
                prev_task = env[step - 1]
                AT_prev = prev_task[4]
                abs_diff = abs(D - (AT - AT_prev))
                if C == 0:
                  Tw_dict['C0'] = 0.5 * Tw_dict['C0'] + max(0, (Q / C_l+ Tw_dict['C0']) - (AT - AT_prev))
                elif C == 1:
                  Tw_dict['C1'] = Tw_dict['C0'] + max(0, (Q / C_l + Tw_dict['C1']) - (AT - AT_prev))
                else:  # C == 2
                  Tw_dict['C2'] = Tw_dict['C0'] + Tw_dict['C1'] + max(0, (Q / C_l + Tw_dict['C2']) - (AT - AT_prev))

                T_w = Tw_dict[f'C{C}']
                Pi_new = Pi *np.log2(T_w / D)
                task[6] = Pi_new
                insert_task(task, Pi_new)
                chosen_action = agent.choose_action(state)
                action = copy.deepcopy(chosen_action)

                rewards_for_action_1 = 0
                rewards_for_action_0 = 0

                if action == 1:
                    AT_prev = 0 if step == 0 else env[step - 1][4]
                    if 5 * (C_l ** 2) * Q < E_t and (Q / C_l) <= D:
                            abs_diff = abs(D - (AT - AT_prev))
                            rewards_for_action_1 = (Pi * abs_diff / SC)
                    else:
                            rewards_for_action_1 = 0
                else:
                    abs_diff = abs(D - (AT - AT_prev))
                    rewards_for_action_0 = Pi * abs_diff / (0.5 * E_t +0.5* T_t)

                rewards = rewards_for_action_1 if rewards_for_action_1 > rewards_for_action_0 else rewards_for_action_0

                next_state = env[step + 1][1:6] if step < max_steps - 1 else None
                state = next_state
                done = (step == max_steps - 1)  # 设置 done 变量
                agent.memory.store_transition(state, action, rewards, next_state, done)

                if len(agent.memory) > agent.batch_size:
                   agent.learn(agent.batch_size)

                if done:
                    break
                total_reward += rewards
            total_rewards.append(total_reward)
            eps_history.append(agent.epsilon)
            agent.decrement_epsilon()

            episode_rewards.append(total_reward)

            print('EP:{} reward:{}  epsilon:{}'.format(episode + 1, total_reward, agent.epsilon))

            save_rewards_to_csv(episode + 1, total_reward, 'C:/Users/12165/Desktop/DQN_R.csv')

            if (episode + 1) % 50 == 0:
                agent.save_models(episode + 1)
        reward_plots.append(episode_rewards)
    # 画图

    for i, alpha in enumerate(batch_size_values):
        plt.plot(reward_plots[i], label=f'batch_size={batch_size}')
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.legend()
    #plt.show()
    path = "C:/Users/12165/Desktop/" + "alpha_values_reward.png"
    plt.savefig(path)

    episodes = [i for i in range(args.max_episodes)]
    #plot_learning_curve(episodes, episode_rewards, 'Reward', 'reward', args.reward_path)
    plot_learning_curve(episodes, eps_history, 'Epsilon', 'epsilon', args.epsilon_path)

def save_rewards_to_csv(episode, reward, csv_file_path):
    # 将奖励值保存到CSV文件
    with open(csv_file_path, 'a', newline='') as csvfile:  # 使用'a'模式追加到文件中
        writer = csv.writer(csvfile)
        writer.writerow([f'EP:{episode} reward:{reward}'])

if __name__ == '__main__':
    main()

import math
import os
import argparse
from DDQN import DDQN
from ss import generate_taskset, insert_task
from utils import plot_learning_curve, create_directory
import matplotlib.pyplot as plt
import numpy as np
import copy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=1000)
parser.add_argument('--ckpt_dir', type=str, default='C:/Users/12165/Desktop/')
parser.add_argument('--reward_path', type=str, default=r'C:/Users/12165/Desktop/avg_reward.png')
parser.add_argument('--epsilon_path', type=str, default=r'C:/Users/12165/Desktop/epsilon.png')
args = parser.parse_args()

def save_to_file(file_path, content):
    with open(file_path, 'a') as file:  # 追加模式打开文件
        file.write(content + '\n')  # 写入内容并换行

def main():
    alpha_values = [0.001]
    reward_plots = []
    action_plots = []

    for alpha in alpha_values:
        agent = DDQN(alpha=0.001, state_dim=5, action_dim=2,
                     fc1_dim=256, fc2_dim=256, ckpt_dir=args.ckpt_dir, gamma=0.99, tau=0.005, epsilon=1.0,
                     eps_end=0.01, eps_dec=5e-6, max_size=10000, batch_size=256)

        create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])

        episode_rewards = []
        episode_actions = []
        eps_history = []
        # Training loop
        for episode in range(args.max_episodes):
            env = generate_taskset(1000)
            episode_selected_tasks_action_0 = []  # 动作值为 0 的任务列表
            episode_selected_tasks_action_1 = []
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
                SC = (0.5 * (Q / C_l + T_w) + 5 * (C_l ** 2) * Q)
                T_t = (B / math.log2(1 + 20 * p))
                E_t = (p * B / math.log2(1 + 20 * p))
                prev_task = env[step - 1]
                AT_prev = prev_task[4]
                abs_diff = abs(D - (AT - AT_prev))
                if C == 0:
                    Tw_dict['C0'] = 0.5 * Tw_dict['C0'] + max(0, (Q / C_l + Tw_dict['C0']) - (AT - AT_prev))
                elif C == 1:
                    Tw_dict['C1'] = Tw_dict['C0'] + max(0, (Q / C_l + Tw_dict['C1']) - (AT - AT_prev))
                else:  # C == 2
                    Tw_dict['C2'] = Tw_dict['C0'] + Tw_dict['C1'] + max(0, (Q / C_l + Tw_dict['C2']) - (AT - AT_prev))

                T_w = Tw_dict[f'C{C}']
                Pi_new = Pi * np.log2(T_w / D)
                task[6] = Pi_new
                insert_task(task, Pi_new)
                chosen_action = agent.choose_action(state)
                action = copy.deepcopy(chosen_action)

                rewards_for_action_1 = 0
                rewards_for_action_0 = 0

                if action == 1:
                    AT_prev = 0 if step == 0 else env[step - 1][4]
                    if (10 ** -27) * (370000000 ** 2) * Q*1000000000 < E_t and (Q / 0.37 + T_w) <= D:
                        abs_diff = abs(D - (AT - AT_prev))
                        rewards_for_action_1 = (Pi * abs_diff / SC)
                    else:
                        rewards_for_action_1 = 0
                else:
                    abs_diff = abs(D - (AT - AT_prev))
                    rewards_for_action_0 = Pi * abs_diff / (0.5 * E_t + 0.5 * T_t)

                rewards = rewards_for_action_1 if rewards_for_action_1 > rewards_for_action_0 else rewards_for_action_0
                next_state = env[step + 1][1:6] if step < max_steps - 1 else None
                state = next_state
                done = (step == max_steps - 1)  # 设置 done 变量
                agent.memory.store_transition(state, action, rewards, next_state, done)

                if chosen_action == 0:
                    episode_selected_tasks_action_0.append(task)
                else:
                    episode_selected_tasks_action_1.append(task)

                if len(agent.memory) > agent.batch_size:
                    agent.learn(agent.batch_size)

                if done:
                    break
                total_reward += rewards

            episode_selected_tasks_action_1.sort(key=lambda x: (x[4],-x[6] ))
            episode_selected_tasks_action_0.sort(key=lambda x: (x[4],-x[6]))
            episode_rewards.append(total_reward)
            eps_history.append(agent.epsilon)
            agent.decrement_epsilon()
            episode_actions.append(action)
            episode_rewards.append(total_reward)

            with open('C:/Users/12165/Desktop/action0.txt', 'a') as file:
                print(f'Episode {episode + 1}, selected tasks for action=0:')
                file.write(f'Episode {episode + 1}, selected tasks for action=0:' + '\n')

            with open('C:/Users/12165/Desktop/action0.txt', 'a') as file:
                for selected_task in episode_selected_tasks_action_0:
                    print(selected_task)
                    file.write(str(selected_task) + '\n')

            with open('C:/Users/12165/Desktop/action1.txt', 'a') as file:
                print(f'Episode {episode + 1}, selected tasks for action=1:')
                file.write(f'Episode {episode + 1}, selected tasks for action=1:' + '\n')

            with open('C:/Users/12165/Desktop/action1.txt', 'a') as file:
                for selected_task in episode_selected_tasks_action_1:
                    print(selected_task)
                    file.write(str(selected_task) + '\n')

           # print('EP:{} reward:{} action:{} '.format(episode + 1, total_reward, action))
            if (episode + 1) % 50 == 0:
                agent.save_models(episode + 1)

        reward_plots.append(episode_rewards)
        action_plots.append(episode_actions)

    fig, ax = plt.subplots()
    for i, episode_actions in enumerate(action_plots):
        steps = np.arange(len(episode_actions))
        ax.plot(steps, episode_actions, marker='o')

    plt.xlabel("Task (Step)")
    plt.ylabel("Action")
    plt.title("Action Taken at Each Task Across Episodes")
    plt.legend()
    plt.show()

    episodes = [i for i in range(args.max_episodes)]
    plot_learning_curve(episodes, eps_history, 'Epsilon', 'epsilon', args.epsilon_path)
if __name__ == '__main__':
    main()


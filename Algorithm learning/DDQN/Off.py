import copy
import csv
import math
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

from ss import generate_taskset, insert_task
from DDQN import DDQN
from utils import plot_learning_curve, create_directory

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=1000)
parser.add_argument('--ckpt_dir', type=str, default='C:/Users/12165/Desktop/demo/')
parser.add_argument('--reward_path', type=str, default=r'C:\Users\12165\Desktop\demo\avg_reward.png')
parser.add_argument('--epsilon_path', type=str, default=r'C:\Users\12165\Desktop\demo\epsilon.png')
parser.add_argument('--energy_path', type=str, default=r'C:\Users\12165\Desktop\demo\local_energy.png')
args = parser.parse_args()


def write_results_to_csv(filename, data):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Total Reward", "Epsilon", "Local Energy"])
        for episode, (reward, epsilon, energy) in enumerate(data):
            writer.writerow([episode + 1, reward, epsilon, energy])


def main():
    alpha_values = [0.01]
    reward_plots = []
    total_local_energies = []
    for alpha in alpha_values:
        agent = DDQN(alpha=0.001, state_dim=5, action_dim=2,
                     fc1_dim=256, fc2_dim=256, ckpt_dir=args.ckpt_dir, gamma=0.99, tau=0.005, epsilon=1.0,
                     eps_end=0.01, eps_dec=5e-6, max_size=10000, batch_size=256)
        create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
        total_rewards, avg_rewards, eps_history = [], [], []
        episode_rewards = []
        local_energies = []

        for episode in range(args.max_episodes):
            env = generate_taskset(1000)
            initial_state_index = np.random.randint(len(env))
            state = env[initial_state_index][1:6]
            total_reward = 0
            max_steps = 500
            total_local_energy = 0
            Tw_dict = {'C0': 0, 'C1': 0, 'C2': 0}
            total_off_energy = 0
            for step in range(max_steps):
                task = env[step]
                state = env[step][1:6]
                B, Q, D, AT, Pi, C, C_l, p = task[1], task[2], task[3], task[4], task[6], task[7], task[9], task[10]
                T_w = Tw_dict[f'C{C}']
                SC = (0.5 * (Q / 0.37 + T_w) + 0.5 * (10 ** -27) * (370000000 ** 2) * Q * 1000000000)
                T_t = (B / 0.43)
                E_t = (0.57 * B / 0.43)

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
                Pi_new = Pi * np.exp(np.log(2) * (T_w / D))
                task[6] = Pi_new
                insert_task(task, Pi_new)

                rewards_for_action_1 = 0
                rewards_for_action_0 = 0

                E_t = (p * B / math.log2(1+20*p))
                total_local_energy += E_t

                rewards = rewards_for_action_1 if rewards_for_action_1 > rewards_for_action_0 else rewards_for_action_0
                next_state = env[step + 1][1:6] if step < max_steps - 1 else None
                state = next_state
                done = (step == max_steps - 1)

                if done:
                    break
                total_reward += rewards

            total_rewards.append(total_reward)
            eps_history.append(agent.epsilon)
            agent.decrement_epsilon()

            episode_rewards.append(total_reward)
            local_energies.append(total_off_energy)
            print('EP:{} Total Local Energy:{} '.format(episode + 1, total_local_energy))

            local_energies.append(total_local_energy)
        if (episode + 1) % 50 == 0:
            agent.save_models(episode + 1)
        reward_plots.append(episode_rewards)

    plt.figure()
    plt.plot(total_local_energies)
    plt.xlabel("Episode")
    plt.ylabel("Total Local Energy Consumption")
    plt.title("Local Energy Consumption Over Episodes")
    plt.savefig(args.energy_path)
    plt.show()

    episodes = [i for i in range(args.max_episodes)]
    plot_learning_curve(episodes, eps_history, 'Epsilon', 'epsilon', args.epsilon_path)
    write_results_to_csv(r'C:\Users\12165\Desktop\Fulloff.csv', zip(total_rewards, eps_history, local_energies))


if __name__ == '__main__':
    main()
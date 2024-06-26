import numpy as np
import math
import csv
import random
def determine_task_category(D):
    if D <= 2.0:
        return 0  # C0类别
    elif D <= 2.5:
        return 1  # C1类别
    else:
        return 2  # C2类别
sorted_tasks = []
def insert_task(task, Pi_new):
    idx = next((i for i, t in enumerate(sorted_tasks) if t[6] > Pi_new), len(sorted_tasks))
    sorted_tasks.insert(idx, task)
def calculate_all_task_Q(all_task_Q):
    sum_all_task_Q = 0
    for Q in all_task_Q:
        sum_all_task_Q += math.sqrt(0.5 * Q)
    return sum_all_task_Q
def generate_taskset(num_tasks):
    tasks = []
    Q_values = []
    for i in range(num_tasks):
        taskID = i + 1  # 任务序号
        B = np.random.uniform(0.01, 0.1)  # 任务输入数据大小
        Q = np.random.uniform(0.3, 0.9)  # 任务所需CPU计算资源
        Q_values.append(Q)  # Add Q value to the list for later calculation
        D = np.random.uniform(1.5, 4)  # 任务截止期
        AT = np.random.uniform(0, 10)  # 任务到达时间（按照泊松过程生成）
        M = np.random.randint(1, 10)  # 任务所属边缘服务器
        Pi = B / D  # 任务初始优先级
        C = determine_task_category(D)  # 确定任务类别
        tasks.append([taskID, B, Q, D, AT, M, Pi, C])
    for task in tasks:
        Q_i = task[2]  # Get the Q value of the task
        D_i = task[3]  # Get the D value of the task
        numerator = 550 * math.sqrt(0.5 * Q_i)
        denominator = calculate_all_task_Q(Q_values)
        C_e = numerator / denominator
        if C_e < 0.5 * Q_i / D_i:
            C_e = 0.5 * Q_i / D_i
        task.append(C_e)  # Add C_e value to the task
    tasks.sort(key=lambda x: (x[4], -x[6]))
    return tasks
tasks = generate_taskset(1000)
csv_file_path = 'C:/Users/12165/Desktop/newtaskset.csv'
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['TaskID', 'B', 'Q', 'D', 'AT', 'M', 'Pi', 'C', 'C_e'])
    writer.writerows(tasks)
def objective_function1(Q, Cl):
    return 0.5 * Q / Cl + 5 * (Cl ** 2) * Q
def initialize_butterflies(n, Q, D):
    butterflies = []
    for _ in range(n):
        Cl = random.uniform(0.5 * Q / D, 2)
        butterflies.append(Cl)
    return butterflies
def butterfly_algorithm(Q_values, D_values, n, iterations=1000):
    butterflies = initialize_butterflies(n, Q_values[0], D_values[0])  # Initialize butterflies with the first task's Q and D values
    optimal_solutions = []
    for _ in range(iterations):
        prev_best = min(butterflies, key=lambda x: objective_function1(Q_values[0], x))
        for i in range(len(butterflies)):
            r = random.uniform(0, 1)
            if r < 0.8:
                g_prime = min(butterflies, key=lambda x: objective_function1(Q_values[0], x))
                butterflies[i] += (g_prime - butterflies[i]) * 0.5 * 0.1
            else:
                x = random.randint(0, len(butterflies) - 1)
                y = random.randint(0, len(butterflies) - 1)
                butterflies[i] += (butterflies[x] - butterflies[y]) * 0.5 * 0.1
        current_best = min(butterflies, key=lambda x: objective_function1(Q_values[0], x))
        if abs(objective_function1(Q_values[0], prev_best) - objective_function1(Q_values[0], current_best)) < 0.1:
            break
    for i, C_l in enumerate(butterflies):
        optimal_solutions.append(C_l)
    return optimal_solutions
Q_values = [task[2] for task in tasks]
D_values = [task[3] for task in tasks]
C_l_values = butterfly_algorithm(Q_values, D_values, len(tasks))
for task, C_l in zip(tasks, C_l_values):
    task.append(C_l)
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['TaskID', 'B', 'Q', 'D', 'AT', 'M', 'Pi', 'C', 'C_e', 'C_l'])
    writer.writerows(tasks)
def objective_function(p, B, Q, C_l, C_e):
    return 0.5 * B / math.log2(1 + 20 * p) + Q / C_l + 0.5 * B * p / math.log2(1 + 20 * p)
def golden_section_search(B, Q, D, C_l, C_e):
    x_init = 2 ** ((0.5 * B / D - B * C_e / Q) - 1) / 20
    y_init = 1
    x = x_init
    y = y_init
    pl = y - 0.618 * (y - x)
    pr = y + 0.618 * (y - x)
    while abs(pl - pr) > 0.61:
        SC_pl = objective_function(pl, B, Q, C_l, C_e)
        SC_pr = objective_function(pr, B, Q, C_l, C_e)
        if SC_pl > SC_pr:
            x = pl
            pl = pr
            SC_pl = SC_pr
            pr = x + 0.618 * (y - x)
        else:
            x = pr
            pr = pl
            SC_pr = SC_pl
            pl = x - 0.618 * (y - x)
    return (pl + pr) / 2
B_value = [task[1] for task in tasks]
Q_value = [task[2] for task in tasks]
D_value = [task[3] for task in tasks]
C_l_value = [task[9] for task in tasks]
C_e_value = [task[8] for task in tasks]
p_value = []
p_i = 0
for _ in range(len(B_value)):
    p = golden_section_search(B_value[p_i], Q_value[p_i], D_value[p_i], C_l_value[p_i], C_e_value[p_i])
    p_i += 1
    print(p)
    p_value.append(p)
for task, p in zip(tasks, p_value):
    task.append(p)
with open(csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['TaskID', 'B', 'Q', 'D', 'AT', 'M', 'Pi', 'C', 'C_e', 'C_l', 'P'])
    writer.writerows(tasks)


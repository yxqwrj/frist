import csv
import random
import math

# 定义目标函数
def objective_function(Q, C_l):
    return 0.5 * Q / C_l + 5 * (C_l ** 2) * Q

# 读取任务数据
def read_task_data(file_path):
    tasks = []
    with open(file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # 跳过标题行
        for row in csv_reader:
            tasks.append({
                'Q': float(row[2]),  # 任务所需CPU计算资源
                'D': float(row[3])   # 任务截止期
            })
    return tasks

# 初始化蝴蝶种群
def initialize_butterflies(n, Q_values, D_values):
    butterflies = []
    for _ in range(n):
        C_l = random.uniform(0.5 * Q_values[0] / D_values[0], 3)  # 满足0.5*Q/D<=C_l<=3条件
        butterflies.append(C_l)
    return butterflies

# 蝴蝶算法求解最优解
def butterfly_algorithm(tasks, n, iterations=100):
    Q_values = [task['Q'] for task in tasks]
    D_values = [task['D'] for task in tasks]
    butterflies = initialize_butterflies(n, Q_values, D_values)
    optimal_solutions = []
    for task in tasks:
        best_solution = None
        for _ in range(iterations):
            prev_best = min(butterflies, key=lambda x: objective_function(task['Q'], x))
            for i in range(len(butterflies)):
                r = random.uniform(0, 1)
                if r < 0.6:
                    g_prime = min(butterflies, key=lambda x: objective_function(task['Q'], x))
                    butterflies[i] += (g_prime - butterflies[i]) * 0.5 * 0.1
                else:
                    x = random.randint(0, len(butterflies) - 1)
                    y = random.randint(0, len(butterflies) - 1)
                    butterflies[i] += (butterflies[x] - butterflies[y]) * 0.5 * 0.1
            current_best = min(butterflies, key=lambda x: objective_function(task['Q'], x))
            if abs(objective_function(task['Q'], prev_best) - objective_function(task['Q'], current_best)) < 0.1:
                best_solution = current_best
                break
        optimal_solutions.append(best_solution)
    return optimal_solutions

# 从CSV文件中读取任务数据
file_path = 'C:/Users/12165/Desktop/1.csv'
tasks = read_task_data(file_path)

# 使用蝴蝶算法求解最优解
n = len(tasks)
optimal_solutions = butterfly_algorithm(tasks, n)

# 输出每个任务的最优解
for solution in optimal_solutions:
    print(solution)

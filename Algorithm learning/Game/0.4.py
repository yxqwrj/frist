import math
import re
import numpy as np
import copy
import sys
import csv


beta = 0.4
beta1 = 0.6
C_edge_max = 40  # 单个边缘服务器资源大小
C_c = 60
#R = 0.43
edge_num = 10  # 边缘服务器数量
episode = 1000  # 轮数
edge_resource = C_edge_max * edge_num
#start_episode = 1  # 初始化第一轮循环
#end_episode = 2  # 初始化第一轮循环结束
file_path = "C:/Users/12165/Desktop/game/action0.txt"
a_e = 1
a_ee = 1
a_c = 1
Episode = 1000
g_max = 10

# 文件读取代码
class Task:
    def __init__(self, num, task_id, size, resources, deadline, arrival_time, edge_server, priority, priority_queue, edgegiven_resources, terminalgiven_resources, task_power):
        self.num = num
        self.task_id = task_id          # 任务序号(id)
        self.size = size                # 任务大小(B)
        self.resources = resources      # 所需资源(Q)
        self.deadline = deadline        # 截止期(D)
        self.arrival_time = arrival_time  # 到达时间(AT)
        self.edge_server = edge_server  # 所属边缘服务器(M)
        self.priority = priority        # 优先级(Pi)
        self.priority_queue = priority_queue  # 优先级队列
        self.edgegiven_resources = edgegiven_resources  # 边缘给任务的资源
        self.terminalgiven_resources = terminalgiven_resources  # 终端给任务的资源
        self.task_power = task_power    # 任务的传输功率

# 文件读取代码
def read_tasks_between_episodes(file_path, start_episode, end_episode):
    tasks = {}
    record = False
    i = 0

    start_pattern = f"Episode {start_episode}, selected tasks for action=0:"
    end_pattern = f"Episode {end_episode}, selected tasks for action=0:"

    with open(file_path, 'r') as file:
        for line in file:
            # Check for the start marker
            if re.match(start_pattern, line):
                record = True
                continue

            # Check for the end marker
            if re.match(end_pattern, line):
                break

            # Record lines between start and end markers
            if record:
                line = line.replace('inf', 'float("inf")')
                # Assuming each task is a list formatted as a string
                try:
                    task_list = eval(line.strip())
                    i += 1
                    #print(task_list)
                    num, task_id, size, resources, deadline, arrival_time, edge_server, priority, priority_queue, edgegiven_resources, terminalgiven_resources, task_power = i, task_list[0], task_list[1], task_list[2], task_list[3], task_list[4], task_list[5], task_list[6], task_list[7], task_list[8], task_list[9], task_list[10]
                    task = Task(int(num), int(task_id), float(size), float(resources), float(deadline), float(arrival_time), int(edge_server), float(priority), int(priority_queue), float(edgegiven_resources), float(terminalgiven_resources), float(task_power))
                    tasks[int(num)] = task

                except SyntaxError:
                    # Handle potential errors in parsing the line
                    continue
    return tasks

"""
# 计算任务的计算资源的分母函数
def calculate_all_task_Q(all_task_Q):
    sum_all_task_Q = 0
    i = 1
    for Q_i in range(len(all_task_Q)):
        sum_all_task_Q += math.sqrt(beta * all_task_Q[i].resources)
        i += 1

    return sum_all_task_Q

# 计算一个任务的计算资源函数
def calculate_task_resource(one_task_Q, sum_all_task_Q):

    C_edge_task_i = math.sqrt(beta * one_task_Q) * edge_resource / sum_all_task_Q

    return C_edge_task_i
"""

# 计算任务主边缘时延
def calculate_host_edge_delay(one_task_Q, wait_time_i, C_edge_task_i):

    task_delay_time = one_task_Q.size / math.log(1 + 20 * one_task_Q.task_power, 2) + one_task_Q.resources / C_edge_task_i + wait_time_i

    return task_delay_time

# 计算任务主边缘能耗
def calculate_host_edge_energy(one_task_Q):

    task_energy = a_e * one_task_Q.task_power * one_task_Q.size

    return task_energy

# 计算任务邻边边缘时延
def calculate_near_edge_delay(one_task_Q, wait_time_i, C_edge_task_i):

    task_near_delay_time = one_task_Q.size / math.log(1 + 20 * one_task_Q.task_power, 2) + one_task_Q.size / 5 + one_task_Q.resources / C_edge_task_i + wait_time_i

    return task_near_delay_time

# 计算任务邻边边缘能耗
def calculate_near_edge_energy(one_task_Q):

    task_near_energy = a_ee * one_task_Q.task_power * one_task_Q.size / math.log(1 + 20 * one_task_Q.task_power, 2)

    return task_near_energy


# 计算任务云时延
def calculate_cloud_delay(one_task_Q):

    task_cloud_time = one_task_Q.size / math.log(1 + 20 * one_task_Q.task_power, 2) + one_task_Q.size / 0.001 + one_task_Q.resources / C_c

    return task_cloud_time

# 计算任务云能耗
def calculate_cloud_energy(one_task_Q):

    task_cloud_energy = a_c * one_task_Q.task_power * one_task_Q.resources / math.log(1 + 20 * one_task_Q.task_power, 2)

    return task_cloud_energy

# 计算主边缘成本
def calculate_host_edge_cost(delay, energy):

    host_edge_cost = beta1 * delay + beta1 * energy

    return host_edge_cost

# 计算邻边缘成本
def calculate_near_edge_cost(delay, energy):

    near_edge_cost =  beta1 * delay + beta1 * energy

    return near_edge_cost

# 计算云成本
def calculate_cloud_cost(delay, energy):

    cloud_cost = beta1 * delay + beta1 * energy

    return cloud_cost

# 查找边缘服务器资源利用率最低的一个
def check_edge_max_resource(edge_num_resource):

    max_value = max(edge_num_resource[1:])
    max_index = edge_num_resource.index(max_value, 1)

    return max_index

# 确定任务的初始队列
def determine_task_queue(D):

    if D <= 2.0:
        return 0  # C0类别
    elif D <= 2.5:
        return 1  # C1类别
    else:
        return 2  # C2类别


"""
if Task_i.resources < edge_resource_list[i]:
    edge_element = Task_i.edge_server
else:
    edge_element = check_edge_max_resource(edge_resource_list)
"""
# 根据任务优先级在队列排序
def sort_task(edge_service_task, Task, edge_element, all_task_wait_time):

    dynamic_priority = {}

    for i in edge_service_task[edge_element]:

        task_i = i

        dynamic_priority_value = Task[task_i].priority

        #for key, value in all_task_wait_time.items():
            #print(f"Key: {key}, Value: {value}")
        #print(task_i)
        #print("task_i",task_i)
        dynamic_priority_value = dynamic_priority_value * np.log2(all_task_wait_time[task_i])

        dynamic_priority[task_i] = dynamic_priority_value

    sorted_keys = sorted(dynamic_priority, key=lambda x: dynamic_priority[x], reverse=True)

    edge_service_task[edge_element] = sorted_keys.copy()
    #print("edge_service_task[edge_element]", edge_service_task[edge_element])
    return edge_service_task[edge_element]

# 计算任务的等待时间计算,其中形参：任务集合，任务排序位置，执行边缘服务器。
# 增加任务优先级
def wait_time(edge_service_task, Task, Task_i, edge_element):

    all_task_wait_time = {}

    edge_service_queue_delay = []

    for i in range(11):
        edge_service_queue_delay.append([0, 0, 0])

    temporary_edge_service_queue = edge_service_queue_delay  # 10个边缘服务器的三个时延队列临时记录列表

    Task_i_queue = determine_task_queue(Task[Task_i].deadline)

    edge_service_task[edge_element].append(Task_i)


    if len(edge_service_task[edge_element]) == 1:

        return 0

    else:

        sort_edge_service_task = edge_service_task.copy()

        wait_time = Task[edge_service_task[edge_element][0]].resources / Task[edge_service_task[edge_element][0]].terminalgiven_resources

        for i in sort_edge_service_task[edge_element]:

            all_task_wait_time[i] = wait_time


        while(len(sort_edge_service_task[edge_element]) != 1):

            pre_task_i = edge_service_task[edge_element][0]

            AT_prev = Task[pre_task_i].arrival_time

            sort_edge_service_task[edge_element].remove(sort_edge_service_task[edge_element][0])

            #print("sort_edge_service_task[edge_element]",sort_edge_service_task[edge_element])
            #for key, value in all_task_wait_time.items():
                #print(f"Key: {key}, Value: {value}")

            sort_edge_service_task[edge_element] = sort_task(sort_edge_service_task, Task, edge_element,
                                                             all_task_wait_time)

            Task_exeist_i = sort_edge_service_task[edge_element][0]

            Task_i_initiation = determine_task_queue(Task[Task_exeist_i].deadline)

            Task_i_edge_service = edge_element

            pre_wait_time = temporary_edge_service_queue[Task_i_edge_service][Task_i_initiation]  #

            if Task_i_initiation == 0:

                temporary_edge_service_queue[Task_i_edge_service][Task_i_initiation] = 0.5 * pre_wait_time + max(0, (
                            Task[pre_task_i].resources / Task[pre_task_i].terminalgiven_resources + pre_wait_time) - (Task[Task_i].arrival_time - AT_prev))
                #print("Task_i_initiation == 0", temporary_edge_service_queue[Task_i_edge_service][Task_i_initiation])


            elif Task_i_initiation == 1:

                temporary_edge_service_queue[Task_i_edge_service][Task_i_initiation] = \
                temporary_edge_service_queue[Task_i_edge_service][Task_i_initiation - 1] + max(0, (
                            Task[pre_task_i].resources / Task[pre_task_i].terminalgiven_resources + pre_wait_time) - (Task[Task_i].arrival_time - AT_prev))


            elif Task_i_initiation == 2:

                temporary_edge_service_queue[Task_i_edge_service][Task_i_initiation] = \
                temporary_edge_service_queue[Task_i_edge_service][Task_i_initiation - 2] + \
                temporary_edge_service_queue[Task_i_edge_service][Task_i_initiation - 1] + max(0, (
                            Task[pre_task_i].resources / Task[pre_task_i].terminalgiven_resources + pre_wait_time) - (Task[Task_i].arrival_time - AT_prev))
                #print("Task_i_initiation == 2",temporary_edge_service_queue[Task_i_edge_service][Task_i_initiation])

            all_task_wait_time = {}

            for i in sort_edge_service_task[edge_element]:

                all_task_wait_time[i] = temporary_edge_service_queue[Task_i_edge_service][Task_i_initiation]

            wait_time = temporary_edge_service_queue[Task_i_edge_service][Task_i_initiation]

            #print(wait_time)

            if wait_time < 0:
                sys.exit()

    #print("wait time", temporary_edge_service_queue[edge_element][Task_i_queue])
    return wait_time

# 记录SC_total
def SC_total(edge_service_task, edge_resource_list, record_task_near_edge, Task, record_a):

    #edge_service_task_backup = copy.deepcopy(edge_service_task)

    SC_total_value = 0
    record_i = 1
    for i in range(len(record_a) - 1):
        edge_service_task_backup = copy.deepcopy(edge_service_task)
        #edge_service_queue_delay_backup = copy.deepcopy(edge_service_queue_delay)

        record_task_num_i_resource = Task[record_i].edgegiven_resources
        record_ae = record_a[record_i][0]
        record_aee = record_a[record_i][1]
        record_ac = record_a[record_i][2]

        if record_ae == 1:
            T_e_value = calculate_host_edge_delay(one_task_Q=Task[record_i],
                                                  wait_time_i=wait_time(edge_service_task, Task, record_i,
                                                                        Task[record_i].edge_server),
                                                  C_edge_task_i=record_task_num_i_resource)

            edge_service_task = copy.deepcopy(edge_service_task_backup)

            E_e_value = calculate_host_edge_energy(one_task_Q=Task[record_i])
            SC_total_value += calculate_host_edge_cost(delay=T_e_value, energy=E_e_value)
            edge_service_task[Task[record_i].edge_server].append(Task[record_i].num)  # 在边缘服务器任务列表进行记录
            edge_resource_list[Task[record_i].edge_server] -= Task[record_i].resources  # 边缘服务器资源记录

        elif record_aee == 1:
            T_ee_value = calculate_near_edge_delay(one_task_Q=Task[record_i],
                                                   wait_time_i=wait_time(edge_service_task, Task, record_i,
                                                                         check_edge_max_resource(edge_resource_list)),
                                                   C_edge_task_i=record_task_num_i_resource)

            edge_service_task = copy.deepcopy(edge_service_task_backup)

            E_ee_value = calculate_near_edge_energy(one_task_Q=Task[record_i])
            SC_total_value += calculate_near_edge_cost(delay=T_ee_value, energy=E_ee_value)

            edge_element = check_edge_max_resource(edge_resource_list)  # 邻边缘服务器编号
            record_task_near_edge[record_i] = edge_element
            edge_service_task[edge_element].append(Task[record_i].num)  # 在边缘服务器任务列表进行记录

            edge_resource_list[edge_element] -= Task[record_i].resources  # 边缘服务器资源记录

        elif record_ac == 1:
            T_c_value = calculate_cloud_delay(one_task_Q=Task[record_i])
            E_c_value = calculate_cloud_energy(one_task_Q=Task[record_i])
            SC_total_value += calculate_cloud_cost(delay=T_c_value, energy=E_c_value)

        record_i += 1

    return SC_total_value, edge_service_task, record_task_near_edge, edge_resource_list

# 改变一条记录SC_total变化函数
def SC_total_chage(edge_service_task, edge_resource_list, record_task_near_edge, Task, SC_total_value, record_a, record_i, new_strategy):
    edge_service_task_backup = copy.deepcopy(edge_service_task)

    logo_SC_total_value = copy.deepcopy(SC_total_value)  # 标记
    logo_edge_service_task = copy.deepcopy(edge_service_task)
    logo_record_task_near_edge = copy.deepcopy(record_task_near_edge)
    logo_edge_resource_list = copy.deepcopy(edge_resource_list)

    record_ae = record_a[record_i][0]
    record_aee = record_a[record_i][1]
    record_ac = record_a[record_i][2]



    record_task_num_i_resource = Task[record_i].edgegiven_resources
    if record_ae == 1:
        T_e_value = calculate_host_edge_delay(one_task_Q=Task[record_i],
                                              wait_time_i=wait_time(edge_service_task, Task, record_i,
                                                                    Task[record_i].edge_server),
                                              C_edge_task_i=record_task_num_i_resource)

        edge_service_task = copy.deepcopy(edge_service_task_backup)

        E_e_value = calculate_host_edge_energy(one_task_Q=Task[record_i])
        SC_total_value -= calculate_host_edge_cost(delay=T_e_value, energy=E_e_value)

        edge_service_task[Task[record_i].edge_server].remove(Task[record_i].num)  # 在边缘服务器任务列表进行记录
        edge_resource_list[Task[record_i].edge_server] += Task[record_i].resources  # 边缘服务器资源记录

    elif record_aee == 1:
        T_ee_value = calculate_near_edge_delay(one_task_Q=Task[record_i],
                                               wait_time_i=wait_time(edge_service_task, Task, record_i,
                                                                     check_edge_max_resource(edge_resource_list)),
                                               C_edge_task_i=record_task_num_i_resource)
        edge_service_task = copy.deepcopy(edge_service_task_backup)


        E_ee_value = calculate_near_edge_energy(one_task_Q=Task[record_i])
        SC_total_value -= calculate_near_edge_cost(delay=T_ee_value, energy=E_ee_value)

        #print("record_task_near_edge[158]",record_task_near_edge[158])
        edge_element = record_task_near_edge[record_i]

        #print("edge_element", edge_element,"edge_service_task[edge_element]", edge_service_task[edge_element], "Task[record_i].num", Task[record_i].num)
        edge_service_task[edge_element].remove(Task[record_i].num)  # 在边缘服务器任务列表进行记录
        edge_resource_list[edge_element] += Task[record_i].resources  # 边缘服务器资源记录

    elif record_ac == 1:
        T_c_value = calculate_cloud_delay(one_task_Q=Task[record_i])
        E_c_value = calculate_cloud_energy(one_task_Q=Task[record_i])
        SC_total_value -= calculate_cloud_cost(delay=T_c_value, energy=E_c_value)

    record_a[record_i] = new_strategy
    record_ae = record_a[record_i][0]
    record_aee = record_a[record_i][1]
    record_ac = record_a[record_i][2]
    edge_service_task_backup = copy.deepcopy(edge_service_task)

    if record_ae == 1:

        if Task[record_i].resources <= edge_resource_list[Task[record_i].edge_server]:
            T_e_value = calculate_host_edge_delay(one_task_Q=Task[record_i],
                                                wait_time_i=wait_time(edge_service_task, Task, record_i,
                                                                        Task[record_i].edge_server),
                                                C_edge_task_i=record_task_num_i_resource)
            edge_service_task = copy.deepcopy(edge_service_task_backup)
            E_e_value = calculate_host_edge_energy(one_task_Q=Task[record_i])
            SC_total_value += calculate_host_edge_cost(delay=T_e_value, energy=E_e_value)
            edge_service_task[Task[record_i].edge_server].append(Task[record_i].num)  # 在边缘服务器任务列表进行记录
            edge_resource_list[Task[record_i].edge_server] -= Task[record_i].resources  # 边缘服务器资源记录


            return SC_total_value, edge_service_task, record_task_near_edge, edge_resource_list

        else:
            SC_total_value = copy.deepcopy(logo_SC_total_value)  # 标记
            edge_service_task = copy.deepcopy(logo_edge_service_task)
            record_task_near_edge = copy.deepcopy(logo_record_task_near_edge)
            edge_resource_list = copy.deepcopy(logo_edge_resource_list)


            return 0, edge_service_task, record_task_near_edge, edge_resource_list

    if record_aee == 1:
        if Task[record_i].resources <= edge_resource_list[record_task_near_edge[record_i]]:
            T_ee_value = calculate_near_edge_delay(one_task_Q=Task[record_i],
                                                wait_time_i=wait_time(edge_service_task, Task, record_i,
                                                                        check_edge_max_resource(edge_resource_list)),
                                                C_edge_task_i=record_task_num_i_resource)
            edge_service_task = copy.deepcopy(edge_service_task_backup)

            E_ee_value = calculate_near_edge_energy(one_task_Q=Task[record_i])
            SC_total_value += calculate_near_edge_cost(delay=T_ee_value, energy=E_ee_value)

            edge_element = check_edge_max_resource(edge_resource_list)  # 邻边缘服务器编号
            record_task_near_edge[record_i] = edge_element
            edge_service_task[edge_element].append(Task[record_i].num)  # 在边缘服务器任务列表进行记录
            edge_resource_list[edge_element] -= Task[record_i].resources  # 边缘服务器资源记录
            return SC_total_value, edge_service_task, record_task_near_edge, edge_resource_list

        else:
            SC_total_value = copy.deepcopy(logo_SC_total_value)  # 标记
            edge_service_task = copy.deepcopy(logo_edge_service_task)
            record_task_near_edge = copy.deepcopy(logo_record_task_near_edge)
            edge_resource_list = copy.deepcopy(logo_edge_resource_list)
            return 0, edge_service_task, record_task_near_edge, edge_resource_list

    if record_ac == 1:
        T_c_value = calculate_cloud_delay(one_task_Q=Task[record_i])
        E_c_value = calculate_cloud_energy(one_task_Q=Task[record_i])
        SC_total_value += calculate_cloud_cost(delay=T_c_value, energy=E_c_value)
        return SC_total_value, edge_service_task, record_task_near_edge, edge_resource_list



# 计算边缘总成本、计算总时延以及总能耗
def calculate_Edge_cost_delay_energy(Task, strategy):

    #all_task_Q = calculate_all_task_Q(Task)

    edge_service_task = []
    for i in range(11):
        edge_service_task.append([])

    # edge_service_queue_delay = []
    # for i in range(11):
    # edge_service_queue_delay.append([0, 0, 0])

    edge_resource_list = []
    for i in range(11):
        edge_resource_list.append(C_edge_max)

    record_task_near_edge = []
    for i in range(len(Task) + 1):
        record_task_near_edge.append(0)

    SC_total_value = 0
    SC_total_delay = 0
    SC_total_energy = 0
    record_i = 1
    for i in range(len(strategy) - 1):
        edge_service_task_backup = copy.deepcopy(edge_service_task)
        #edge_service_queue_delay_backup = copy.deepcopy(edge_service_queue_delay)

        record_task_num_i_resource = Task[record_i].edgegiven_resources
        record_ae = strategy[record_i][0]
        record_aee = strategy[record_i][1]
        record_ac = strategy[record_i][2]

        if record_ae == 1:
            T_e_value = calculate_host_edge_delay(one_task_Q=Task[record_i],
                                                  wait_time_i=wait_time(edge_service_task, Task, record_i,
                                                                        Task[record_i].edge_server),
                                                  C_edge_task_i=record_task_num_i_resource)

            edge_service_task = copy.deepcopy(edge_service_task_backup)

            E_e_value = calculate_host_edge_energy(one_task_Q=Task[record_i])

            SC_total_delay += T_e_value
            SC_total_energy += E_e_value
            SC_total_value += calculate_host_edge_cost(delay=T_e_value, energy=E_e_value)

            edge_service_task[Task[record_i].edge_server].append(Task[record_i].num)  # 在边缘服务器任务列表进行记录
            edge_resource_list[Task[record_i].edge_server] -= Task[record_i].resources  # 边缘服务器资源记录

        elif record_aee == 1:
            T_ee_value = calculate_near_edge_delay(one_task_Q=Task[record_i],
                                                   wait_time_i=wait_time(edge_service_task, Task, record_i,
                                                                         check_edge_max_resource(edge_resource_list)),
                                                   C_edge_task_i=record_task_num_i_resource)

            edge_service_task = copy.deepcopy(edge_service_task_backup)

            E_ee_value = calculate_near_edge_energy(one_task_Q=Task[record_i])

            SC_total_delay += T_ee_value
            SC_total_energy += E_ee_value
            SC_total_value += calculate_near_edge_cost(delay=T_ee_value, energy=E_ee_value)

            edge_element = check_edge_max_resource(edge_resource_list)  # 邻边缘服务器编号
            record_task_near_edge[record_i] = edge_element
            edge_service_task[edge_element].append(Task[record_i].num)  # 在边缘服务器任务列表进行记录

            edge_resource_list[edge_element] -= Task[record_i].resources  # 边缘服务器资源记录

        elif record_ac == 1:
            T_c_value = calculate_cloud_delay(one_task_Q=Task[record_i])
            E_c_value = calculate_cloud_energy(one_task_Q=Task[record_i])

            SC_total_delay += T_c_value
            SC_total_energy += E_c_value
            SC_total_value += calculate_cloud_cost(delay=T_c_value, energy=E_c_value)

        record_i += 1

    return SC_total_value, SC_total_delay, SC_total_energy

# 计算负载均衡
def calculate_load_balance(Task, strategy, record_task_near_edge):
    C_e_edge = C_edge_max
    avg_edge_load_balance = 0
    C_e = []  # 记录10个边缘服务器的资源使用情况
    for i in range(11):
        C_e.append(0)
    every_edge_load_balance = []  # 记录10个边缘服务器的负载
    for i in range(11):
        every_edge_load_balance.append(0)

    strategy_i = 1

    #print("len(strategy)", len(strategy))

    for i in range(len(strategy)-1):

        #print('strategy_i',strategy_i)
        ae = strategy[strategy_i][0]
        aee = strategy[strategy_i][1]
        # ac = strategy[strategy_i][2]

        if ae == 1:
            C_e[Task[strategy_i].edge_server] += Task[strategy_i].resources

        elif aee == 1:
            C_e[record_task_near_edge[strategy_i]] += Task[strategy_i].resources

        strategy_i += 1

    load_balance = 1

    for i in range(10):
        every_edge_load_balance[load_balance] = C_e[load_balance] / C_e_edge
        avg_edge_load_balance += every_edge_load_balance[load_balance]

        load_balance += 1

    avg_edge_load_balance = avg_edge_load_balance / 10

    lb_load_balance = 1
    sum = 0  # 临时记录
    for i in range(10):
        Load_m = every_edge_load_balance[lb_load_balance]

        sum += (Load_m - avg_edge_load_balance) * (Load_m - avg_edge_load_balance)

        lb_load_balance += 1

    Lb_every_edge_load_balance = math.sqrt(sum / 10)

    return every_edge_load_balance, avg_edge_load_balance, Lb_every_edge_load_balance


# 博弈论算法
def Game_theory_algorithm(start_episode, end_episode):

    Task = read_tasks_between_episodes(file_path, start_episode, end_episode)
    edge_service_task = []
    for i in range(11):
        edge_service_task.append([])

    edge_resource_list = []
    for i in range(11):
        edge_resource_list.append(C_edge_max)
    record_task_near_edge = []
    for i in range(len(Task) + 1):
        record_task_near_edge.append(0)

    task_num = len(Task)
    record_a = []
    for i in range(len(Task) + 1):
        record_a.append([0, 0, 0])
    #all_task_Q = calculate_all_task_Q(Task)

    task_num_i = 1

    for i in range(task_num):

        edge_service_task_backup = copy.deepcopy(edge_service_task)
        #edge_service_queue_delay_backup = copy.deepcopy(edge_service_queue_delay)

        D_i = Task[task_num_i].deadline

        task_num_i_resource = Task[task_num_i].edgegiven_resources

        T_e_value = calculate_host_edge_delay(one_task_Q=Task[task_num_i], wait_time_i=wait_time(edge_service_task, Task, task_num_i, Task[task_num_i].edge_server), C_edge_task_i=task_num_i_resource)

        #edge_service_queue_delay1 = edge_service_queue_delay.copy()
        #print(T_e_value)
        #edge_service_queue_delay = copy.deepcopy(edge_service_queue_delay_backup)
        edge_service_task = copy.deepcopy(edge_service_task_backup)  # 恢复

        T_ee_value = calculate_near_edge_delay(one_task_Q=Task[task_num_i], wait_time_i=wait_time(edge_service_task, Task, task_num_i, check_edge_max_resource(edge_resource_list)), C_edge_task_i=task_num_i_resource)
        #edge_service_queue_delay2 = edge_service_queue_delay.copy()

        #edge_service_queue_delay = copy.deepcopy(edge_service_queue_delay_backup)
        edge_service_task = copy.deepcopy(edge_service_task_backup)  # 恢复

        T_c_value = calculate_cloud_delay(one_task_Q=Task[task_num_i])
        E_e_value = calculate_host_edge_energy(one_task_Q=Task[task_num_i])
        E_ee_value = calculate_near_edge_energy(one_task_Q=Task[task_num_i])
        E_c_value = calculate_cloud_energy(one_task_Q=Task[task_num_i])
        SC_e_value = calculate_host_edge_cost(delay=T_e_value, energy=E_e_value)
        SC_ee_value = calculate_near_edge_cost(delay=T_ee_value, energy=E_ee_value)
        SC_c_value = calculate_cloud_cost(delay=T_c_value, energy=E_c_value)

        if T_e_value > D_i and T_ee_value > D_i and T_c_value > D_i:

            a_e, a_ee, a_c = 0, 0, 0
            record_a[task_num_i] = [a_e, a_ee, a_c]

        elif T_e_value <= D_i and SC_e_value <= min(SC_ee_value, SC_c_value) and Task[task_num_i].resources <= edge_resource_list[Task[task_num_i].edge_server]:

            edge_service_task[Task[task_num_i].edge_server].append(Task[task_num_i].num)  # 在边缘服务器任务列表进行记录
            edge_resource_list[Task[task_num_i].edge_server] -= Task[task_num_i].resources  # 边缘服务器资源记录
            #edge_service_queue_delay = copy.deepcopy(edge_service_queue_delay1)
            a_e, a_ee, a_c = 1, 0, 0
            record_a[task_num_i] = [a_e, a_ee, a_c]

        elif T_ee_value <= D_i and SC_ee_value <= SC_c_value and Task[task_num_i].resources <= edge_resource_list[check_edge_max_resource(edge_resource_list)]:

            edge_element = check_edge_max_resource(edge_resource_list)  # 邻边缘服务器编号
            record_task_near_edge[task_num_i] = edge_element
            edge_service_task[edge_element].append(Task[task_num_i].num)  # 在边缘服务器任务列表进行记录
            edge_resource_list[edge_element] -= Task[task_num_i].resources  # 边缘服务器资源记录
            #edge_service_queue_delay = copy.deepcopy(edge_service_queue_delay2)
            a_e, a_ee, a_c = 0, 1, 0
            record_a[task_num_i] = [a_e, a_ee, a_c]

        else:

            a_e, a_ee, a_c = 0, 0, 1
            record_a[task_num_i] = [a_e, a_ee, a_c]

        task_num_i += 1


    record_a_mxa = record_a.copy()  # 初始策略
    #print(record_a_mxa)

    task_num = len(Task)
    record_a = []
    for i in range(len(Task) + 1):
        record_a.append([0, 0, 0])

    g = 1
    for g_i in range(g_max):

        task_num_i = 1

        # 重新记录
        edge_service_task = []
        for i in range(11):
            edge_service_task.append([])

        #edge_service_queue_delay = []
        #for i in range(11):
            #edge_service_queue_delay.append([0, 0, 0])

        edge_resource_list = []
        for i in range(11):
            edge_resource_list.append(C_edge_max)

        record_task_near_edge = []
        for i in range(len(Task) + 1):
            record_task_near_edge.append(0)

        SC_total_value, edge_service_task, record_task_near_edge, edge_resource_list = SC_total(edge_service_task, edge_resource_list, record_task_near_edge, Task, record_a_mxa)

        #print("我再看看", record_task_near_edge[186], "edge_service_task[10]", edge_service_task[10])

        while task_num_i <= task_num:

            edge_resource_list_backup = copy.deepcopy(edge_resource_list)
            record_task_near_edge_backup = copy.deepcopy(record_task_near_edge)
            edge_service_task_backup = copy.deepcopy(edge_service_task)
            record_a_mxa_backup = copy.deepcopy(record_a_mxa)
            #edge_service_queue_delay_backup = copy.deepcopy(edge_service_queue_delay)

            #print("new_strategy = [1, 0, 0]task_num_i", task_num_i,"主函数", record_task_near_edge[task_num_i],"edge_service_task[10]", edge_service_task[10])
            new_strategy = [1, 0, 0]
            SC_total_value1, edge_service_task1, record_task_near_edge1, edge_resource_list1 = SC_total_chage(edge_service_task, edge_resource_list, record_task_near_edge, Task, SC_total_value, record_a_mxa, task_num_i, new_strategy)

            edge_resource_list = copy.deepcopy(edge_resource_list_backup)
            record_task_near_edge = copy.deepcopy(record_task_near_edge_backup)
            edge_service_task = copy.deepcopy(edge_service_task_backup)
            record_a_mxa = copy.deepcopy(record_a_mxa_backup)
            #edge_service_queue_delay = copy.deepcopy(edge_service_queue_delay_backup)

            #print("new_strategy = [0, 1, 0]task_num_i", task_num_i, "主函数", record_task_near_edge[task_num_i],"edge_service_task[10]", edge_service_task[10])
            new_strategy = [0, 1, 0]
            SC_total_value2, edge_service_task2, record_task_near_edge2, edge_resource_list2 = SC_total_chage(edge_service_task, edge_resource_list, record_task_near_edge, Task, SC_total_value, record_a_mxa , task_num_i, new_strategy)

            edge_resource_list = copy.deepcopy(edge_resource_list_backup)
            record_task_near_edge = copy.deepcopy(record_task_near_edge_backup)
            edge_service_task = copy.deepcopy(edge_service_task_backup)
            record_a_mxa = copy.deepcopy(record_a_mxa_backup)
            #edge_service_queue_delay = copy.deepcopy(edge_service_queue_delay_backup)

            #print("new_strategy = [0, 0, 1]task_num_i", task_num_i, "主函数", record_task_near_edge[task_num_i],"edge_service_task[10]", edge_service_task[10])
            new_strategy = [0, 0, 1]
            SC_total_value3, edge_service_task3, record_task_near_edge3, edge_resource_list3 = SC_total_chage(edge_service_task, edge_resource_list, record_task_near_edge, Task, SC_total_value, record_a_mxa, task_num_i, new_strategy)

            edge_resource_list = copy.deepcopy(edge_resource_list_backup)
            record_task_near_edge = copy.deepcopy(record_task_near_edge_backup)
            edge_service_task = copy.deepcopy(edge_service_task_backup)
            record_a_mxa = copy.deepcopy(record_a_mxa_backup)
            #edge_service_queue_delay = copy.deepcopy(edge_service_queue_delay_backup)

            SC_total_value_max = min(SC_total_value1, SC_total_value2, SC_total_value3)

            if SC_total_value_max == SC_total_value1:
                record_a_mxa[task_num_i] = [1, 0, 0]
                edge_service_task = edge_service_task1
                record_task_near_edge = record_task_near_edge1
                edge_resource_list = edge_resource_list1
                #edge_service_queue_delay = edge_service_queue_delay1
                #print('g_i',g_i, 'task_num_i SC_total_value1 compete', task_num_i, 'edge_service_task1', edge_service_task)
            elif SC_total_value_max == SC_total_value2:
                record_a_mxa[task_num_i] = [0, 1, 0]
                edge_service_task = edge_service_task2
                record_task_near_edge = record_task_near_edge2
                edge_resource_list = edge_resource_list2
                #edge_service_queue_delay = edge_service_queue_delay2
                #print('g_i',g_i, 'task_num_i SC_total_value2 compete', task_num_i, 'edge_service_task2', edge_service_task)
            elif SC_total_value_max == SC_total_value3:
                record_a_mxa[task_num_i] = [0, 0, 1]
                edge_service_task = edge_service_task3
                record_task_near_edge = record_task_near_edge3
                edge_resource_list = edge_resource_list3
                #edge_service_queue_delay = edge_service_queue_delay3
                #print('g_i',g_i, 'task_num_i SC_total_value3 compete', task_num_i, 'edge_service_task3', edge_service_task)


            task_num_i += 1

            #print('g_i', g_i, 'task_num_i SC_total_value3 compete', task_num_i, 'edge_service_task3', edge_service_task)
        g += 1

    every_edge_load_balance, avg_edge_load_balance, Lb_every_edge_load_balance = calculate_load_balance(Task, record_a_mxa, record_task_near_edge)
    SC_total_value, SC_total_delay, SC_total_energy = calculate_Edge_cost_delay_energy(Task, record_a_mxa)

    return every_edge_load_balance, avg_edge_load_balance, Lb_every_edge_load_balance, record_a_mxa, SC_total_value, SC_total_delay, SC_total_energy

"""
把列表存文件中
"""
def append_list_to_csv(lst, filename, n):
    with open(filename, 'a', newline='') as file:
        # 请求文件锁
        #msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, os.path.getsize(filename))
        writer = csv.writer(file)
        for i in range(n):
            writer.writerow(lst)
        # 释放文件锁
        #msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, os.path.getsize(filename))
"""
把数值存文件中
"""
def value_to_csv(value, filename, n):
    with open(filename, mode='a', newline='') as file:
        # 请求文件锁
        #msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, os.path.getsize(filename))
        writer = csv.writer(file)
        for i in range(n):
            writer.writerow([value])
        # 释放文件锁
        #msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, os.path.getsize(filename))


if __name__ == '__main__':
    start_episode = 1
    end_episode = 2
    Episode = 1000


    for i in range(Episode):
        print("轮数", start_episode)
        every_edge_load_balance, avg_edge_load_balance, Lb_every_edge_load_balance, record_a_mxa, SC_total_value, SC_total_delay, SC_total_energy = Game_theory_algorithm(start_episode, end_episode)


        append_list_to_csv(every_edge_load_balance, 'C:/Users/12165/Desktop/game/0.4/every_edge_load_balance.csv', 1)  # 每次迭代记录10个边缘服务器的负载均衡，每行是一代，列表
        value_to_csv(avg_edge_load_balance, 'C:/Users/12165/Desktop/game/0.4/avg_edge_load_balance.csv', 1)  # 每次迭代记录10个边缘服务器的负载均衡均值，每行是一代，数值
        value_to_csv(Lb_every_edge_load_balance, 'C:/Users/12165/Desktop/game/0.4/Lb_every_edge_load_balance.csv', 1)  # 每次迭代记录各个边缘服务器的负载均衡均值，每行是一代，数值
        append_list_to_csv(record_a_mxa, 'C:/Users/12165/Desktop/game/0.4/record_a_mxa.csv', 1)  # 每次迭代记录该代策略，每行是一代，列表
        value_to_csv(SC_total_value, 'C:/Users/12165/Desktop/game/0.4/SC_total_value.csv', 1)  # 每次迭代记录总体成本，每行是一代，数值
        value_to_csv(SC_total_delay, 'C:/Users/12165/Desktop/game/0.4/SC_total_delay.csv', 1)  # 每次迭代记录总体时延，每行是一代，数值
        value_to_csv(SC_total_energy, 'C:/Users/12165/Desktop/game/0.4/SC_total_energy.csv', 1)  # 每次迭代记录总体能量，每行是一代，数值

        start_episode += 1
        end_episode += 1

        print(every_edge_load_balance, avg_edge_load_balance, Lb_every_edge_load_balance, record_a_mxa, SC_total_value, SC_total_delay, SC_total_energy)


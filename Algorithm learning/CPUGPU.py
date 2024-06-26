import torch
import time

def train(size_list, epochs):

    for s in size_list:

        # CPU
        start_time1 = time.time()
        a = torch.ones(s,s)
        for _ in range(epochs):
            a += a
        cpu_time = time.time() - start_time1

        # GPU
        start_time2 = time.time()
        b = torch.ones(s,s).cuda()
        for _ in range(epochs):
            b += b
        gpu_time = time.time() - start_time2

        print('s = %d, CPU_time = %.4fs, GPU_time = %.4fs'%(s, cpu_time, gpu_time))

size_list = [8, 32, 128, 512]
epochs = 100000
train(size_list, 100000)


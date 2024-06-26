import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib

matplotlib.use('TkAgg')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
mpl.rcParams['font.family']='SimHei'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus']=False
import matplotlib.pyplot as plt

# 数据
experiments = ['1', '2', '3', '4', '5', 'Average']
methods = ['所提方法', 'HBMG方法', 'PDIV方法']
data = {
    '所提方法': [3.34, 3.18, 2.97, 3.23, 3.09, 3.16],
    'HBMG方法': [5.98, 5.87, 5.76, 7.05, 6.99, 6.33],
    'PDIV方法': [8.79, 8.93, 9.65, 9.96, 8.97, 9.15]
}

# 绘图
plt.figure(figsize=(10, 6)) # 设置图像大小

bar_width = 0.25 # 设置柱状图的宽度
index = range(len(experiments)) # 设置柱状图的位置

#colors = ['blue', 'green', 'red'] # 不同方法的颜色

# 绘制柱状图
for i, method in enumerate(methods):
    plt.bar([x + i * bar_width for x in index], data[method], width=bar_width, label=method)

plt.xlabel('实验次数')
plt.ylabel('旅游胜地选择偏好的预测时间(s)')
plt.xticks([idx + bar_width for idx in index], experiments) # 设置 x 轴刻度
plt.legend()

# 显示图像
plt.show()

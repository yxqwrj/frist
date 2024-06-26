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
import numpy as np

# 横坐标刻度
x_values = [200, 400, 600, 800]

# 数据
data1_values = [330.76, 287.23, 236.82, 198.99]
data2_values = [410.17, 362.47, 312.90, 270.66]
data3_values = [553.90, 500.10, 449.17, 396.95]
data4_values = [614.16, 563.25, 502.91, 455.29]

data1_values = [330.76, 410.17, 553.90, 614.16]
data2_values = [287.23, 362.47, 500.10, 563.25]
data3_values = [236.82, 312.90, 449.17, 502.91]
data4_values = [198.99, 270.66, 396.95, 455.29]

# 设置柱状图的宽度
bar_width = 0.15

# 计算每个柱状图的位置
bar1_positions = np.arange(len(x_values))
bar2_positions = [pos + bar_width for pos in bar1_positions]
bar3_positions = [pos + bar_width for pos in bar2_positions]
bar4_positions = [pos + bar_width for pos in bar3_positions]

# 绘图
plt.figure(figsize=(10, 6))

plt.bar(bar1_positions, data1_values, width=bar_width, label='β=0.2',color = 'blue')
plt.bar(bar2_positions, data2_values, width=bar_width, label='β=0.4',color = 'red')
plt.bar(bar3_positions, data3_values, width=bar_width, label='β=0.6',color = 'green')
plt.bar(bar4_positions, data4_values, width=bar_width, label='β=0.8',color = 'orange')

# 添加轴标签和标题
plt.xlabel('任务数量',fontsize=15, weight='bold')
plt.ylabel('系统平均成本Cost',fontsize=15, weight='bold')
plt.xticks(bar1_positions + 1.5 * bar_width, x_values)
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()

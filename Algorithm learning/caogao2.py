import pandas as pd
import matplotlib.pyplot as plt
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

# 职业及其对应的百分比
occupations = ['0.5-1k', '1k-2k', '2k-5k', '5k+']
percentages = [20.45,32.1, 36.8, 100 - (18 + 14.7 + 15.6 + 25 + 21.9)]  # 计算其他为退休人员的百分比

# 绘制饼图
plt.figure(figsize=(8, 8))
plt.pie(percentages, labels=occupations, autopct='%1.1f%%', startangle=140)
plt.title('经费分布饼图')
plt.axis('equal')  # 使饼图比例相等
plt.show()



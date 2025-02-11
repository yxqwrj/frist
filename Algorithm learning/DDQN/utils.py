import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def plot_learning_curve(episodes, records, title, ylabel, figure_file):

    plt.figure()
    plt.plot(episodes, records, linestyle='-')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)
    plt.savefig(figure_file, format='png')
    plt.show()

def create_directory(path: str, sub_dirs: list):
    for sub_dir in sub_dirs:
        if os.path.exists(path + sub_dir):
            print(path + sub_dir + ' is already exist!')
        else:
            os.makedirs(path + sub_dir, exist_ok=True)
            print(path + sub_dir + ' create successfully!')
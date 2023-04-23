import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sbn
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from mpl_toolkits.mplot3d import Axes3D


font_path = 'C:\Windows\Fonts\STXIHEI.TTF'
# font = FontProperties(fname=r"simsun.ttc", size=18)
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()

# set some initial paths

# path to the directory where this script lives
thisdir = os.path.abspath('')

# path to the scripts directory of the repository
scriptsdir = os.path.split(thisdir)[0]

# path to the main directory of the repository
maindir = os.path.split(scriptsdir)[0]

# path to the results subdirectory
resultsdir = os.path.join(os.path.split(scriptsdir)[0], 'results')

# path to the data subdirectory
datadir = os.path.join(maindir, 'data')

# path to the output_source subsubdirectory
output_source_dir = os.path.join(datadir, 'output_source')

# %% 年龄计数，合成的人口分布
def get_ages(location, country, level, num_agebrackets=18):
    """
    Get the age count for the synthetic population of the location.

    Args:
        location (str)        : name of the location
        country (str)         : name of the country
        level (str)           : name of level (country or subnational)
        num_agebrackets (int) : the number of age brackets

    Returns:
        dict: A dictionary of the age count.
    """

    if country == 'Europe':
        country = location
        level = 'country'
    if level == 'country':
        file_name = country + '_' + level + '_level_age_distribution_' + '%i' % num_agebrackets + '.csv'
    else:
        file_name = country + '_' + level + '_' + location + '_age_distribution_' + '%i' % num_agebrackets + '.csv'
    file_path = os.path.join(datadir, 'origin_resource', 'age_distributions', file_name)
    df = pd.read_csv(file_path, delimiter=',', header=None)
    df.columns = ['age', 'age_count']
    return dict(zip(df.age.values.astype(int), df.age_count.values))

# 刻度标签数组
def get_label(num_agebrackets=18):
    if num_agebrackets == 85:
        age_brackets = [str(i) + '-' + str((i + 1)) for i in range(0, 84)] + ['85+']
    elif num_agebrackets == 18:
        age_brackets = [str(5 * i) + '-' + str(5 * (i + 1) - 1) for i in range(17)] + ['85+']
    return age_brackets

def read_cumulative_incidence(location,p,p_v,l,l_v,k,k_v,num_agebrackets):
    file_name = f"{location}_cumulative_incidence_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', f'{p}_{p_v}', location, file_name)

    M = np.loadtxt(file_path, delimiter=',')
    return M

def read_cumulative_incidence_age(location,p,p_v,l,l_v,k,k_v,num_agebrackets):
    file_name = f"{location}_cumulative_incidence_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', "bracket", p+"_"+str(p_v), location, file_name)

    M = np.loadtxt(file_path, delimiter=',')
    return M

def plot_p_cumulative_incidence(location, p, l, l_v, k, k_v, num_agebrackets, isSave=False):
    pv_arr = [0.1, 0.5, 0.9]
    data1 = read_cumulative_incidence(location, p, pv_arr[0],l,l_v,k,k_v,num_agebrackets).T
    data2 = read_cumulative_incidence(location, p, pv_arr[1],l,l_v,k,k_v,num_agebrackets).T
    data3 = read_cumulative_incidence(location, p, pv_arr[2],l,l_v,k,k_v,num_agebrackets).T

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(5, 4))
    plt.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.90, wspace=0.1, hspace=0.1)
    axes.plot(data1[0], data1[1], label="p = " + str(pv_arr[0]))
    axes.plot(data2[0], data2[1], label="p = " + str(pv_arr[1]))
    axes.plot(data3[0], data3[1], label="p = " + str(pv_arr[2]))
    axes.set_xlabel('时间（天）', labelpad=10, fontsize=14)
    axes.set_ylabel('发病率（%）', labelpad=15, fontsize=14)
    plt.ylim(0, 100)
    plt.xlim(0, 700)
    plt.rcParams.update({'font.size': 12})
    axLine, axLabel = axes.get_legend_handles_labels()
    fig.legend(axLine, axLabel, loc='upper center', shadow=False, frameon=False, ncol=3)
    # plt.show()

    if isSave:
        file_name = f'{location}_cumulative_incidence_age_{num_agebrackets}_p_{l}_{l_v}_{k}_{k_v}.pdf'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'{location}_cumulative_incidence_age_{num_agebrackets}_p_{l}_{l_v}_{k}_{k_v}.eps'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'{location}_cumulative_incidence_age_{num_agebrackets}_p_{l}_{l_v}_{k}_{k_v}.svg'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')

def plot_cumulative_incidence_age(location, p, l, l_v, k, k_v,num_agebrackets=18, isSave=False):
    data = read_cumulative_incidence_age(location, p, p_v,l,l_v,k,k_v,num_agebrackets).T
    print(data)
    age_brackets = get_label()
    age_brackets = np.array(age_brackets).reshape(3, 6)
    print(age_brackets)
    left = 0.08
    right = 0.97
    top = 0.9
    bottom = 0.15
    fig, axes = plt.subplots(3, 6, figsize=(9, 4))
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=0.3, hspace=0.8)
    for i in range(3):
        axes[i][0].set_ylabel('发病率（%）', fontsize=14)
        for j in range(6):
            axes[2][j].set_xlabel('时间（天）', fontsize=14)
            axes[i][j].set_title(f'年龄组:{age_brackets[i][j]}', fontsize=12)
            axes[i][j].xaxis.set_major_locator(plt.MaxNLocator(3))
            axes[i][j].yaxis.set_major_locator(plt.MaxNLocator(2))
            axes[i][j].set_ylim((0, 50))
            axes[i][j].tick_params(labelsize=10)
            axes[i][j].plot(data[0][0:600], data[(i+1)*(j+1)][0:600])
            axes[i][j].axhline(25, linewidth=0.5, linestyle='--', color='r')

    plt.show()

    if isSave:
        file_name = f'{location}_cumulative_incidence_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.pdf'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'{location}_cumulative_incidence_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.eps'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'{location}_cumulative_incidence_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.svg'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')

def plot_cumulative_incidence_l_k(location, p, p_v, l, l_v, k, k_v,num_agebrackets=18, isSave=False):
    data = []
    for l_v in range(11):
        for k_v in range(11):
            val = read_cumulative_incidence(location, p, p_v, l, l_v/10, k, k_v/10, num_agebrackets).T[1][-1]
            data.append(val)
    data = np.array(data).reshape(11, 11)

    print(data.round(1))
    left = 0.08
    right = 0.97
    top = 0.9
    bottom = 0.15

    fig = plt.figure(figsize=(9, 4))
    ax = fig.gca(projection='3d')  # 三维坐标轴



    # plt.show()

    if isSave:
        file_name = f'{location}_cumulative_incidence_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.pdf'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'{location}_cumulative_incidence_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.eps'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'{location}_cumulative_incidence_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.svg'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')


location = 'Shanghai'
country = 'China'
level = 'subnational'
num_agebrackets = 18
p = 'p'
w = 'w'
# beta = 'beta'
q = 'q'
l = 'l'
# param = 'sigma_inverse'
k = 'k'
# param = 'gamma_inverse'
state = 'all_states'

# Dou-SEAIQ model parameters
p_v = 0.1  # 0.1 0.5  0.9   # percent of unvaccinated
w_v = 0.5  # 感染率降低系数
beta_v = 0.3  # 感染率
q_v = 0.5  # 有症状的概率
l_v = 0.8  # 0-1 接种导致有症状概率降低系数
sigma_inverse = 5.2  # mean latent period
k_v = 0.5  # 0-1 无症状被发现的概率降低系数
gamma_inverse = 1.6  # mean quarantined period

ages = get_ages(location, country, level, num_agebrackets)
total_population = sum(ages.values())

# 累计发病率
plot_p_cumulative_incidence(location, p, l, l_v, k, k_v,num_agebrackets, isSave=True)
# plot_cumulative_incidence_age(location, p, l, l_v, k, k_v,num_agebrackets, isSave=False)

# plot_cumulative_incidence_l_k(location, p, p_v, l, l_v, k, k_v, num_agebrackets=18, isSave=False)
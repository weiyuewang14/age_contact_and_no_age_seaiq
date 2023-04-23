import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sbn
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import font_manager

from pylab import *
from matplotlib.font_manager import FontProperties

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


# 替换字符串中的空格为"_"
def replaceSpace(s: str) -> str:
    # 定义一个列表用来存储结果
    res = []

    # 遍历循环字符串s
    # 当 i 是空格的时候，res拼接“%20”
    # 当 i 非空格时，res拼接当前字符i
    for i in s:
        if i == ' ':
            res.append("_")
        else:
            res.append(i)

    # 将列表转化为字符串返回
    return "".join(res)


def swap(matrix):
    len = matrix.shape
    a = np.ones(len)

    for i in range(len[0]):
        a[i, :] = matrix[len[0] - 1 - i, :]
    return a


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


location = 'Shanghai'
country = 'China'
level = 'subnational'
num_agebrackets = 18  # number of age brackets for the contact matrices

ages = get_ages(location, country, level, num_agebrackets)
total_population = sum(ages.values())


# print(type(total_population[0]))

# 读取需要对比状态的数据
def read_data_param(state_type, p, p_value, l, l_value, k, k_value):
    res_data = []
    for kv in k_value:
        data = []
        for lv in l_value:
            file_name = f"new_case_per_day_no_age_{p}={p_value:.2f}_{l}={lv:.2f}_{k}={kv:.2f}.csv"
            file_path = os.path.join(output_source_dir, 'no_age-structured_output_source', file_name)

            if state_type == 'asymptomaticN':
                #  asymptomatic
                M = np.loadtxt(file_path, delimiter=',', usecols=4).sum()
            elif state_type == 'asymptomaticV':
                #  asymptomatic
                M = np.loadtxt(file_path, delimiter=',', usecols=5).sum()
            elif state_type == 'symptomaticN':
                #  symptomaticN
                M = np.loadtxt(file_path, delimiter=',', usecols=6).sum()
            elif state_type == 'symptomaticV':
                #  symptomaticV
                M = np.loadtxt(file_path, delimiter=',', usecols=7).sum()
            elif state_type == 'asymptomatic':
                #  total asymptomatic
                M = np.loadtxt(file_path, delimiter=',', usecols=(4, 5)).sum(axis=1).sum()
            elif state_type == "symptomatic":
                #  total symptomatic
                M = np.loadtxt(file_path, delimiter=',', usecols=(6, 7)).sum(axis=1).sum()
            elif state_type == "total":
                #  total infected
                M = np.loadtxt(file_path, delimiter=',', usecols=(4, 5, 6, 7)).sum(axis=1).sum()
            else:
                #  total infected
                M = np.loadtxt(file_path, delimiter=',', usecols=(4, 5, 6, 7)).sum(axis=1).sum()

            data.append(M)
        res_data.append(data)
    res = np.array(res_data)

    return res


def plot_p_l_k(state_type, data, p, p_value, isSave=True):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(9, 8))
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(state_type, fontsize=20)
    x_label = [i / 10.0 for i in range(0, 11)]
    y_label = [i / 10.0 for i in range(10, -1, -1)]
    my_colormap = LinearSegmentedColormap.from_list("black", ["green", "yellow", "orange", "red"])
    sbn.heatmap(swap(data) / total_population, vmin=0, cmap=my_colormap, yticklabels=y_label, xticklabels=x_label,
                ax=axes)

    axes.set_xlabel('k', labelpad=15, fontsize=15)
    axes.set_ylabel('l', labelpad=10, fontsize=15)
    plt.show()

    if isSave:
        file_name = state_type + '_' + p + '=' + str(p_value) + '.pdf'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', 'param_test', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = state_type + '_' + p + '=' + str(p_value) + '.eps'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', 'param_test', file_name)
        fig.savefig(fig_path, format='eps')


def plot_p_l_k_mutil(state_type, data, p, p_value, isSave=True):
    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(12, 5))
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.18, top=0.80, wspace=0.1, hspace=0.1)
    plt.suptitle(state_type + " infection ratio", fontsize=20)
    x_label = [i / 10.0 for i in range(0, 11)]
    y_label = [i / 10.0 for i in range(10, -1, -1)]

    sbn.heatmap(swap(data[0]) / total_population, vmin=0, cmap='GnBu', yticklabels=y_label, xticklabels=x_label,
                ax=axes[0])
    axes[0].set_title("p = 0.1")
    axes[0].set_xlabel('k', labelpad=15, fontsize=15)
    axes[0].set_ylabel('l', labelpad=10, fontsize=15)

    sbn.heatmap(swap(data[1]) / total_population, vmin=0, cmap='GnBu', yticklabels=y_label, xticklabels=x_label,
                ax=axes[1])
    axes[1].set_title("p = 0.5")
    axes[1].set_xlabel('k', labelpad=15, fontsize=15)

    sbn.heatmap(swap(data[2]) / total_population, vmin=0, cmap='GnBu', yticklabels=y_label, xticklabels=x_label,
                ax=axes[2])
    axes[2].set_title("p = 0.9")
    axes[2].set_xlabel('k', labelpad=15, fontsize=15)

    plt.show()

    if isSave:
        file_name = state_type + '_infection_ratio_' + p + '.pdf'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', 'param_test', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = state_type + '_infection_ratio_' + p + '.eps'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', 'param_test', file_name)
        fig.savefig(fig_path, format='eps')


def read_attack_ratio(p, p_value):
    file_name = f"attack_ratio_{p}={p_value:.1f}_l_k.csv"
    file_path = os.path.join(output_source_dir, 'no_age-structured_output_source', 'mutil_params', file_name)

    M = np.loadtxt(file_path, delimiter=',')

    return M


def plt_attack_ratio(attack_data, p, p_value, l, k, isSave=True):
    x = [i / 10 for i in range(11)]
    Y = X = np.array(x)
    X, Y = np.meshgrid(X, Y)
    Z = attack_data

    # 绘制表面
    fig = plt.figure(figsize=(6, 5))

    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.set_xlabel(l)
    ax.set_ylabel(k)
    # 添加将值映射到颜色的颜色栏
    fig.colorbar(surf, shrink=0.7, aspect=6)
    ax.set_title("attack ratio with p = " + str(p_value), fontsize=16)
    plt.show()

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    # plt.suptitle('attack rate', fontsize=15)
    # plt.subplots_adjust(left=0.15, right=0.90, bottom=0.14, top=0.90, wspace=0.1, hspace=0.4)
    #
    # ax.set_xlabel(l)
    # ax.set_ylabel('k')

    if isSave:
        file_name = 'attack_ratio' + '_' + p + '=' + str(p_value) + '_l_k.pdf'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = 'attack_ratio' + '_' + p + '=' + str(p_value) + '_l_k.eps'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')


p = "p"
p_value = 0.9  # 0.1 0.5  0.9   # percent of unvaccinated
l = "l"
l_value = [i / 10 for i in range(0, 11)]
k = "k"
k_value = [i / 10 for i in range(0, 11)]
state_type = "total"  # asymptomaticN, asymptomaticV, symptomaticN, symptomaticV, asymptomatic, symptomatic, total

# data = read_data_param(state_type, p, p_value, l, l_value, k, k_value)
# print(data)
# print(data.shape)
# plot_p_l_k(state_type, data, p, p_value, isSave=False)
#
# data1 = read_data_param(state_type, p, 0.1, l, l_value, k, k_value)
# data2 = read_data_param(state_type, p, 0.5, l, l_value, k, k_value)
# data3 = read_data_param(state_type, p, 0.9, l, l_value, k, k_value)
#
# data = [data1, data2, data3]
# plot_p_l_k_mutil(state_type, data, p, p_value, isSave=True)

#  画不同接种比例下的发病率情况
# data = read_attack_ratio(p, p_value)
# plt_attack_ratio(data, p, p_value, l, k, isSave=False)

""""                     接种率对发病率的影响                         """


def read_cumulative_incidence(p, p_value, l, l_value, k, k_v):
    file_name = f"cumulative_incidence_{p}={p_value:.1f}_{l}_{l_value}_{k}_{k_v}.csv"
    file_path = os.path.join(output_source_dir, 'no_age-structured_output_source', file_name)

    M = np.loadtxt(file_path, delimiter=',')

    return M


def plot_p_cumulative_incidence(p, l, l_v, k, k_v, isSave=False):
    pv_arr = [0.1, 0.5, 0.9]
    data1 = read_cumulative_incidence(p, pv_arr[0], l, l_v, k, k_v).T
    data2 = read_cumulative_incidence(p, pv_arr[1], l, l_v, k, k_v).T
    data3 = read_cumulative_incidence(p, pv_arr[2], l, l_v, k, k_v).T

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(5, 4))
    plt.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.90, wspace=0.1, hspace=0.1)
    axes.plot(data1[0][:500], data1[1][:500], label="p = " + str(pv_arr[0]))
    axes.plot(data2[0][:500], data2[1][:500], label="p = " + str(pv_arr[1]))
    axes.plot(data3[0][:500], data3[1][:500], label="p = " + str(pv_arr[2]))
    axes.set_xlabel('时间（天）', labelpad=10, fontsize=14)
    axes.set_ylabel('发病率（%）', labelpad=15, fontsize=14)
    plt.ylim(0, 100)
    plt.xlim(0, 500)
    plt.rcParams.update({'font.size': 12})
    axLine, axLabel = axes.get_legend_handles_labels()
    fig.legend(axLine, axLabel, loc='upper center', shadow=False, frameon=False, ncol=3)
    # plt.show()

    if isSave:
        file_name = f'cumulative_incidence_no_age_p_{l}_{l_v}_{k}_{k_v}.pdf'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'cumulative_incidence_no_age_p_{l}_{l_v}_{k}_{k_v}.eps'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'cumulative_incidence_no_age_p_{l}_{l_v}_{k}_{k_v}.svg'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')


"""                        接种率对无症状感染者的影响                        """


def read_asymptomatic(p, p_v, l, l_v, k, k_v):
    file_name = f"all_states_numbers_no_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'no_age-structured_output_source', 'mutil_params', file_name)
    M = np.loadtxt(file_path, delimiter=',', usecols=(0, 4, 5))

    return M


def plot_p_asymptomatic(p, l, l_v, k, k_v, isSave=False):
    pv_arr = [0.1, 0.5, 0.9]
    data1 = read_asymptomatic(p, pv_arr[0], l, l_v, k, k_v).T
    data2 = read_asymptomatic(p, pv_arr[1], l, l_v, k, k_v).T
    data3 = read_asymptomatic(p, pv_arr[2], l, l_v, k, k_v).T
    total_population = 499504.0
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(8, 5))
    plt.subplots_adjust(left=0.12, right=0.9, bottom=0.15, top=0.90, wspace=0.1, hspace=0.1)
    axes.plot(data1[0][:500], (data1[1:] / total_population * 100).sum(axis=0)[:500], label="p = " + str(pv_arr[0]))
    axes.plot(data2[0][:500], (data2[1:] / total_population * 100).sum(axis=0)[:500], label="p = " + str(pv_arr[1]))
    axes.plot(data3[0][:500], (data3[1:] / total_population * 100).sum(axis=0)[:500], label="p = " + str(pv_arr[2]))
    axes.set_xlabel('时间（天）', labelpad=10, fontsize=20)
    axes.set_ylabel('无症状感染者数量（%）', labelpad=15, fontsize=20)
    plt.rcParams.update({'font.size': 18})
    plt.legend()
    plt.show()

    if isSave:
        file_name = f'asymptomatic_no_age_p_{l}_{l_v}_{k}_{k_v}.pdf'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'asymptomatic_no_age_p_{l}_{l_v}_{k}_{k_v}.eps'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'asymptomatic_no_age_p_{l}_{l_v}_{k}_{k_v}.svg'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')


"""                        接种率对无症状感染者的影响                        """


def read_infectious(p, p_v, l, l_v, k, k_v):
    file_name = f"all_states_numbers_no_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'no_age-structured_output_source', 'mutil_params', file_name)
    M = np.loadtxt(file_path, delimiter=',', usecols=(0, 4, 5, 6, 7))

    return M


def plot_p_infectious(p, l, l_v, k, k_v, isSave=False):
    pv_arr = [0.1, 0.5, 0.9]
    data1 = read_infectious(p, pv_arr[0], l, l_v, k, k_v).T
    data2 = read_infectious(p, pv_arr[1], l, l_v, k, k_v).T
    data3 = read_infectious(p, pv_arr[2], l, l_v, k, k_v).T
    total_population = 499504.0
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(8, 5))
    plt.subplots_adjust(left=0.12, right=0.9, bottom=0.15, top=0.90, wspace=0.1, hspace=0.1)
    axes.plot(data1[0][:500], (data1[1:] / total_population * 100).sum(axis=0)[:500], label="p = " + str(pv_arr[0]))
    axes.plot(data2[0][:500], (data2[1:] / total_population * 100).sum(axis=0)[:500], label="p = " + str(pv_arr[1]))
    axes.plot(data3[0][:500], (data3[1:] / total_population * 100).sum(axis=0)[:500], label="p = " + str(pv_arr[2]))
    axes.set_xlabel('时间（天）', labelpad=10, fontsize=20)
    axes.set_ylabel('感染者数量（%）', labelpad=15, fontsize=20)
    plt.rcParams.update({'font.size': 18})
    plt.legend()
    plt.show()

    if isSave:
        file_name = f'infectious_no_age_p_{l}_{l_v}_{k}_{k_v}.pdf'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'infectious_no_age_p_{l}_{l_v}_{k}_{k_v}.eps'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'infectious_no_age_p_{l}_{l_v}_{k}_{k_v}.svg'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')


def plot_p_infectious_asy_no_age(p, l, l_v, k, k_v, isSave=False):
    pv_arr = [0.1, 0.5, 0.9]

    total_population = 499504.0
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(7, 4))
    plt.subplots_adjust(left=0.12, right=0.9, bottom=0.17, top=0.8, wspace=0.1, hspace=0.1)
    lines = []
    labels = []
    for pv in pv_arr:
        infectious = read_infectious(p, pv, l, l_v, k, k_v).T
        axes[0].plot(infectious[0], (infectious[1:] / total_population * 100).sum(axis=0), label="p = " + str(pv))

        asymptomatic = read_asymptomatic(p, pv, l, l_v, k, k_v).T
        axes[1].plot(asymptomatic[0], (asymptomatic[1:] / total_population * 100).sum(axis=0), label="p = " + str(pv))

    axLine, axLabel = axes[0].get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)

    # axes.xaxis.set_major_locator(plt.MaxNLocator(8))
    # axes.yaxis.set_major_locator(plt.MaxNLocator(5))
    # axes.set_ylim((0, 5))
    # axes.tick_params(labelsize=12)
    axes[0].set_xlabel('时间（天）', labelpad=10, fontsize=14)
    axes[0].set_title('感染者', fontsize=16)
    axes[1].set_xlabel('时间（天）', labelpad=10, fontsize=14)
    axes[1].set_title('无症状感染者', fontsize=16)
    axes[0].set_ylabel('数量（%）', labelpad=15, fontsize=14)
    plt.rcParams.update({'font.size': 12})
    # plt.ylim(0, 5)
    plt.xlim(0, 500)
    fig.legend(lines, labels, loc='upper center', shadow=False, frameon=False, ncol=3)

    if isSave:
        file_name = f'infectious_asymptomatic_no_age_p_{l}_{l_v}_{k}_{k_v}.pdf'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'infectious_asymptomatic_no_age_p_{l}_{l_v}_{k}_{k_v}.eps'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'infectious_asymptomatic_no_age_p_{l}_{l_v}_{k}_{k_v}.svg'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')


"""                R0的等线图           """


def plot_R0_l_k(beta, p, p_v, w, w_v, q_v, l, l_v, sigma_inverse, k, k_v, gamma_inverse, isSave=False):
    # 建立步长为0.01，即每隔0.01取一个点
    step = 0.01
    l_v = np.arange(0, 1.01, step)
    k_v = np.arange(0.1, 1.01, step)

    # 将原始数据变成网格数据形式
    L, K = np.meshgrid(l_v, k_v)
    # 写入函数，z是大写
    R0 = (beta / (K * 1 / gamma_inverse)) * (
            p_v * ((1 - q_v) + K * q_v) + (1 - p_v) * w_v * ((1 - q_v * L) + K * q_v * L))
    # 设置打开画布大小,长10，宽6
    fig = plt.figure(figsize=(6, 5))
    # 填充颜色，f即filled
    # plt.contourf(L, K, R0)
    # 画等高线
    # plt.contour(L, K, R0)
    contourf = plt.contour(L, K, R0, 7)
    manual_locations = [(.2, .87), (.5, .85), (.6, .18),
                        (0.2, 0.55), (0.5, 0.55), (0.8, 0.6),
                        (0.5, 0.27), (0.2, .28), (.7, .36),
                        (0.12, 0.11), (0.2, 0.2), (0.8, 0.18),
                        (0.4, 0.1), (0.3, 0.15),
                        (0.06, 0.15)]
    plt.clabel(contourf, inline=5, fontsize=12, inline_spacing=2, colors='r', fmt='%.1f', manual=manual_locations)
    # plt.clabel(contourf, inline=True, fontsize=12, inline_spacing=2, colors='r', fmt='%.1f')
    # plt.colorbar()
    plt.title("基本再生数R0", fontsize=16)
    plt.xlabel("接种后出现症状的概率降低系数（l）", fontsize=14)
    plt.ylabel("无症状被发现的概率降低系数（k）", fontsize=14)
    # plt.show()

    if isSave:
        file_name = f'R0_no_age_p_{p_v}_{l}_{k}1.pdf'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'R0_no_age_p_{p_v}_{l}_{k}1.eps'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'R0_no_age_p_{p_v}_{l}_{k}1.svg'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')


def plot_R0_l_w(beta, p, p_v, w, w_v, q_v, l, l_v, sigma_inverse, k, k_v, gamma_inverse, isSave=False):
    # 建立步长为0.01，即每隔0.01取一个点
    step = 0.01
    l_v = np.arange(0, 1.01, step)
    w_v = np.arange(0, 1.01, step)

    # 将原始数据变成网格数据形式
    L, W = np.meshgrid(l_v, w_v)
    # 写入函数，z是大写
    R0 = (beta / (k_v * 1 / gamma_inverse)) * (
            p_v * ((1 - q_v) + k_v * q_v) + (1 - p_v) * W * ((1 - q_v * L) + k_v * q_v * L))
    # 设置打开画布大小,长10，宽6
    fig = plt.figure(figsize=(6, 5))
    # 填充颜色，f即filled
    # plt.contourf(L, K, R0)
    # 画等高线
    # plt.contour(L, K, R0)
    contourf = plt.contour(L, W, R0, 10)
    manual_locations = [(0.2, 0.05), (0.5, 0.05), (0.8, 0.02),
                        (0.2, 0.15), (0.5, 0.15), (0.8, 0.2),
                        (0.2, 0.25), (0.5, 0.3), (0.8, .35),
                        (0.2, 0.45), (0.5, 0.45), (0.8, 0.46),
                        (0.2, 0.55), (0.5, 0.55), (0.8, 0.6),
                        (0.2, 0.65), (0.5, 0.7), (0.8, 0.8),
                        (0.2, 0.75), (0.5, 0.8), (0.8, 0.9),
                        (0.2, 0.9), (0.5, 0.9),
                        (0.06, 0.95)]
    plt.clabel(contourf, inline=1, fontsize=12, inline_spacing=2, colors='r', fmt='%.1f', manual=manual_locations)
    # plt.clabel(contourf, inline=True, fontsize=12, inline_spacing=2, colors='r', fmt='%.1f')
    # plt.colorbar()
    plt.title("基本再生数R0", fontsize=16)
    plt.xlabel("接种后出现症状的概率降低系数（l）", fontsize=14)
    plt.ylabel("感染率降低系数（w）", fontsize=14)
    plt.show()

    if isSave:
        file_name = f'R0_no_age_p_{p_v}_{l}_{w}.pdf'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'R0_no_age_p_{p_v}_{l}_{w}.eps'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'R0_no_age_p_{p_v}_{l}_{w}.svg'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')

def plot_R0_w_l_k(beta, p, p_v, w, w_v, q_v, l, l_v, sigma_inverse, k, k_v, gamma_inverse, isSave=False):
    # 建立步长为0.01，即每隔0.01取一个点
    step = 0.01
    l_v_list = np.arange(0, 1.01, step)
    w_v_list = np.arange(0, 1.01, step)

    # 设置打开画布大小,长10，宽6
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(8, 4))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.25, hspace=0.1)

    # 将原始数据变成网格数据形式
    L, W = np.meshgrid(l_v_list, w_v_list)
    # 写入函数，z是大写
    R01 = (beta / (k_v * 1 / gamma_inverse)) * (
            p_v * ((1 - q_v) + k_v * q_v) + (1 - p_v) * W * ((1 - q_v * L) + k_v * q_v * L))

    contourf1 = axes[0].contour(L, W, R01, 10)
    manual_locations1 = [(0.2, 0.05), (0.5, 0.05), (0.8, 0.02),
                         (0.2, 0.15), (0.5, 0.15), (0.8, 0.2),
                         (0.2, 0.25), (0.5, 0.3), (0.8, .35),
                         (0.2, 0.45), (0.5, 0.45), (0.8, 0.46),
                         (0.2, 0.55), (0.5, 0.55), (0.8, 0.6),
                         (0.2, 0.65), (0.5, 0.7), (0.8, 0.8),
                         (0.2, 0.75), (0.5, 0.8), (0.8, 0.9),
                         (0.2, 0.9), (0.5, 0.9),
                         (0.06, 0.95)]

    axes[0].clabel(contourf1, inline=1, fontsize=12, inline_spacing=2, colors='r', fmt='%.1f', manual=manual_locations1)
    axes[0].set_title("基本再生数R0", fontsize=16)
    # axes[0].set_xlabel("接种后出现症状的概率降低系数（l）", fontsize=14)
    axes[0].set_ylabel("感染率降低系数（w）", fontsize=14)

    l_v_list = np.arange(0, 1.01, step)
    k_v_list = np.arange(0.1, 1.01, step)
    L, K = np.meshgrid(l_v_list, k_v_list)
    # 写入函数，z是大写
    R02 = (beta / (K * 1 / gamma_inverse)) * (
            p_v * ((1 - q_v) + K * q_v) + (1 - p_v) * w_v * ((1 - q_v * L) + K * q_v * L))
    contourf2 = axes[1].contour(L, K, R02, 7)
    manual_locations2 = [(.2, .87), (.5, .85), (.6, .18),
                        (0.2, 0.55), (0.5, 0.55), (0.8, 0.6),
                        (0.5, 0.27), (0.2, .28), (.7, .36),
                        (0.12, 0.11), (0.2, 0.2), (0.8, 0.18),
                        (0.4, 0.1), (0.3, 0.15),
                        (0.06, 0.15)]

    axes[1].clabel(contourf2, inline=True, fontsize=12, inline_spacing=2, colors='r', fmt='%.1f', manual=manual_locations2)

    axes[1].set_title("基本再生数R0", fontsize=16)
    # axes[1].set_xlabel("接种后出现症状的概率降低系数（l）", fontsize=14)
    axes[1].set_ylabel("无症状被发现的概率降低系数（k）", fontsize=14)
    fig.text(0.3, 0.01, "接种后出现症状的概率降低系数（l）", fontsize=14)
    # plt.show()

    if isSave:
        file_name = f'R0_no_age_p_{p_v}_{w}_{l}_{k}.pdf'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'R0_no_age_p_{p_v}_{w}_{l}_{k}.eps'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'R0_no_age_p_{p_v}_{w}_{l}_{k}.svg'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')


"""              所有状态的基本变化图            """


# 读取需要对比状态的数据
def read_all_states_data(p, p_v, l, l_v, k, k_v):
    file_name = f"all_states_numbers_no_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'no_age-structured_output_source', "mutil_params", file_name)
    M = np.loadtxt(file_path, delimiter=',')

    return M


def plot_all_states(all_state_data, p, p_v, l, l_v, k, k_v, isSave=False):
    # 画图
    fontsizes = {'colorbar': 30, 'colorbarlabels': 22, 'title': 44, 'ylabel': 28, 'xlabel': 28, 'xticks': 24,
                 'yticks': 24}
    total_population = 499504.0
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(8, 5))
    plt.subplots_adjust(left=0.12, right=0.9, bottom=0.15, top=0.90, wspace=0.1, hspace=0.1)
    for i in range(8):
        axes.plot(all_state_data[:500, 0], all_state_data[:500, i + 1] / total_population)
    # ax.plot(all_state_data[:, 0], all_state_data[:, 2])
    # ax.plot(all_state_data[:, 0], all_state_data[:, 3])
    # ax.plot(all_state_data[:, 0], all_state_data[:, 4])
    # ax.plot(all_state_data[:, 0], all_state_data[:, 5])
    # ax.plot(all_state_data[:, 0], all_state_data[:, 6])
    # ax.plot(all_state_data[:, 0], all_state_data[:, 7])
    # ax.plot(all_state_data[:, 0], all_state_data[:, 8])

    axes.set_title('SEAIQ模型', fontsize=16)
    axes.set_xlabel('时间（天）', labelpad=10, fontsize=14)
    axes.set_ylabel('数量（%）', labelpad=15, fontsize=14)
    plt.rcParams.update({'font.size': 12})

    axins = inset_axes(axes, width=2.2, height=1.3)

    axins.plot(all_state_data[100:100, 0], all_state_data[[0 for i in range(100, 100)], 0])
    for i in range(2, 8):
        axins.plot(all_state_data[100:500, 0], all_state_data[100:500, i] / total_population)

    axes.legend(['易感染者', '未接种的暴露者', '已接种的暴露者', '未接种的无症状感染者', '已接种的无症状感染者', '未接种的有症状感染者', '已接种的有症状感染者', '隔离者'])
    # plt.show()

    # 是否保存
    if (isSave):
        file_name = f'all_states_numbers_no_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.pdf'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'all_states_numbers_no_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.eps'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'all_states_numbers_no_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.svg'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')


def plot_infectious_with_param(data, p, p_v, l, l_v, k, k_v, isSave=False):
    # 画图
    total_population = 499504.0
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(8, 5))
    plt.subplots_adjust(left=0.12, right=0.9, bottom=0.15, top=0.90, wspace=0.1, hspace=0.1)
    for d in data:
        axes.plot(d[0], (d[1:] / total_population * 100).sum(axis=0))

    axes.set_title('感染者', fontsize=22)
    axes.set_xlabel('时间（天）', labelpad=10, fontsize=20)
    axes.set_ylabel('比例（%）', labelpad=15, fontsize=20)
    plt.rcParams.update({'font.size': 16})

    axes.legend([f'{k} = {param_v / 10}' for param_v in range(0, 11)])
    plt.show()

    # 是否保存
    if (isSave):
        file_name = f'infectious_no_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}.pdf'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'infectious_no_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}.eps'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'infectious_no_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}.svg'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')

def plot_infectious_with_l_or_k(p, p_v, l, l_v, k, k_v, isSave=False):
    # 画图
    total_population = 499504.0
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4.3))
    plt.subplots_adjust(left=0.12, right=0.9, bottom=0.2, top=0.90, wspace=0.15, hspace=0.1)

    param = [i / 10 for i in range(11)]
    for v in param:
        data_k = read_infectious(p, p_v, l, l_v, k, v).T
        axes[1].plot(data_k[0], (data_k[1:] / total_population * 100).sum(axis=0), label=f'k = {v}')

        data_l = read_infectious(p, p_v, l, v, k, k_v).T
        axes[0].plot(data_l[0], (data_l[1:] / total_population * 100).sum(axis=0), label=f'l = {v}')

    axes[0].set_xlabel('时间（天）\n （a）接种后有症状概率降低系数l', labelpad=10, fontsize=14)
    axes[0].set_ylabel('感染者比例（%）', labelpad=15, fontsize=14)
    axes[1].set_xlabel('时间（天）\n（b）接种后有症状概率降低系数k', labelpad=10, fontsize=14)
    # axes[1].set_ylabel('比例（%）', labelpad=15, fontsize=20)

    plt.rcParams.update({'font.size': 12})
    # axLine, axLabel = axes[0].get_legend_handles_labels()
    # axes[0].legend(axLine, axLabel, loc='upper right', shadow=False, frameon=False, ncol=4)
    axes[0].legend(shadow=False, frameon=False)
    axes[1].legend(shadow=False, frameon=False)
    # axes.legend([f'{k} = {param_v / 10}' for param_v in range(0, 11)])
    # plt.show()

    # 是否保存
    if (isSave):
        file_name = f'infectious_no_age_{p}={p_v:.2f}_{l}_{k}.pdf'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'infectious_no_age_{p}={p_v:.2f}_{l}_{k}.eps'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'infectious_no_age_{p}={p_v:.2f}_{l}_{k}.svg'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')


def plot_asymptomatic_with_param(data, p, p_v, l, l_v, k, k_v, isSave=False):
    # 画图
    total_population = 499504.0
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(8, 5))
    plt.subplots_adjust(left=0.12, right=0.9, bottom=0.15, top=0.90, wspace=0.1, hspace=0.1)
    for d in data:
        axes.plot(d[0], (d[1:] / total_population * 100).sum(axis=0))

    axes.set_title('无症状感染者', fontsize=22)
    axes.set_xlabel('时间（天）', labelpad=10, fontsize=20)
    axes.set_ylabel('比例（%）', labelpad=15, fontsize=20)
    plt.rcParams.update({'font.size': 15})

    axes.legend([f'{l} = {param_v / 10}' for param_v in range(0, 11)])
    plt.show()

    # 是否保存
    if (isSave):
        file_name = f'asymptomatic_no_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}.pdf'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'asymptomatic_no_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}.eps'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'asymptomatic_no_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}.svg'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')

p = 'p'
w = 'w'
l = 'l'
k = 'k'

p_v = 0.1  # 0.1 0.5  0.9   # percent of unvaccinated
w_v = 0.5  # 感染率降低系数
beta = 0.6  # 感染率
q_v = 0.5  # 有症状的概率
l_v = 0.8  # 0-1 接种导致有症状概率降低系数
sigma_inverse = 5.2  # mean latent period
k_v = 0.5  # 0-1 无症状被发现的概率降低系数
gamma_inverse = 1.6  # mean quarantined period

# 接种对发病率的影响
# plot_p_cumulative_incidence(p, l, l_v, k, k_v, isSave=True)

# 接种率对无症状感染者的影响
# plot_p_asymptomatic(p, l, l_v, k, k_v, isSave=True)

# 接种率对感染者的影响
# plot_p_infectious(p, l, l_v, k, k_v, isSave=True)

# plot_p_infectious_asy_no_age(p, l, l_v, k, k_v, isSave=True)

# R0 随 l 和 w 的等高图
# plot_R0_l_w(beta, p, p_v, w, w_v, q_v, l, l_v, sigma_inverse, k, k_v, gamma_inverse, isSave=True)
#
# # R0 随 l 和 k 的等高图
# plot_R0_l_k(beta, p, p_v, w, w_v, q_v, l, l_v, sigma_inverse, k, k_v, gamma_inverse, isSave=True)

# plot_R0_w_l_k(beta, p, p_v, w, w_v, q_v, l, l_v, sigma_inverse, k, k_v, gamma_inverse, isSave=True)

# 状态基本变化图
# all_state_data = read_all_states_data(p, p_v, l, l_v, k, k_v)
# plot_all_states(all_state_data, p, p_v, l, l_v, k, k_v, isSave=True)


# data_list = []
# param = [i / 10 for i in range(11)]
# for k_v in param:
#     data = read_infectious(p, p_v, l, l_v, k, k_v).T
#     data_list.append(data)
# 感染者随l/k的变化图
# plot_infectious_with_param(data_list, p, p_v, l, l_v, k, k_v, isSave=True)

plot_infectious_with_l_or_k(p, p_v, l, l_v, k, k_v, isSave=True)

# 无症状感染者随l/k的变化图
# plot_asymptomatic_with_param(data_list, p, p_v, l, l_v, k, k_v, isSave=True)





def compute(beta, p, w, q, l, sigma, k, gamma):
    return (beta / (k * gamma)) * (p * ((1 - q) + k * q) + (1 - p) * w * ((1 - q * l) + k * q * l))


R0 = compute(beta, p_v, w_v, q_v, l_v, 1 / sigma_inverse, k_v, 1 / gamma_inverse)
print("基本再生数R0：", R0)

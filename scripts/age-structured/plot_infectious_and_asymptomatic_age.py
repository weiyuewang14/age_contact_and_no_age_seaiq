import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sbn
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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


"""   无症状感染者   """
def read_asymptomatic_age(location, p, p_v, l, l_v, k, k_v, num_agebrackets):
    file_name = f"{location}_all_states_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', f'{p}_{p_v:.1f}', location, file_name)
    M = np.loadtxt(file_path, delimiter=',', usecols=(0, 4, 5))

    return M
def plot_p_asymptomatic(location, total_population, p, l, l_v, k, k_v, num_agebrackets, isSave=False):
    pv_arr = [0.1, 0.5, 0.9]
    data1 = read_asymptomatic_age(location, p, pv_arr[0], l, l_v, k, k_v, num_agebrackets).T
    data2 = read_asymptomatic_age(location, p, pv_arr[1], l, l_v, k, k_v, num_agebrackets).T
    data3 = read_asymptomatic_age(location, p, pv_arr[2], l, l_v, k, k_v, num_agebrackets).T

    # total_population = 499504.0
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(8, 5))
    plt.subplots_adjust(left=0.12, right=0.9, bottom=0.15, top=0.90, wspace=0.1, hspace=0.1)
    axes.plot(data1[0][:500], (data1[1:] / total_population * 100).sum(axis=0)[:500], label="p = " + str(pv_arr[0]))
    axes.plot(data2[0][:500], (data2[1:] / total_population * 100).sum(axis=0)[:500], label="p = " + str(pv_arr[1]))
    axes.plot(data3[0][:500], (data3[1:] / total_population * 100).sum(axis=0)[:500], label="p = " + str(pv_arr[2]))
    axes.xaxis.set_major_locator(plt.MaxNLocator(8))
    axes.yaxis.set_major_locator(plt.MaxNLocator(5))
    axes.set_ylim((0, 5))
    axes.tick_params(labelsize=16)
    axes.set_xlabel('时间（天）', labelpad=10, fontsize=20)
    axes.set_ylabel('无症状感染者数量（%）', labelpad=15, fontsize=20)
    plt.rcParams.update({'font.size': 18})
    plt.legend()
    plt.show()

    if isSave:
        file_name = f'{location}_asymptomatic_age_{num_agebrackets}_p_{l}_{l_v}_{k}_{k_v}.pdf'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'{location}_asymptomatic_age_{num_agebrackets}_p_{l}_{l_v}_{k}_{k_v}.eps'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'{location}_asymptomatic_age_{num_agebrackets}_p_{l}_{l_v}_{k}_{k_v}.svg'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')

"""   感染者   """
def read_infectious_age(location, p, p_v, l, l_v, k, k_v, num_agebrackets):
    file_name = f"{location}_all_states_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', f'{p}_{p_v:.1f}', location, file_name)
    M = np.loadtxt(file_path, delimiter=',', usecols=(0, 4, 5, 6, 7))

    return M


def plot_p_infectious_age(location, total_population, p, l, l_v, k, k_v, num_agebrackets, isSave=False):
    pv_arr = [0.1, 0.5, 0.9]
    data1 = read_infectious_age(location, p, pv_arr[0], l, l_v, k, k_v, num_agebrackets).T
    data2 = read_infectious_age(location, p, pv_arr[1], l, l_v, k, k_v, num_agebrackets).T
    data3 = read_infectious_age(location, p, pv_arr[2], l, l_v, k, k_v, num_agebrackets).T

    # total_population = 499504.0
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(8, 5))
    plt.subplots_adjust(left=0.12, right=0.9, bottom=0.15, top=0.90, wspace=0.1, hspace=0.1)
    axes.plot(data1[0][:500], (data1[1:] / total_population * 100).sum(axis=0)[:500], label="p = " + str(pv_arr[0]))
    axes.plot(data2[0][:500], (data2[1:] / total_population * 100).sum(axis=0)[:500], label="p = " + str(pv_arr[1]))
    axes.plot(data3[0][:500], (data3[1:] / total_population * 100).sum(axis=0)[:500], label="p = " + str(pv_arr[2]))
    axes.xaxis.set_major_locator(plt.MaxNLocator(8))
    axes.yaxis.set_major_locator(plt.MaxNLocator(5))
    axes.set_ylim((0, 5))
    axes.tick_params(labelsize=16)
    axes.set_xlabel('时间（天）', labelpad=10, fontsize=20)
    axes.set_ylabel('感染者数量（%）', labelpad=15, fontsize=20)
    plt.rcParams.update({'font.size': 18})
    plt.legend()
    plt.show()

    if isSave:
        file_name = f'{location}_infectious_all_age_{num_agebrackets}_p_{l}_{l_v}_{k}_{k_v}.pdf'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'{location}_infectious_all_age_{num_agebrackets}_p_{l}_{l_v}_{k}_{k_v}.eps'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'{location}_infectious_all_age_{num_agebrackets}_p_{l}_{l_v}_{k}_{k_v}.svg'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')

def plot_p_infectious_asy_age(location, total_population, p, l, l_v, k, k_v, num_agebrackets, isSave=False):
    pv_arr = [0.1, 0.5, 0.9]

    # total_population = 499504.0
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(7, 4))
    plt.subplots_adjust(left=0.12, right=0.9, bottom=0.17, top=0.8, wspace=0.1, hspace=0.1)
    lines = []
    labels = []
    for pv in pv_arr:
        infectious = read_infectious_age(location, p, pv, l, l_v, k, k_v, num_agebrackets).T
        axes[0].plot(infectious[0], (infectious[1:] / total_population * 100).sum(axis=0), label="p = " + str(pv))

        asymptomatic = read_asymptomatic_age(location, p, pv, l, l_v, k, k_v, num_agebrackets).T
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
    plt.ylim(0, 5)
    plt.xlim(0, 700)
    fig.legend(lines, labels, loc='upper center', shadow=False, frameon=False, ncol=3)

    if isSave:
        file_name = f'{location}_infectious_asymptomatic_all_age_{num_agebrackets}_p_{l}_{l_v}_{k}_{k_v}.pdf'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'{location}_infectious_asymptomatic_all_age_{num_agebrackets}_p_{l}_{l_v}_{k}_{k_v}.eps'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'{location}_infectious_asymptomatic_all_age_{num_agebrackets}_p_{l}_{l_v}_{k}_{k_v}.svg'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')

def read_infectious_all_age(location, age, p, p_v, l, l_v, k, k_v, num_agebrackets):
    file_name = f"{location}_all_states_numbers_{num_agebrackets}_age={age}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', 'bracket', f'{p}_{p_v:.1f}', location, file_name)
    M = np.loadtxt(file_path, delimiter=',', usecols=(0, 4, 5, 6, 7))

    return M

def plot_infectious_all_age(location, ages, p, l, l_v, k, k_v,num_agebrackets=18, isSave=False):
    age_brackets = get_label()
    # age_brackets = np.array(age_brackets).reshape(3, 6)
    print(age_brackets)
    left = 0.05
    right = 0.97
    top = 0.85
    bottom = 0.15
    fig, axes = plt.subplots(3, 6, figsize=(9, 4))
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=0.3, hspace=1.5)
    for ax, age_title, age in zip(axes.flat, age_brackets, range(num_agebrackets)):
        data = read_infectious_all_age(location, age, p, p_v, l, l_v, k, k_v, num_agebrackets).T
        data1 = data[1:].sum(axis=0)/ages[age] * 100
        ax.set_xlabel('时间（天）', fontsize=10)
        ax.set_title(f'年龄组:{age_title}', fontsize=10)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_ylim((0, 6))
        ax.tick_params(labelsize=8)
        ax.plot(data[0][0:200], data1[0:200])
        # ax.axhline(25, linewidth=0.5, linestyle='--', color='r')
    plt.suptitle(f'{location_dic[location]}')
    plt.show()

    if isSave:
        file_name = f'{location}_infectious_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.pdf'
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'infectious_all_age', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'{location}_infectious_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.eps'
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'infectious_all_age', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'{location}_infectious_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.svg'
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'infectious_all_age', file_name)
        fig.savefig(fig_path, format='svg')


def plot_infectious_all_age_one_fig(location_dic, p, l, l_v, k, k_v,num_agebrackets=18, isSave=False):
    age_brackets = get_label()
    # age_brackets = np.array(age_brackets).reshape(3, 6)
    print(age_brackets)
    left = 0.07
    right = 0.97
    top = 0.85
    bottom = 0.15
    fig, axes = plt.subplots(3, 6, sharex=True, sharey=True, figsize=(9, 4))
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=0.2, hspace=0.8)
    fig.text(0.01, 0.5, '感染者数量（%）', va='center', rotation='vertical')
    lines = []
    labels = []
    for ax, age_title, age in zip(axes.flat, age_brackets, range(num_agebrackets)):
        if age > 11:
            ax.set_xlabel('时间（天）', fontsize=10)
        ax.set_title(f'年龄组:{age_title}', fontsize=10)
        for location in location_dic:
            ages = get_ages(location, country, level, num_agebrackets)
            data = read_infectious_all_age(location, age, p, p_v, l, l_v, k, k_v, num_agebrackets).T
            data1 = data[1:].sum(axis=0)/ages[age] * 100
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.set_ylim((0, 1))
            ax.tick_params(labelsize=8)
            ax.plot(data[0], data1, label=location_dic[location])
        # ax.axhline(25, linewidth=0.5, linestyle='--', color='r')
        if age == 0:
            axLine, axLabel = ax.get_legend_handles_labels()
            lines.extend(axLine)
            labels.extend(axLabel)

    fig.legend(lines, labels, loc='upper center', shadow=False, frameon=False, ncol=6)
    plt.show()

    if isSave:
        file_name = f'infectious_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.pdf'
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'infectious_all_age', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'infectious_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.eps'
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'infectious_all_age', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'infectious_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.svg'
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'infectious_all_age', file_name)
        fig.savefig(fig_path, format='svg')

def plot_asy_one_fig(location_dic, p, l, l_v, k, k_v, num_agebrackets=18, isSave=False):
    age_brackets = get_label()
    # age_brackets = np.array(age_brackets).reshape(3, 6)
    print(age_brackets)
    left = 0.07
    right = 0.97
    top = 0.85
    bottom = 0.15
    fig, axes = plt.subplots(3, 6, sharex=True, sharey=True, figsize=(9, 4))
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=0.2, hspace=0.8)
    fig.text(0.01, 0.5, '无症状感染者数量（%）', va='center', rotation='vertical')
    lines = []
    labels = []
    for ax, age_title, age in zip(axes.flat, age_brackets, range(num_agebrackets)):
        if age > 11:
            ax.set_xlabel('时间（天）', fontsize=10)
        ax.set_title(f'年龄组:{age_title}', fontsize=10)
        for location in location_dic:
            ages = get_ages(location, country, level, num_agebrackets)
            data = read_infectious_all_age(location, age, p, p_v, l, l_v, k, k_v, num_agebrackets).T
            data1 = data[1:].sum(axis=0)/ages[age] * 100
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(4))
            ax.set_ylim((0, 1))
            ax.tick_params(labelsize=8)
            ax.plot(data[0], data1, label=location_dic[location])
        # ax.axhline(25, linewidth=0.5, linestyle='--', color='r')
        if age == 0:
            axLine, axLabel = ax.get_legend_handles_labels()
            lines.extend(axLine)
            labels.extend(axLabel)

    fig.legend(lines, labels, loc='upper center', shadow=False, frameon=False, ncol=6)
    plt.show()

    if isSave:
        file_name = f'infectious_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.pdf'
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'infectious_all_age', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'infectious_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.eps'
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'infectious_all_age', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'infectious_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.svg'
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'infectious_all_age', file_name)
        fig.savefig(fig_path, format='svg')


location_dic = {'Hubei':'湖北省', 'Shanghai':'上海市', 'Jiangsu':'江苏省', 'Beijing':'北京市', 'Tianjin':'天津市', 'Guangdong':'广东省'}
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





# 接种率对无症状感染者的影响
# plot_p_asymptomatic(location, p, l, l_v, k, k_v, num_agebrackets, isSave=False)
# plot_p_infectious_age(location, p, l, l_v, k, k_v, num_agebrackets, isSave=False)
# for location in location_dic:
#     ages = get_ages(location, country, level, num_agebrackets)
#     total_population = sum(ages.values())
#     plot_infectious_all_age(location, ages, p, l, l_v, k, k_v, num_agebrackets, isSave=False)

# plot_infectious_all_age_one_fig(location_dic, p, l, l_v, k, k_v, num_agebrackets=18, isSave=True)

ages = get_ages(location, country, level, num_agebrackets)
total_population = sum(ages.values())
plot_p_infectious_asy_age(location, total_population, p, l, l_v, k, k_v, num_agebrackets, isSave=True)
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


"""   每日新增病例   """
def read_new_case_age(location, p, p_v, l, l_v, k, k_v, num_agebrackets):
    file_name = f"{location}_new_case_per_day_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', f'{p}_{p_v:.1f}', location, file_name)
    M = np.loadtxt(file_path, delimiter=',', usecols=(0, 4, 5, 6, 7))

    return M

def plot_new_case_infectious_asymptomatic_l(location, p, l, l_v, k, k_v, num_agebrackets, isSave=False):

    ages = get_ages(location, country, level, num_agebrackets)
    total_population = sum(ages.values())
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(9, 5))
    plt.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.90, wspace=0.3, hspace=0.1)


    for l_v in range(11):
        l_v = l_v/10
        data = read_new_case_age(location, p, p_v, l, l_v, k, k_v, num_agebrackets).T

        axes[1].plot(data[0], (data[1:] / total_population * 100).sum(axis=0), label=f'{l}={l_v}')
        axes[0].plot(data[0], (data[1:3] / total_population * 100).sum(axis=0), label=f'{l}={l_v}')
    # axes.xaxis.set_major_locator(plt.MaxNLocator(8))
    # axes.yaxis.set_major_locator(plt.MaxNLocator(5))
    # fig.tick_params(labelsize=16)
    axes[0].set_title('无症状感染者', fontsize=16)
    axes[0].set_xlabel('时间（天）', labelpad=10, fontsize=14)
    axes[0].set_ylabel('每日新增病例（%）', labelpad=15, fontsize=14)
    axes[1].set_title('感染者', fontsize=20)
    axes[1].set_xlabel('时间（天）', labelpad=10, fontsize=14)
    axes[1].set_ylabel('每日新增病例（%）', labelpad=15, fontsize=14)
    plt.rcParams.update({'font.size': 12})
    plt.ylim(0, 5)
    plt.xlim(0, 700)
    plt.legend()
    # plt.show()

    if isSave:
        file_name = f'{location}_new_case_per_day_age_{num_agebrackets}_{p}_{p_v}_{l}_{k}_{k_v}.pdf'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'{location}_new_case_per_day_age_{num_agebrackets}_{p}_{p_v}_{l}_{k}_{k_v}.eps'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'{location}_new_case_per_day_age_{num_agebrackets}_{p}_{p_v}_{l}_{k}_{k_v}.svg'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')

def plot_new_case_infectious_asymptomatic_k(location, p, l, l_v, k, k_v, num_agebrackets, isSave=False):

    ages = get_ages(location, country, level, num_agebrackets)
    total_population = sum(ages.values())
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(8, 4))
    plt.subplots_adjust(left=0.10, right=0.85, bottom=0.15, top=0.85, wspace=0.2, hspace=0.1)

    lines = []
    labels = []
    for k_v in range(11):
        k_v = k_v/10
        data = read_new_case_age(location, p, p_v, l, l_v, k, k_v, num_agebrackets).T

        axes[1].plot(data[0], (data[1:] / total_population * 100).sum(axis=0), label=f'{k}={k_v}')
        axes[0].plot(data[0], (data[1:3] / total_population * 100).sum(axis=0), label=f'{k}={k_v}')
    axLine, axLabel = axes[0].get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)
    axes[0].set_title('无症状感染者', fontsize=16)
    axes[0].set_xlabel('时间（天）', labelpad=10, fontsize=14)
    axes[0].set_ylabel('每日新增病例（%）', labelpad=15, fontsize=14)
    axes[1].set_title('感染者', fontsize=16)
    axes[1].set_xlabel('时间（天）', labelpad=10, fontsize=14)
    fig.legend(lines, labels, loc='right', shadow=False, frameon=False, ncol=1)
    plt.rcParams.update({'font.size': 12})
    plt.ylim(0, 5)
    plt.xlim(0, 700)
    # plt.show()

    if isSave:
        file_name = f'{location}_new_case_per_day_age_{num_agebrackets}_{p}_{p_v}_{l}_{l_v}_{k}.pdf'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'{location}_new_case_per_day_age_{num_agebrackets}_{p}_{p_v}_{l}_{l_v}_{k}.eps'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'{location}_new_case_per_day_age_{num_agebrackets}_{p}_{p_v}_{l}_{l_v}_{k}.svg'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')

def plot_new_case_infectious_by_location_l(location, p, l, l_v, k, k_v, num_agebrackets, isSave=False):

    ages = get_ages(location, country, level, num_agebrackets)
    total_population = sum(ages.values())
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(9, 5))
    plt.subplots_adjust(left=0.1, right=0.87, bottom=0.12, top=0.9, wspace=0.2, hspace=0.3)


    lines = []
    labels = []
    for ax, location in zip(axes.flat, location_dic):
        ax.set_title(location_dic[location])
        total_population = sum(get_ages(location, country, level, num_agebrackets).values())

        for l_v in range(11):
            l_v = l_v/10
            file_name = f"{location}_new_case_per_day_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
            file_path = os.path.join(output_source_dir, 'age-structured_output_source', f'{p}_{p_v}', location, file_name)

            data = np.loadtxt(file_path, delimiter=',', usecols=(0, 4, 5, 6, 7)).T
            ax.plot(data[0], data[1:].sum(axis=0), label=f'{l} = {l_v}')
            ax.set_title(location_dic[location], fontsize=16)
        if(location == 'Shanghai'):
            axLine, axLabel = ax.get_legend_handles_labels()
            lines.extend(axLine)
            labels.extend(axLabel)

    fig.legend(lines, labels, loc='right', shadow=False, frameon=False, ncol=1)
    fig.text(0.5, 0.05, '时间（天）', va='center', fontsize=14)
    fig.text(0.01, 0.5, '每日新增病例（个）', va='center', fontsize=14, rotation='vertical')
    plt.rcParams.update({'font.size': 12})
    # plt.ylim(0, 5)
    plt.xlim(0, 700)
    # plt.show()

    if isSave:
        file_name = f'six_new_case_per_day_{num_agebrackets}_{p}_{p_v}_{l}_{k}_{k_v}.pdf'
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'six_new_case_per_day_{num_agebrackets}_{p}_{p_v}_{l}_{k}_{k_v}.eps'
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'six_new_case_per_day_{num_agebrackets}_{p}_{p_v}_{l}_{k}_{k_v}.svg'
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='svg')


def plot_new_case_infectious_by_location_k(location, p, l, l_v, k, k_v, num_agebrackets, isSave=False):

    ages = get_ages(location, country, level, num_agebrackets)
    total_population = sum(ages.values())
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(9, 5))
    plt.subplots_adjust(left=0.1, right=0.87, bottom=0.12, top=0.9, wspace=0.2, hspace=0.3)


    lines = []
    labels = []
    for ax, location in zip(axes.flat, location_dic):
        ax.set_title(location_dic[location], fontsize=16)
        total_population = sum(get_ages(location, country, level, num_agebrackets).values())

        for k_v in range(0, 11, 2):
            k_v = k_v/10
            file_name = f"{location}_new_case_per_day_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
            file_path = os.path.join(output_source_dir, 'age-structured_output_source', f'{p}_{p_v}', location, file_name)

            data = np.loadtxt(file_path, delimiter=',', usecols=(0, 4, 5, 6, 7)).T
            ax.plot(data[0], data[1:].sum(axis=0), label=f'{k} = {k_v}')
            ax.set_title(location_dic[location], fontsize=16)
        if(location == 'Shanghai'):
            axLine, axLabel = ax.get_legend_handles_labels()
            lines.extend(axLine)
            labels.extend(axLabel)

    fig.legend(lines, labels, loc='right', shadow=False, frameon=False, ncol=1)
    fig.text(0.5, 0.05, '时间（天）', va='center', fontsize=14)
    fig.text(0.01, 0.5, '每日新增病例', va='center', fontsize=14, rotation='vertical')
    plt.rcParams.update({'font.size': 12})
    # plt.ylim(0, 5)
    plt.xlim(0, 700)
    # plt.show()

    if isSave:
        file_name = f'six_new_case_per_day_{num_agebrackets}_{p}_{p_v}_{l}_{l_v}_{k}.pdf'
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'six_new_case_per_day_{num_agebrackets}_{p}_{p_v}_{l}_{l_v}_{k}.eps'
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'six_new_case_per_day_{num_agebrackets}_{p}_{p_v}_{l}_{l_v}_{k}.svg'
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
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


plot_new_case_infectious_asymptomatic_k(location, p, l, l_v, k, k_v, num_agebrackets, isSave=True)
# plot_new_case_infectious_asymptomatic_l(location, p, l, l_v, k, k_v, num_agebrackets, isSave=True)

# plot_new_case_infectious_by_location_l(location, p, l, l_v, k, k_v, num_agebrackets, isSave=True)
# plot_new_case_infectious_by_location_k(location, p, l, l_v, k, k_v, num_agebrackets, isSave=True)


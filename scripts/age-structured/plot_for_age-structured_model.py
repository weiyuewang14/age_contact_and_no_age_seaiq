import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sbn
import pandas as pd
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import cmocean
import matplotlib as mplt

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

# path to the origin_source subsubdirectory
origin_source_dir = os.path.join(datadir, 'origin_resource')


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

# %% 读取接触矩阵
def read_contact_matrix(location, setting, num_agebrackets=85):
    """
    Read in the contact for each setting.

    Args:
        location (str)        : name of the location
        setting (str)         : name of the contact setting
        num_agebrackets (int) : the number of age brackets for the matrix

    Returns:
        A numpy matrix of contact.
    """
    setting_type, setting_suffix = 'F', 'setting'
    if setting == 'overall':
        setting_type, setting_suffix = 'M', 'contact_matrix'

    file_name = 'China_subnational_' + location + '_' + setting_type + '_' + setting + '_' + setting_suffix + '_' + '%i' % num_agebrackets + '.csv'
    file_path = os.path.join(datadir, 'origin_resource', 'contact_matrices', file_name)
    return np.loadtxt(file_path, delimiter=',')

# %% 获取接触矩阵字典
def get_contact_matrix_dic(location, num_agebrackets=85):
    """
    Get a dictionary of the setting contact matrices for a location.

    Args:
        location (str)        : name of the location
        country (str)         : name of the country
        level (str)           : name of level (country or subnational)
        num_agebrackets (int) : the number of age brackets for the matrix

    Returns:
        dict: A dictionary of the setting contact matrices for the location.
    """
    settings = ['household', 'school', 'work', 'community']
    matrix_dic = {}
    for setting in settings:
        matrix_dic[setting] = read_contact_matrix(location, setting, num_agebrackets)
    return matrix_dic

# %% 混合矩阵
def combine_synthetic_matrices(contact_matrix_dic, weights, num_agebrackets=85):
    """
    A linear combination of contact matrices for different settings to create an overall contact matrix given the weights for each setting.

    Args:
        contact_matrix_dic (dict) : a dictionary of contact matrices for different settings of contact. All setting contact matrices must be square and have the dimensions (num_agebrackets, num_agebrackets).
        weights (dict)            : a dictionary of weights for each setting of contact
        num_agebrackets (int)     : the number of age brackets of the contact matrix

    Returns:
        np.ndarray: A linearly combined overall contact matrix.
    """
    contact_matrix = np.zeros((num_agebrackets, num_agebrackets))
    for setting in weights:
        contact_matrix += contact_matrix_dic[setting] * weights[setting]
    return contact_matrix

def normalized(matrix):
    for a in range(len(matrix[0])):
        if np.sum(matrix[a, :]) != 0:
            matrix[a, :] = matrix[a, :]/np.sum(matrix[a, :])
    matrix = matrix/np.sum(matrix)
    return matrix

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

def get_ages1(location, country, level, num_agebrackets=18):
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
    if country == location:
        level = 'country'

    if country == 'Europe':
        country = location
        level = 'country'

    if level == 'country':
        file_name = country + '_' + level + '_level_age_distribution_' + '%i' % num_agebrackets + '.csv'
    else:
        file_name = country + '_' + level + '_' + location + '_age_distribution_' + '%i' % num_agebrackets + '.csv'
    file_path = os.path.join(origin_source_dir, 'age_distributions', file_name)

    ages = np.loadtxt(file_path, delimiter=',')
    # df = pd.read_csv(file_path, delimiter=',', header=None)
    # df.columns = ['age', 'age_count']
    # ages = dict(zip(df.age.values.astype(int), df.age_count.values))
    return ages

# 刻度标签数组
def get_label(num_agebrackets=18):
    if num_agebrackets == 85:
        age_brackets = [str(i) + '-' + str((i + 1)) for i in range(0, 84)] + ['85+']
    elif num_agebrackets == 18:
        age_brackets = [str(5 * i) + '-' + str(5 * (i + 1) - 1) for i in range(17)] + ['85+']
    return age_brackets


def read_data(location, state, p, p_v, l, l_v, k, k_v, num_agebrackets=18):
    file_name = f"{location}_{state}_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', p + "_" + str(p_v), file_name)

    M = np.loadtxt(file_path, delimiter=',')
    return M


def plot_all_states_age(all_states_age_data, total_population, location, p, p_v, l, l_v, k, k_v, isSave=False):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(8, 5))
    plt.subplots_adjust(left=0.12, right=0.9, bottom=0.15, top=0.90, wspace=0.1, hspace=0.1)
    for i in range(8):
        axes.plot(all_states_age_data[:700, 0], all_states_age_data[:700, i + 1] / total_population)

    axes.set_title('SEAIQ模型', fontsize=22)
    axes.set_xlabel('时间（天）', labelpad=10, fontsize=20)
    axes.set_ylabel('数量（%）', labelpad=15, fontsize=20)
    plt.rcParams.update({'font.size': 12})

    axins = inset_axes(axes, width=2.0, height=1.1, loc=5)

    axins.plot(all_states_age_data[100:100, 0], all_states_age_data[[0 for i in range(100, 100)], 0])
    for i in range(2, 8):
        axins.plot(all_states_age_data[200:600, 0], all_states_age_data[200:600, i] / total_population)

    axes.legend(['易感染者', '未接种的暴露者', '已接种的暴露者', '未接种的无症状感染者', '已接种的无症状感染者', '未接种的有症状感染者', '已接种的有症状感染者', '隔离者'])
    # plt.show()

    # 是否保存
    if (isSave):
        file_name = f'{location}_all_states_numbers_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.pdf'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'{location}_all_states_numbers_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.eps'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'{location}_all_states_numbers_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.svg'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')


def read_sus(location, state, p, p_v, l, l_v, k, k_v, num_agebrackets=18):
    file_name = f"{location}_{state}_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', p + "_" + str(p_v), file_name)

    M = np.loadtxt(file_path, delimiter=',')
    return M


def plot_sus(location, state, ages, p, p_v, l, l_v, k, k_v, isSave=False):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(6, 5))
    plt.subplots_adjust(left=0.16, right=0.95, bottom=0.2, top=0.90, wspace=0.1, hspace=0.1)

    labels = get_label(num_agebrackets)
    intial_list = np.zeros(num_agebrackets)
    final_list = np.zeros(num_agebrackets)
    for age in ages:
        file_name = f"{location}_{state}_numbers_{num_agebrackets}_age={age}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
        file_path = os.path.join(output_source_dir, 'age-structured_output_source', 'bracket', p + "_" + str(p_v), location, file_name)
        intial_final_sus = np.loadtxt(file_path, delimiter=',', usecols=1)
        intial_list[age] = intial_final_sus[0]
        final_list[age] = intial_final_sus[-1]

    axes.set_xticklabels(labels, rotation=60)
    up = intial_list - final_list
    axes.bar(labels, final_list, color='#db8231', label="最终易感染者")
    # 第二根柱子“堆积”在第一根柱子上方，通过'bottom'调整，显示第二种产品的销量
    axes.bar(labels, up, bottom=final_list, color='#4c92c3', label="初始易感染者")

    axes.set_xlabel('年龄组', labelpad=10, fontsize=16)
    axes.set_ylabel('数量（个）', labelpad=15, fontsize=16)
    plt.rcParams.update({'font.size': 12})

    axes.legend()
    # plt.show()

    # 是否保存
    if (isSave):
        file_name = f'{location}_sus_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.pdf'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'{location}_sus_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.eps'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'{location}_sus_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.svg'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')

def plot_location_contact_and_distribution(location, weights, cmap=cmocean.cm.deep_r, num_agebrackets=18, isSave=False):
    if isinstance(cmap, str):
        cmap = mplt.cm.get_cmap(cmap)

    fontsizes = {'colorbar': 18, 'colorbarlabels': 10, 'title': 20, 'ylabel': 16, 'xlabel': 16, 'xticks': 10, 'yticks': 10}
    settings_dic = {'household': '家庭', 'school': '学校', 'work': "工作场所", 'community': "其他", 'overall': '总体', 'distribution': '年龄分布'}
    # settings_dic = {'overall': '', 'household': '家  庭', 'school': '学  校', 'work': "工作场所", 'community': "其  他"}

    # defaults without colorbar label
    left = 0.1
    right = 0.99
    top = 0.95
    bottom = 0.1
    age_brackets = get_label(num_agebrackets)

    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(9, 6))
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=0.16, hspace=0.05)

    min_CM = 1e-4
    max_CM = 1e-1


    axes = axes.flatten()
    for setting, ax, num in zip(settings_dic, axes, [i for i in range(6)]):

        print(num, ax)
        if setting == 'combine':
            contact_matrix_dic = get_contact_matrix_dic(location, num_agebrackets)
            matrix = combine_synthetic_matrices(contact_matrix_dic, weights, num_agebrackets)
            matrix = normalized(matrix)
        elif setting == 'distribution':
            matrix = matrix
        else:
            matrix = read_contact_matrix(location, setting, num_agebrackets)
            matrix = normalized(matrix)
        im = ax.imshow(matrix.T, origin='lower', interpolation='nearest', cmap=cmap, norm=LogNorm(vmin=min_CM, vmax=max_CM))
        if setting == 'combine':
            ax.set_title(location_dic[location], fontsize=16)
        else:
            ax.set_title(settings_dic[setting], fontsize=16)
        if num == 3 or num == 4:
            ax.set_xlabel('年龄组', fontsize=14)
        if num == 0 or num == 3:
            ax.set_ylabel('接触的年龄组', fontsize=14)
        ax.set_xticks(np.arange(len(age_brackets)))
        ax.set_xticklabels(age_brackets, rotation=60)

        ax.set_yticks(np.arange(len(age_brackets)))
        ax.set_yticklabels(age_brackets)
        ax.tick_params(labelsize=7)
    plt.delaxes(axes[-1])
    for i, cot in zip([i for i in range(5)], ["A", "B", "C", "D", "E"]):
        axes[i].text(-3, 20, cot, fontsize=16)


    # fig.add_subplot(2, 3, 6)
    # fig.subplot('Position', [0.45,0.1,0.1,0.1])
    axes6 = plt.axes([0.598, 0.15, 0.215, 0.319])
    axes6.set_ylabel('人数（%）', fontsize=12)
    #  年龄分布
    ages = get_ages1(location, country, level, num_agebrackets)
    data = ages.T
    total_population = data.sum()
    print("总人数：", total_population)

    axes6.bar(data[0], data[1]/total_population * 100)
    axes6.plot(data[0], data[1]/total_population * 100, color='r')
    axes6.set_xticks(np.arange(len(age_brackets)))
    axes6.set_xticklabels(age_brackets, rotation=60)
    axes6.tick_params(labelsize=7)
    axes6.yaxis.tick_right()
    axes6.set_ylim((0, 13))
    axes6.set_title("年龄分布", fontsize=16)
    axes6.set_xlabel("年龄组", fontsize=14)

    axes6.text(-3, 15, "F", fontsize=16)

    cb = fig.colorbar(im, ax=axes, label='接触频率')
    cb.set_label('接触频率', fontsize=16)

    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # plt.show()

    if isSave:
        file_name = f'{location}_{num_agebrackets}_contact_matrix_and_age_distribution.pdf'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'{location}_{num_agebrackets}_contact_matrix_and_age_distribution.eps'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'{location}_{num_agebrackets}_contact_matrix_and_age_distribution.svg'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='svg')

location_dic = {'Hubei': '湖北省', 'Shanghai': '上海市', 'Jiangsu': '江苏省', 'Beijing': '北京市', 'Tianjin': '天津市',
                'Guangdong': '广东省'}
weights1 = {'household': 4.11, 'school': 11.41, 'work': 8.07, 'community': 2.79}
weights2 = {'household': 0.31, 'school': 0.24, 'work': 0.16, 'community': 0.29}

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
beta_v = 0.26  # 感染率
q_v = 0.5  # 有症状的概率
l_v = 0.8  # 0-1 接种导致有症状概率降低系数
sigma_inverse = 5.2  # mean latent period
k_v = 0.5  # 0-1 无症状被发现的概率降低系数
gamma_inverse = 1.6  # mean quarantined period
l_value = [i / 10 for i in range(0, 11)]
k_value = [i / 10 for i in range(0, 11)]

ages = get_ages(location, country, level, num_agebrackets)
# total_population = sum(ages.values())
#
# all_states_age_data = read_data(location, state, p, p_v, l, l_v, k, k_v, num_agebrackets)
# plot_all_states_age(all_states_age_data, total_population, location, p, p_v, l, l_v, k, k_v, isSave=True)

# plot_sus(location, state, ages, p, p_v, l, l_v, k, k_v, isSave=True)
plot_location_contact_and_distribution(location, weights=weights2, cmap=cmocean.cm.deep_r, num_agebrackets=18, isSave=True)

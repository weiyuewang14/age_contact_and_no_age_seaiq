import csv
import matplotlib as mplt
import matplotlib.pyplot as plt
import cmocean
import numpy as np
import matplotlib.cm as cm
import copy
import os
import pandas as pd
import datetime
import seaborn as sbn
import cmocean
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib import font_manager

# from pylab import *
from matplotlib.font_manager import FontProperties


# print(mplt.matplotlib_fname())

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
resultsdir = os.path.join(maindir, 'results')

# path to the data subdirectory
datadir = os.path.join(maindir, 'data')

# path to the output_source subsubdirectory
output_source_dir = os.path.join(datadir, 'output_source')

# path to the origin_source subsubdirectory
origin_source_dir = os.path.join(datadir, 'origin_resource')


# 反转矩阵，或者使用 np.flipud
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

# 刻度标签数组
def get_label(num_agebrackets=18):
    if num_agebrackets == 85:
        age_brackets = [str(i) + '-' + str((i + 1)) for i in range(0, 84)] + ['85+']
    elif num_agebrackets == 18:
        age_brackets = [str(5 * i) + '-' + str(5 * (i + 1) - 1) for i in range(17)] + ['85+']
    return age_brackets

# 地区接触矩阵
def plot_contact_matrix(location, location_dic, setting, weights, cmap=cmocean.cm.deep_r, num_agebrackets=18, isSave=False):
    """

    Args:
        location (str): name of the location
        matrix_type (str) : the type of contact matrix, either 'Model' for the overall contact matrix from a linear combination of the synthetic setting contact matrices, or 'Survey' for the empirical survey matrix.
        cmap (str or matplotlib.colors.LinearSegmentedColormap) : name of the colormap to use
        isSave (bool)                                             : If True, save the figure

    Returns:
        Matplotlib figure.
    """

    if isinstance(cmap, str):
        cmap = mplt.cm.get_cmap(cmap)

    fontsizes = {'colorbar': 18, 'colorbarlabels': 10, 'title': 20, 'ylabel': 16, 'xlabel': 16, 'xticks': 10, 'yticks': 10}
    settings_dic = {'overall': '总体', 'household': '：家庭', 'school': '：学校', 'work': "：工作场所", 'community': "：其他"}
    # settings_dic = {'overall': '', 'household': '家  庭', 'school': '学  校', 'work': "工作场所", 'community': "其  他"}

    # defaults without colorbar label
    left = 0.175
    right = 0.935
    top = 0.98
    bottom = 0.08

    # move margins in a little
    left -= 0.00
    right -= 0.09
    top -= 0.01
    bottom += 0.00



    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    ax = fig.add_subplot(111)


    if setting == 'combine':
        contact_matrix_dic = get_contact_matrix_dic(location, num_agebrackets)
        matrix = combine_synthetic_matrices(contact_matrix_dic, weights, num_agebrackets)
        matrix = normalized(matrix)
    else:
        matrix = read_contact_matrix(location, setting, num_agebrackets)
        matrix = normalized(matrix)


    min_CM = 1e-1
    max_CM = 1e-4
    im = ax.imshow(matrix.T, origin='lower', interpolation='nearest', cmap=cmap, norm=LogNorm(vmin=min_CM, vmax=max_CM))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.15)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=fontsizes['colorbarlabels'])
    cbar.ax.set_ylabel('接触频率', fontsize=fontsizes['ylabel'])
    if setting == 'combine':
        ax.set_title(location_dic[location] + '接触矩阵', fontsize=fontsizes['title'])
    else:
        ax.set_title(location_dic[location] + settings_dic[setting], fontsize=fontsizes['title'])
    ax.set_ylabel('接触的年龄组', fontsize=fontsizes['ylabel'])
    ax.set_xlabel('年龄组', fontsize=fontsizes['xlabel'])

    age_brackets = get_label(num_agebrackets)

    ax.set_xticks(np.arange(len(age_brackets)))
    ax.set_xticklabels(age_brackets, rotation=60)
    ax.set_yticks(np.arange(len(age_brackets)))
    ax.set_yticklabels(age_brackets)
    ax.tick_params(labelsize=fontsizes['xticks'])

    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.show()

    if isSave:

        file_name = f'{location}_{setting}_contact_matrix_{num_agebrackets}.pdf'
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'contact_matrices', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'{location}_{setting}_contact_matrix_{num_agebrackets}.eps'
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'contact_matrices', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'{location}_{setting}_contact_matrix_{num_agebrackets}.svg'
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'contact_matrices', file_name)
        fig.savefig(fig_path, format='svg')

def plot_contact_matrix1(location_dic, setting, weights, cmap=cmocean.cm.deep_r, num_agebrackets=18, isSave=False):
    """

    Args:
        location (str): name of the location
        matrix_type (str) : the type of contact matrix, either 'Model' for the overall contact matrix from a linear combination of the synthetic setting contact matrices, or 'Survey' for the empirical survey matrix.
        cmap (str or matplotlib.colors.LinearSegmentedColormap) : name of the colormap to use
        isSave (bool)                                             : If True, save the figure

    Returns:
        Matplotlib figure.
    """

    if isinstance(cmap, str):
        cmap = mplt.cm.get_cmap(cmap)

    fontsizes = {'colorbar': 18, 'colorbarlabels': 10, 'title': 20, 'ylabel': 16, 'xlabel': 16, 'xticks': 10, 'yticks': 10}
    settings_dic = {'overall': '总体', 'household': '：家庭', 'school': '：学校', 'work': "：工作场所", 'community': "：其他"}
    # settings_dic = {'overall': '', 'household': '家  庭', 'school': '学  校', 'work': "工作场所", 'community': "其  他"}

    # defaults without colorbar label
    left = 0.1
    right = 0.99
    top = 0.95
    bottom = 0.1

    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(9, 6))
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=0.16, hspace=0.05)

    min_CM = 1e-4
    max_CM = 1e-1
    for ax, location, num in zip(axes.flat, location_dic, [i for i in range(6)]):
        if setting == 'combine':
            contact_matrix_dic = get_contact_matrix_dic(location, num_agebrackets)
            matrix = combine_synthetic_matrices(contact_matrix_dic, weights, num_agebrackets)
            matrix = normalized(matrix)
        else:
            matrix = read_contact_matrix(location, setting, num_agebrackets)
            matrix = normalized(matrix)

        im = ax.imshow(matrix.T, origin='lower', interpolation='nearest', cmap=cmap, norm=LogNorm(vmin=min_CM, vmax=max_CM))

        if setting == 'combine':
            ax.set_title(location_dic[location], fontsize=16)
        else:
            ax.set_title(location_dic[location] + settings_dic[setting], fontsize=fontsizes['title'])
        if num > 2:
            ax.set_xlabel('年龄组', fontsize=14)
        if num == 0 or num == 3:
            ax.set_ylabel('接触的年龄组', fontsize=14)



        age_brackets = get_label(num_agebrackets)

        ax.set_xticks(np.arange(len(age_brackets)))
        ax.set_xticklabels(age_brackets, rotation=60)
        ax.set_yticks(np.arange(len(age_brackets)))
        ax.set_yticklabels(age_brackets)
        ax.tick_params(labelsize=7)
    cb = fig.colorbar(im, ax=[ax for ax in axes.flatten()], label='接触频率')
    cb.set_label('接触频率', fontsize=16)

    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # plt.show()

    if isSave:

        file_name = f'location_{setting}_contact_matrix_{num_agebrackets}.pdf'
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'contact_matrices', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'location_{setting}_contact_matrix_{num_agebrackets}.eps'
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'contact_matrices', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'location_{setting}_contact_matrix_{num_agebrackets}.svg'
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'contact_matrices', file_name)
        fig.savefig(fig_path, format='svg')

# 地区年龄分布
def plot_age_distribution(location, location_dic, data, total_population, num_agebrackets, isSave=False):
    fontsizes = {'colorbar': 30, 'colorbarlabels': 22, 'title': 44, 'ylabel': 28, 'xlabel': 28, 'xticks': 24, 'yticks': 24}

    # defaults without colorbar label
    left = 0.155
    right = 0.935
    top = 0.91
    bottom = 0.12

    # move margins in a little
    left -= 0.04
    right -= 0.02
    top -= 0.00
    bottom += 0.03



    fig = plt.figure(figsize=(5, 5))
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    ax = fig.add_subplot(111)
    plt.bar(data[0], data[1]/total_population*100)
    plt.plot(data[0], data[1]/total_population*100, color='r')

    age_brackets = get_label(num_agebrackets)
    ax.set_xticks(np.arange(len(age_brackets)))
    ax.set_xticklabels(age_brackets, rotation=60)
    ax.set_ylim((0, 13))
    ax.set_title(location_dic[location] + '年龄分布', fontsize=20)
    ax.set_xlabel("年龄组", fontsize=16)
    ax.set_ylabel("人数（%）", fontsize=16)

    # plt.show()

    if isSave:
        file_name = f'{location}_{num_agebrackets}_age_distribution.pdf'
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'age_distributions', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'{location}_{num_agebrackets}_age_distribution.eps'
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'age_distributions', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'{location}_{num_agebrackets}_age_distribution.svg'
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'age_distributions', file_name)
        fig.savefig(fig_path, format='svg')

def plot_age_distribution1(location_dic, num_agebrackets, isSave=False):
    fontsizes = {'colorbar': 30, 'colorbarlabels': 22, 'title': 44, 'ylabel': 28, 'xlabel': 28, 'xticks': 24, 'yticks': 24}

    # defaults without colorbar label
    left = 0.09
    right = 0.95
    top = 0.9
    bottom = 0.12

    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(9, 6))
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=0.2, hspace=0.22)

    age_brackets = get_label(num_agebrackets)
    for ax, loc, num in zip(axes.flat, location_dic, [i for i in range(6)]):
        ages = get_ages(loc, country, level, num_agebrackets)
        data = ages.T
        total_population = data[1].sum()
        print("总人数：", total_population)

        ax.bar(data[0], data[1]/total_population*100)
        ax.plot(data[0], data[1]/total_population*100, color='r')


        ax.set_xticks(np.arange(len(age_brackets)))
        ax.set_xticklabels(age_brackets, rotation=60)
        ax.tick_params(labelsize=7)
        ax.set_ylim((0, 13))
        ax.set_title(location_dic[loc], fontsize=16)
        if num > 2:
            ax.set_xlabel("年龄组", fontsize=14)
        if num == 0 or num == 3:
            ax.set_ylabel("人数（%）", fontsize=14)

    # plt.show()

    if isSave:
        file_name = f'location_{num_agebrackets}_age_distribution.pdf'
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'age_distributions', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'location_{num_agebrackets}_age_distribution.eps'
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'age_distributions', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'location_{num_agebrackets}_age_distribution.svg'
        fig_path = os.path.join(resultsdir, 'age-structured_result', 'age_distributions', file_name)
        fig.savefig(fig_path, format='svg')

location_dic= {'Jiangsu':'江苏省', 'Shanghai':'上海市', 'Guangdong':'广东省', 'Beijing':'北京市', 'Tianjin':'天津市', 'Hubei':'湖北省'}
weights1 = {'household': 4.11, 'school': 11.41, 'work': 8.07, 'community': 2.79}
weights2 = {'household': 0.31, 'school': 0.24, 'work': 0.16, 'community': 0.29}
settings = ['household', 'school', 'work', 'community']
# location = 'Shanghai'
country = 'China'
level = 'subnational'
num_agebrackets = 18

"""   年龄分布   """

for loc in location_dic:
    ages = get_ages(loc, country, level, num_agebrackets)
    data = ages.T
    total_population = data[1].sum()
    print("总人数：", total_population)
    # plot_age_distribution(loc, location_dic, data, total_population, num_agebrackets, isSave=True)
# plot_age_distribution1(location_dic, num_agebrackets, isSave=True)

def normalized(matrix):
    for a in range(len(matrix[0])):
        if np.sum(matrix[a, :]) != 0:
            matrix[a, :] = matrix[a, :]/np.sum(matrix[a, :])
    matrix = matrix/np.sum(matrix)
    return matrix

"""   合成矩阵   """
# contact_matrix_dic = get_contact_matrix_dic("Hubei", num_agebrackets)
# contact1 = combine_synthetic_matrices(contact_matrix_dic, weights1, num_agebrackets)
# contact2 = combine_synthetic_matrices(contact_matrix_dic, weights2, num_agebrackets)
# overall_contact = read_contact_matrix("Hubei", 'overall', num_agebrackets)
# print("合成1：", normalized(contact1))
# print("合成2：", normalized(contact2))
# print("overall", normalized(overall_contact))

"""   接触矩阵   """

# for loc in location_dic:
#     plot_contact_matrix(loc, location_dic, setting="combine", weights=weights2, num_agebrackets=18, isSave=True)
# plot_contact_matrix1(location_dic, setting="combine", weights=weights2, num_agebrackets=18, isSave=True)
# for setting in settings:
#     plot_contact_matrix('Shanghai', location_dic, setting="overall", weights=weights2, num_agebrackets=18, isSave=True)
"""
location_list = ['Hubei', 'Shanghai', 'Jiangsu', 'Beijing', 'Tianjin', 'Guangdong']
location = 'Shanghai'
country = 'China'
level = 'subnational'
num_agebrackets = 18  # number of age brackets for the contact matrices

weights = {'household': 4.11, 'school': 11.41, 'workplace': 8.07,
           'community': 2.79}  # effective setting weights as found in our study

contact_matrix_dic = get_contact_matrix_dic(location, country, level, num_agebrackets)
# print(contact_matrix_dic)
combine_synthetic_matric = combine_synthetic_matrices(contact_matrix_dic, weights, num_agebrackets)

# print(combine_synthetic_matric)
label = []

if num_agebrackets == 85:
    age_brackets = [str(i) + '-' + str((i + 1)) for i in range(0, 85)] + ['85+']
elif num_agebrackets == 18:
    age_brackets = [str(5 * i) + '-' + str(5 * (i + 1) - 1) for i in range(18)] + ['85+']

fig = plt.figure(figsize=(5, 5))
plt.title('Synthetic', fontsize=20)

y_label = np.flipud(age_brackets)
x_label = age_brackets

sbn.heatmap(swap(combine_synthetic_matric), cmap='GnBu', yticklabels=y_label, xticklabels=x_label)

plt.xlabel('Age', labelpad=15)
plt.ylabel('Age of contact', labelpad=20)
plt.show()

fig_name = 'combine_synthetic_matrix_18.pdf'
fig_path = os.path.join(resultsdir, 'age-structured_result', fig_name)
fig.savefig(fig_path, format='pdf')
fig_name = 'combine_synthetic_matrix_18.eps'
fig_path = os.path.join(resultsdir, 'age-structured_result', fig_name)
fig.savefig(fig_path, format='eps')

settings = ['household', 'school', 'workplace', 'community']

for setting in settings:
    matrix = contact_matrix_dic[setting]
    fig = plt.figure(figsize=(5, 5))
    if setting == "community":
        setting = 'others'
    plt.title(setting.capitalize(), fontsize=20)
    y_label = np.flipud(age_brackets)
    x_label = age_brackets

    sbn.heatmap(swap(matrix), cmap='GnBu', yticklabels=y_label, xticklabels=x_label)

    plt.xlabel('Age', labelpad=15)
    plt.ylabel('Age of contact', labelpad=50)
    plt.show()
    fig_name = setting.capitalize() + '_contact_matrix_18.pdf'
    fig_path = os.path.join(resultsdir, 'age-structured_result', fig_name)
    fig.savefig(fig_path, format='pdf')
    fig_name = setting.capitalize() + '_contact_matrix_18.eps'
    fig_path = os.path.join(resultsdir, 'age-structured_result', fig_name)
    fig.savefig(fig_path, format='eps')

"""

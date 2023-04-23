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


# 每个地区的累计发病率
def plot_cumulative_incidence_by_location(location_dic, p, p_v, l, l_v, k, k_v, num_agebrackets, isSave=True):

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(5, 4))
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.92, wspace=0.1, hspace=0.1)

    for location in location_dic:
        file_name = f"{location}_cumulative_incidence_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
        file_path = os.path.join(output_source_dir, 'age-structured_output_source', file_name)

        data = np.loadtxt(file_path, delimiter=',').T
        axes.plot(data[0], data[1], label=location_dic[location])

    axes.set_ylabel('累计发病率（%）', labelpad=10, fontsize=14)
    axes.set_xlabel('时间（天）', labelpad=10, fontsize=14)
    plt.ylim(0, 100)
    plt.xlim(0, 700)
    plt.legend(fontsize=12)

    # plt.show()
    if isSave:
        file_name = f'cumulative_incidence_by_location_{p}_{p_v}.pdf'
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'cumulative_incidence_by_location_{p}_{p_v}.eps'
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'cumulative_incidence_by_location_{p}_{p_v}.svg'
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='svg')
"""
def plot_infectious_by_location(location_dic, p, p_v, l, l_v, k, k_v, num_agebrackets, isSave=True):

    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(8, 3))
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.18, top=0.90, wspace=0.1, hspace=0.1)
    for ax, p_v in zip(axes.flat, [0.1, 0.5, 0.9]):
        
        for location in location_dic:
            ages = get_ages(location, country, level, num_agebrackets)
            total_p = sum(ages.values())
            file_name = f"{location}_all_states_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
            file_path = os.path.join(output_source_dir, 'age-structured_output_source', f'{p}_{p_v}', file_name)

            data1 = np.loadtxt(file_path, delimiter=',', usecols=0).T
            data = np.loadtxt(file_path, delimiter=',', usecols=(4, 5, 6, 7)).sum(axis=1).T / total_p * 100
            ax.plot(data1, data, label=location)
            ax.set_xlabel('时间（天）', labelpad=10, fontsize=15)

    axes[0].set_ylabel('感染者（%）', labelpad=10, fontsize=15)

    plt.ylim(0, 10)
    plt.xlim(0, 400)
    plt.legend()

    # plt.show()
    if isSave:
        file_name = f'infectious_by_location_{p}_{p_v}.pdf'
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'infectious_by_location_{p}_{p_v}.eps'
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'infectious_by_location_{p}_{p_v}.svg'
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='svg')
"""
def plot_infectious_by_location(location_dic, p, p_v, l, l_v, k, k_v, num_agebrackets, isSave=True):

    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(8, 3))
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.2, top=0.80, wspace=0.1, hspace=0.1)

    lines = []
    labels = []
    for ax, p_v in zip(axes.flat, [0.9, 0.5, 0.1]):
        ax.set_title(f'未接种率 p = {p_v}')

        for location in location_dic:
            ages = get_ages(location, country, level, num_agebrackets)
            total_p = sum(ages.values())
            file_name = f"{location}_all_states_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
            file_path = os.path.join(output_source_dir, 'age-structured_output_source', f'{p}_{p_v}', location, file_name)

            data1 = np.loadtxt(file_path, delimiter=',', usecols=0).T
            data = np.loadtxt(file_path, delimiter=',', usecols=(4, 5, 6, 7)).sum(axis=1).T / total_p * 100
            ax.plot(data1, data, label=location_dic[location])
            ax.set_xlabel('时间（天）', labelpad=10, fontsize=15)

        if p_v == 0.1:
            axLine, axLabel = ax.get_legend_handles_labels()
            lines.extend(axLine)
            labels.extend(axLabel)

    axes[0].set_ylabel('感染者（%）', labelpad=10, fontsize=15)
    fig.legend(lines, labels, loc='upper right', shadow=False, frameon=False, ncol=6)
    plt.ylim(0, 6)
    plt.xlim(0, 400)


    # plt.show()
    if isSave:
        file_name = f'infectious_by_location.pdf'
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'infectious_by_location.eps'
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'infectious_by_location.svg'
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='svg')

def plot_asymptomatic_by_location(location_dic, p, p_v, l, l_v, k, k_v, num_agebrackets, isSave=True):

    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(8, 5))
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.18, top=0.90, wspace=0.1, hspace=0.1)
    for ax in axes:
        for location in location_dic:
            file_name = f"{location}_cumulative_incidence_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
            file_path = os.path.join(output_source_dir, 'age-structured_output_source', file_name)

            data = np.loadtxt(file_path, delimiter=',').T
            axes.plot(data[0], data[1], label=location)

    axes.set_ylabel('无症状感染者（%）', labelpad=10, fontsize=15)
    axes.set_xlabel('时间（天）', labelpad=10, fontsize=15)
    plt.ylim(0, 100)
    plt.xlim(0, 300)
    plt.legend()

    # plt.show()
    if isSave:
        file_name = f'cumulative_incidence_by_location_{p}_{p_v}.pdf'
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'cumulative_incidence_by_location_{p}_{p_v}.eps'
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'cumulative_incidence_by_location_{p}_{p_v}.svg'
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='svg')

def plot_asymptomatic_by_location_k(location_dic, p, p_v, l, l_v, k, k_v, num_agebrackets, isSave=True):

    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(9, 5))
    plt.subplots_adjust(left=0.1, right=0.87, bottom=0.12, top=0.9, wspace=0.2, hspace=0.3)

    lines = []
    labels = []
    for ax, location in zip(axes.flat, location_dic):
        ax.set_title(location_dic[location], fontsize=16)
        total_population = sum(get_ages(location, country, level, num_agebrackets).values())
        for k_v in range(11):
            k_v = k_v/10
            file_name = f"{location}_all_states_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
            file_path = os.path.join(output_source_dir, 'age-structured_output_source', f'{p}_{p_v}', location, file_name)

            data = np.loadtxt(file_path, delimiter=',', usecols=(0, 4, 5, 6, 7)).T
            ax.plot(data[0], data[1:].sum(axis=0)/total_population, label=f'{k}={k_v}')
        if(location == 'Shanghai'):
            axLine, axLabel = ax.get_legend_handles_labels()
            lines.extend(axLine)
            labels.extend(axLabel)

    fig.legend(lines, labels, loc='right', shadow=False, frameon=False, ncol=1)
    fig.text(0.5, 0.05, '时间（天）', va='center', fontsize=14)
    fig.text(0.01, 0.5, '无症状感染者（%）', va='center', fontsize=14, rotation='vertical')
    # plt.ylim(0, 1)
    plt.xlim(0, 700)

    # plt.show()
    if isSave:
        file_name = f'asymptomatic_by_location_{p}_{p_v}_{l}_{l_v}_{k}.pdf'
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f'asymptomatic_by_location_{p}_{p_v}_{l}_{l_v}_{k}.eps'
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='eps')

        file_name = f'asymptomatic_by_location_{p}_{p_v}_{l}_{l_v}_{k}.svg'
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

plot_cumulative_incidence_by_location(location_dic, p, p_v, l, l_v, k, k_v, num_agebrackets, isSave=True)

# plot_infectious_by_location(location_dic, p, p_v, l, l_v, k, k_v, num_agebrackets, isSave=True)

# plot_asymptomatic_by_location_k(location_dic, p, p_v, l, l_v, k, k_v, num_agebrackets, isSave=True)
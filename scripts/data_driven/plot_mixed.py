import csv

import numpy as np
import copy
import os
from decimal import Decimal
import string
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
# import cartopy.crs as ccrs

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

# def read_cumulative_incidence(age_type, p, p_v, l, l_v, k, k_v):

"""  无年龄结构  """


def plot_no_age_cumulative_incidence(p, p_v, l, l_v, k, k_v, isSave=True):
    file1_name = f"cumulative_incidence_{p}={0.1}_{l}_{l_v:.1f}_{k}.csv"
    file1_path = os.path.join(output_source_dir, 'no_age-structured_output_source', file1_name)
    M1 = np.loadtxt(file1_path, delimiter=',', usecols=int(k_v * 10))
    file2_name = f"cumulative_incidence_{p}={0.5}_{l}_{l_v:.1f}_{k}.csv"
    file2_path = os.path.join(output_source_dir, 'no_age-structured_output_source', file2_name)
    M2 = np.loadtxt(file2_path, delimiter=',', usecols=int(k_v * 10))
    file3_name = f"cumulative_incidence_{p}={0.9}_{l}_{l_v:.1f}_{k}.csv"
    file3_path = os.path.join(output_source_dir, 'no_age-structured_output_source', file3_name)
    M3 = np.loadtxt(file3_path, delimiter=',', usecols=int(k_v * 10))

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(8, 5))
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.18, top=0.80, wspace=0.1, hspace=0.1)
    # plt.suptitle("infection ratio", fontsize=20)
    # axes.set_xlabel('k', labelpad=15, fontsize=15)
    axes.set_ylabel('Cumulative incidence(%)', labelpad=10, fontsize=15)
    axes.set_xlabel('Time', labelpad=10, fontsize=15)
    plt.ylim(0, 100)
    plt.plot(M1[:400])
    plt.plot(M2[:400])
    plt.plot(M3[:400])
    plt.legend(["p=0.1", "p=0.5", "p=0.9"])

    plt.show()
    if isSave:
        file_name = 'cumulative_incidence_no_age' + '_' + p + '.pdf'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = 'cumulative_incidence_no_age' + '_' + p + '.eps'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')


def plot_no_age_new_case_per_day(p, p_v, l, l_v, k, k_v, isSave=True):
    file1_name = f"new_case_per_day_no_age_{p}={0.1:.2f}_{l}={l_v:.2f}_{k}={k_v:0.2f}.csv"
    file1_path = os.path.join(output_source_dir, 'no_age-structured_output_source', file1_name)
    M1 = np.loadtxt(file1_path, delimiter=',', usecols=(4, 5, 6, 7)).sum(axis=1)

    file2_name = f"new_case_per_day_no_age_{p}={0.5:.2f}_{l}={l_v:.2f}_{k}={k_v:0.2f}.csv"
    file2_path = os.path.join(output_source_dir, 'no_age-structured_output_source', file2_name)
    M2 = np.loadtxt(file2_path, delimiter=',', usecols=(4, 5, 6, 7)).sum(axis=1)

    file3_name = f"new_case_per_day_no_age_{p}={0.9:.2f}_{l}={l_v:.2f}_{k}={k_v:0.2f}.csv"
    file3_path = os.path.join(output_source_dir, 'no_age-structured_output_source', file3_name)
    M3 = np.loadtxt(file3_path, delimiter=',', usecols=(4, 5, 6, 7)).sum(axis=1)
    # print(M1)

    max_idx1 = np.argmax(np.array(M1))
    max_v1 = max(np.array(M1))

    max_idx2 = np.argmax(np.array(M2))
    max_v2 = max(np.array(M2))

    max_idx3 = np.argmax(np.array(M3))
    max_v3 = max(np.array(M3))

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(8, 5))
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.18, top=0.80, wspace=0.1, hspace=0.1)
    plt.plot(M1[:400], color="b")

    plt.plot(M2[:400], color="g")

    plt.plot(M3[:400], color="r")
    plt.plot([0, max_idx2], [max_v2, max_v2], color="g", linestyle="--")
    plt.plot([max_idx2, max_idx2], [0, max_v2], color="g", linestyle="--")
    plt.plot([0, max_idx1], [max_v1, max_v1], color="b", linestyle="--")
    plt.plot([max_idx1, max_idx1], [0, max_v1], color="b", linestyle="--")
    plt.plot([0, max_idx3], [max_v3, max_v3], color="r", linestyle="--")
    plt.plot([max_idx3, max_idx3], [0, max_v3], color="r", linestyle="--")

    axes.set_ylabel('New cases per day', labelpad=10, fontsize=15)
    axes.set_xlabel('Time', labelpad=10, fontsize=15)
    plt.legend(["p=0.1", "p=0.5", "p=0.9"])
    plt.show()

    if isSave:
        file_name = 'new_cases_per_day_no_age' + '_' + p + '.pdf'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = 'new_cases_per_day_no_age' + '_' + p + '.eps'
        fig_path = os.path.join(resultsdir, 'no_age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')


"""  有年龄结构  """

def plot_age_cumulative_incidence(location, country, state, p, p_v, l, l_v, k, k_v, isSave=True):
    file1_name = f"cumulative_incidence_{p}={0.1}_{l}_{l_v:.1f}_{k}.csv"
    file1_path = os.path.join(output_source_dir, 'age_structured_output_source', file1_name)
    M1 = np.loadtxt(file1_path, delimiter=',', usecols=int(k_v * 10))
    file2_name = f"cumulative_incidence_{p}={0.5}_{l}_{l_v:.1f}_{k}.csv"
    file2_path = os.path.join(output_source_dir, 'age_structured_output_source', file2_name)
    M2 = np.loadtxt(file2_path, delimiter=',', usecols=int(k_v * 10))
    file3_name = f"cumulative_incidence_{p}={0.9}_{l}_{l_v:.1f}_{k}.csv"
    file3_path = os.path.join(output_source_dir, 'age_structured_output_source', file3_name)
    M3 = np.loadtxt(file3_path, delimiter=',', usecols=int(k_v * 10))

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(8, 5))
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.18, top=0.80, wspace=0.1, hspace=0.1)
    # plt.suptitle("infection ratio", fontsize=20)
    # axes.set_xlabel('k', labelpad=15, fontsize=15)
    axes.set_ylabel('Cumulative incidence(%)', labelpad=10, fontsize=15)
    axes.set_xlabel('Time', labelpad=10, fontsize=15)
    plt.ylim(0, 100)
    plt.plot(M1[:400])
    plt.plot(M2[:400])
    plt.plot(M3[:400])
    plt.legend(["p=0.1", "p=0.5", "p=0.9"])

    plt.show()
    if isSave:
        file_name = 'cumulative_incidence' + '_' + p + '.pdf'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = 'cumulative_incidence' + '_' + p + '.eps'
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')


def plot_age_new_case_per_day(location, country, state, p, p_v, l, l_v, k, k_v, num_agebrackets=18, isSave=True):
    file1_name = f"{country}_{location}_new_case_per_day_{num_agebrackets}_{p}={0.1:.2f}_{l}={l_v:.2f}_{k}={k_v:0.2f}.csv"
    file1_path = os.path.join(output_source_dir, 'age-structured_output_source', f"{p}_{0.1:.1f}", file1_name)
    M1 = np.loadtxt(file1_path, delimiter=',', usecols=(5, 6, 7, 8)).sum(axis=1)

    file2_name = f"{country}_{location}_new_case_per_day_{num_agebrackets}_{p}={0.5:.2f}_{l}={l_v:.2f}_{k}={k_v:0.2f}.csv"
    file2_path = os.path.join(output_source_dir, 'age-structured_output_source', f"{p}_{0.5:.1f}", file2_name)
    M2 = np.loadtxt(file2_path, delimiter=',', usecols=(5,6,7,8)).sum(axis=1)

    file3_name = f"{country}_{location}_new_case_per_day_{num_agebrackets}_{p}={0.9:.2f}_{l}={l_v:.2f}_{k}={k_v:0.2f}.csv"
    file3_path = os.path.join(output_source_dir, 'age-structured_output_source', f"{p}_{0.9:.1f}", file3_name)
    M3 = np.loadtxt(file3_path, delimiter=',', usecols=(5,6,7,8)).sum(axis=1)
    # print(M1)

    max_idx1 = np.argmax(np.array(M1))
    max_v1 = max(np.array(M1))

    max_idx2 = np.argmax(np.array(M2))
    max_v2 = max(np.array(M2))

    max_idx3 = np.argmax(np.array(M3))
    max_v3 = max(np.array(M3))

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(8, 5))
    plt.subplots_adjust(left=0.12, right=0.95, bottom=0.18, top=0.80, wspace=0.1, hspace=0.1)
    plt.plot(M1[:80], color="b")

    plt.plot(M2[:80], color="g")

    plt.plot(M3[:80], color="r")
    plt.plot([0, max_idx2], [max_v2, max_v2], color="g", linestyle="--")
    plt.plot([max_idx2, max_idx2], [0, max_v2], color="g", linestyle="--")
    plt.plot([0, max_idx1], [max_v1, max_v1], color="b", linestyle="--")
    plt.plot([max_idx1, max_idx1], [0, max_v1], color="b", linestyle="--")
    plt.plot([0, max_idx3], [max_v3, max_v3], color="r", linestyle="--")
    plt.plot([max_idx3, max_idx3], [0, max_v3], color="r", linestyle="--")

    axes.set_ylabel('New cases per day', labelpad=10, fontsize=15)
    axes.set_xlabel('Time', labelpad=10, fontsize=15)
    plt.legend(["p=0.1", "p=0.5", "p=0.9"])
    plt.show()

    if isSave:
        file_name = f"{country}_{location}_new_cases_per_day_{p}.pdf"
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f"{country}_{location}_new_cases_per_day_{p}.eps"
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        fig.savefig(fig_path, format='eps')

def plot_attack_ratio_bracket_age(location, country, p, p_v, l, l_v, k, k_v, num_agebrackets=18, isSave=True):
    attack = []
    file_name = f"{country}_{location}_cumulative_incidence_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:0.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', 'bracket', f"{p}_{p_v:.1f}", file_name)
    with open(file_path,"r",encoding="gbk") as f:
        r = f.readlines()
        print(r[0])  # 读第一行
        print(r[-1])


    plt.show()

    if isSave:
        file_name = f"{country}_{location}_bracket_attack_ratio_{p}={p_v:.1f}.pdf"
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        # fig.savefig(fig_path, format='pdf')

        file_name = f"{country}_{location}_bracket_attack_ratio_{p}={p_v:.1f}.eps"
        fig_path = os.path.join(resultsdir, 'age-structured_result', file_name)
        # fig.savefig(fig_path, format='eps')


def plot_mixed_new_case_per_day(location, country, p, p_v, l, l_v, k, k_v, num_agebrackets=18, isSave=True):
    age_file_name = f"{country}_{location}_new_case_per_day_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:0.2f}.csv"
    age_file_path = os.path.join(output_source_dir, 'age-structured_output_source', f"{p}_{p_v:.1f}", age_file_name)
    ageM = np.loadtxt(age_file_path, delimiter=',', usecols=(5, 6, 7, 8)).sum(axis=1)

    no_age_file_name = f"new_case_per_day_no_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    no_age_file_path = os.path.join(output_source_dir, 'no_age-structured_output_source', no_age_file_name)
    no_ageM = np.loadtxt(no_age_file_path, delimiter=',', usecols=(5,6,7,8)).sum(axis=1)

    # print(M1)

    max_idx1 = np.argmax(np.array(ageM))
    max_v1 = max(np.array(ageM))

    max_idx2 = np.argmax(np.array(no_ageM))
    max_v2 = max(np.array(no_ageM))

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(8, 5))
    plt.subplots_adjust(left=0.14, right=0.95, bottom=0.18, top=0.80, wspace=0.1, hspace=0.1)
    plt.plot(ageM, color="b")

    plt.plot(no_ageM[:400], color="g")

    plt.plot([0, max_idx2], [max_v2, max_v2], color="g", linestyle="--")
    plt.plot([max_idx2, max_idx2], [0, max_v2], color="g", linestyle="--")
    plt.plot([0, max_idx1], [max_v1, max_v1], color="b", linestyle="--")
    plt.plot([max_idx1, max_idx1], [0, max_v1], color="b", linestyle="--")

    axes.set_title(p + " = " + str(p_v))
    axes.set_ylabel('New cases per day', labelpad=10, fontsize=15)
    axes.set_xlabel('Time', labelpad=10, fontsize=15)
    plt.legend(["age", "no age"])
    plt.show()

    if isSave:
        file_name = f"compare_new_cases_per_day_{p}={p_v:.1f}.pdf"
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f"compare_new_cases_per_day_{p}={p_v:.1f}.eps"
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='eps')

def plot_mixed_period(location, country,state, p, p_v, l, l_v, k, k_v, num_agebrackets=18, isSave=True):
    age = []
    no_age = []
    for p_v in [0.1, 0.5, 0.9]:
        age_file_name = f"{country}_{location}_{state}_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:0.2f}.csv"
        age_file_path = os.path.join(output_source_dir, 'age-structured_output_source', f"{p}_{p_v:.1f}", age_file_name)
        ageM = np.sum(np.loadtxt(age_file_path, delimiter=',', usecols=(4, 5, 6, 7)), axis=1)

        no_age_file_name = f"{state}_numbers_no_age_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
        no_age_file_path = os.path.join(output_source_dir, 'no_age-structured_output_source', no_age_file_name)
        no_ageM = np.sum(np.loadtxt(no_age_file_path, delimiter=',', usecols=(4, 5, 6, 7)), axis=1)

        age.append(ageM)
        no_age.append(no_ageM)

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, figsize=(8, 12))
    plt.subplots_adjust(left=0.15, right=0.9, bottom=0.1, top=0.90, wspace=0.1, hspace=0.1)
    # plt.title("")
    # print(no_ageM)
    #
    for i in range(3):
        # max_idx1 = np.argmax(np.array(age[i]))
        # max_v1 = max(np.array(age[i]))
        #
        # max_idx2 = np.argmax(np.array(no_age[i]))
        # max_v2 = max(np.array(no_age[i]))

        axes[i].plot(age[i], color="b")

        axes[i].plot(no_age[i][:400], color="g")

    """
        0.5 :
            xy=[5,0] width = 80
            xy=[60,0] width = 150
        0.9:
            xy=[6,0] width = 80
            xy=[36,0] width = 130
    """
    # axes.set_rasterized(True)
    max_v1 = max(np.array(age[0]))
    max_v2 = max(np.array(no_age[0]))
    axes[0].add_patch(patches.Rectangle(xy=[6, 0], width=80, height=max_v1,
                                        facecolor='blue',
                                        alpha=0.2))
    axes[0].add_patch(patches.Rectangle(xy=[120, 0], width=240, height=max_v2,
                                        facecolor='green',
                                        alpha=0.2))
    max_v1 = max(np.array(age[1]))
    max_v2 = max(np.array(no_age[1]))
    axes[1].add_patch(patches.Rectangle(xy=[5, 0], width=80, height=max_v1,
                                        facecolor='blue',
                                        alpha=0.2))
    axes[1].add_patch(patches.Rectangle(xy=[60, 0], width=150, height=max_v2,
                                        facecolor='green',
                                        alpha=0.2))
    max_v1 = max(np.array(age[2]))
    max_v2 = max(np.array(no_age[2]))
    axes[2].add_patch(patches.Rectangle(xy=[6, 0], width=80, height=max_v1,
                                    facecolor='blue',
                                    alpha=0.2))
    axes[2].add_patch(patches.Rectangle(xy=[36, 0], width=130, height=max_v2,
                                     facecolor='green',
                                     alpha=0.2))

    # plt.plot([0, max_idx2], [max_v2, max_v2], color="g", linestyle="--")
    # plt.plot([max_idx2, max_idx2], [0, max_v2], color="g", linestyle="--")
    # plt.plot([0, max_idx1], [max_v1, max_v1], color="b", linestyle="--")
    # plt.plot([max_idx1, max_idx1], [0, max_v1], color="b", linestyle="--")

    # axes.set_title(p + " = " + str(p_v))
    axes[0].set_ylabel('Cases per day', labelpad=10, fontsize=15)
    axes[1].set_ylabel('Cases per day', labelpad=10, fontsize=15)
    axes[2].set_ylabel('Cases per day', labelpad=10, fontsize=15)
    axes[2].set_xlabel('Time', labelpad=10, fontsize=15)
    axes[0].legend(["age", "no age"])
    axes[1].legend(["age", "no age"])
    axes[2].legend(["age", "no age"])

    axes[0].text(x=300, y=100000,fontsize=20,c="red", s="p = 0.1")
    axes[1].text(x=300, y=100000,fontsize=20,c="red", s="p = 0.5")
    axes[2].text(x=300, y=100000,fontsize=20,c="red", s="p = 0.9")
    plt.show()

    if isSave:
        # file_name = f"compare_cases_per_day_{p}={p_v:.1f}.pdf"
        # fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        # fig.savefig(fig_path, format='pdf')
        #
        # file_name = f"compare_cases_per_day_{p}={p_v:.1f}.eps"
        # fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        # fig.savefig(fig_path, format='eps')
        file_name = f"compare_cases_per_day.pdf"
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='pdf')

        file_name = f"compare_cases_per_day.eps"
        fig_path = os.path.join(resultsdir, 'data_driven_result', file_name)
        fig.savefig(fig_path, format='eps')



location = 'Shanghai'
country = 'China'
level = 'subnational'
state = "all_states"
p = "p"
l = "l"
k = "k"
p_v = 0.1  # 0.1 0.5 0.9
l_v = 0.8
k_v = 0.5

# 画无年龄结构的累计发病率
# plot_no_age_cumulative_incidence(p, p_v, l, l_v, k, k_v, isSave=True)

# 画无年龄结构的每日新病例
# plot_no_age_new_case_per_day(p, p_v, l, l_v, k, k_v, isSave=True)

# 画有年龄结构的每日新增病例
# plot_age_new_case_per_day(location, country, state, p, p_v, l, l_v, k, k_v, num_agebrackets=18, isSave=True)

# 画每个年龄段的发病率
# plot_attack_ratio_bracket_age(location, country, p, p_v, l, l_v, k, k_v, num_agebrackets=18, isSave=False)

# 画有无年龄结构的每日新增对比
# plot_mixed_new_case_per_day(location, country, p, p_v, l, l_v, k, k_v, num_agebrackets=18, isSave=True)

# 画有无年龄结构的感染对比
plot_mixed_period(location, country, state, p, p_v, l, l_v, k, k_v, num_agebrackets=18, isSave=True)
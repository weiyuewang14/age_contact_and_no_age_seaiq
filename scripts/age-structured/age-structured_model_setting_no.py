import csv

import numpy as np
import copy
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# set some initial paths

# path to the directory where this script lives
thisdir = os.path.abspath('')

# path to the scripts directory of the repository
scriptsdir = os.path.split(thisdir)[0]

# path to the main directory of the repository
maindir = os.path.split(scriptsdir)[0]

# path to the results subdirectory
resultsdir = os.path.join(os.path.split(maindir)[0], 'results')

# path to the data subdirectory
datadir = os.path.join(maindir, 'data')

# path to the output_source subsubdirectory
output_source_dir = os.path.join(datadir, 'output_source')


# %% 将得到的结果读入csv文件中
def write_compare_data(country, location, state, num_agebrackets, mode, param, param_value, allocation_capacity, value,
                       overwrite=True):
    if mode == 'baseline':
        file_name = f"{country}_{location}_{state}_numbers_{num_agebrackets:.0f}_{mode}_{param}={param_value:.4f}.csv"
    else:
        file_name = f"{country}_{location}_{state}_numbers_{num_agebrackets:.0f}_{mode}_{allocation_capacity:.0f}_{param}={param_value:.4f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', file_name)
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists.')
        if overwrite:
            print('Overwriting the file.')
            f = open(file_path, 'w+')
            for v in range(len(value)):
                f.write(
                    f"{v:.2f},{value[v][0]:.2f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f},{value[v][4]:.2f},{value[v][5]:.2f},{value[v][6]:.2f},{value[v][7]:.2f}\n")
            f.close()
        else:
            print('Not overwriting the existing file.')
    else:
        f = open(file_path, 'w+')
        for v in range(len(value)):
            f.write(
                f"{v:.2f},{value[v][0]:.2f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f},{value[v][4]:.2f},{value[v][5]:.2f},{value[v][6]:.2f},{value[v][7]:.2f}\n")
        f.close()
    return file_path


# %% 将得到的结果读入csv文件中
def write_data_param(country, location, state, value, p, p_value, l, l_value, k, num_agebrackets=18, overwrite=True):
    file_name = f"{country}_{location}_{state}_numbers_{num_agebrackets:.0f}_{p}={p_value:.1f}_{l}={l_value:.1f}_{k}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', file_name)
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists.')
        if overwrite:
            print('Overwriting the file.')
            f = open(file_path, 'w+')
            for v in range(len(value)):
                f.write(
                    f"{value[v][0]:.0f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f}\n")
            f.close()
        else:
            print('Not overwriting the existing file.')
    else:
        print(f'{file_path} was written')
        f = open(file_path, 'w+')
        for v in range(len(value)):
            f.write(
                f"{value[v][0]:.2f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f}\n")
        f.close()
    return file_path


# %% 将得到的结果读入csv文件中
def write_data(country, location, state, value, p, p_v, l, l_v, k, k_v, num_agebrackets=18, overwrite=True):
    file_name = f"{country}_{location}_{state}_numbers_all_age_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', p + "_" + str(p_v), file_name)
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists.')
        if overwrite:
            print('Overwriting the file.')
            f = open(file_path, 'w+')
            for v in range(len(value)):
                f.write(
                    f"{v:.2f},{value[v][0]:.2f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f},{value[v][4]:.2f},{value[v][5]:.2f},{value[v][6]:.2f},{value[v][7]:.2f}\n")
            f.close()
        else:
            print('Not overwriting the existing file.')
    else:
        print(f'{file_path} was written')
        f = open(file_path, 'w+')
        for v in range(len(value)):
            f.write(
                f"{v:.2f},{value[v][0]:.2f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f},{value[v][4]:.2f},{value[v][5]:.2f},{value[v][6]:.2f},{value[v][7]:.2f}\n")
        f.close()
    return file_path


# %% 将得到的结果读入csv文件中
def write_data_bracket_age(country, location, state, value, age, p, p_v, l, l_v, k, k_v, num_agebrackets=18,
                           overwrite=True):
    file_name = f"{country}_{location}_{state}_numbers_{num_agebrackets}_age={age}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', "bracket", p + "_" + str(p_v),
                             file_name)
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists.')
        if overwrite:
            print('Overwriting the file.')
            f = open(file_path, 'w+')
            for v in range(len(value)):
                f.write(
                    f"{v:.2f},{value[v][0]:.2f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f},{value[v][4]:.2f},{value[v][5]:.2f},{value[v][6]:.2f},{value[v][7]:.2f}\n")
            f.close()
        else:
            print('Not overwriting the existing file.')
    else:
        print(f'{file_path} was written.')
        f = open(file_path, 'w+')
        for v in range(len(value)):
            f.write(
                f"{v:.2f},{value[v][0]:.2f},{value[v][1]:.2f},{value[v][2]:.2f},{value[v][3]:.2f},{value[v][4]:.2f},{value[v][5]:.2f},{value[v][6]:.2f},{value[v][7]:.2f}\n")
        f.close()
    return file_path


# %% 保存每日新增病例数
def write_per_day_new_cases(country, location, new_case_per_day, p, p_v, l, l_v, k, k_v, num_agebrackets=18,
                            overwrite=True):
    file_name = f"{country}_{location}_new_case_per_day_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'age-structured_output_source', p + "_" + str(p_v), file_name)
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists.')
        if overwrite:
            print('Overwriting the file.')
            f = open(file_path, 'w+')
            # f.write(
            #     f"susceptible, exposedN, exposedV, asymptomaticN, asymptomaticV, infectedN, infectedV, quarantined")
            for v in range(len(new_case_per_day)):
                f.write(
                    f"{v:.2f},{new_case_per_day[v][0]:.2f},{new_case_per_day[v][1]:.2f},{new_case_per_day[v][2]:.2f},{new_case_per_day[v][3]:.2f},{new_case_per_day[v][4]:.2f},{new_case_per_day[v][5]:.2f},{new_case_per_day[v][6]:.2f},{new_case_per_day[v][7]:.2f}\n")
            f.close()
        else:
            print('Not overwriting the existing file.')
    else:
        print(f'{file_path} was written.')
        f = open(file_path, 'w+')
        # f.write(
        #     f"susceptible, exposedN, exposedV, asymptomaticN, asymptomaticV, infectedN, infectedV, quarantined")
        for v in range(len(new_case_per_day)):
            f.write(
                f"{v:.2f},{new_case_per_day[v][0]:.2f},{new_case_per_day[v][1]:.2f},{new_case_per_day[v][2]:.2f},{new_case_per_day[v][3]:.2f},{new_case_per_day[v][4]:.2f},{new_case_per_day[v][5]:.2f},{new_case_per_day[v][6]:.2f},{new_case_per_day[v][7]:.2f}\n")
        f.close()
    return file_path


# %% 保存每次运行的参数
def write_param_value(p_v, w_v, beta, q_v, l_v, sigma_inverse, k_v, gamma_inverse, result_file_path):
    date = datetime.datetime.now()
    p = str(p_v)
    w = str(w_v)
    beta = str(beta)
    q = str(q_v)
    l = str(l_v)
    sigma_inverse = str(sigma_inverse)
    k = str(k_v)
    gamma_inverse = str(gamma_inverse)
    file_path = os.path.join(datadir, 'param_run_log_age-structured.csv')
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists, a new line of data was written')
        file = open(file_path, mode='a+', encoding='utf-8', newline='')
        sWrite = csv.writer(file)
        sWrite.writerow([date, p, w, beta, q, l, sigma_inverse, k, gamma_inverse, result_file_path])
        file.close()
    else:
        print(f'{file_path} is created, a row of data is written')
        file = open(file_path, mode='w', encoding='utf-8', newline='')
        sWrite = csv.writer(file)
        sWrite.writerow(['date', 'p', 'w', 'beta', 'q', 'l', 'sigma_inverse', 'k', 'gamma_inverse', 'file_path'])
        sWrite.writerow([date, p, w, beta, q, l, sigma_inverse, k, gamma_inverse, result_file_path])
        file.close()
    return


# %% 模型的模拟
def dou_seaiq_with_age_specific_contact_martix(contact_matrix, ages, p, w, beta, q, l, k,
                                               susceptibility_factor_vector,
                                               sigma_inverse, gamma_inverse, initial_infected_age,
                                               percent_of_initial_infected_seeds,
                                               num_agebrackets, timesteps):
    """
    Args:
        contact_matrix_dic (dict)                 : a dictionary of contact matrices for different settings of contact. All setting contact matrices must be square and have the dimensions (num_agebrackets, num_agebrackets).
        weights (dict)                            : a dictionary of weights for each setting of contact
        p (float)                                 : percent of the population with unvaccinated
        w (float)                                 : percent of the population with unvaccinated
        beta (float)                              : the transmissibility for unvaccinated
        susceptibility_factor_vector (np.ndarray) : vector of age specific susceptibility, where the value 1 means fully susceptibility and 0 means unsusceptible.
        q (float)                                 : percentage of unvaccinated exposed becoming symptomatic infections
        l (float)                                 : percentage of vaccinated exposed becoming symptomatic infections
        sigma_inverse (float)                     : the incubation period for unvaccinated exposed
        k (float)                                 : percentage of vaccinated exposed becoming symptomatic infections
        gamma_inverse (float)                     : the quarantined period for asyptomatic infections
        initial_infected_age (int)                : the initial age seeded with infections
        percent_of_initial_infected_seeds (float) : percent of the population initially seeded with infections.
        num_agebrackets (int)                     : the number of age brackets of the contact matrix
        timesteps (int)                           : the number of timesteps

    Returns:
        A numpy array with the number of people in each disease state (Dou S-E-A-I-Q) by age for each timestep,
        a numpy array with the incidence by age for each timestep and the disease state indices.
    """
    sigma = 1. / sigma_inverse
    gamma = 1. / gamma_inverse

    total_population = sum(ages.values())  # 总人口数：各个年龄段的人数总和
    # print('总人数为：', total_population)
    # 初始化起初的感染人数
    # initial_infected_number = min(total_population * percent_of_initial_infected_seeds, ages[initial_infected_age])
    initial_infected_number = min(1, ages[initial_infected_age])
    # print('初始感染的人数为：', initial_infected_number)
    # 总共有多少个年龄段
    # num_agebrackets = len(ages)

    # simulation output
    states = np.zeros((8, num_agebrackets, timesteps + 1))
    states_increase = np.zeros((8, num_agebrackets, timesteps + 1))
    cumulative_incidence_age = np.zeros((num_agebrackets, timesteps))
    cumulative_incidence = np.zeros((timesteps))
    incidenceN = np.zeros((num_agebrackets, timesteps))
    incidenceV = np.zeros((num_agebrackets, timesteps))

    # indices
    indices = {'susceptible': 0, 'exposedN': 1, 'exposedV': 2, 'asymptomaticN': 3, 'asymptomaticV': 4, 'infectedN': 5,
               'infectedV': 6, 'quarantined': 7}

    # initial conditions
    states[indices['infectedN']][initial_infected_age][0] = initial_infected_number
    for age in range(num_agebrackets):
        states[indices['susceptible']][age][0] = copy.deepcopy(ages[age]) - states[indices['infectedN']][age][0]

    age_effective_contact_matrix = get_age_effective_contact_matrix_with_factor_vector(contact_matrix,
                                                                                       susceptibility_factor_vector)
    for t in range(timesteps):
        for i in range(num_agebrackets):
            for j in range(num_agebrackets):
                incidenceN[i][t] += p * beta * age_effective_contact_matrix[i][j] * states[indices['susceptible']][i][
                    t] * (states[indices['infectedN']][j][t] +
                          states[indices['infectedV']][j][t] +
                          states[indices['asymptomaticN']][j][t] +
                          states[indices['asymptomaticV']][j][t]) / ages[j]
                incidenceV[i][t] += (1 - p) * w * beta * age_effective_contact_matrix[i][j] * \
                                    states[indices['susceptible']][i][t] * (
                                            states[indices['infectedN']][j][t] +
                                            states[indices['infectedV']][j][t] +
                                            states[indices['asymptomaticN']][j][t] +
                                            states[indices['asymptomaticV']][j][t]) / ages[j]
            print("aaaaaaa",incidenceN[i][t] + incidenceV[i][t])
            if (states[indices['susceptible']][i][t] - incidenceN[i][t] - incidenceV[i][t] < 0):
                states[indices['susceptible']][i][t + 1] = 0
            else:
                states[indices['susceptible']][i][t + 1] = states[indices['susceptible']][i][t] - incidenceN[i, t] - \
                                                       incidenceV[i, t]
            # print(states[indices['susceptible']][i][t + 1])
            states[indices['exposedN']][i][t + 1] = states[indices['exposedN']][i][t] + incidenceN[i, t] - sigma * \
                                                    states[indices['exposedN']][i][t]
            states[indices['exposedV']][i][t + 1] = states[indices['exposedV']][i][t] + incidenceV[i, t] - sigma * \
                                                    states[indices['exposedV']][i][t]
            states[indices['infectedN']][i][t + 1] = states[indices['infectedN']][i][t] + q * sigma * \
                                                     states[indices['exposedN']][i][t] - gamma * \
                                                     states[indices['infectedN']][i][t]
            states[indices['infectedV']][i][t + 1] = states[indices['infectedV']][i][t] + l * q * sigma * \
                                                     states[indices['exposedV']][i][t] - gamma * \
                                                     states[indices['infectedV']][i][t]
            states[indices['asymptomaticN']][i][t + 1] = states[indices['asymptomaticN']][i][t] + (1 - q) * sigma * \
                                                         states[indices['exposedN']][i][t] - k * gamma * \
                                                         states[indices['asymptomaticN']][i][t]
            states[indices['asymptomaticV']][i][t + 1] = states[indices['asymptomaticV']][i][t] + (1 - q * l) * sigma * \
                                                         states[indices['exposedV']][i][t] - k * gamma * \
                                                         states[indices['asymptomaticV']][i][t]
            states[indices['quarantined']][i][t + 1] = states[indices['quarantined']][i][t] + gamma * (
                        states[indices['infectedV']][i][t] + states[indices['infectedN']][i][t]) + k * gamma * (
                                                                   states[indices['asymptomaticN']][i][t] +
                                                                   states[indices['asymptomaticV']][i][t])

            states_increase[indices['asymptomaticN']][i][t + 1] = (1 - q) * sigma * states[indices['exposedN']][i][t]
            states_increase[indices['asymptomaticV']][i][t + 1] = (1 - q * l) * sigma * states[indices['exposedV']][i][
                t]
            states_increase[indices['infectedN']][i][t + 1] = q * sigma * states[indices['exposedN']][i][t]
            states_increase[indices['infectedV']][i][t + 1] = l * q * sigma * states[indices['exposedV']][i][t]
            states_increase[indices['quarantined']][i][t + 1] = gamma * (
                    states[indices['infectedV']][i][t] + states[indices['infectedN']][i][t]) + k * gamma * (
                                                                        states[indices['asymptomaticN']][i][t] +
                                                                        states[indices['asymptomaticV']][i][t])
            cumulative_incidence_age[i][t] = (states_increase[indices['asymptomaticN']][i].sum() +
                                          states_increase[indices['asymptomaticV']][i].sum() +
                                          states_increase[indices['infectedN']][i].sum() +
                                          states_increase[indices['infectedV']][i].sum()) / ages[i] * 100
        cumulative_incidence[t] = states_increase[3:7].sum()/total_population*100
    return states, incidenceN, incidenceV, states_increase, cumulative_incidence_age, cumulative_incidence, indices


# %% 获得有效的年龄接触矩阵
def get_age_effective_contact_matrix_with_factor_vector(contact_matrix, susceptibility_factor_vector):
    """
    Get an effective age specific contact matrix with an age dependent susceptibility drop factor.

    Args:
        contact_matrix (np.ndarray)        : the contact matrix
        susceptibility_factor_vector (int): vector of age specific susceptibility, where the value 1 means fully susceptibility and 0 means unsusceptible.


    Returns:
        np.ndarray: A numpy square matrix that gives the effective contact matrix given an age dependent susceptibility drop factor.
    """
    effective_matrix = contact_matrix * susceptibility_factor_vector
    return effective_matrix


# %% 获得C的值，资源分配的方式
def get_allocation_method(states_all_increase, indices, population_total, allocation_capacity, mode='baseline'):
    res = 0
    if (mode == 'baseline'):
        res = 0.01
    if (mode == 'case_based'):
        asymptomatic_newly = states_all_increase[indices['asymptomaticN']] + states_all_increase[
            indices['asymptomaticV']]
        all_newly = states_all_increase[indices['asymptomaticN']] + states_all_increase[indices['asymptomaticV']] + \
                    states_all_increase[indices['infectedN']] + states_all_increase[indices['asymptomaticV']]
        if (all_newly == 0):
            res = (allocation_capacity / population_total)
        else:
            res = (asymptomatic_newly / all_newly) * (allocation_capacity / population_total)
    return res


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


# %% 计算最大特征值

def get_eigenvalue(matrix):
    """
    Get the real component of the leading eigenvalue of a square matrix.

    Args:
        matrix (np.ndarray): square matrix

    Returns:
        float: Real component of the leading eigenvalue of the matrix.
    """
    eigenvalue = max(np.linalg.eigvals(matrix)).real
    return eigenvalue


# %% 计算R0

def get_R0_with_factor_vector(p, w, beta, l, q, sigma_inverse, k, gamma_inverse, susceptibility_factor_vector,
                              num_agebrackets,
                              contact_matrix):
    """
    Get the basic reproduction number R0 for a SIR compartmental model with an age dependent susceptibility drop factor and the age specific contact matrix.

    Args:
        beta (float)                              : the transmissibility
        susceptibility_factor_vector (np.ndarray) : vector of age specific susceptibility, where the value 1 means fully susceptibility and 0 means unsusceptible.
        num_agebrackets (int)                     : the number of age brackets of the contact matrix
        gamma_inverse (float)                     : the mean recovery period
        contact_matrix (np.ndarray)               : the contact matrix

    Returns:
        float: The basic reproduction number for a SEAIQ compartmental model with an age dependent susceptibility drop factor and age specific contact patterns.
    """

    sigma = 1.0 / sigma_inverse
    gamma = 1.0 / gamma_inverse
    effective_matrix = get_age_effective_contact_matrix_with_factor_vector(contact_matrix.T,
                                                                           susceptibility_factor_vector)
    eig = get_eigenvalue(effective_matrix)
    print('接触矩阵最大特征值', eig)
    return (beta / (k * gamma)) * (p * ((1 - q) + k * q) + (1 - p) * w * ((1 - q * l) + k * q * l)) * eig


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


# %% 读取接触矩阵
def read_contact_matrix(location, setting, num_agebrackets=85):
    """
    Read in the contact for each setting.

    Args:
        location (str)        : name of the location
        country (str)         : name of the country
        level (str)           : name of level (country or subnational)
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
    return np.round(np.loadtxt(file_path, delimiter=','), 6)


# %% 合成综合矩阵
def combine_synthetic_matrices(contact_matrix_dic, weights, num_agebrackets=18):
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


# %% 定义易感染因子  define the vector of susceptibility by age
def get_susceptibility_factor_vector(num_agebrackets, example):
    """
        :param num_agebrackets: the total age brackets
        :param example:
        :return: susceptibility_factor_vector

        Example 1: past 18 the suseptibility of individuals drops to a relative factor of 0.6 of those under 18
        susceptibility_factor_vector = np.ones((num_agebrackets, 1))
        for age in range(18, num_agebrackets):
            susceptibility_factor_vector[age, :] *= 0.6

        Example 2: under 18 the susceptibility of individuals drops to 0.2, from 18 to 65 the susceptibility is 0.8,
        and those 65 and over are fully susceptible to the disease.

        """
    susceptibility_factor_vector = np.ones((num_agebrackets, 1))
    if num_agebrackets == 85:
        if (example == 1):
            for age in range(18, num_agebrackets):
                susceptibility_factor_vector[age, :] *= 0.6
        if (example == 2):
            for age in range(0, 18):
                susceptibility_factor_vector[age, :] *= 0.2
            for age in range(18, 65):
                susceptibility_factor_vector[age, :] *= 0.8
    if (num_agebrackets == 18):
        if (example == 1):
            for age in range(4, num_agebrackets):
                susceptibility_factor_vector[age, :] *= 0.6
        if (example == 2):
            for age in range(0, 4):
                susceptibility_factor_vector[age, :] *= 0.8
            for age in range(4, 12):
                susceptibility_factor_vector[age, :] *= 0.6
    return susceptibility_factor_vector


# 画基本传播图
def plot_basic_fig(all_state_data):
    # 画图
    fontsizes = {'colorbar': 30, 'colorbarlabels': 22, 'title': 44, 'ylabel': 28, 'xlabel': 28, 'xticks': 24,
                 'yticks': 24}
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    time = [i for i in range(151)]
    ax.plot(time, all_state_data[0])
    ax.plot(time, all_state_data[1])
    ax.plot(time, all_state_data[2])
    ax.plot(time, all_state_data[3])
    ax.plot(time, all_state_data[4])
    ax.plot(time, all_state_data[5])
    ax.plot(time, all_state_data[6])
    ax.plot(time, all_state_data[7])

    # ax = plt.gca()
    # ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    # ax.get_yaxis().get_offset_text().set(va='bottom', ha='left')
    # ax.yaxis.get_offset_text().set_fontsize(16)  # 设置1e6的大小与位置

    ax.legend(['S', 'En', 'Ev', 'An', 'Av', 'In', 'Iv', 'Q'], loc='upper right')
    axins = inset_axes(ax, width=2.5, height=1.5, loc='center right')

    # axins.plot(all_state_data[0], all_state_data[[0 for i in range(10, 60)], 0])
    for i in range(8):
        axins.plot(time[10:50], all_state_data[i][10:50])

    title = 'SEAIQ with age structured'
    ax.set_title(title, fontsize=32)
    ax.set_ylabel('number of states', fontsize=fontsizes['ylabel'])
    ax.set_xlabel('time', fontsize=fontsizes['xlabel'])

    plt.show()


# 保存每个年龄段的累计发病率
def write_cumulative_incidence(country, location, cumulative_incidence, p, p_v, l, l_v, k, k_v, num_agebrackets=18,
                               overwrite=True):
    file_name = f"{country}_{location}_cumulative_incidence_{num_agebrackets}_{p}={p_v:.2f}_{l}={l_v:.2f}_{k}={k_v:.2f}.csv"
    file_path = os.path.join(output_source_dir, 'mixed_output_source', "age", "bracket", p + "_" + str(p_v), file_name)
    e = os.path.exists(file_path)
    if e:
        print(f'{file_path} already exists.')
        if overwrite:
            print('Overwriting the file.')
            f = open(file_path, 'w+')
            for v in range(len(cumulative_incidence)):
                f.write(
                    f"{v:.2f},{cumulative_incidence[v][0]:.2f},{cumulative_incidence[v][1]:.2f},{cumulative_incidence[v][2]:.2f},{cumulative_incidence[v][3]:.2f},{cumulative_incidence[v][4]:.2f},{cumulative_incidence[v][5]:.2f},{cumulative_incidence[v][6]:.2f},{cumulative_incidence[v][7]:.2f},{cumulative_incidence[v][8]:.2f},{cumulative_incidence[v][9]:.2f},{cumulative_incidence[v][10]:.2f},{cumulative_incidence[v][11]:.2f},{cumulative_incidence[v][12]:.2f},{cumulative_incidence[v][13]:.2f},{cumulative_incidence[v][14]:.2f},{cumulative_incidence[v][15]:.2f},{cumulative_incidence[v][16]:.2f},{cumulative_incidence[v][17]:.2f}\n")
            f.close()
        else:
            print('Not overwriting the existing file.')
    else:
        print(f'{file_path} was written.')
        f = open(file_path, 'w+')
        # f.write(
        #     f"susceptible, exposedN, exposedV, asymptomaticN, asymptomaticV, infectedN, infectedV, quarantined, asymptomatic_total, infected_total, indectious_total")
        for v in range(len(cumulative_incidence)):
            f.write(
                f"{v:.2f},{cumulative_incidence[v][0]:.2f},{cumulative_incidence[v][1]:.2f},{cumulative_incidence[v][2]:.2f},{cumulative_incidence[v][3]:.2f},{cumulative_incidence[v][4]:.2f},{cumulative_incidence[v][5]:.2f},{cumulative_incidence[v][6]:.2f},{cumulative_incidence[v][7]:.2f},{cumulative_incidence[v][8]:.2f},{cumulative_incidence[v][9]:.2f},{cumulative_incidence[v][10]:.2f},{cumulative_incidence[v][11]:.2f},{cumulative_incidence[v][12]:.2f},{cumulative_incidence[v][13]:.2f},{cumulative_incidence[v][14]:.2f},{cumulative_incidence[v][15]:.2f},{cumulative_incidence[v][16]:.2f},{cumulative_incidence[v][17]:.2f}\n")
        f.close()
    return file_path


# %%

num_agebrackets = 18  # number of age brackets for the contact matrices# R0_star = 1.6  # basic reproduction number
initial_infected_age = 0
if num_agebrackets == 85:
    initial_infected_age = 20  # some initial age to seed infections within the population
elif num_agebrackets == 18:
    initial_infected_age = 4  # some initial age to seed infections within the population

# 归一化矩阵
def normalized(matrix):
    for a in range(len(matrix[0])):
        if np.sum(matrix[a, :]) != 0:
            matrix[a, :] = matrix[a, :]/np.sum(matrix[a, :])
    matrix = matrix/np.sum(matrix)
    return matrix

percent_of_initial_infected_seeds = 1e-5
timesteps = 300  # how long to run the SEAIQ model

p = 'p'
w = 'w'
beta = 'beta'
q = 'q'
l = 'l'
# param = 'sigma_inverse'
k = 'k'
# param = 'gamma_inverse'

# Dou-SEAIQ model parameters

p_v = 0.1  # 0.1 0.5 0.9  # percent of unvaccinated
w_v = 0.5
beta_v = 0.6  #
q_v = 0.5
l_v = 0.8
sigma_inverse = 5.2  # mean incubation period
k_v = 0.5
gamma_inverse = 1.6  # mean quarantined period

location_dic = {'Hubei': '湖北省', 'Shanghai': '上海市', 'Jiangsu': '江苏省', 'Beijing': '北京市', 'Tianjin': '天津市',
                'Guangdong': '广东省'}
weights1 = {'household': 4.11, 'school': 11.41, 'work': 8.07, 'community': 2.79}
weights2 = {'household': 0.31, 'school': 0.24, 'work': 0.16, 'community': 0.29}
weights4 = {'household': 0.31, 'school': 0.24, 'work': 0.16, 'community': 0.29}
weights3 = {'household': 2.11, 'school': 6.24, 'work': 4.16, 'community': 1.29}
settings = ['household', 'school', 'work', 'community']
location = 'Shanghai'
country = 'China'
level = 'subnational'
num_agebrackets = 18

# contact_matrix = read_contact_matrix(location, 'overall', num_agebrackets)

ages = get_ages(location, country, level, num_agebrackets)
contact_matrix_dic = get_contact_matrix_dic(location, num_agebrackets)
susceptibility_factor_vector = get_susceptibility_factor_vector(num_agebrackets, example=2)
ccm = combine_synthetic_matrices(contact_matrix_dic, weights1, num_agebrackets)
# ccmn = normalized(ccm)


# print(contact_matrix)
# print(ecm)
states, incidenceN, incidenceV, states_increase, cumulative_incidence_age, cumulative_incidence, indices = dou_seaiq_with_age_specific_contact_martix(
    ccm, ages, p_v, w_v, beta_v, q_v, l_v, k_v, susceptibility_factor_vector, sigma_inverse, gamma_inverse,
    initial_infected_age, percent_of_initial_infected_seeds, num_agebrackets, timesteps)
#

print(cumulative_incidence)
R0 = get_R0_with_factor_vector(p_v, w_v, beta_v, l_v, q_v, sigma_inverse, k_v, gamma_inverse, susceptibility_factor_vector, num_agebrackets, ccm)
print('R0:', R0)
print('总的感染人数：', states_increase[3:7].sum())


# print(cumulative_incidence)
# R0 = get_R0_with_factor_vector(p_v, w_v, beta, l_v, q_v, sigma_inverse, k_v, gamma_inverse, susceptibility_factor_vector, num_agebrackets, ccm)
# print(R0)

print(f'{location}总人数:{sum(ages.values())}')
# for i in range(11):
#     l_v = i / 10.0
#     attack_rate_list = []
#     subresult = []
#     for j in range(0, 11):
#         k_v = j / 10
#         states, incidenceN, incidenceV, states_increase, cumulative_incidence, indices = dou_seaiq_with_age_specific_contact_martix(
#             combine_matrix, ages, p_v, w_v, beta_v, q_v, l_v, k_v, susceptibility_factor_vector,
#             sigma_inverse, gamma_inverse, initial_infected_age, percent_of_initial_infected_seeds,
#             num_agebrackets, timesteps)
#
#         # 写入每个年龄段的累计发病率
#         # write_cumulative_incidence(country, location, cumulative_incidence.T, p, p_v, l, l_v, k, k_v)
#
#         total_asymptomaticN = states_increase[indices['asymptomaticN']].sum()
#         total_asymptomaticV = states_increase[indices['asymptomaticV']].sum()
#         total_infectedN = states_increase[indices['infectedN']].sum()
#         total_infectedV = states_increase[indices['infectedV']].sum()

# subresult.append([total_asymptomaticN, total_asymptomaticV, total_infectedN, total_infectedV])
#
# attack_rate = (total_asymptomaticN + total_asymptomaticV + total_infectedV + total_infectedN) / sum(
#     ages.values()) * 100
# attack_rate_list.append(float("%.2f" % attack_rate))
#
# all_states = np.round(np.sum(states, axis=1))
# write_data(country, location, state, np.round(all_states.T), p, p_v, l, l_v, k, k_v, num_agebrackets=18, overwrite=True)
# plot_basic_fig(all_states)
#
# for age in range(num_agebrackets):
#     # print(states[:, age, :].T.shape)
#     write_data_bracket_age(country, location, state, states[:, age, :].T, age, p,p_v,l,l_v,k,k_v,num_agebrackets=18, overwrite=True)


# # 将每日增长的病例保存起来
# new_case_per_day = np.round(np.sum(states_increase, axis=1).T)
# # # print(new_case_per_day.shape)
# write_per_day_new_cases(country, location, new_case_per_day, p, p_v, l, l_v, k, k_v, num_agebrackets=18, overwrite=True)


# result_file_path = write_data_param(country=country, location=location, state='four_states',
#                                     num_agebrackets=num_agebrackets, value=subresult, p="p", p_value=p, l="l",
#                                     l_value=l, k="k")
# # write_per_day_new_cases(country, location, np.round(per_day_newly_increase.T), num_agebrackets=num_agebrackets, param=param, param_value=param_value, overwrite=True)
# # 将每次运行的参数保存起来
# write_param_value(p, w, beta, q, l, sigma_inverse, "[0-1]", gamma_inverse, result_file_path)
#
#
# print('发病率', attack_rate_list)
# attack.append(attack_rate_list)

#!/usr/bin/env python
# coding: utf-8

# In[1]:


from itertools import product
import numpy as np
import scipy as sp
from scipy import optimize
from copy import deepcopy
import datetime
from sklearn.preprocessing import MinMaxScaler
import time
from scipy import math
from scipy import linalg
import sys
import random
import pandas as pd
import pickle
import datetime
import os


# In[2]:Relative Entropy Inverse Reinforcement Learning


def irl_gradient_ascent(sample_trajectories, feature_matrix, feature_e, scaler, error_term, thre, learning_rate, base_p,
                        execu_p, show_flag=False, show_count=50):
    '''
    Relative Entropy Inverse Reinforcement Learning
    '''
    print('feature_matrix='+feature_matrix)
    print('feature_e='+feature_e)
    feature_expectation = feature_e
    diff = 10000
    last_diff = 0
    n_feature = len(feature_e)
    max_fe = []
    for l_fe in feature_matrix.T:
        max_fe.append(np.max(l_fe))
    theta = np.random.random(size=(n_feature,))
    base_vector = np.zeros(n_feature)

    prob_ratio = get_prob_ratio_from_policies(sample_trajectories, base_p, execu_p)
    loop_counter = 0
    last_theta = []
    ll_diff = 0
    back_flag = False
    mem_alpha = learning_rate
    set_alpha = False

    while abs(diff) > thre:
        if not back_flag:
            last_diff = diff
        iterated_rewards = np.dot(theta, feature_matrix.T) + prob_ratio
        alphas = theta / np.abs(theta) * error_term
        feature_count = np.zeros(n_feature)
        sample_importance = 0
        m_reward = max(iterated_rewards)
        ind_re = np.argmax(iterated_rewards)
        iterated_rewards -= m_reward + 20
        for i in range(len(feature_matrix)):
            if prob_ratio[i] == 0:
                continue
            try:
                feature_count += math.exp(iterated_rewards[i]) * feature_matrix[i]
                sample_importance += math.exp(iterated_rewards[i])
            except:
                print(feature_matrix[i])
                print(theta)
                print(iterated_rewards[i])
                print(prob_ratio[i])
                raise (ValueError)
        gradient = feature_expectation - feature_count / sample_importance
        theta = theta + learning_rate * gradient
        t = judge_norm(theta, base_vector, 2)
        diff = judge_norm(gradient, base_vector, 2)

        if loop_counter % show_count == 0 and show_flag:
            print(loop_counter, diff)
            print('reward', m_reward)
            print('fe', feature_count / sample_importance)
            print('differ', abs(diff - last_diff))
            print('theta', theta)
            print('grad', gradient)
            print('    ')
        loop_counter += 1
        if loop_counter >= 5000000:
            break
        last_theta = theta
    print(loop_counter, diff)
    print('theta', theta)
    return diff, theta


# In[3]:get_prob_ratio_from_policies


def get_prob_ratio_from_policies(sample_trajectories, base_p, execu_p):
    ratios = []
    for traj in sample_trajectories:
        ratio = 1
        for i in range(len(traj) - 1):
            step = traj[i]
            next_step = traj[i + 1]
            print(str(next_step))
            try:
                grid = str(step[1]) + '|' + str(step[2])
                action = str(next_step[1]) + '|' + str(next_step[2])
                ratio += math.log(base_p[grid][action] / 1.0 / execu_p[grid][action])
            except:
                ratio += 0
        ratios.append(ratio)
    return np.array(ratios)


# In[4]:get_feature_matrix
def get_feature_matrix(trajectories, features):
    feature_matrix = []
    for traj in trajectories:
        feature_matrix.append(get_feature_vector(traj, features[0], features[1], features[2]))
    return np.array(feature_matrix)


# In[5]:judge_norm


def judge_norm(first, second, norm):
    if len(first) != len(second):
        raise (ValueError('Not same length'))
    else:
        max_norm = 0
        for i in range(len(first)):
            max_norm += abs(first[i] - second[i]) ** norm
        return max_norm


# In[6]:cont_mdp


class cont_mdp(object):
    """
    MDP example
    """

    def __init__(self, states, actions, state_action):
        self.states = states
        self.actions = actions
        self.state_action = state_action

        # generate uniform policy
        self.uniform_policy = {}
        grid_set = set()
        for state in self.states:
            items = state.split('|')
            grid_set.add('|'.join(items[:2]))
        grids = list(grid_set)
        for grid in grids:
            policy = {}
            grid_l = list(map(int, grid.split('|')))
            x = grid_l[0]
            y = grid_l[1]
            acts = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    acts.append('|'.join(map(str, [x + i, y + j])))
            for act in acts:
                policy[act] = 1 / 9
            self.uniform_policy[grid] = policy


# In[7]:trajs_2_states


def trajs_2_states(trajs):
    state_set_plates = {}
    action_set_plates = {}
    for plate in trajs:
        state_set = set()
        action_set = set()
        for traj in trajs[plate]:
            for i in range(len(traj) - 1):
                state = '|'.join(map(str, traj[i][1:]))
                action = '|'.join(map(str, traj[i + 1][1:3]))
                result = '|'.join(map(str, traj[i + 1][1:]))
                state_set.add(state)
                action_set.add(action)
                state_set.add(result)
        #                 print(state)
        #                 print('--------------------------')
        #                 print(action)
        #                 print('--------------------------')
        #                 print(result)
        #                 print('--------------------------')
        #                 print('--------------------------')
        state_set_plates[plate] = state_set
        action_set_plates[plate] = action_set

    states_plates = {}
    actions_plates = {}
    for plate in state_set_plates:
        states = list(state_set_plates[plate])
        actions = list(action_set_plates[plate])
        states_plates[plate] = states
        actions_plates[plate] = actions

    state_index = {}
    action_index = {}
    for plate in states_plates:
        state_index[plate] = {}
        action_index[plate] = {}
        for i in range(len(states_plates[plate])):
            state_index[plate][states_plates[plate][i]] = i
        for i in range(len(actions_plates[plate])):
            action_index[plate][actions_plates[plate][i]] = i

    state_action_plates = {}
    for plate in trajs:
        state_action_plates[plate] = {}
        for traj in trajs[plate]:
            for i in range(len(traj) - 1):
                state = '|'.join(map(str, traj[i][1:]))
                action = '|'.join(map(str, traj[i + 1][1:3]))
                s_i = state_index[plate][state]
                a_i = action_index[plate][action]
                if s_i not in state_action_plates[plate]:
                    state_action_plates[plate][s_i] = [a_i]
                elif a_i not in state_action_plates[plate][s_i]:
                    state_action_plates[plate][s_i].append(a_i)
    return states_plates, actions_plates, state_action_plates


# In[8]:load
def load(month):
    trajectories = pickle.load(open("D:/Software Engineering/thesis/Driver/datas/mdp_trajs_07.pkl", 'rb'))
    states, actions, state_action = trajs_2_states(trajectories)
    profile_info = pickle.load(open("D:/Software Engineering/thesis/Driver/datas/profile_info.pkl", 'rb'))
    [fa, mf, hl, bt] = profile_info
    hf = pickle.load(open('D:/Software Engineering/thesis/Driver/datas/profile_info.pkl', 'rb'))
    return states, actions, state_action, trajectories, mf, fa, hl, bt, hf


def extract_plate_info(plate, states_plates, actions_plates, state_action_plates, trajectories_plates):
    return states_plates[plate], actions_plates[plate], state_action_plates[plate], trajectories_plates[plate]


# In[9]:gps2grid
def gps2grid(lat, lgt):
    return [int((lat - 22.44) / 0.009) + 1, int((lgt - 113.75) / 0.01) + 1]


# In[10]:get_feature_vector
def get_feature_vector(trajectory, hf, mf, fa):
    pois = [gps2grid(22.639444, 113.810833), gps2grid(22.534167, 114.111667)]  # airport, train station.
    feature = np.zeros(11)
    plate = trajectory[0][0]
    bl = list(map(int, mf[plate].split('|')))
    hl = gps2grid(hl_plates[plate][1], hl_plates[plate][0])
    bt = bt_plates[plate][0] // (60 * 5) + 1
    last_step = []
    for i in range(len(trajectory)):
        step = trajectory[i]

        ind = '|'.join(map(str, step[1:]))

        if ind in fa[plate]:
            fml = fa[plate][ind]
        else:
            fml = 0
        db = math.sqrt((bl[0] - step[1]) ** 2 + (bl[1] - step[2]) ** 2)
        dp = [math.sqrt((poi[0] - step[1]) ** 2 + (poi[1] - step[2]) ** 2) for poi in pois]
        dh = math.sqrt((hl[0] - step[1]) ** 2 + (hl[1] - step[2]) ** 2)
        dt = step[3] - bt
        try:
            f = deepcopy(hf[ind])[:-1]
        except:
            f = [0, 0, 0, 0]
        f.extend(dp)
        f.extend([fml, db, dh, dt])
        if i > 0 and last_step[1] == step[1] and last_step[2] == step[2]:
            f.append(1)
        else:
            f.append(0)
        fea = np.array(f)
        feature += fea
        last_step = step
    return feature


# In[11]:get_policy_from_traj


def get_policy_from_traj(trajs):
    raw_counter = {}
    for traj in trajs:
        for i in range(len(traj) - 1):
            step = traj[i]
            next_step = traj[i + 1]
            print(next_step)
            x1 = step[1]
            y1 = step[2]
            x2 = next_step[1]
            y2 = next_step[2]
            if abs(x2 - x1) > 1 or abs(y2 - y1) > 1:
                continue
            grid = str(x1) + '|' + str(y1)
            action = str(x2) + '|' + str(y2)
            if grid not in raw_counter:
                raw_counter[grid] = {action: 1}
            elif action not in raw_counter[grid]:
                raw_counter[grid][action] = 1
            else:
                raw_counter[grid][action] += 1
    policy = {}
    for grid in raw_counter:
        sum_p = 0
        policy_grid = {}
        for action in raw_counter[grid]:
            sum_p += raw_counter[grid][action]
        for action in raw_counter[grid]:
            policy_grid[action] = raw_counter[grid][action] / 1.0 / sum_p
        policy[grid] = policy_grid
    return policy


# In[1]:match_feature_tr


def match_feature_tr(trajs, hf0, mf0, fa0):
    scaler = MinMaxScaler()
    scaler_14 = MinMaxScaler()
    t0_all = []
    features = []
    t14_all = []
    features_14 = []
    for traj in trajs:
        t0_all.append(traj)
        t14_all.append(traj)
        fea = get_feature_vector(traj, hf0, mf0, fa0)
        features.append(fea)
        features_14.append(fea)
    feature_matri_14 = np.array(features_14)
    scaler_14 = scaler_14.fit(feature_matri_14)

    return scaler_14, t14_all, feature_matri_14


# In[5]:Main Method
month = '07'
states_plates, actions_plates, state_action_plates, trajectories_plates, mf_plates, fa_plates, hl_plates, bt_plates, hf = load(
    month=month)

sample_plates = list(trajectories_plates.keys())

theta_plates_ = {}
k = 0
t0 = datetime.datetime.now()
for plate in sample_plates[:]:
    k += 1

    states_1_plate, actions_1_plate, state_action_1_plate, trajs_1_plate = extract_plate_info(plate, states_plates,
                                                                                              actions_plates,
                                                                                              state_action_plates,
                                                                                              trajectories_plates)
    if (len(trajs_1_plate) == 0):
        print('useless plate:', plate)
        continue
    exam_mdp_1_plate = cont_mdp(states_1_plate, actions_1_plate, state_action_1_plate)

    scaler_, t_all, feature_matri_ = match_feature_tr(trajs_1_plate, hf, mf_plates, fa_plates)
    #print('Collection Of All Feature MinMaxScaler\n'+str(t_all))
    ori_policy_ = get_policy_from_traj(t_all)
    tr_fea_ = scaler_.transform(feature_matri_)
    feature_e_ = sum(tr_fea_) / len(feature_matri_)
    t1 = datetime.datetime.now()
    diff_, re_ = irl_gradient_ascent(error_term=0, feature_matrix=tr_fea_, learning_rate=1, scaler=scaler_,
                                     feature_e=feature_e_, sample_trajectories=t_all, thre=1e-7,
                                     base_p=exam_mdp_1_plate.uniform_policy, execu_p=ori_policy_, show_flag=False,
                                     show_count=20000)
    pickle.dump(re_, open('D:/Software Engineering/thesis/Driver/datas/mdp_trajs_07.pkl', 'wb'))
    theta_plates_[plate] = re_
    print(k, '*******************', (datetime.datetime.now() - t1), (datetime.datetime.now() - t0))



import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from functools import partial

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-nocv', dest='show_nocv', default=False, action='store_true', help='show no control variate')
    parser.add_argument('--dir', type=str, default='./data', help='path to dir contains data')
    args = parser.parse_args()

    return args

def compare(name, dr_data, index, model_index, true_mean, ax=None):

    if name == 'State':
        inner_name = 'state'
    elif 'DR' in name:
        inner_name = 'dr'
    elif name == 'MC':
        inner_name = 'nocv'
    elif name == 'StateAction':
        inner_name = 'sa'
    elif name == 'Traj-CV':
        inner_name = 'trajwise'
    else:
        raise NotImplementedError
    
    os.chdir(inner_name)

    np_files = os.listdir('.')

    prefix = 'iter_' + str(model_index) + '_'
    for d in np_files:
        if prefix not in d:
            continue        
        data = np.load('./' + d)
        
    os.chdir('../')

    assert true_mean.shape == (1, 194)

    if name == 'DR':
        return np.sum(np.mean((data - true_mean) ** 2, axis=0))
    else:        
        sum_var = np.sum(np.mean((data - true_mean) ** 2, axis=0))    
        sum_var = (sum_var - dr_data) / sum_var

    ax.bar(name, sum_var, label=name, color=colors[name], width=0.3)
    
    return 0

def get_true_mean(prefix):
    dirs = os.listdir('.')
    for d in dirs:
        if d.startswith(prefix):
            return np.load(d)

def get_title(model_index):
    prefix = 'iter_' + str(model_index) + '_'
    for d in os.listdir('../Model'):
        if prefix not in d:
            continue
        else:
            reward = d.split('sim')[0].split('_')[3]
    title = 'Iter' + str(model_index)
    title = title + ", Reward " + str(reward)
    return title

def get_model_info():
    dirs = os.listdir('dr')
    pic_num  = len(dirs)

    model_indices = []
    for d in dirs:
        index = int(d.split('_')[1])
        model_indices.append(index)
    model_indices.sort()

    return pic_num, model_indices

args = get_args()
os.chdir('./data')

colors = {
    'MC': '#1f77b4',
    'State': '#ff7f0e',
    'StateAction': '#2ca02c',
    'Traj-CV': '#d62728',
    'DR': '#9467bd',
}

pic_num, model_indices = get_model_info()
name_list = ['State', 'StateAction', 'Traj-CV']
if args.show_nocv:
    name_list.insert(0, 'MC')

fig, axs = plt.subplots(1, pic_num, figsize=(16, 4), sharex=True)

for pic_index in range(pic_num):
    ax=axs[pic_index]
    model_index = model_indices[pic_index]

    prefix = 'iter_' + str(model_index) + '_'    
    true_mean = get_true_mean(prefix)
    dr_var = compare('DR', None, 4, true_mean=true_mean, model_index=model_index, ax=ax)

    index = 0
    for n in name_list:
        compare(n, dr_var, index, model_index, true_mean=true_mean, ax=ax)
        index += 1
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_title(get_title(model_index))
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
    
plt.tight_layout()
plt.show()
plt.clf()
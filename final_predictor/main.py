#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:38:14 2018

@author: roman
"""
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pylab import rcParams
from prepare_data import DataLoader
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os

# ---------- load inpputs and models ---------- #
with open('inputs.json') as f:
    inputs = json.load(f)
months = inputs['months']
model_name = inputs['model_name']
models = inputs['models']
dataset_original = inputs['dataset']
dataset_name = inputs['dataset_name']
crash_threshold = inputs['crash_threshold']
n_lookback = inputs['n_days_lookback']
n_plot = inputs['n_days_plot']

# ---------- load data ---------- #
os.chdir('./data')
data = DataLoader([dataset_original], [dataset_name])
dataset_revised, crashes = data.get_data_revised([crash_threshold])
dfs_x, dfs_y = data.get_dfs_xy_predict(months=months)
X, _, _, _ = data.get_train_test(dfs_x, dfs_y, dataset_name, test_data=None)
os.chdir('..')

# ---------- make predictions ---------- #
y_pred_bin_all = []
y_pred_weighted_all = []
for month, model in zip(months, models):
    model = pickle.load(open(model, 'rb'))
    y_pred_bin = model.predict(X).astype(int)
    y_pred_weighted = []
    for i in range(-n_plot, -1):
        y_pred_bin_ = y_pred_bin[:i] 
        y_pred_weighted.append(np.dot(np.linspace(0,1,21) / \
                sum(np.linspace(0, 1, n_lookback)), y_pred_bin_[-n_lookback:]))
    y_pred_weighted.append(np.dot(np.linspace(0, 1, n_lookback) / \
                sum(np.linspace(0, 1, n_lookback)), y_pred_bin[-n_lookback:]))
    y_pred_weighted_all.append(y_pred_weighted)

# ---------- print and plot results ---------- #
df = dataset_revised[0].iloc[-n_plot:, :]
df['y_pred_weighted_1m'] = y_pred_weighted_all[0]
df['y_pred_weighted_3m'] = y_pred_weighted_all[1]
df['y_pred_weighted_6m'] = y_pred_weighted_all[2]
last_date = str(df.index[-1])[:10]

print(str(dataset_name) + ' crash prediction ' + str(model_name) + ' model as of '\
      + str(last_date))
print('probabilities as weighted average of binary predictions over last '\
      + str(n_lookback) + str(' days'))
print('* crash within 6 months: ' + str(np.round(100 \
        * df['y_pred_weighted_6m'][-1], 2)) + '%')
print('* crash within 3 months: ' + str(np.round(100 \
        * df['y_pred_weighted_3m'][-1], 2)) + '%')
print('* crash within 1 month:  ' + str(np.round(100 \
        * df['y_pred_weighted_1m'][-1], 2)) + '%')

plt.style.use('seaborn-darkgrid')
rcParams['figure.figsize'] = 10, 6
rcParams.update({'font.size': 12})
gs = gridspec.GridSpec(3, 1, height_ratios=[2.5, 1, 1])
plt.subplot(gs[0])
plt.plot(df['price'], color='blue')
plt.ylabel(str(dataset_name) + ' index')
plt.title(str(dataset_name) + ' crash prediction ' + str(model_name) + ' ' \
          + str(last_date))
plt.xticks([])
plt.subplot(gs[1])
plt.plot(df['y_pred_weighted_6m'], color='salmon')
plt.plot(df['y_pred_weighted_3m'], color='red')
plt.plot(df['y_pred_weighted_1m'], color='brown')
plt.ylabel('crash probability')
plt.ylim([0, 1.1])
plt.xticks(rotation=45)
plt.legend(['crash in 6 months', 'crash in 3 months', 'crash in 1 month'])
plt.show()



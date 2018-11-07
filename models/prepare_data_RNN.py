#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 21:15:58 2018

@author: roman
"""
import numpy as np
import pandas as pd
from datetime import timedelta

class DataLoader():
    def __init__(self, datasets_original, dataset_names):
        self.num_datasets = len(datasets_original)
        self.datasets_original = datasets_original
        self.dataset_names = dataset_names
        self.datasets_revised = None
        self.drawdowns = None
        self.crashes = None

    def get_data_revised(self, crash_thresholds):
        datasets = []
        for d in self.datasets_original:
            data_original = pd.read_csv(d, index_col = 'Date')
            data_original.index = pd.to_datetime(data_original.index, format='%Y/%m/%d')
            data_norm = data_original['Close'] / data_original['Close'][-1]
            data_ch = data_original['Close'].pct_change()
            window = 10
            data_vol = data_original['Close'].pct_change().rolling(window).std()
            data = pd.concat([data_original['Close'], data_norm, data_ch, data_vol], axis=1).dropna()
            data.columns = ['price', 'norm', 'ch', 'vol']
            datasets.append(data)
        self.drawdowns = []
        self.crashes = []
        for df, ct in zip(datasets, crash_thresholds):
            pmin_pmax = (df['price'].diff(-1) > 0).astype(int).diff() # <- -1 indicates pmin, +1 indicates pmax
            pmax = pmin_pmax[pmin_pmax == 1]
            pmin = pmin_pmax[pmin_pmax == -1]
            # make sure drawdowns start with pmax, end with pmin:
            if pmin.index[0] < pmax.index[0]:
                pmin = pmin.drop(pmin.index[0])
            if pmin.index[-1] < pmax.index[-1]:
                pmax = pmax.drop(pmax.index[-1])
            D = (np.array(df['price'][pmin.index]) - np.array(df['price'][pmax.index])) \
            / np.array(df['price'][pmax.index])
            d = {'Date':pmax.index, 'drawdown':D, 'd_start': pmax.index, 'd_end':\
                 pmin.index}    
            df_d = pd.DataFrame(d).set_index('Date')
            df_d.index = pd.to_datetime(df_d.index, format='%Y/%m/%d')
            df_d = df_d.reindex(df.index).fillna(0)
            df_d = df_d.sort_values(by='drawdown')
            df_d['rank'] = list(range(1,df_d.shape[0]+1))
            self.drawdowns.append(df_d)
            df_c = df_d[df_d['drawdown'] < ct]
            df_c.columns = ['drawdown', 'crash_st', 'crash_end', 'rank']
            self.crashes.append(df_c)
        self.datasets_revised = []  
        for i in range(len(datasets)):
            self.datasets_revised.append(pd.concat([datasets[i], self.drawdowns[i]],\
                    axis=1))
        return self.datasets_revised, self.crashes

    def get_df_xy(self, months, sequence, batch_size, additional_feat=False):   
        dfs_x1, dfs_x2, dfs_y = [], [], [] 
        for df, c in zip(self.datasets_revised, self.crashes):
            df['ch'] = df['ch'] / abs(df['ch']).mean()
            df['vol'] = df['vol'] / abs(df['vol']).mean()
            x1, x2, y = {}, {}, {}
            if additional_feat == False:
                for i in range(sequence, df.shape[0] - 126):
                    date = df.index[i]
                    x1[date] = [df['ch'].iloc[i-j] for j in range(sequence)]
                    x2[date] = [df['vol'].iloc[i-j] for j in range(sequence)]
                    y[date] = [max([date <= c and date+timedelta(month * 21) > c \
                          for c in c['crash_st']]) for month in months]
                df_x1 = pd.DataFrame.from_dict(x1, orient='index')
                df_x2 = pd.DataFrame.from_dict(x2, orient='index')
                df_y = pd.DataFrame.from_dict(y, orient='index')
                len_adj = df_x1.shape[0] - df_x1.shape[0] % batch_size 
                df_x1 = df_x1.iloc[:len_adj, :]
                df_x2 = df_x2.iloc[:len_adj, :]
                df_y = df_y.iloc[:len_adj, :]
                dfs_x1.append(df_x1)
                dfs_x2.append(df_x2)
                dfs_y.append(df_y)
            if additional_feat == True:
                for i in range(252, df.shape[0]-126):
                    date = df.index[i]    
                    x1[date] = [df['ch'].iloc[i-j] for j in range(sequence)]
                    x1[date].append(df['ch'][(date - timedelta(21)):(date - \
                      timedelta(sequence))].mean())
                    x1[date].append(df['ch'][(date - timedelta(3*21)):(date - \
                      timedelta(21))].mean())
                    x1[date].append(df['ch'][(date - timedelta(6*21)):(date - \
                      timedelta(3*21))].mean())
                    x1[date].append(df['ch'][(date - timedelta(252)):(date - \
                      timedelta(6*21))].mean())
                    x2[date] = [df['vol'].iloc[i-j] for j in range(sequence)]
                    x2[date].append(df['vol'][(date - timedelta(21)):(date - \
                      timedelta(sequence))].mean())
                    x2[date].append(df['vol'][(date - timedelta(3*21)):(date -\
                      timedelta(21))].mean())
                    x2[date].append(df['vol'][(date - timedelta(6*21)):(date -\
                      timedelta(3*21))].mean())
                    x2[date].append(df['vol'][(date - timedelta(252)):(date - \
                      timedelta(6*21))].mean())
                    y[date] = [max([date <= c and date+timedelta(month * 21) > c \
                          for c in c['crash_st']]) for month in months]
                df_x1 = pd.DataFrame.from_dict(x1, orient='index')
                df_x2 = pd.DataFrame.from_dict(x2, orient='index')
                df_y = pd.DataFrame.from_dict(y, orient='index')
                len_adj = df_x1.shape[0] - df_x1.shape[0] % batch_size 
                df_x1 = df_x1.iloc[:len_adj, :]
                df_x2 = df_x2.iloc[:len_adj, :]
                df_y = df_y.iloc[:len_adj, :]
                dfs_x1.append(df_x1)
                dfs_x2.append(df_x2)
                dfs_y.append(df_y)
        return dfs_x1, dfs_x2, dfs_y

    def get_train_stateful(self, dfs_x1, dfs_x2, dfs_y, dataset_names, test_data):
        for i, name in enumerate(dataset_names):
            if name == test_data:
                index = i
        dfs_x1_copy = list(dfs_x1)
        dfs_x1_copy.pop(index)
        dfs_x2_copy = list(dfs_x2)
        dfs_x2_copy.pop(index)
        dfs_y_copy = list(dfs_y)
        dfs_y_copy.pop(index)
        np_train_x1_l = [np.array(x) for x in dfs_x1_copy]
        np_train_x2_l = [np.array(x) for x in dfs_x2_copy]
        np_train_y_l = [np.array(y) for y in dfs_y_copy]
        return np_train_x1_l, np_train_x2_l, np_train_y_l
    
    def get_train_test(self, dfs_x1, dfs_x2, dfs_y, dataset_names, test_data):
        for i, name in enumerate(dataset_names):
            if name == test_data:
                index = i
        dfs_x1_copy = list(dfs_x1)
        dfs_x2_copy = list(dfs_x2)
        dfs_y_copy = list(dfs_y)
        df_x1_test = dfs_x1_copy.pop(index)
        df_x2_test = dfs_x2_copy.pop(index)
        df_y_test = dfs_y_copy.pop(index)
        np_x1_test = np.array(df_x1_test)
        np_x2_test = np.array(df_x2_test)
        np_y_test = np.array(df_y_test).astype(int)
        np_x1_test = np.expand_dims(np_x1_test, axis=2)
        np_x2_test = np.expand_dims(np_x2_test, axis=2)
        np_x1_train = np.concatenate(([np.array(x1) for x1 in dfs_x1_copy]))
        np_x2_train = np.concatenate(([np.array(x2) for x2 in dfs_x2_copy]))
        np_y_train = np.concatenate(([np.array(y) for y in dfs_y_copy])).astype(int)
        np_x1_train = np.expand_dims(np_x1_train, axis=2)
        np_x2_train = np.expand_dims(np_x2_train, axis=2)
        return np_x1_train, np_x2_train, np_y_train, np_x1_test, np_x2_test, np_y_test 

    def split_results(self, df_combined, dfs_xy, dataset_names, test_data, y_pred_t_bin,\
                      y_pred_tr_bin, y_train, y_test):
        df_combined = [dfc.reindex(dfs.index) for dfc, dfs in zip(df_combined, dfs_xy)]
        dfs_predict = []
        n = 0
        for df, name in zip(df_combined, dataset_names):
            if name == test_data:
                df['y'] = y_test
                df['y_pred'] = y_pred_t_bin
                dfs_predict.append(df)
            else:
                df['y'] = y_train[n:n+df.shape[0]]
                df['y_pred'] = y_pred_tr_bin[n:n+df.shape[0]]
                dfs_predict.append(df)
                n += df.shape[0]
        return dfs_predict
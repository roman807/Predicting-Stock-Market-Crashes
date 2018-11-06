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
        self.datases_revised = None
        self.drawdowns = None
        self.crashes = None
    
    def get_data_revised(self, crash_thresholds):
        datasets = []
        for d in self.datasets_original:
            data_original = pd.read_csv(d, index_col = 'Date')
            data_original.index = pd.to_datetime(data_original.index, \
                format='%Y/%m/%d')
            data_norm = data_original['Close'] / data_original['Close'][-1]
            data_ch = data_original['Close'].pct_change()
            window = 10
            data_vol = data_original['Close'].pct_change().rolling(window).std()
            data = pd.concat([data_original['Close'], data_norm, data_ch, data_vol],\
                axis=1).dropna()
            data.columns = ['price', 'norm', 'ch', 'vol']
            datasets.append(data)
        self.drawdowns = []
        self.crashes = []
        for df, ct in zip(datasets, crash_thresholds):
            pmin_pmax = (df['price'].diff(-1) > 0).astype(int).diff()
            pmax = pmin_pmax[pmin_pmax == 1]
            pmin = pmin_pmax[pmin_pmax == -1]
            # make sure drawdowns start with pmax, end with pmin:
            if pmin.index[0] < pmax.index[0]:
                pmin = pmin.drop(pmin.index[0])
            if pmin.index[-1] < pmax.index[-1]:
                pmax = pmax.drop(pmax.index[-1])
            D = (np.array(df['price'][pmin.index]) - np.array(df['price']\
                 [pmax.index]))/ np.array(df['price'][pmax.index])
            d = {'Date':pmax.index, 'drawdown':D, 'd_start': pmax.index,\
                 'd_end': pmin.index}    
            df_d = pd.DataFrame(d).set_index('Date')
            df_d.index = pd.to_datetime(df_d.index, format='%Y/%m/%d')
            df_d = df_d.reindex(df.index).fillna(0)
            df_d = df_d.sort_values(by='drawdown')
            df_d['rank'] = list(range(1,df_d.shape[0]+1))
            self.drawdowns.append(df_d)
            df_d = df_d.sort_values(by='Date')
            df_c = df_d[df_d['drawdown'] < ct]
            df_c.columns = ['drawdown', 'crash_st', 'crash_end', 'rank']
            self.crashes.append(df_c)
        self.datasets_revised = []  
        for i in range(len(datasets)):
            self.datasets_revised.append(pd.concat([datasets[i], \
                    self.drawdowns[i]], axis=1))
        return self.datasets_revised, self.crashes

    def get_dfs_xy(self, months):
    ### dfs_xy: dataframe for each dataset x (columns 0:-1) and  y (column -1)     
        dfs_x, dfs_y = [], []
        for df, c in zip(self.datasets_revised, self.crashes):
            df['ch'] = df['ch'] / abs(df['ch']).mean()
            df['vol'] = df['vol'] / abs(df['vol']).mean()
            xy = {}
            for date in df.index[252:-126]: # <--subtract 126 days in the end
                xy[date] = list([df['ch'][(date-timedelta(5)):date].mean()])
                xy[date].append(df['ch'][(date-timedelta(10)):(date-timedelta(5))].mean())
                xy[date].append(df['ch'][(date-timedelta(15)):(date-timedelta(10))].mean())
                xy[date].append(df['ch'][(date-timedelta(21)):(date-timedelta(15))].mean())
                xy[date].append(df['ch'][(date-timedelta(42)):(date-timedelta(21))].mean())
                xy[date].append(df['ch'][(date-timedelta(63)):(date-timedelta(42))].mean())
                xy[date].append(df['ch'][(date-timedelta(126)):(date-timedelta(63))].mean())
                xy[date].append(df['ch'][(date-timedelta(252)):(date-timedelta(126))].mean())
                xy[date].append(df['vol'][(date-timedelta(5)):date].mean())
                xy[date].append(df['vol'][(date-timedelta(10)):(date-timedelta(5))].mean())
                xy[date].append(df['vol'][(date-timedelta(15)):(date-timedelta(10))].mean())
                xy[date].append(df['vol'][(date-timedelta(21)):(date-timedelta(15))].mean())
                xy[date].append(df['vol'][(date-timedelta(42)):(date-timedelta(21))].mean())
                xy[date].append(df['vol'][(date-timedelta(63)):(date-timedelta(42))].mean())
                xy[date].append(df['vol'][(date-timedelta(126)):(date-timedelta(63))].mean())
                xy[date].append(df['vol'][(date-timedelta(252)):(date-timedelta(126))].mean())
                xy[date] = xy[date] + [max([date <= c and date + timedelta(month * 21) > c \
                          for c in c['crash_st']]) for month in months]
            df_xy = pd.DataFrame.from_dict(xy, orient='index').dropna()
            df_x = df_xy.iloc[:, :-len(months)]
            df_y = df_xy.iloc[:, -len(months):]
            dfs_x.append(df_x)
            dfs_y.append(df_y)
        return dfs_x, dfs_y
    
    def get_dfs_xy_predict(self, months):
    ### dfs_xy: dataframe for each dataset x (columns 0:-1) and  y (column -1)     
        dfs_x, dfs_y = [], []
        for df, c in zip(self.datasets_revised, self.crashes):
            df['ch'] = df['ch'] / abs(df['ch']).mean()
            df['vol'] = df['vol'] / abs(df['vol']).mean()
            xy = {}
            for date in df.index: # <--subtract 126 days in the end
                xy[date] = list([df['ch'][(date-timedelta(5)):date].mean()])
                xy[date].append(df['ch'][(date-timedelta(10)):(date-timedelta(5))].mean())
                xy[date].append(df['ch'][(date-timedelta(15)):(date-timedelta(10))].mean())
                xy[date].append(df['ch'][(date-timedelta(21)):(date-timedelta(15))].mean())
                xy[date].append(df['ch'][(date-timedelta(42)):(date-timedelta(21))].mean())
                xy[date].append(df['ch'][(date-timedelta(63)):(date-timedelta(42))].mean())
                xy[date].append(df['ch'][(date-timedelta(126)):(date-timedelta(63))].mean())
                xy[date].append(df['ch'][(date-timedelta(252)):(date-timedelta(126))].mean())
                xy[date].append(df['vol'][(date-timedelta(5)):date].mean())
                xy[date].append(df['vol'][(date-timedelta(10)):(date-timedelta(5))].mean())
                xy[date].append(df['vol'][(date-timedelta(15)):(date-timedelta(10))].mean())
                xy[date].append(df['vol'][(date-timedelta(21)):(date-timedelta(15))].mean())
                xy[date].append(df['vol'][(date-timedelta(42)):(date-timedelta(21))].mean())
                xy[date].append(df['vol'][(date-timedelta(63)):(date-timedelta(42))].mean())
                xy[date].append(df['vol'][(date-timedelta(126)):(date-timedelta(63))].mean())
                xy[date].append(df['vol'][(date-timedelta(252)):(date-timedelta(126))].mean())
                xy[date] = xy[date] + [max([date <= c and date + timedelta(month * 21) > c \
                          for c in c['crash_st']]) for month in months]
            df_xy = pd.DataFrame.from_dict(xy, orient='index').dropna()
            df_x = df_xy.iloc[:, :-len(months)]
            df_y = df_xy.iloc[:, -len(months):]
            dfs_x.append(df_x)
            dfs_y.append(df_y)
        return dfs_x, dfs_y

    def get_train_test(self, dfs_x, dfs_y, datasets, test_data):
        for i, name in enumerate(datasets):
            if name == test_data:
                index = i
        dfs_x_copy = list(dfs_x)
        dfs_y_copy = list(dfs_y)
        np_x_test = None
        np_y_test = None
        if test_data:
            df_x_test = dfs_x_copy.pop(index)
            df_y_test = dfs_y_copy.pop(index)
            np_x_test = np.array(df_x_test)
            np_y_test = np.array(df_y_test)
        np_x_train = np.concatenate(([np.array(x) for x in dfs_x_copy]))
        np_y_train = np.concatenate(([np.array(y) for y in dfs_y_copy]))
        return np_x_train, np_y_train, np_x_test, np_y_test

    def split_results(self, df_combined, dfs_xy, dataset_names, test_data, y_pred_t_bin, \
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
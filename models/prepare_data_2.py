#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 21:15:58 2018

@author: roman
"""

import numpy as np
import pandas as pd
#import datetime
from datetime import timedelta

class DataLoader():
    def __init__(self, datasets_original, dataset_names):
        self.num_datasets = len(datasets_original)
        self.datasets_original = datasets_original
        self.dataset_names = dataset_names
    
    def get_df_combined(self, crash_thresholds):
    ### df_combined: dataframe for each dataset with price and drawdown information
    ###    drawdonw: dataframe for each dataset with drawdowns (ranked)
    ###    crashes: dataframe for each dataset with crashes (chronologically) 
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
            #datasets[5] = datasets[5].loc['1990-11-09':,:]  
        drawdowns = []
        crashes = []
        # crash thresholds identified in 'eploration.ipynb', 99.5% quartile):
        #crash_thresholds = [-0.091, -0.109, -0.120, -0.144, -0.166, -0.110, -0.233]
        for df, ct in zip(datasets, crash_thresholds):
            pmin_pmax = (df['price'].diff(-1) > 0).astype(int).diff() #<- -1 indicates pmin, +1 indicates pmax
            pmax = pmin_pmax[pmin_pmax == 1]
            pmin = pmin_pmax[pmin_pmax == -1]
            # make sure drawdowns start with pmax, end with pmin:
            if pmin.index[0] < pmax.index[0]:
                pmin = pmin.drop(pmin.index[0])
            if pmin.index[-1] < pmax.index[-1]:
                pmax = pmax.drop(pmax.index[-1])
            D = (np.array(df['price'][pmin.index]) - np.array(df['price'][pmax.index])) \
            / np.array(df['price'][pmax.index])
            d = {'Date':pmax.index, 'drawdown':D, 'd_start': pmax.index, 'd_end': pmin.index}    
            df_d = pd.DataFrame(d).set_index('Date')
            df_d.index = pd.to_datetime(df_d.index, format='%Y/%m/%d')
            df_d = df_d.reindex(df.index).fillna(0)
            df_d = df_d.sort_values(by='drawdown')
            df_d['rank'] = list(range(1,df_d.shape[0]+1))
            drawdowns.append(df_d)
            df_d = df_d.sort_values(by='Date')
            df_c = df_d[df_d['drawdown'] < ct]
            df_c.columns = ['drawdown', 'crash_st', 'crash_end', 'rank']
            c_st = list(df_c['crash_st'])
            d_st = [df['price'][(c_s-timedelta(int(252*5/12))):c_s].idxmax() for c_s in c_st] 
            d_st_adj = [max(d, c_prev) for d, c_prev in zip(d_st[1:], c_st[:-1])]
            d_st_adj = [d_st[0]] + d_st_adj
            df_c['down_st'] = d_st_adj
            crashes.append(df_c)
        df_combined = []  
        for i in range(len(datasets)):
            df_combined.append(pd.concat([datasets[i], drawdowns[i]], axis=1))
        return df_combined, drawdowns, crashes

    def get_df_xy(self, months, sequence, df_combined, crashes, select_features=False, vol=False):
    ### dfs_xy: dataframe for each dataset x (columns 0:-1) and  y (column -1)     
        dfs_xy = []
        if select_features == False:    
            for df, c in zip(df_combined, crashes):
                xy = {}
                #for date in df.index[255:-126]: # <--subtract 126 days in the end
                for i in range(sequence, df.shape[0]-126):
                    date = df.index[i]
                    x_ch = [df['ch'].iloc[i-j] for j in range(sequence)]
                    x_vol = []
                    if vol == True:
                        x_vol = [df['vol'].iloc[i-j] for j in range(sequence)]
                    xy[date] = x_ch + x_vol
                    xy[date].append(max([date <= c and date+timedelta(months * 21) > c \
                      for c in c['crash_st']]))
                df_xy = pd.DataFrame.from_dict(xy, orient='index').dropna()
                dfs_xy.append(df_xy)
        if select_features == True:
            for df, c in zip(df_combined, crashes):
                xy = {}
                for date in df.index[252:-126]: # <--subtract 126 days in the end
                    x_ch_12_6m = df['ch'][(date-timedelta(252)):(date-timedelta(126))].mean()
                    x_ch_6_3m = df['ch'][(date-timedelta(125)):(date-timedelta(63))].mean()
                    x_ch_3_2m = df['ch'][(date-timedelta(62)):(date-timedelta(42))].mean()
                    x_ch_2_1m = df['ch'][(date-timedelta(41)):(date-timedelta(21))].mean()
                    x_ch_4_3w = df['ch'][(date-timedelta(20)):(date-timedelta(15))].mean()
                    x_ch_3_2w = df['ch'][(date-timedelta(15)):(date-timedelta(10))].mean()
                    x_ch_2_1w = df['ch'][(date-timedelta(10)):(date-timedelta(5))].mean()
                    x_ch_1_0w = df['ch'][(date-timedelta(5)):(date-timedelta(0))].mean()
                    x_vol_12_6m = df['vol'][(date-timedelta(252)):(date-timedelta(126))].mean()
                    x_vol_6_3m = df['vol'][(date-timedelta(125)):(date-timedelta(63))].mean()
                    x_vol_3_2m = df['vol'][(date-timedelta(62)):(date-timedelta(42))].mean()
                    x_vol_2_1m = df['vol'][(date-timedelta(41)):(date-timedelta(21))].mean()
                    x_vol_4_3w = df['vol'][(date-timedelta(20)):(date-timedelta(15))].mean()
                    x_vol_3_2w = df['vol'][(date-timedelta(15)):(date-timedelta(10))].mean()
                    x_vol_2_1w = df['vol'][(date-timedelta(10)):(date-timedelta(5))].mean()
                    x_vol_1_0w = df['vol'][(date-timedelta(5)):(date-timedelta(0))].mean()
                    y_c = max([date <= c and date+timedelta(months*21) > c for c in c['crash_st']])
                    xy[date] = [x_ch_12_6m, x_ch_6_3m, x_ch_3_2m, x_ch_2_1m, x_ch_4_3w, \
                          x_ch_3_2w, x_ch_2_1w, x_ch_1_0w, x_vol_12_6m, x_vol_6_3m, x_vol_3_2m, \
                          x_vol_2_1m, x_vol_4_3w, x_vol_3_2w, x_vol_2_1w, x_vol_1_0w, y_c]
                df_xy = pd.DataFrame.from_dict(xy, orient='index').dropna()
                df_xy.columns = ['x_ch_12_6m', 'x_ch_6_3m', 'x_ch_3_2m', 'x_ch_2_1m', 'x_ch_4_3w', 
                          'x_ch_3_2w', 'x_ch_2_1w', 'x_ch_1_0w', 'x_vol_12_6m', 'x_vol_6_3m', 
                          'x_vol_3_2m', 'x_vol_2_1m', 'x_vol_4_3w', 'x_vol_3_2w', 'x_vol_2_1w', 
                          'x_vol_1_0w', 'y_c']
                dfs_xy.append(df_xy)
        return dfs_xy

    def get_train_test(self, dfs_xy, dataset_names, test_data):
    ### get n_datasets * 2 train/test splits: first and last n years of each dataset are chosen
    ### as test set.
    ### np_train: list of all training sets, np_test: list of all corresponding test sets """
        for i, name in enumerate(dataset_names):
            if name == test_data:
                index = i
        dfs_xy_copy = list(dfs_xy)
        df_test = dfs_xy_copy.pop(index)
        np_test = np.array(df_test)
        np_train = np.concatenate(([np.array(xy) for xy in dfs_xy_copy]))
        return np_train, np_test

    def split_results(self, df_combined, dfs_xy, dataset_names, test_data, y_pred_t_bin, \
                      y_pred_tr_bin, y_train, y_test):
    ### split results into individual datasets to plot results
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



    def get_train_test_old(self, dfs_xy, years):
    ### get n_datasets * 2 train/test splits: first and last n years of each dataset are chosen
    ### as test set.
    ### np_train: list of all training sets, np_test: list of all corresponding test sets """
        np_xy = np.concatenate(([np.array(xy) for xy in dfs_xy]))
        split = [0]
        i = 0
        for df in dfs_xy:
            i += df.shape[0]-1
            split.append(i)
        np_train = [] 
        np_test = []        
        n = years * 252
        for j in range(len(dfs_xy) * 2):
            i = round(j/2 + 0.1)
            if j % 2 == 0:
                np_test.append(np_xy[split[i]:split[i]+n, :])
                np_train.append(np.concatenate(([np_xy[0:split[i],:], np_xy[split[i]+n:,:]])))
            if j % 2 == 1:
                np_test.append(np_xy[split[i]-n:split[i], :])
                np_train.append(np.concatenate(([np_xy[0:split[i]-n,:], np_xy[split[i]:,:]])))
        return np_train, np_test
    
    def split_results_old(self, df_combined, dfs_xy, crashes, dataset_names, y_pred_bin_t_all, \
                      y_actual, years):
    ### split results into individual datasets to plot results
        df_combined = [dfc.reindex(dfs.index) for dfc, dfs in zip(df_combined, dfs_xy)]
        dfs_predict = []
        n = 252 * years
        for j in range(len(df_combined)*2):
            #i = round(j/2 + 0.1)
            k = round(j/2 - 0.1)
            if j % 2 == 0:
                df = df_combined[k].iloc[:n, :]
                df['y_pred'] = y_pred_bin_t_all[j]
                df['y'] = y_actual[j]
                dfs_predict.append(df)
            if j % 2 == 1:       
                df = df_combined[k].iloc[-n:, :]
                df['y_pred'] = y_pred_bin_t_all[j]
                df['y'] = y_actual[j]
                dfs_predict.append(df)
        cr_ext = []
        for c in crashes:
            cr_ext.append(c)
            cr_ext.append(c)
        ds_name_ext = []
        for ds_name in dataset_names:
            ds_name_ext.append(ds_name)
            ds_name_ext.append(ds_name)
        return dfs_predict, cr_ext, ds_name_ext

























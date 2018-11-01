#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 20:15:33 2018

@author: roman
"""

import numpy as np
import pandas as pd
from sklearn import metrics
from datetime import datetime, timedelta
from matplotlib import gridspec
import matplotlib.pyplot as plt

class EvaluateResults():
    def __init__(self, y_train_all, y_test_all, y_pred_tr_all, y_pred_t_all, model_name, test_data):
        self.y_train_all = y_train_all
        self.y_test_all = y_test_all
        self.y_pred_t_all = y_pred_t_all
        self.y_pred_tr_all = y_pred_tr_all
        self.model_name = model_name
        self.test_data = test_data
    
    def find_threshold(self, beta, threshold_min, threshold_max, resolution=20):
        precision_tr_all, recall_tr_all, accuracy_tr_all = [], [], []
        precision_t_all, recall_t_all, accuracy_t_all = [], [], [] 
        fbeta_tr_all, fbeta_t_all = [], []
        thresholds = list(np.linspace(threshold_min, threshold_max, resolution))
        for threshold in thresholds:
            precision_tr, recall_tr, accuracy_tr = [], [], []
            precision_t, recall_t, accuracy_t = [], [], []
            y_pred_t_bin_all, y_pred_tr_bin_all = [], []
            score_fbeta_tr, score_fbeta_t = [], []
            for y_train, y_test, y_pred_tr, y_pred_t in zip(self.y_train_all, 
                                                            self.y_test_all, \
                                                            self.y_pred_tr_all,\
                                                            self.y_pred_t_all):
                y_pred_tr_bin = y_pred_tr > threshold
                y_pred_tr_bin = y_pred_tr_bin.astype(int)
                y_pred_tr_bin_all.append(y_pred_tr_bin)
                precision_tr.append(metrics.precision_score(y_train, y_pred_tr_bin))
                recall_tr.append(metrics.recall_score(y_train, y_pred_tr_bin))
                accuracy_tr.append(metrics.accuracy_score(y_train, y_pred_tr_bin))
                score_fbeta_tr.append(metrics.fbeta_score(y_train, y_pred_tr_bin, beta=beta))
                y_pred_t_bin = y_pred_t > threshold
                y_pred_t_bin = y_pred_t_bin.astype(int)
                y_pred_t_bin_all.append(y_pred_t_bin)
                precision_t.append(metrics.precision_score(y_test, y_pred_t_bin))
                recall_t.append(metrics.recall_score(y_test, y_pred_t_bin))
                accuracy_t.append(metrics.accuracy_score(y_test, y_pred_t_bin))
                score_fbeta_t.append(metrics.fbeta_score(y_test, y_pred_t_bin, beta=beta))
            precision_tr_all.append(np.mean(precision_tr)) 
            precision_t_all.append(np.mean(precision_t)) 
            recall_tr_all.append(np.mean(recall_tr)) 
            recall_t_all.append(np.mean(recall_t))
            accuracy_tr_all.append(np.mean(accuracy_tr)) 
            accuracy_t_all.append(np.mean(accuracy_t))
            fbeta_tr_all.append(np.mean(score_fbeta_tr))
            fbeta_t_all.append(np.mean(score_fbeta_t))
        plt.subplot(1,3,1)
        plt.plot(thresholds, precision_tr_all, color='blue')
        plt.plot(thresholds, precision_t_all, color='red')
        plt.title('Precision by threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Precision')
        plt.legend(['training set', 'test set'])
        plt.grid()
        plt.subplot(1,3,2)
        plt.plot(thresholds, recall_tr_all, color='blue')
        plt.plot(thresholds, recall_t_all, color='red')
        plt.title('Recall by threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Recall')
        plt.legend(['training set', 'test set'])
        plt.grid()
        plt.subplot(1,3,3)
        plt.plot(thresholds, fbeta_tr_all, color='blue')
        plt.plot(thresholds, fbeta_t_all, color='red')
        plt.title('F-beta score by threshold')
        plt.xlabel('Threshold')
        plt.ylabel('F-beta score')
        plt.legend(['training set', 'test set'])
        plt.grid()
        plt.tight_layout()
        plt.show()
    
    def training_results(self, threshold, training_set_names, beta=2):
        precision_tr, recall_tr, accuracy_tr, score_fbeta_tr = [], [], [], []
        precision_t, recall_t, accuracy_t, score_fbeta_t = [], [], [], []
        y_pred_tr_bin_all, y_pred_t_bin_all = [], []
        for y_train, y_test, y_pred_tr, y_pred_t in zip(self.y_train_all, self.y_test_all, \
                                                        self.y_pred_tr_all, self.y_pred_t_all):
            y_pred_tr_bin = y_pred_tr > threshold
            y_pred_tr_bin = y_pred_tr_bin.astype(int)
            y_pred_tr_bin_all.append(y_pred_tr_bin)
            precision_tr.append(metrics.precision_score(y_train, y_pred_tr_bin))
            recall_tr.append(metrics.recall_score(y_train, y_pred_tr_bin))
            accuracy_tr.append(metrics.accuracy_score(y_train, y_pred_tr_bin))
            score_fbeta_tr.append(metrics.fbeta_score(y_train, y_pred_tr_bin, beta=beta))
            y_pred_t_bin = y_pred_t > threshold
            y_pred_t_bin = y_pred_t_bin.astype(int)
            y_pred_t_bin_all.append(y_pred_t_bin)
            precision_t.append(metrics.precision_score(y_test, y_pred_t_bin))
            recall_t.append(metrics.recall_score(y_test, y_pred_t_bin))
            accuracy_t.append(metrics.accuracy_score(y_test, y_pred_t_bin))
            score_fbeta_t.append(metrics.fbeta_score(y_test, y_pred_t_bin, beta=beta))
        
        y_tr_pos = [np.mean(y) for y in self.y_train_all]
        y_tr_pred_pos = [np.mean(y_pred) for y_pred in y_pred_tr_bin_all]
        y_t_pos = [np.mean(y) for y in self.y_test_all]
        y_t_pred_pos = [np.mean(y_pred) for y_pred in y_pred_t_bin_all]
        d = {'positive actual tr': np.round(y_tr_pos, 2), \
             'positive pred tr': np.round(y_tr_pred_pos, 2), \
             'precision tr': np.round(precision_tr,2), \
             'recall tr': np.round(recall_tr,2), 'accuracy_tr': np.round(accuracy_tr,2), \
             'score_fbeta tr': np.round(score_fbeta_tr,2), \
             'positive actual t': np.round(y_t_pos, 2), \
             'positive pred t': np.round(y_t_pred_pos, 2), \
             'precision t': np.round(precision_t,2), \
             'recall t': np.round(recall_t,2), 'accuracy_t': np.round(accuracy_t,2), \
             'score fbeta t': np.round(score_fbeta_t,2)}
        results_all = pd.DataFrame.from_dict(d, orient='index')
        results_all.columns = training_set_names  ##############
        print('Results for each train/val split:')
        print(results_all)
        print('\n')
        
        # calculate precision, recall, accuracy for comparable random model
        sum_tr = sum([len(tr) for tr in self.y_train_all])
        pos_tr = sum([sum(tr) for tr in self.y_train_all])
        sum_t = sum([len(t) for t in self.y_test_all])
        pos_t = sum([sum(t) for t in self.y_test_all])
        sum_tr_pred = sum([len(tr) for tr in y_pred_tr_bin_all])
        pos_tr_pred = sum([sum(tr) for tr in y_pred_tr_bin_all])
        sum_t_pred = sum([len(t) for t in y_pred_t_bin_all])
        pos_t_pred = sum([sum(t) for t in y_pred_t_bin_all])

        y_train_pos_actual = pos_tr / sum_tr
        y_train_pos_pred = pos_tr_pred / sum_tr_pred
        rnd_TP = y_train_pos_pred * y_train_pos_actual
        rnd_FP = y_train_pos_pred * (1 - y_train_pos_actual)
        rnd_TN = (1 - y_train_pos_pred) * (1 - y_train_pos_actual)
        rnd_FN = (1 - y_train_pos_pred) * y_train_pos_actual
        rnd_pr_tr = rnd_TP / (rnd_TP+rnd_FP)
        rnd_re_tr = rnd_TP / (rnd_TP+rnd_FN)
        rnd_ac_tr = rnd_TP + rnd_TN
        y_test_pos_actual = pos_t / sum_t
        y_test_pos_pred = pos_t_pred / sum_t_pred
        rnd_TP = y_test_pos_pred * y_test_pos_actual
        rnd_FP = y_test_pos_pred * (1 - y_test_pos_actual)
        rnd_TN = (1 - y_test_pos_pred) * (1 - y_test_pos_actual)
        rnd_FN = (1 - y_test_pos_pred) * y_test_pos_actual
        rnd_pr_t = rnd_TP / (rnd_TP+rnd_FP)
        rnd_re_t = rnd_TP / (rnd_TP+rnd_FN)
        rnd_ac_t = rnd_TP + rnd_TN
        rnd_fbeta_tr = (1 + beta ** 2) * (rnd_pr_tr * rnd_re_t) / ((beta ** 2 * rnd_pr_tr) + rnd_re_tr)
        rnd_fbeta_t = (1 + beta ** 2) * (rnd_pr_t * rnd_re_t) / ((beta ** 2 * rnd_pr_t) + rnd_re_t)
        
        print('Results - average over all train/val splits:')
        print('Positive train cases actual:        '+ str(round(y_train_pos_actual, 2)))
        print('Positive train cases predicted:     '+ str(round(y_train_pos_pred, 2)))
        print('Avg precision train (model/random): '+ str(round(np.mean(precision_tr), 2))\
              + ' / ' + str(round(rnd_pr_tr, 2)))
        print('Avg recall train (model/random):    '+ str(round(np.mean(recall_tr), 2))\
              + ' / ' + str(round(rnd_re_tr, 2)))
        print('Avg accuracy train (model/random):  '+ str(round(np.mean(accuracy_tr), 2))\
              + ' / ' + str(round(rnd_ac_tr, 2)))
        print('Score train fbeta:                  '+ str(round(np.mean(score_fbeta_tr), 2))\
              + ' / ' + str(round(rnd_fbeta_tr, 2)))
        print('Positive test cases actual:         '+ str(round(y_test_pos_actual, 2)))
        print('Positive test cases predicted:      '+ str(round(y_test_pos_pred, 2)))
        print('Avg precision test (model/random):  '+ str(round(np.mean(precision_t), 2))\
              + ' / ' + str(round(rnd_pr_t, 2)))
        print('Avg recall test (model/random):     '+ str(round(np.mean(recall_t), 2))\
              + ' / ' + str(round(rnd_re_t, 2)))
        print('Avg accuracy test (model/random):   '+ str(round(np.mean(accuracy_t), 2))\
              + ' / ' + str(round(rnd_ac_t, 2)))
        print('Score test fbeta:                   '+ str(round(np.mean(score_fbeta_t), 2))\
              + ' / ' + str(round(rnd_fbeta_t, 2)))

    def test_results(self, y_test, y_pred_t, threshold, beta=2):
        y_pred_t_bin = y_pred_t > threshold
        y_pred_t_bin = y_pred_t_bin.astype(int)
        precision_t = metrics.precision_score(y_test, y_pred_t_bin)
        recall_t = metrics.recall_score(y_test, y_pred_t_bin)
        accuracy_t = metrics.accuracy_score(y_test, y_pred_t_bin)
        score_fbeta_t = metrics.fbeta_score(y_test, y_pred_t_bin, beta=beta)
        y_t_pos = np.mean(y_test)
        y_t_pos_actual = sum(y_test) / len(y_test)
        y_t_pos_pred = np.mean(y_pred_t_bin)
        rnd_TP = y_t_pos_pred * y_t_pos
        rnd_FP = y_t_pos_pred * (1 - y_t_pos)
        rnd_TN = (1 - y_t_pos_pred) * (1 - y_t_pos)
        rnd_FN = (1 - y_t_pos_pred) * y_t_pos
        rnd_pr_t = rnd_TP / (rnd_TP+rnd_FP)
        rnd_re_t = rnd_TP / (rnd_TP+rnd_FN)
        rnd_ac_t = rnd_TP + rnd_TN
        rnd_fbeta = (1 + beta ** 2) * (rnd_pr_t * rnd_re_t) / ((beta ** 2 * rnd_pr_t) + rnd_re_t)
        print('Test results (test set: S&P 500):')
        print('Positive test cases actual:         '+ str(round(y_t_pos_actual, 2)))
        print('Positive test cases predicted:      '+ str(round(y_t_pos_pred, 2)))
        print('Precision test (model/random):      '+ str(round(np.mean(precision_t), 2))\
              + ' / ' + str(round(rnd_pr_t, 2)))
        print('Recall test (model/random):         '+ str(round(np.mean(recall_t), 2))\
              + ' / ' + str(round(rnd_re_t, 2)))
        print('Accuracy test (model/random):       '+ str(round(np.mean(accuracy_t), 2))\
              + ' / ' + str(round(rnd_ac_t, 2)))
        print('Score test fbeta:                   '+ str(round(np.mean(score_fbeta_t), 2))\
              + ' / ' + str(round(rnd_fbeta, 2)))
        return y_pred_t_bin
        
    def plot_test_results(self, df, c, t_start, t_end):
        t_start = [datetime.strptime(t, '%Y-%m-%d') for t in t_start]
        t_end = [datetime.strptime(t, '%Y-%m-%d') for t in t_end]
        for t1, t2 in zip(t_start, t_end):
            gs = gridspec.GridSpec(3, 1, height_ratios=[2.5, 1, 1])
            plt.subplot(gs[0])
            y_start = list(df[t1:t2][df.loc[t1:t2, 'y'].diff(-1) < 0].index)
            y_end = list(df[t1:t2][df.loc[t1:t2, 'y'].diff(-1) > 0].index)
            crash_st = list(filter(lambda x: x > t1 and x < t2, c['crash_st']))
            crash_end = list(filter(lambda x: x > t1 and x < t2, c['crash_end']))
            [plt.axvspan(x1, x2, alpha=0.2, color='red') for x1, x2 in zip(y_start, y_end)]
            [plt.axvspan(x1, x2, alpha=0.5, color='red') for x1, x2 in zip(crash_st, crash_end)]
            df_norm = df['price'][t1:t2] / df['price'][t1:t2].max()
            plt.plot(df_norm[t1:t2], color='blue') 
            plt.title(self.model_name + ' Testcase: ' + self.test_data + ' ' + str(t1.year) + '-' \
                      + str(t2.year))
            plt.legend(['price', 'downturn / crash'])
            plt.xticks([])
            plt.grid()     
            plt.subplot(gs[1])
            plt.plot(df.loc[t1:t2, 'vol'])
            [plt.axvspan(x1, x2, alpha=0.2, color='red') for x1, x2 in zip(y_start, y_end)]
            [plt.axvspan(x1, x2, alpha=0.5, color='red') for x1, x2 in zip(crash_st, crash_end)]
            plt.legend(['Volatility'])
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.grid()
            plt.xticks([])
            plt.subplot(gs[2])
            plt.plot(df['y'][t1:t2], color='black')
            plt.plot(df['y_pred'][t1:t2].rolling(10).mean(), color='darkred', linewidth=0.8)
            [plt.axvspan(x1, x2, alpha=0.2, color='red') for x1, x2 in zip(y_start, y_end)]
            [plt.axvspan(x1, x2, alpha=0.5, color='red') for x1, x2 in zip(crash_st, crash_end)]
            plt.legend(['crash within 6m', 'crash predictor'])
            plt.ylim(0, 1.1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.grid()
            plt.show()  
        
        
        
        
        
        
        
        
        
        
        
















    
    
    
    
    
    
    
    
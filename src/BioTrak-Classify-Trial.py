
# coding: utf-8

# # Brian McKean 
# # ===========
# # BioTrak Health 
# https://www.biotrakhealth.com/
# ##  Session Management
# ## Identify ineffective sessions during user trials
# # ===========
# ## Galvanize Data Science Immersion
# ## Capstone Project
#
# This program takes as input 
#   text files with sessions information
#   pickled model
#
# as output
#   report.csv -- report format including classification
#   model.csv -- the model features with classification
#   sessions.csv -- sessin data with classification

from docx import Document
from docx.shared import Inches
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.parasite_axes import SubplotHost
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import seaborn as sns
import pymongo
import requests
import scipy.stats as stats
from sklearn.externals import joblib
import datetime


# ## Get Data from files


# Read in data
df_sessions = pd.read_json('sessions_latest.txt')
print df_sessions.columns
df_trial_users = pd.read_csv('TrialUsers.csv',header=None)

#
# ### Data Status Summary
# 
print len(df_sessions), "sessions are uploaded"


# ### Set up function to count cycles and biggest swing

def count_cycles(s, pts=10, intv=600):
    '''
    Looks at series
    for each 'intv' points
    - count crossings of 'pts' lookback moving average
    INPUTS:
        s = list of measurements
        pts = how many pts to collect for moving average
        intv = interval to measure crossing (600 = 10 sec)
    OUTPUTS
        cycles / sec
        biggest_move = on a crossing biggest move

    '''
    if len(s) < intv:
        return 0, 0
    if pts >= intv:
        return 0,0
    s = np.array(s)
    crossing_counts = [0]  
    index = pts
    max_swing = 0
    up = True # True for last cross up, False for down
    while (index + intv < len(s)):
        #print index
        interval_crossings = 0
        for i in range(intv):
            last_n = s[index-pts:index]
            avg = last_n.mean()
            if up and s[index] < avg:
                up = False
                interval_crossings += 1
                swing = abs(s[index]-s[index-1])
                if swing > max_swing:
                    max_swing = swing
            if not up and s[index] > avg:
                up = True
                interval_crossings += 1
                swing = abs(s[index]-s[index-1])
                if swing > max_swing:
                    max_swing = swing
            index += 1
        crossing_counts.append(interval_crossings)
   
    time_sec = intv/60.0
    return np.array(crossing_counts).mean()/(2*time_sec), max_swing
            


# #### Create Feature Set for Model Analysis
# 
# Calculated from the 'average_data' field  
# 1. Length
# 2. Min
# 3. Max
# 4. Mean
# 5. Std
# 6. Skew
# 7. Kurt
# 8. Cycles - average frequency
# 9. Swing - max swing on a direction change
# 
# In order to incorporate the heuristics I add the following features  
# 1. min_len (T/F) -- is the length at least 3600 units (60 seconds)  
# 2. pegged_L (T/F) -- at least one value of 0 after 60 seconds  
# 3. pegged_H (T/F) -- at least one value of 4095 afte 60 seconds  


def make_features(df, use_skew_kurt=False):
    '''
    Makes the features that can be used for analysis
    INPUTS:
        df - a data frame containing session data
        use_skew_kut - if True include skew and kurtosis in analysis
    OUTPUTS:
        df_features - a data frame with info and created features
        feature_list - the lists of features to use for analysis
        
    '''
    #
    # Set up minimal data frame and feature list
    #
    df_copy = df.copy()
    feature_list = ['avg_data_mean',
        'avg_data_max',  'avg_data_min',  'avg_data_std', 'avg_data_len', 
            'cycles','max_swing','min_len','pegged_low','pegged_high']
    if use_skew_kurt:
        feature_list.extend(['avg_data_skew','avg_data_kurt'])
    df_features = df_copy[['Session_name','userid']]
    df_features['session_num'] = df_copy.index
    
    #
    # Get details of sesison data by coverting to dictionary
    #
    dfSession = df_copy.join(pd.DataFrame(df_sessions["session_data"].to_dict()).T)
    avg_data = dfSession['average_data'].str.replace(" ","").str.split(',').str[:-1].map(lambda xx: np.array([int(yy) for yy in xx]))
    # Some of the time series are empty, if so insert an entry of one sample at 0
    # Signal is inverted -- need to change it 
    avg_data = avg_data.map(lambda x: np.array( [0] if not len(x) else x ))
    avg_data = avg_data.map(lambda x: 4095 - x)
    
    #
    # Add features
    #
    df_features['avg_data_len'] = avg_data.map(lambda x: len(x))
    df_features['avg_data_max'] = avg_data.map(lambda x: x.max()).fillna(0)
    df_features['avg_data_min'] = avg_data.map(lambda x: x.min()).fillna(0)
    df_features['avg_data_mean'] = avg_data.map(lambda x: x.mean()).fillna(0)
    df_features['avg_data_std'] = avg_data.map(lambda x: x.std()).fillna(0)
    if use_skew_kurt:
        df_features['avg_data_skew'] = avg_data.map(lambda x: stats.skew(x)).fillna(0)
        df_features['avg_data_kurt'] = avg_data.map(lambda x: stats.kurtosis(x)).fillna(0)
    
    # Count cycles and swings for each avg_data series
    cycles = []
    max_swing = []
    for idx, elem in enumerate(avg_data):
        a,b = count_cycles(elem)
        cycles.append(a)
        max_swing.append(b)
    df_features['cycles'] = cycles
    df_features['max_swing'] = max_swing
    #
    # Add Heuristics

    df_features['min_len'] = df_features['avg_data_len']>3600
    df_features['pegged_low'] = [False if (len(x)<3600 or min(x[3600:])>0) else True for x in avg_data]
    df_features['pegged_high'] = [False if (len(x)<3600 or max(x[3600:])<4095) else True for x in avg_data]
    
    
    return df_features, feature_list
    

df_results, df_key_columns = make_features(df_sessions)

clf = joblib.load('brtakrf_class.pkl')

X = df_results[df_key_columns].as_matrix()
y_pred = clf.predict(X)

df_sessions['Not_effective'] = y_pred 
df_results['Not_effective'] = y_pred

#print "Bad=",len( [x  for x in y_pred if x])
#print "Good=",len( [x for x in y_pred if not x])

df_trial_results = df_results[df_results['userid'].isin(df_trial_users[0])]
df_trial_sessions = df_sessions[df_sessions['userid'].isin(df_trial_users[0])]

df_trial_sessions.reset_index(inplace=True)
df_trial_sessions['session_num']=df_trial_sessions['index']


columns =  ['session_num','userid', 'Local_date','Start_local_time','Session_name', 'halocalm_score', 'baseline_score','below_threshold', 'min', 'above_threshold','max','mean','Good_Session']
index = np.arange(0,len(df_trial_sessions))
df_report = pd.DataFrame(index = index, columns = columns)
df_report.fillna('N/A')
df_trial_session_info =  df_trial_sessions.join(pd.DataFrame(df_trial_sessions["Session_data_header"].to_dict()).T)

for col in columns[:-1]:
    df_report[col] = df_trial_session_info[col]
    print col, df_trial_session_info[col], df_report[col]
    

df_report['Good_Session'] = np.where(df_trial_sessions['Not_effective']==False, 'Yes', 'No')

fname = 'Trial_report.csv'
df_report.to_csv(fname)
fname = 'Trial_model_report.csv'
df_trial_results.to_csv(fname)
fname = 'Trial_sessions_report.csv'
df_trial_sessions.to_csv(fname)


















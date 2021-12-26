import matplotlib.pyplot as plt
import matplotlib.patches as pat
import csv
import pandas as pd
import numpy as np

# Summary
#-------------------------------------------------------------------
#
#    This file contains the functions to calculate ECE (a measure 
#      of calibration) and to make some simple calibration graphs.
#      As it is, these are ultility functions and the lines of code 
#      at the bottom are an example only.
#
#-------------------------------------------------------------------

# Notes on input
#-------------------------------------------------------------------
#
#   These functions assume that the outputs of the model to have 
#     stored the values to be evaluated in a csv of the following
#     format listed below. However, if you wish to use your own
#     format, the column names must be changed throughout.
#
#   Format:
#   The column headers "correct,softmax"
#   Each row contains a 0 or 1 for correct and the softmax as double.
#
#   example.csv
#   correct,softmax
#   0,.9
#   1,.75
#   
#-------------------------------------------------------------------




#Get the pandas dataframe from the input csv 
def get_cali_df(filename):
    df = pd.read_csv(filename)
    df = bin_and_count(df)
    df['x'] = rrange(0,1,0.05)
    return df




#gets the ece value
def get_ece(df):
    df = get_ece_column(df)
    ece = df['ece_comp'].sum()
    return ece




#Simple calibration graph that saves to savefile
def graph_calibration(df, savefile):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    
    df.plot.bar('x','correct',ax=ax,legend=False)
    ax.set_xticklabels(df.bin)
    ax.plot([0,20],[0,1],'k--')
    ax.set_xlabel('Bins')
    ax.set_ylabel('Accuracy')

    fig.savefig(savefile)

    
    

#Both bins the softmax values and counts the objects in each bin
def bin_and_count(df):
    df['bin'] = pd.cut(df['softmax'],np.linspace(0,1,int(1/0.05)+1),include_lowest=False)
    df_a = df.groupby('bin').mean().reset_index()
    df_b = df[['bin']].groupby('bin').size().rename('count').reset_index()
    df = df_a.merge(df_b,on='bin')
    return df




#Builds a df with all the column for ece for each bin, 'ece_comp', if you want
def get_ece_column(df):
    df['n']=sum(df['count'])
    df['ece_comp']=df.apply(lambda r : (r['count']/r['n'])*abs(r['correct']-r['softmax']),axis=1)
    return df


#Utility for binning
def rrange(start,stop,step=1):
    vals=[]
    while start<=stop:
        vals.append(start)
        start+=step
    return vals




#Example use

df=get_cali_df('./cali_test.csv')
print(get_ece(df))
graph_calibration(df,'test.png')
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 20:39:55 2018

@author: jiangjiechu
"""
import cv2
from copy import deepcopy
from numpy import *
def GaussianFilter(n):
    filter_list = zeros(n)
    mu = (n+1)/2
    sigma = mu/3
    for i in arange(n):
        filter_list[i]=exp(-square(i+1-mu)/(2*square(sigma)))
    return filter_list/sum(filter_list)

def Convolution(data,n):
    filter_list = GaussianFilter(n)
    di = int((n-1)/2)
    data_new = zeros(data.size)
    for i in arange(di,data.size-di):
        #-print("%d\n"%i)
        data_new[i] = sum(data[i-di:i+di+1]*filter_list)
    data_new[0:di]=data[0:di]
    data_new[data.size-di:data.size]=data[data.size-di:data.size]
    return data_new

def FindCandidates(data,threshold,window):
    candidates = zeros(data.size,uint)
    for i in arange(data.size-window):
        if candidates[i]==0:
            if data[i]<threshold:
                candidates[i:min(data.size-1,i+window)] = 1    
    return candidates

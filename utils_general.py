# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:15:54 2024

@author: dowel
"""

import numpy as np
import pickle
import pandas as pd
#%%
class utils_general():
    def __init__(self):
        self.version = 2401016
        
    def circ_subtract(a,b):
        adiff = a-b
        sindiff = np.sin(adiff)
        cosdiff = np.cos(adiff)
        return np.arctan2(sindiff,cosdiff)
        
    def round_to_sig_figs(x, sig_figs):
        if x == 0:
            return 0
        return np.round(x, sig_figs - int(np.floor(np.log10(abs(x)))) - 1)
    def find_blocks(x,mergeblocks=False,merg_threshold = 5):
        x = np.append(x,0)
        x = np.append(0,x)
        dx =np.diff(x)
        bst = np.where(dx==1)[0]
        bed = np.where(dx==-1)[0]
        blocksize = bed-bst
        
        blockstart = bst
        if mergeblocks: # Merges blocks with small separations
            blockstart2 = blockstart+blocksize
            bdist = blockstart[1:]-blockstart2[:-1]
            bmergers = np.where(bdist<merg_threshold)[0]
            blockstart3 = np.delete(blockstart,bmergers+1)
            blocksize2 = np.zeros_like(blockstart3)
            for ib,b in enumerate(blockstart3):
               
                if ib<len(blockstart3)-1:
                    ibdx = np.where(blockstart==b)[0][0]
                    nbdx = np.where(blockstart==blockstart3[ib+1])[0][0]
                    blocksize2[ib] = blockstart[nbdx-1]+blocksize[nbdx-1]-blockstart[ibdx]
                else:
                    blocksize2[-1] = blocksize[-1]
            
            
            # blocksize[bmergers] = blocksize[bmergers]+blocksize[bmergers+1] 
            # blocksize = np.delete(blocksize,bmergers+1)
            # blockstart = np.delete(blockstart,bmergers+1)
            
            blockstart = blockstart3
            blocksize = blocksize2
        
        return blockstart, blocksize
    def find_nearest(search,x):
        df = search-x
        i = np.argmin(np.abs(df))
        
        return i
    def find_nearest_euc(search,x):
        df = search-x
        df = np.sum(df**2,axis=1)
        i = np.argmin(np.abs(df))
        
        return i
    def save_pick(data,savename):
        

        with open(savename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def load_pick(savename):
        with open(savename, 'rb') as handle:
            data = pickle.load(handle)
            
        return data
    
    def fictrac_repair(x,y):
        dx = np.abs(np.diff(x))
        dy = np.abs(np.diff(y))
        lrgx = dx>5 
        lrgy = dy>5 
        bth = np.logical_or(lrgx, lrgy)
        
        fixdx =[i+1 for i,b in enumerate(bth) if b]
        for i,f in enumerate(fixdx):
            
            x[f:] =x[f:]- (x[f]-x[f-1])
            
            y[f:] = y[f:]- (y[f]-y[f-1])
        return x, y

    def get_ft_time(df):
        t = pd.Series.to_numpy(df['timestamp'])
        t = np.array(t,dtype='str')
        t_real = np.empty(len(t),dtype=float)
        for i,it in enumerate(t):
            tspl = it.split('T')
            tspl2 = tspl[1].split(':')
            t_real[i] = float(tspl2[0])*3600+float(tspl2[1])*60+float(tspl2[2])
        t_real = t_real-t_real[0]
        return t_real
    
    def time_varying_correlation(x,y,window):
        """ 
        Function gets the time varying correlation between two signals, measured
        over a window specified
        """
        
        iter = len(x)-window
        output = np.zeros(len(x))
        for i in range(iter):
            idx = np.arange(i,i+window)
            cor = np.corrcoef(x[idx],y[idx])
            
            output[i+window] = cor[0,1] 
        return output
        
        
        
#         function [blockstart, blocksize] = findblocks(data,condition,val)
# % function takes array of ones and zeros and finds instances where there
# % is a contiguous block and returns the start and size
# % CD 03/03/2020
# if size(data,2)>size(data,1)
#     data = data';
# end
# if sum(abs(data)>1)>0
#     error('Function only takes vectors of ones and zeros')
# end
# data = [0;data;0]; %pad data to find edges
# df = diff(data);
# if sum(abs(df))==0
#     blockstart = [];
#     blocksize = [];
#     return
# end
# bst = find(df==1);
# bed = find(df==-1);
# blocksize = bed-bst;
# blockstart = bst;
# [blocksize,i] = sort(blocksize,'descend');
# blockstart = blockstart(i);
# if nargin>1
#     switch condition
#         case 'runorder'
#             [blockstart,i] = sort(blockstart,'ascend');
#             blocksize = blocksize(i);
#         case 'max'
#             blockstart = blockstart(1);
#             blocksize = blocksize(1);
#         case 'first'
#             [blockstart,i] = min(blockstart);
#             blocksize = blocksize(i);
#         case 'minsize'
#             [blockstart,i] = sort(blockstart,'ascend');
#             blocksize = blocksize(i);
#             blockstart = blockstart(blocksize>=val);
#             blocksize = blocksize(blocksize>=val);
#         case 'maxsize'
#             [blockstart,i] = sort(blockstart,'ascend');
#             blocksize = blocksize(i);
#             blockstart = blockstart(blocksize<=val);
#             blocksize = blocksize(blocksize<=val);
#     end
# end
# end
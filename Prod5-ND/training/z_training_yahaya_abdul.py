#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import time
import math
import warnings
import struct
import binascii
import pandas as pd
import numpy as np
import pickle
import h5py
import tensorflow as tf
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from datetime import datetime
from sklearn import preprocessing
from tqdm import tqdm,tqdm_notebook
from IPython.display import display, clear_output
from matplotlib.image import imread # read images
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from numpy.polynomial.polynomial import polyfit
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Activation,concatenate
from tensorflow.keras.optimizers import Adam #optimizer
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
from tensorflow.python.client import device_lib
from sklearn.preprocessing import MinMaxScaler # normalize and scale data
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score,r2_score


# In[ ]:


# create directory paths for train & test data sets
data_dir='/home/m962g264/ondemand/research_repo/data-preprocess'
train_path=data_dir+'/ND_MC_Nominal_RHC_traindata/'
test_path=data_dir+'/ND_MC_Nominal_RHC_testdata/'
print('Training files path: \t{}'.format(train_path))
print('Validation files path: \t{}'.format(test_path))


# In[ ]:


train_files = list(set(n for n in os.listdir(train_path) if n.endswith(".h5")))
test_files = list(set(n for n in os.listdir(test_path) if n.endswith(".h5")))
# train & test size
train_idx=0
test_idx=0
for h5_filename in os.listdir(train_path):
    #Adding these two lines to avoid errors because there are extra directories in the FHC set for some reason
    if os.path.isdir(h5_filename)==True:
        continue
    
    train_idx=train_idx+len((h5py.File(train_path+h5_filename,'r')['run'][:]))
for h5_filename in os.listdir(test_path):
    test_idx=test_idx+len((h5py.File(test_path+h5_filename,'r')['run'][:]))

# training events are broad scoped while validation events are quality cut ND contained. further train/test splitting is done later.
print('Number of training files:\t{}\nNumber of validation files:\t{}'.format(len(os.listdir(train_path)), len(os.listdir(test_path))))
print('Number of training events:\t{}\nNumber of validation events:\t{}'.format((train_idx), (test_idx)))


# In[ ]:


# test interaction is canoncial for training files
df=h5py.File(train_path+os.listdir(train_path)[0],'r')
print(list(df.keys()))

f=h5py.File(test_path+os.listdir(test_path)[0],'r')
print(list(f.keys()))


# In[ ]:


# read all the h5 files
idx=0
energy,pdg,cvnmap,test_cvnmap,truerecovtxx,truerecovtxy,truerecovtxz,test_mode,test_iscc,testtruerecovtxx,testtruerecovtxy,testtruerecovtxz=([] for i in range(12))
for h5_filename in os.listdir(train_path):
        #Adding these two lines to avoid errors because there are extra directories in the FHC set for some reason
    if os.path.isdir(h5_filename)==True:
        continue
    print('Processing... {} of {}'.format(idx,len(os.listdir(train_path))), end="\r", flush=True)
    energy=np.append(energy, h5py.File(train_path+h5_filename,'r')['E'][:],axis=0)
    pdg=np.append(pdg,h5py.File(train_path+h5_filename,'r')['PDG'][:],axis=0)
    cvnmap.append(h5py.File(train_path+h5_filename,'r')['cvnmap'][:])
    truerecovtxz.append(h5py.File(train_path+h5_filename,'r')['TrueRecoVtxZ'][:])
    test_cvnmap.append(h5py.File(test_path+h5_filename,'r')['cvnmap'][:])
    train_mode=np.append(test_mode,h5py.File(train_path+h5_filename,'r')['Mode'][:],axis=0)
    train_iscc=np.append(test_iscc,h5py.File(train_path+h5_filename,'r')['isCC'][:],axis=0)
    test_mode=np.append(test_mode,h5py.File(test_path+h5_filename,'r')['Mode'][:],axis=0)
    test_iscc=np.append(test_iscc,h5py.File(test_path+h5_filename,'r')['isCC'][:],axis=0)
    testtruerecovtxz.append(h5py.File(test_path+h5_filename,'r')['TrueRecoVtxZ'][:])
    idx+=1

# convert to np array
truerecovtxz=np.array(truerecovtxz)
testtruerecovtxz=np.array(testtruerecovtxz)
print('Test & Validation files read successful') 


# In[ ]:


# normalize cvnmap for CNN processing
# training data
idx=file=0
cvnmap_norm,test_cvnmap_norm=([] for i in range(2))
while idx < (len(os.listdir(train_path))):
        #seeing if we can fix the index problem
    #if idx==2001:
        #break
    
    cvnmap_norm.append(preprocessing.normalize(cvnmap[idx],axis=1))
    idx+=1
# testing data
while file < (len(os.listdir(test_path))):
    test_cvnmap_norm.append(preprocessing.normalize(test_cvnmap[file],axis=1))
    file+=1
# convert to np array
cvnmap_norm=np.array(cvnmap_norm)
test_cvnmap_norm=np.array(test_cvnmap_norm)


# In[ ]:


# extract true vertices for model training processing and model validation
truevtxz,recovtxz,testtruevtxz,testrecovtxz=([] for i in range(4))
idx=0
while idx < (len(os.listdir(train_path))):
        print('Processing...', end="\r", flush=True)
        event=0
        #seeing if we can fix the index problem
        
        #if idx==2001:
            #break
        
        while event < (truerecovtxz[idx].shape[0]):
            truevtxz=np.append(truevtxz,truerecovtxz[idx][event][0])
            recovtxz=np.append(recovtxz,truerecovtxz[idx][event][1])
            event+=1
        idx+=1
print('Training preprocessing complete\n', end="\r", flush=True)
idx=0
while idx < (len(os.listdir(test_path))):
        print('Processing...', end="\r", flush=True)
        event=0
        while event < (testtruerecovtxz[idx].shape[0]):
            testtruevtxz=np.append(testtruevtxz,testtruerecovtxz[idx][event][0])
            testrecovtxz=np.append(testrecovtxz,testtruerecovtxz[idx][event][1])
            event+=1
        idx+=1
print('Testing preprocessing complete\n', end="\r", flush=True)
# convert to np arrays
testtruevtxz=np.array(testtruevtxz)
testrecovtxz=np.array(testrecovtxz)


# In[ ]:


# split normalized cvnmap into reshaped events with multi-views
a,b,c,d,cvnmap_norm_resh,cvnmap_norm_resh_xz,cvnmap_norm_resh_yz,test_cvnmap_norm_resh,test_cvnmap_norm_resh_xz,test_cvnmap_norm_resh_yz=([] for i in range(10))
file=event=0
## training CVN map view split ##
while file < (len(os.listdir(train_path))):
        #seeing if we can fix the index problem
    #if idx==2001:
        #break
    
    a=cvnmap_norm[file]
    print('Processing train cvnmap file {} of {}'.format(file+1, (len(os.listdir(train_path)))), end="\r", flush=True)
    event=0
    while event < (a.shape[0]):
        b=a[event].reshape(2,100,80)
        cvnmap_norm_resh.append(b)
        cvnmap_norm_resh_xz.append(b[0])
        cvnmap_norm_resh_yz.append(b[1])
        event+=1
    file+=1
file=event=0
while file < (len(os.listdir(test_path))):
    c=test_cvnmap_norm[file]
    print('Processing tests cvnmap file {} of {}'.format(file+1, (len(os.listdir(test_path)))), end="\r", flush=True)
    event=0
    while event < (c.shape[0]):
        d=c[event].reshape(2,100,80)
        test_cvnmap_norm_resh.append(d)
        test_cvnmap_norm_resh_xz.append(d[0])
        test_cvnmap_norm_resh_yz.append(d[1])
        event+=1
    file+=1
print('\ncvnmap processing complete')


# In[ ]:


#Defines the finalstate array! Need to make a numpy array from the extracted final state pdg information
pdgpath='/home/m962g264/ondemand/research_repo/data-preprocess/ND_MC_Nominal_Files_RHC_mominkhan/'
idx=0
for h5_filename in os.listdir(pdgpath):
        #Adding these two lines to avoid errors because there are extra directories in the FHC set for some reason
    if os.path.isdir(h5_filename)==True:
        continue
        
    print('Processing... {} of {}'.format(idx,len(os.listdir(train_path))), end="\r", flush=True)
   
    g=h5py.File(pdgpath+h5_filename,'r')
    finalstate=g['rec.mc.nu.prim']['pdg'][:]
 
    idx+=1
    
# convert to np array
print('Test & Validation files read successful') 


# **Split Datasets by Interaction Types**

# In[ ]:


test_cvnmap_qe_xz,test_cvnmap_qe_yz,test_cvnmap_res_xz,test_cvnmap_res_yz,test_cvnmap_coh_xz,test_cvnmap_coh_yz,test_cvnmap_mec_xz,test_cvnmap_mec_yz,test_cvnmap_dis_xz,test_cvnmap_dis_yz=([] for i in range(10))
qe_true_vtxz,res_true_vtxz,dis_true_vtxz,coh_true_vtxz,mec_true_vtxz,qe_reco_vtxz,res_reco_vtxz,mec_reco_vtxz,coh_reco_vtxz,dis_reco_vtxz=([] for i in range(10))
x=idx=0
print('Processing {} test files. Splitting events by interaction mode.'.format((len(test_mode))))
#############MODE####################################################
while idx < (len(test_mode)):
    time.sleep(0.001) # hesitates for 0.001 seconds to prevent server comm errors
    print('MODE RUN: Processing train file {}'.format(idx), end="\r", flush=True)
    if test_mode[idx] == 0.0: # quasi-elastic
        test_cvnmap_qe_xz.append(test_cvnmap_norm_resh_xz[idx])
        test_cvnmap_qe_yz.append(test_cvnmap_norm_resh_yz[idx])
        qe_true_vtxz.append(testtruevtxz[idx])
        qe_reco_vtxz.append(testrecovtxz[idx])
    elif test_mode[idx] == 1.0: # resonance state
        test_cvnmap_res_xz.append(test_cvnmap_norm_resh_xz[idx])
        test_cvnmap_res_yz.append(test_cvnmap_norm_resh_yz[idx])
        res_true_vtxz.append(testtruevtxz[idx])
        res_reco_vtxz.append(testrecovtxz[idx])
    elif test_mode[idx] == 2.0: # deep inelastic
        test_cvnmap_dis_xz.append(test_cvnmap_norm_resh_xz[idx])
        test_cvnmap_dis_yz.append(test_cvnmap_norm_resh_yz[idx])
        dis_true_vtxz.append(testtruevtxz[idx])
        dis_reco_vtxz.append(testrecovtxz[idx])
    elif test_mode[idx] == 3.0: # Coh
        test_cvnmap_coh_xz.append(test_cvnmap_norm_resh_xz[idx])
        test_cvnmap_coh_yz.append(test_cvnmap_norm_resh_yz[idx])
        coh_true_vtxz.append(testtruevtxz[idx])
        coh_reco_vtxz.append(testrecovtxz[idx])
    elif test_mode[idx] == 10.0: #Mec
        test_cvnmap_mec_xz.append(test_cvnmap_norm_resh_xz[idx])
        test_cvnmap_mec_yz.append(test_cvnmap_norm_resh_yz[idx])
        mec_true_vtxz.append(testtruevtxz[idx])
        mec_reco_vtxz.append(testrecovtxz[idx])
    idx+=1
print('\nJob complete.')


# In[ ]:


#############CC/NC###################################################
test_cvnmap_cc_xz,test_cvnmap_cc_yz,test_cvnmap_nc_xz,test_cvnmap_nc_yz=([] for i in range(4))
cc_true_vtxz,nc_true_vtxz,cc_reco_vtxz,nc_reco_vtxz=([] for i in range(4))
x=idx=0
print('Processing {} test files. Splitting events by interactions current charge.'.format((len(test_mode))))
while idx < (len(test_mode)):
    time.sleep(0.001) # hesitates for 0.001 seconds to prevent server comm errors
    print('CC/NC RUN: Processing train file {}'.format(idx), end="\r", flush=True)
    if test_iscc[idx] == 1.0: # cc
        test_cvnmap_cc_xz.append(test_cvnmap_norm_resh_xz[idx])
        test_cvnmap_cc_yz.append(test_cvnmap_norm_resh_yz[idx])
        cc_true_vtxz.append(testtruevtxz[idx])
        cc_reco_vtxz.append(testrecovtxz[idx])
    elif test_iscc[idx] == 0.0: # nc
        test_cvnmap_nc_xz.append(test_cvnmap_norm_resh_xz[idx])
        test_cvnmap_nc_yz.append(test_cvnmap_norm_resh_yz[idx])
        nc_true_vtxz.append(testtruevtxz[idx])
        nc_reco_vtxz.append(testrecovtxz[idx])
    idx+=1
print('\nJob complete.')


# In[ ]:


# extract reco data sets from training
qe_reco_vtxz,res_reco_vtxz,dis_reco_vtxz,coh_reco_vtxz,mec_reco_vtxz=([] for i in range(5))
qe_reco_true_vtxz,res_reco_true_vtxz,dis_reco_true_vtxz,coh_reco_true_vtxz,mec_reco_true_vtxz=([] for i in range(5))
x=idx=0
print('Processing {} train files. Splitting events by interaction mode.'.format((len(train_mode))))
#############MODE####################################################
while idx < (len(train_mode)):
    time.sleep(0.001) # hesitates for 0.001 seconds to prevent server comm errors
    print('MODE RUN: Processing train file {}'.format(idx), end="\r", flush=True)
    if train_mode[idx] == 0.0: # quasi-elastic
        qe_reco_vtxz.append(recovtxz[idx])
        qe_reco_true_vtxz.append(truevtxz[idx])
    elif train_mode[idx] == 1.0: # resonance state
        res_reco_vtxz.append(recovtxz[idx])
        res_reco_true_vtxz.append(truevtxz[idx])
    elif train_mode[idx] == 2.0: # deep inelastic
        dis_reco_vtxz.append(recovtxz[idx])
        dis_reco_true_vtxz.append(truevtxz[idx])
    elif train_model[idx] == 3.0: #coh
        coh_reco_vtxz.append(recovtxz[idx])
        coh_reco_true_vtxz.append(truevtxz[idx])
    elif train_model[idx] == 10.0: #Mec
        mec_reco_vtxz.append(recovtxz[idx])
        mec_reco_true_vtxz.append(truevtxz[idx])
    idx+=1
print('\nJob complete.')


# In[ ]:


#############CC/NC###################################################
cc_reco_vtxz,nc_reco_vtxz,cc_reco_true_vtxz,nc_reco_true_vtxz=([] for i in range(4))
x=idx=0
print('Processing {} test files. Splitting events by interactions current charge.'.format((len(train_mode))))
while idx < (len(train_mode)):
    time.sleep(0.001) # hesitates for 0.001 seconds to prevent server comm errors
    print('CC/NC RUN: Processing train file {}'.format(idx), end="\r", flush=True)
    if train_iscc[idx] == 1.0: # cc
        cc_reco_vtxz.append(recovtxz[idx])
        cc_reco_true_vtxz.append(truevtxz[idx])
    elif train_iscc[idx] == 0.0: # nc
        nc_reco_vtxz.append(recovtxz[idx])
        nc_reco_true_vtxz.append(truevtxz[idx])
    idx+=1
print('\nJob complete.')


# In[ ]:


##### Split by pdg number for pions #######
finalstatepizero,finalstatepiplus,pizerorecovtxz,pizerotruevtxz,pizerotestrecovtxz,pizerotesttruevtxz,piplusrecovtxz,piplustruevtxz,piplustestrecovtxz,piplustesttruevtxz=([] for i in range(10))
x=idx=0
print('Processing {} test files. Splitting events by final state pdg code.'.format((len(finalstate))))
while idx < (len(finalstate)):
    time.sleep(0.001) # hesitates for 0.001 seconds to prevent server comm errors
    print('PI 0 RUN: Processing train file {}'.format(idx), end="\r", flush=True)
    if finalstate[idx] == 111: # pdg code for Pi 0 particle
        
        pizerorecovtxz.append(recovtxz[idx])
        pizerotruevtxz.append(truevtxz[idx])
    
        
        pizerotestrecovtxz.append(testrecovtxz[idx])
        pizerotesttruevtxz.append(testtruevtxz[idx])
    
        
        
    elif finalstate[idx] == 211: #pd code for a pi +
        finalstatepiplus.append(finalstate[idx])
        
        piplusrecovtxz.append(recovtxz[idx])
        piplustruevtxz.append(truevtxz[idx])
        
        piplustestrecovtxz.append(testrecovtxz[idx])
        piplustesttruevtxz.append(testtruevtxz[idx])
       

    idx+=1
print('\nJob complete.')


# In[ ]:


##### Split by pdg number for muon #######
finalstatemuon,finalstateantimuon,muonrecovtxz,muontruevtxz,muontestrecovtxz,muontesttruevtxz,antimuonrecovtxz,antimuontruevtxz,antimuontestrecovtxz,antimuontesttruevtxz=([] for i in range(10))
x=idx=0
print('Processing {} test files. Splitting events by final state pdg code.'.format((len(pdg))))
while idx < (len(finalstate)):
    time.sleep(0.001) # hesitates for 0.001 seconds to prevent server comm errors
    print('MUON RUN: Processing train file {}'.format(idx), end="\r", flush=True)
    if finalstate[idx] == 13: # pdg code for normal muon particle
       
        
        finalstatemuon.append(finalstate[idx])
        
        muonrecovtxz.append(recovtxz[idx])
        muontruevtxz.append(truevtxz[idx])
        
        muontestrecovtxz.append(testrecovtxz[idx])
        muontesttruevtxz.append(testtruevtxz[idx])
        
        
    elif finalstate[idx] == -13: #pd code for an anti muon
        finalstateantimuon.append(finalstate[idx])
        
        antimuonrecovtxz.append(recovtxz[idx])
        antimuontruevtxz.append(truevtxz[idx])
        
        antimuontestrecovtxz.append(testrecovtxz[idx])
        antimuontesttruevtxz.append(testtruevtxz[idx])
       

    idx+=1
print('\nJob complete.')


# In[ ]:


##### Split by pdg number for electron #######
finalstateelectron,finalstatepositron,electronrecovtxz,electrontruevtxz,electrontestrecovtxz,electrontesttruevtxz,positronrecovtxz,positrontruevtxz,positrontestrecovtxz,positrontesttruevtxz=([] for i in range(10))
x=idx=0
print('Processing {} test files. Splitting events by final state pdg code.'.format((len(pdg))))
while idx < (len(finalstate)):
    time.sleep(0.001) # hesitates for 0.001 seconds to prevent server comm errors
    print('ELECTRON RUN: Processing train file {}'.format(idx), end="\r", flush=True)
    if finalstate[idx] == 11: # pdg code for electron
       
        
        finalstateelectron.append(finalstate[idx])
        
        electronrecovtxz.append(recovtxz[idx])
        electrontruevtxz.append(truevtxz[idx])
        
        electrontestrecovtxz.append(testrecovtxz[idx])
        electrontesttruevtxz.append(testtruevtxz[idx])
        
        
    elif finalstate[idx] == -11: #pd code for an anti muon
        finalstatepositron.append(finalstate[idx])
        
        positronrecovtxz.append(recovtxz[idx])
        positrontruevtxz.append(truevtxz[idx])
        
        positrontestrecovtxz.append(testrecovtxz[idx])
        positrontesttruevtxz.append(testtruevtxz[idx])
       

    idx+=1
print('\nJob complete.')


# In[ ]:


# mutli dimensional xz & yz views for each event. this is used for plotting events only
cvnmap_norm_resh=np.array(cvnmap_norm_resh)


# In[ ]:


cvnmap_norm_resh_xz=np.array(cvnmap_norm_resh_xz) # xz views only


# In[ ]:


cvnmap_norm_resh_yz=np.array(cvnmap_norm_resh_yz) # yz views only


# In[ ]:


test_cvnmap_qe_xz=np.array(test_cvnmap_qe_xz)
test_cvnmap_qe_yz=np.array(test_cvnmap_qe_yz)


# In[ ]:


test_cvnmap_res_xz=np.array(test_cvnmap_res_xz)
test_cvnmap_res_yz=np.array(test_cvnmap_res_yz)


# In[ ]:


test_cvnmap_dis_xz=np.array(test_cvnmap_dis_xz)
test_cvnmap_dis_yz=np.array(test_cvnmap_dis_yz)


# In[ ]:


test_cvnmap_cc_xz=np.array(test_cvnmap_cc_xz)
test_cvnmap_cc_yz=np.array(test_cvnmap_cc_yz)


# In[ ]:


test_cvnmap_nc_xz=np.array(test_cvnmap_nc_xz)
test_cvnmap_nc_yz=np.array(test_cvnmap_nc_yz)


# In[ ]:


test_cvnmap_norm_resh_xz=np.array(test_cvnmap_norm_resh_xz) # xz views only


# In[ ]:


test_cvnmap_norm_resh_yz=np.array(test_cvnmap_norm_resh_yz) # yz views only


# In[ ]:


def plot_event_with_vtx(event,idx=0):
    pixelmap=event[idx]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(21, 7))
    # added [:-20] to end of pixelmap[X] to preview square images for CNN
    sns.heatmap(pixelmap[0][:-20],cmap='coolwarm',cbar=False,square=True,xticklabels=10,yticklabels=10,ax=axes[0])
    sns.heatmap(pixelmap[1][:-20],cmap='coolwarm',cbar=False,square=True,xticklabels=10,yticklabels=10,ax=axes[1])
#    axes[0].scatter(x=vtxx,y=vtxz,c='yellow',marker='x',s=50)       # comment/uncomment to plot/unplot vertex point
#    axes[1].scatter(x=vtxy,y=vtxz,c='yellow',marker='x',s=50)       # comment/uncomment to plot/unplot vertex point
    plt.suptitle("XZ & YZ Plot", fontsize=30)
    print('UX Specified Fields\nEvent Number:\t{}'.format(idx))
#    print('CVN Vertex Position (x,y,z) = ({},{},{})'.format(f'{vtxx:.3}',f'{vtxy:.3}',f'{vtxz:.3}'))
    axes[0].set_xlabel("Cell", fontsize=25)
    axes[0].set_ylabel("Plane", fontsize=25)
    axes[1].set_xlabel("Cell", fontsize=25)
    axes[1].set_ylabel("Plane", fontsize=25)
#    plt.savefig('event.pdf')


# In[ ]:


# see if GPU is recognized
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.test.is_built_with_cuda())
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
print(sess)
print(device_lib.list_local_devices())


# In[ ]:


# XZ view. Trains on the x-coordinate
X1_train, X1_test, y1_train, y1_test = train_test_split(cvnmap_norm_resh_xz, truevtxz, test_size=0.25, random_state=101)


# In[ ]:


# YZ view. Trains on the y-cooridnate
X2_train, X2_test, y2_train, y2_test = train_test_split(cvnmap_norm_resh_yz, truevtxz, test_size=0.25, random_state=101)


# In[ ]:


# add one more dimension to let the CNN know we are dealing with one color dimension
x1_train=X1_train.reshape(X1_train.shape[0],100,80,1)
x1_test=X1_test.reshape(X1_test.shape[0],100,80,1)
x2_train=X2_train.reshape(X2_train.shape[0],100,80,1)
x2_test=X2_test.reshape(X2_test.shape[0],100,80,1)
#batch_size,width,heigh,color_channels


# In[ ]:


# add one more dimension to let the CNN know we are dealing with one color dimension
test_cvnmap_qe_xz=test_cvnmap_qe_xz.reshape(test_cvnmap_qe_xz.shape[0],100,80,1)
test_cvnmap_qe_yz=test_cvnmap_qe_yz.reshape(test_cvnmap_qe_yz.shape[0],100,80,1)
test_cvnmap_res_xz=test_cvnmap_res_xz.reshape(test_cvnmap_res_xz.shape[0],100,80,1)
test_cvnmap_res_yz=test_cvnmap_res_yz.reshape(test_cvnmap_res_yz.shape[0],100,80,1)
test_cvnmap_dis_xz=test_cvnmap_dis_xz.reshape(test_cvnmap_dis_xz.shape[0],100,80,1)
test_cvnmap_dis_yz=test_cvnmap_dis_yz.reshape(test_cvnmap_dis_yz.shape[0],100,80,1)
test_cvnmap_coh_xz=test_cvnmap_coh_xz.reshape(test_cvnmap_coh_xz.shape[0],100,80,1)
test_cvnmap_coh_yz=test_cvnmap_coh_yz.reshape(test_cvnmap_coh_yz.shape[0],100,80,1)
test_cvnmap_mec_xz=test_cvnmap_mec_xz.reshape(test_cvnmap_mec_xz.shape[0],100,80,1)
test_cvnmap_mec_yz=test_cvnmap_mec_yz.reshape(test_cvnmap_mec_yz.shape[0],100,80,1)
#batch_size,width,heigh,color_channels


# In[ ]:


# add one more dimension to let the CNN know we are dealing with one color dimension
test_cvnmap_cc_xz=test_cvnmap_cc_xz.reshape(test_cvnmap_cc_xz.shape[0],100,80,1)
test_cvnmap_cc_yz=test_cvnmap_cc_yz.reshape(test_cvnmap_cc_yz.shape[0],100,80,1)
test_cvnmap_nc_xz=test_cvnmap_nc_xz.reshape(test_cvnmap_nc_xz.shape[0],100,80,1)
test_cvnmap_nc_yz=test_cvnmap_nc_yz.reshape(test_cvnmap_nc_yz.shape[0],100,80,1)
#batch_size,width,heigh,color_channels


# In[ ]:


# custom regression loss functions
# huber loss
def huber(true, pred, delta):
    loss = np.where(np.abs(true-pred) < delta , 0.5*((true-pred)**2), delta*np.abs(true - pred) - 0.5*(delta**2))
    return np.sum(loss)

# log cosh loss
def logcosh(true, pred):
    loss = np.log(np.cosh(pred - true))
    return np.sum(loss)


# In[ ]:


# instantiate the models
model_regCNN_xz = Sequential()
model_regCNN_yz = Sequential()
# add two fully connected 2-dimensional convolutional layers for the XZ and YZ views
model_regCNN_xz.add(Conv2D(filters=32,kernel_size=(2,2),strides=(1,1),
                  input_shape=(100,80,1),activation='relu'))
model_regCNN_yz.add(Conv2D(filters=32,kernel_size=(2,2),strides=(1,1),
                 input_shape=(100,80,1),activation='relu'))
# specify 2-dimensional pooling
model_regCNN_xz.add(MaxPool2D(pool_size=(2,2)))
model_regCNN_yz.add(MaxPool2D(pool_size=(2,2)))
# flatten the datasets
model_regCNN_xz.add(Flatten())
model_regCNN_yz.add(Flatten())
# add dense layers for each view. 256 neurons per layer
model_regCNN_xz.add(Dense(256,activation='relu'))
model_regCNN_yz.add(Dense(256,activation='relu'))
model_regCNN_xz.add(Dense(256,activation='relu'))
model_regCNN_yz.add(Dense(256,activation='relu'))
model_regCNN_xz.add(Dense(256,activation='relu'))
model_regCNN_yz.add(Dense(256,activation='relu'))
model_regCNN_xz.add(Dense(256,activation='relu'))
model_regCNN_yz.add(Dense(256,activation='relu'))
model_regCNN_xz.add(Dense(256,activation='relu'))
model_regCNN_yz.add(Dense(256,activation='relu'))
model_regCNN_xz.add(Dense(256,activation='relu'))
model_regCNN_yz.add(Dense(256,activation='relu'))
# no. of classes (output)
n_classes=1
# tf concatenate the models
model_regCNN_concat = concatenate([model_regCNN_xz.output, model_regCNN_yz.output],axis=-1)
model_regCNN_concat = Dense(n_classes)(model_regCNN_concat)
model_regCNN = Model(inputs=[model_regCNN_xz.input, model_regCNN_yz.input], outputs=model_regCNN_concat)
# compile the concatenated model
model_regCNN.compile(loss='logcosh', optimizer='adam') # loss was 'mse' then 'mae'
# print a summary of the model
print(model_regCNN.summary())


# In[ ]:


# z-coordinate system
model_regCNN.fit(x=[x1_train,x2_train],y=y1_train,epochs=200)


# In[ ]:


# Save the model as .h5 file
model_regCNN.save('z_model.h5')

# Save as .pb file
tf.saved_model.save(model_regCNN, 'z_model.pb')


# In[ ]:


qe_predictions=model_regCNN.predict([test_cvnmap_qe_xz,test_cvnmap_qe_yz])
res_predictions=model_regCNN.predict([test_cvnmap_res_xz,test_cvnmap_res_yz])
dis_predictions=model_regCNN.predict([test_cvnmap_dis_xz,test_cvnmap_dis_yz])
coh_predictions=model_regCNN.predict([test_cvnmap_coh_xz,test_cvnmap_coh_yz])
mec_predictions=model_regCNN.predict([test_cvnmap_mec_xz,test_cvnmap_mec_yz])


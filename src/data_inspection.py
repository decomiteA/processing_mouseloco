import os, sys 
import numpy as np 
import copy
# import pickle 
import pandas as pd 
# from tqdm import tqdm
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt 
from scipy.signal import savgol_filter
# import scipy.io as spio
import warnings
warnings.filterwarnings('ignore')
# from sklearn.decomposition import PCA


#########################################################################
# Put the following to True if you want to see the figures during running
bool_plot = False

figure_path = os.path.join(os.getcwd(),'FiguresFolder')
os.makedirs(figure_path, exist_ok=True)

str_list = ['Snout','RF','LF','RH','LH','Tail']
input_folder = os.path.join(os.getcwd())
tot_list_data = []
for file in os.listdir(input_folder):
    if file.endswith('.h5'):
        input_path_file = os.path.join(input_folder, file)
        input_data = pd.read_hdf(input_path_file)
        list_data = []
        for str in str_list:
            list_data.append(input_data[('DLC_HrnetW32_openfield_v3Sep10shuffle2_detector_170_snapshot_160',f'{str}')].values)
        tot_list_data.append(list_data)
        fig, axs = plt.subplots(1,1,figsize=(5,5))
        axs.spines[['top','right']].set_visible(False)
        axs.plot(list_data[0][:,0], list_data[0][:,1], color='k',lw=0.5)
        axs.set_xlabel('x-position'), axs.set_ylabel('y-position'), axs.set_aspect('equal')
        plt.tight_layout()
        fig.savefig(os.path.join(figure_path,f'{file}_two_dimensional_snout.png'))


        fig, axs = plt.subplots(1,1,figsize=(20,5))
        axs.spines[['top','right']].set_visible(False)
        axs.plot(list_data[0][:,0],'k',lw=1)
        axs.plot(list_data[1][:,0],'r',lw=1)
        axs.plot(list_data[2][:,0],'g',lw=1)
        axs.plot(list_data[3][:,0],'b',lw=1)
        axs.plot(list_data[4][:,0],'m',lw=1)
        plt.tight_layout()
        fig.savefig(os.path.join(figure_path,f'{file}_x_axis_position.png'))
        if bool_plot:
            plt.show()
        else:
            plt.close('all')


# Filtering the data
b_body,a_body = scipy.signal.butter(6, 0.1, 'lowpass')
b_paws,a_paws = scipy.signal.butter(6, 0.5, 'lowpass')

tot_list_data_filtered = copy.deepcopy(tot_list_data)

for local_data in tot_list_data:
    local_data[0][:,0] = scipy.signal.filtfilt(b_body,a_body,local_data[0][:,0])
    local_data[5][:,0] = scipy.signal.filtfilt(b_body,a_body,local_data[5][:,0])
    local_data[0][:,1] = scipy.signal.filtfilt(b_body,a_body,local_data[0][:,1])
    local_data[5][:,1] = scipy.signal.filtfilt(b_body,a_body,local_data[5][:,1])

    local_data[1][:,0] = scipy.signal.filtfilt(b_paws,a_paws,local_data[1][:,0])
    local_data[2][:,0] = scipy.signal.filtfilt(b_paws,a_paws,local_data[2][:,0])
    local_data[3][:,0] = scipy.signal.filtfilt(b_paws,a_paws,local_data[3][:,0])
    local_data[4][:,0] = scipy.signal.filtfilt(b_paws,a_paws,local_data[4][:,0])
    local_data[1][:,1] = scipy.signal.filtfilt(b_paws,a_paws,local_data[1][:,1])
    local_data[2][:,1] = scipy.signal.filtfilt(b_paws,a_paws,local_data[2][:,1])
    local_data[3][:,1] = scipy.signal.filtfilt(b_paws,a_paws,local_data[3][:,1])
    local_data[4][:,1] = scipy.signal.filtfilt(b_paws,a_paws,local_data[4][:,1])


# Attempting to filter the data
fig, axs = plt.subplots(1,1,figsize=(20,5))
axs.spines[['top','right']].set_visible(False)
axs.plot(tot_list_data_filtered[0][0][:,0],'r',lw=2)
axs.plot(tot_list_data_filtered[0][1][:,0],'k',lw=2)
axs.plot(tot_list_data[0][0][:,0],'m',lw=2)
axs.plot(tot_list_data[0][1][:,0],'c',lw=2)
plt.tight_layout()
plt.show()



import os, sys
import numpy as np
import h5py
import pickle
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt
import scipy.io as spio
import warnings
from sklearn.decomposition import PCA
from utils.utils_mouse_processing import *
warnings.filterwarnings('ignore')


input_folder = os.path.join(os.getcwd(),'data','preprocessed')
output_folder = os.path.join(os.getcwd(),'data','processed')
os.makedirs(output_folder, exist_ok=True)

with open(os.path.join(input_folder,'preprocessed_data.pkl'),'rb') as f1:
    list_data = pickle.load(f1)

tot_foot_contact, tot_foot_raw, tot_time = np.full((2,4,1),np.nan), np.full((4,6,1),np.nan), np.zeros((1,2))

for jj in tqdm(range(len(list_data))):
    video_data = list_data[jj]
    video_data_with_speed = compute_velocity_markers(video_data.T, framerate=60).T
    time_vector = np.linspace(0,1/60*video_data.shape[2],video_data.shape[2])
    head_velocity = video_data_with_speed[2,0,:]

    
    head_velocity_f = np.zeros(head_velocity.shape)
    for ii in range(5,len(head_velocity)-5):
        head_velocity_f[ii] = np.nanmean(head_velocity[ii-5:ii+5])

    timings_f = findHighSpeed(time_vector,head_velocity_f)
    diff_f = np.abs(np.diff(timings_f,1))
    idx_long = np.where(diff_f>40)[0]
    for line in range(len(idx_long)):
        idx_begin = timings_f[idx_long[line],0].astype(int)
        idx_end = timings_f[idx_long[line],1].astype(int)
        local_data = video_data[:,:6,idx_begin:idx_end]
        local_x_data = np.reshape(video_data[0,:6,idx_begin:idx_end], (local_data.shape[1]*local_data.shape[2],1))
        local_y_data = np.reshape(video_data[1,:6,idx_begin:idx_end], (local_data.shape[1]*local_data.shape[2],1))
        pca_data = np.concatenate((local_x_data, local_y_data), 1)

        pca = PCA(n_components=2)
        pca.fit(pca_data)
        local_angle = np.arctan2(pca.components_[0,1], pca.components_[0,0])
        rot_matrix = np.array([[np.cos(-local_angle), -np.sin(-local_angle)],
                            [np.sin(-local_angle), np.cos(-local_angle)]])
        rotated_local_data = np.zeros(local_data.shape)
        for time in range(rotated_local_data.shape[2]):
            for marker in range(rotated_local_data.shape[1]):
                rotated_local_data[0,marker,time] = rot_matrix[0,0] * local_data[0,marker,time] + rot_matrix[0,1] * local_data[1,marker,time]
                rotated_local_data[1,marker,time] = rot_matrix[1,0] * local_data[0,marker,time] + rot_matrix[1,1] * local_data[1,marker,time]


        # Determine whether this is a straight line locomotion movement
        local_angle_vector = np.arctan2(rotated_local_data[1,0,:]-rotated_local_data[1,-2,:], rotated_local_data[0,0,:]-rotated_local_data[0,-1,:])
        local_angle_degree = np.rad2deg(np.unwrap(local_angle_vector))
        idx_in = np.where(np.abs(local_angle_degree-local_angle_degree[int(len(local_angle_degree)//2)])<50)[0]
        bool_straight = (len(idx_in)/len(local_angle_degree)> 0.75)
        # bool_straight = np.all(np.abs(local_angle_degree - np.nanmean(local_angle_degree)) < 50)

        if bool_straight:
            rotated_local_data = compute_velocity_markers(rotated_local_data.T).T 
            local_foot_contact = get_foot_contact(rotated_local_data, framerate=60, bool_rett=True)
            tot_foot_contact = np.concatenate((tot_foot_contact, local_foot_contact,np.full((2,4,1),np.nan)),2)
            tot_foot_raw = np.concatenate((tot_foot_raw, rotated_local_data, np.full((4,6,1),np.nan)),2)
            tot_time = np.concatenate((tot_time, np.expand_dims(np.array([idx_begin, idx_end]),0)),0)



np.save(os.path.join(output_folder,'tot_foot_contact.npy'), tot_foot_contact)
np.save(os.path.join(output_folder,'tot_foot_raw.npy'), tot_foot_raw)
np.save(os.path.join(output_folder,'tot_time.npy'), tot_time)
print(f'Data saved')
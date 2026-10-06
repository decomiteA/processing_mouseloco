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

# You need to change this line to whichever path you have saved your data in
input_folder = os.path.join(os.getcwd(),'..','datasets','mice','test_code')

# You also need to change the following parameters to adjust the framerate at which the data was collected and the resolution used on the gopros.

framerate = 30
resolution = 0.0006

output_folder = input_folder
str_population = ["WildType","R270X","ABE"]

#############################
# Preprocessing of the data #
#############################
for population in range(len(str_population)): 
    list_file_path, list_file_name = [], []
    local_output_path = os.path.join(output_folder, 'preprocessed',str_population[population])
    os.makedirs(local_output_path, exist_ok=True)
    list_total_matrix = []
    for file in os.listdir(os.path.join(input_folder,str_population[population],'raw')):
        # this will be h5 files 
        list_file_path.append(os.path.join(input_folder,str_population[population],'raw',file))
        list_file_name.append(file)

        with h5py.File(list_file_path[-1],'r') as f:
            input_file = np.squeeze(fill_missing_7markers(f['tracks'][:].T).T)
        filtered_total_matrix = np.zeros(input_file.shape)
        for dir in range(2):
            for point in range(input_file.shape[1]):
                filtered_total_matrix[dir,point,:] = savgol_filter(input_file[dir,point,:],window_length=3,polyorder=1)


        list_total_matrix.append(filtered_total_matrix)

    # Saving the data

    with open(os.path.join(local_output_path, 'preprocessed_data_updated.pkl'),'wb') as f:
        pickle.dump(list_total_matrix, f)

    with open(os.path.join(local_output_path, 'filenames.pkl'),'wb') as f:
        pickle.dump(list_file_name, f)
    #############################################################
    # Extracting the straight line segments and foot placements #
    ############################################################# 
    local_output_path = os.path.join(output_folder, 'processed',str_population[population])
    os.makedirs(local_output_path, exist_ok=True)

    tot_foot_contact, tot_foot_raw, tot_time = np.full((2,4,1),np.nan), np.full((4,6,1),np.nan), np.zeros((1,2))


    for jj in tqdm(range(len(list_total_matrix))):
        video_data = list_total_matrix[jj] * resolution
        video_data_with_speed = compute_velocity_markers(video_data.T).T
        time_vector = np.linspace(0,1/30*video_data.shape[2], video_data.shape[2])
        head_velocity = video_data_with_speed[2,0,:]


        head_velocity_f = np.zeros(head_velocity.shape)
        for ii in range(5,len(head_velocity)-5):
            head_velocity_f[ii] = np.nanmean(head_velocity[ii-5:ii+5])

        
        timings_f = findHighSpeed(time_vector,head_velocity_f)
        diff_f = np.abs(np.diff(timings_f,1))
        idx_long = np.where(diff_f>60)[0]
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

            if bool_straight:
                rotated_local_data = compute_velocity_markers(rotated_local_data.T).T 
                local_foot_contact = get_foot_contact(rotated_local_data,bool_rett=True)
                tot_foot_contact = np.concatenate((tot_foot_contact, local_foot_contact,np.full((2,4,1),np.nan)),2)
                tot_foot_raw = np.concatenate((tot_foot_raw, rotated_local_data, np.full((4,6,1),np.nan)),2)
                tot_time = np.concatenate((tot_time, np.expand_dims(np.array([idx_begin, idx_end]),0)),0)

    plot_foot_contact = copy.deepcopy(tot_foot_contact)
    plot_foot_contact[plot_foot_contact==0] = np.nan


    np.save(os.path.join(local_output_path,'tot_foot_contact.npy'), tot_foot_contact)
    np.save(os.path.join(local_output_path,'tot_foot_raw.npy'), tot_foot_raw)
    np.save(os.path.join(local_output_path,'tot_time.npy'), tot_time)

    #############################################
    # Extracting the gait cycle related metrics #
    #############################################
    local_output_path = os.path.join(output_folder, 'io_processed',str_population[population])
    os.makedirs(local_output_path, exist_ok=True)
    input_raw_data = tot_foot_raw
    input_foot_data = tot_foot_contact
    input_tot_time = tot_time



    tmp_diff = np.where(np.diff(input_tot_time[:,0],axis=0)<0)[0]
    tot_video = np.zeros((input_tot_time.shape[0],1))
    for ii in range(len(tmp_diff)):
        tot_video[tmp_diff[ii-1]+1:tmp_diff[ii]+1] = ii
    tot_video[tmp_diff[-1]+1:] = ii+1

    idx_nans = np.where(np.isnan(input_raw_data[0,0,:]))[0]

    total_video = np.zeros((input_raw_data.shape[2],1))
    total_video[idx_nans] = np.nan
    for ii in range(len(idx_nans)-1):
        total_video[idx_nans[ii]+1:idx_nans[ii+1]] = tot_video[ii]


    phasor_metrics_leg_0 = extract_phasor_metrics(input_raw_data, input_foot_data, total_video, leg_id=0)
    phasor_metrics_leg_1 = extract_phasor_metrics(input_raw_data, input_foot_data, total_video, leg_id=1)
    phasor_metrics_leg_2 = extract_phasor_metrics(input_raw_data, input_foot_data, total_video, leg_id=2)
    phasor_metrics_leg_3 = extract_phasor_metrics(input_raw_data, input_foot_data, total_video, leg_id=3)


    with open(os.path.join(local_output_path,'phasor_metrics_leg_0.pkl'),'wb') as f1:
        pickle.dump(phasor_metrics_leg_0, f1)
    with open(os.path.join(local_output_path,'phasor_metrics_leg_1.pkl'),'wb') as f1:
        pickle.dump(phasor_metrics_leg_1, f1)
    with open(os.path.join(local_output_path,'phasor_metrics_leg_2.pkl'),'wb') as f1:
        pickle.dump(phasor_metrics_leg_2, f1)
    with open(os.path.join(local_output_path,'phasor_metrics_leg_3.pkl'),'wb') as f1:
        pickle.dump(phasor_metrics_leg_3, f1)

    output_metrics = extract_metrics(input_raw_data, input_foot_data, total_video, framerate=framerate)

    with open(os.path.join(local_output_path,'list_metrics.pkl'),'wb') as f1:
        pickle.dump(output_metrics, f1)

    with open(os.path.join(local_output_path,'list_metrics.pkl'),'rb') as f1:
        output_metrics = pickle.load(f1)
    # Get the information for the interlimb coordination
    idx_to_keep = np.where((output_metrics[:,1]==0) & (output_metrics[:,2]==0) & (np.abs(output_metrics[:,4])>0.02) & (np.abs(output_metrics[:,4])<0.10))[0]
    total_input, total_output = np.zeros((len(idx_to_keep),11,4)), np.zeros((len(idx_to_keep),14))
    total_animal = output_metrics[idx_to_keep,0]
    for line in tqdm(range(len(idx_to_keep))):
        tmp_input, tmp_output = get_io_time_model_cycle_front(input_foot_data, input_raw_data, output_metrics[idx_to_keep[line],:], output_metrics)
        if tmp_output is None:
            total_input[line,:], total_output[line,:] = np.nan, np.nan
        elif tmp_output.shape[1]!=0:
            total_input[line,:] = tmp_input
            total_output[line,0] = tmp_output[0][0]
            total_output[line,1] = tmp_output[1][0]
            total_output[line,2] = tmp_output[2][0]
            total_output[line,3] = tmp_output[3][0]
            total_output[line,4] = tmp_output[4][0]
            total_output[line,5] = tmp_output[5][0]
            total_output[line,6] = tmp_output[6][0]
            total_output[line,7] = tmp_output[7][0]
            total_output[line,8] = tmp_output[8][0]
            total_output[line,9] = tmp_output[9][0]
            total_output[line,10] = tmp_output[10][0]
            total_output[line,11] = tmp_output[11][0]
            total_output[line,12] = tmp_output[12][0]
            total_output[line,13] = output_metrics[idx_to_keep[line],0]
        else:
            total_input[line,:], total_output[line,:] = np.nan, np.nan

    np.save(os.path.join(local_output_path, 'stride_level_data_input.npy'), total_input)
    np.save(os.path.join(local_output_path, 'stride_level_data_output.npy'), total_output)
    np.save(os.path.join(local_output_path, 'stride_level_data_animal.npy'), total_animal)

    # Get the information for the body-limb coordination
    total_input = np.zeros((len(idx_to_keep),11,8))
    total_animal = output_metrics[idx_to_keep,0]
    for line in tqdm(range(len(idx_to_keep))):
        tmp_input = get_info_body_limb(input_foot_data, input_raw_data, output_metrics[idx_to_keep[line],:], output_metrics)

        if tmp_input is None:
            total_input[line,:] = np.nan
        elif tmp_input.shape[1]!=0:
            total_input[line,:] = tmp_input
        else:
            total_input[line,:] = np.nan
    np.save(os.path.join(local_output_path, 'body_limb_coordination.npy'), total_input)
    np.save(os.path.join(local_output_path, 'body_limb_animal.npy'), total_animal)



    idx_to_keep = np.where((output_metrics[:,1]==0) & (output_metrics[:,2]==0) & (np.abs(output_metrics[:,4])>0.02) & (np.abs(output_metrics[:,4])<0.10))[0]

    total_input, total_output, total_input_self = np.zeros((len(idx_to_keep),21,8)), np.zeros((len(idx_to_keep),6)), np.zeros((len(idx_to_keep),21,8))
    total_animal = output_metrics[idx_to_keep,0]
    idx_nans = np.where(np.isnan(input_raw_data[0,0,:]))[0]
    for line in tqdm(range(len(idx_to_keep))):
        tmp_input, tmp_output = get_io_time_model_fr(input_foot_data, input_raw_data, output_metrics[idx_to_keep[line],:], output_metrics)
        tmp_self, _ = get_io_time_model_fr_self(input_foot_data, input_raw_data, output_metrics[idx_to_keep[line],:], output_metrics)
        if tmp_output is None:
            total_input[line,:], total_output[line,:], total_input_self[line,:] = np.nan, np.nan, np.nan
        elif tmp_output.shape[1]!=0:
            total_input[line,:] = tmp_input
            total_input_self[line,:] = tmp_self
            total_output[line,0] = tmp_output[0][0]
            total_output[line,1] = tmp_output[1][0]
            total_output[line,2] = tmp_output[2][0]
            total_output[line,3] = tmp_output[3][0]
            total_output[line,4] = tmp_output[4][0]
            total_output[line,5] = tmp_output[5][0]
        else:
            total_input[line,:], total_output[line,:], total_input_self[line,:] = np.nan, np.nan, np.nan
        
    idx_flip = np.where(np.nanmean(total_input[:,:,2],axis=1)<0)[0]
    total_input[idx_flip,:,:] = - total_input[idx_flip,:,:]
    total_output[idx_flip,:] = - total_output[idx_flip,:]
    total_input_self[idx_flip,:,:] = - total_input_self[idx_flip,:,:]

    np.save(os.path.join(local_output_path, 'step_level_data_input.npy'), total_input)
    np.save(os.path.join(local_output_path, 'step_level_data_output.npy'), total_output)
    np.save(os.path.join(local_output_path, 'step_level_data_input_self.npy'), total_input_self)
    np.save(os.path.join(local_output_path, 'step_level_data_animal.npy'), total_animal)

    ################################
    # Extracting the gross metrics #
    ################################

    tmp_diff = np.where(np.diff(input_tot_time[:,0],axis=0)<0)[0]
    tot_video = np.zeros((input_tot_time.shape[0],1))
    for ii in range(len(tmp_diff)):
        tot_video[tmp_diff[ii-1]+1:tmp_diff[ii]+1] = ii
    tot_video[tmp_diff[-1]+1:] = ii+1

    idx_nans = np.where(np.isnan(input_raw_data[0,0,:]))[0]

    total_video = np.zeros((input_raw_data.shape[2],1))
    total_video[idx_nans] = np.nan
    for ii in range(len(idx_nans)-1):
        total_video[idx_nans[ii]+1:idx_nans[ii+1]] = tot_video[ii]

    gross_metrics = extract_high_level_metrics(input_raw_data, total_video,framerate=framerate)

    np.save(os.path.join(local_output_path,'gross_locomotion_metrics.npy'), gross_metrics)

    ############################
    # Extracting animal length #
    ############################
    idx_nans = np.where(np.isnan(input_raw_data[0,0,:]))[0]
    diff_time = input_tot_time[1:,1] - input_tot_time[:-1,1]
    idx_switch = np.where(diff_time<0)[0]
    diff_positions = np.sqrt(np.square(input_raw_data[0,5,:]-input_raw_data[0,0,:]) + np.square(input_raw_data[1,5,:] - input_raw_data[1,0,:]))

    n_animal = len(idx_switch)+1
    local_animal_length = np.zeros((n_animal,2))
    local_animal_length[0,0] = np.nanmedian(diff_positions[1:idx_switch[0]])
    local_animal_length[0,1] = np.nanmean(diff_positions[1:idx_switch[0]])

    local_animal_length[-1,0] = np.nanmedian(diff_positions[idx_switch[-1]:])
    local_animal_length[-1,1] = np.nanmean(diff_positions[idx_switch[-1]:])

    for animal in range(1,n_animal-1):
        idx_begin = idx_nans[idx_switch[animal-1]]+1
        idx_end = idx_nans[idx_switch[animal]]
        local_length = np.nanmedian(diff_positions[idx_begin:idx_end])
        local_length_mean = np.nanmean(diff_positions[idx_begin:idx_end])
        if animal==n_animal-1:
            local_animal_length[animal,0] = local_length_mean
            local_animal_length[animal,1] = local_length
        else:
            local_animal_length[animal,0] = local_length_mean
            local_animal_length[animal,1] = local_length
    np.save(os.path.join(local_output_path,'animal_length.npy'),local_animal_length)

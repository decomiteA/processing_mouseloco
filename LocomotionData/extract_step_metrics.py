import pandas as pd 
import pickle 
import numpy as np
import scipy.io as spio
from tqdm import tqdm
import os, sys 
import warnings
import matplotlib.pyplot as plt 
from utils.utils_mouse_processing import *
warnings.filterwarnings('ignore')

input_path = os.path.join(os.getcwd(),'data','processed')
output_path = os.path.join(os.getcwd(),'data','io_processed')
os.makedirs(output_path,exist_ok=True)
input_raw_data = np.load(os.path.join(input_path,'tot_foot_raw.npy'))
input_foot_data = np.load(os.path.join(input_path,'tot_foot_contact.npy'))
input_tot_time = np.load(os.path.join(input_path,'tot_time.npy'))
print(input_tot_time.shape, input_raw_data.shape)

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


with open(os.path.join(output_path,'phasor_metrics_leg_0.pkl'),'wb') as f1:
    pickle.dump(phasor_metrics_leg_0, f1)
with open(os.path.join(output_path,'phasor_metrics_leg_1.pkl'),'wb') as f1:
    pickle.dump(phasor_metrics_leg_1, f1)
with open(os.path.join(output_path,'phasor_metrics_leg_2.pkl'),'wb') as f1:
    pickle.dump(phasor_metrics_leg_2, f1)
with open(os.path.join(output_path,'phasor_metrics_leg_3.pkl'),'wb') as f1:
    pickle.dump(phasor_metrics_leg_3, f1)

output_metrics = extract_metrics(input_raw_data, input_foot_data, total_video, framerate=60)
# output_metrics_normalized = extract_metrics_norm(input_raw_data, input_foot_data, total_video, animal_length)[1:,:]

with open(os.path.join(output_path,'list_metrics.pkl'),'wb') as f1:
    pickle.dump(output_metrics, f1)
# Get the information for the interlimb coordination
idx_to_keep = np.where((output_metrics[:,1]==0) & (output_metrics[:,2]==0) & (np.abs(output_metrics[:,4])>0.02) & (np.abs(output_metrics[:,4])<1000))[0]
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

np.save(os.path.join(output_path, 'stride_level_data_input.npy'), total_input)
np.save(os.path.join(output_path, 'stride_level_data_output.npy'), total_output)
np.save(os.path.join(output_path, 'stride_level_data_animal.npy'), total_animal)

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
np.save(os.path.join(output_path, 'body_limb_coordination.npy'), total_input)
np.save(os.path.join(output_path, 'body_limb_animal.npy'), total_animal)


idx_to_keep = np.where((output_metrics[:,1]==0) & (output_metrics[:,2]==0) & (np.abs(output_metrics[:,4])>0.02) & (np.abs(output_metrics[:,4])<1000))[0]

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

np.save(os.path.join(output_path, 'step_level_data_input.npy'), total_input)
np.save(os.path.join(output_path, 'step_level_data_output.npy'), total_output)
np.save(os.path.join(output_path, 'step_level_data_input_self.npy'), total_input_self)
np.save(os.path.join(output_path, 'step_level_data_animal.npy'), total_animal)
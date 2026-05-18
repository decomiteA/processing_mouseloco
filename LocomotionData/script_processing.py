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


input_folder = os.path.join(os.getcwd(),'data','raw')
output_folder = os.path.join(os.getcwd(),'data','preprocessed')
os.makedirs(output_folder,exist_ok=True)

list_total_matrix = []
b,a = butter(6,0.5)

for file in os.listdir(input_folder):
    with h5py.File(os.path.join(input_folder,file),'r') as f:
        input_file = data2array(pd.read_hdf(os.path.join(input_folder,file)))
    input_data = reshape_kojiro_data(input_file)
    filtered_total_matrix = np.zeros(input_data.shape)
    for dir in range(2):
        for point in range(input_data.shape[1]):
            filtered_total_matrix[dir,point,:] = savgol_filter(input_data[dir,point,:],window_length=3,polyorder=1)
            filtered_total_matrix[dir,point,:] = filtfilt(b,a,filtered_total_matrix[dir,point,:])

    list_total_matrix.append(filtered_total_matrix)

with open(os.path.join(output_folder,'preprocessed_data.pkl'),'wb') as f:
    pickle.dump(list_total_matrix, f)

print('Data preprocessed and saved')
import os, sys 
import numpy as np
import matplotlib.pyplot as plt 
import pickle

with open(os.path.join(os.getcwd(),'data','io_processed','list_metrics.pkl'),'rb') as f:
    input_data = pickle.load(f)

print(input_data.shape)

idx_same = np.where((input_data[:,1]==input_data[:,2]))[0]

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.scatter(np.abs(input_data[idx_same,6]), np.abs(input_data[idx_same,4]),color='k',s=5)
plt.tight_layout()



fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.hist(np.abs(input_data[idx_same,6]),alpha=0.5, bins=20,density=True,color='k')
plt.tight_layout()

fig, axs = plt.subplots(1,1,figsize=(3,3))
axs.spines[['top','right']].set_visible(False)
axs.hist(np.abs(input_data[idx_same,4]),alpha=0.5, bins=20,density=True,color='k', range=(0,500))
plt.tight_layout()
plt.show()
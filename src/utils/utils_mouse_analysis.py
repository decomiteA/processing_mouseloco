import numpy as np 
import os, sys 
import copy
import scipy.stats
import scipy.io as spio
from tqdm import tqdm
import matplotlib.pyplot as plt 
import scikit_posthocs as spph
import scipy.signal as signal
import matplotlib.cm as cm
import scipy as sp
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

figure_path = os.path.join(os.getcwd(),'..','results','figures','mice','rett_female')


def normalize_gross(gross_metrics, animal_length, bool_klibaite=False):
    """
    Normalize the gross locomotion metrics based on the animal length
    """
    g = 9.81
    normalized_gross = copy.deepcopy(gross_metrics)
    if bool_klibaite:
        normalized_gross[:,0] = normalized_gross[:,0]//4
        normalized_gross[:,1] /= 1000
        normalized_gross[:,-1] /= 1000
    for animal in range(animal_length.shape[0]):
        idx_animal = np.where(normalized_gross[:,0]==animal)[0]
        if bool_klibaite:
            l0 = animal_length[animal,0]
        else:
            l0 = animal_length[animal,0]
        normalized_gross[idx_animal,1] /= l0
        normalized_gross[idx_animal,2] /= np.sqrt(l0/g)
        normalized_gross[idx_animal,3] /= np.sqrt(l0*g)

    return normalized_gross



def normalize_contact_mode(speed_data, output_data, animal_data, animal_length):
    """
    Normalize the contact mode metrics based on the animal length
    """
    g, dt = 9.81, 1/80
    normalized_speed = copy.deepcopy(speed_data)/1000
    normalized_output = copy.deepcopy(output_data)
    for animal in range(animal_length.shape[0]):
        idx_animal = np.where(animal_data==animal)[0]
        l0 = animal_length[animal,0]
        normalized_speed[idx_animal] /= np.sqrt(l0*g)
        normalized_output[idx_animal,2] /= np.sqrt(l0/g)
        normalized_output[idx_animal,3] = (normalized_output[idx_animal,3] * dt ) / (np.sqrt(l0/g))
        normalized_output[idx_animal,4] /= np.sqrt(l0/g)
        normalized_output[idx_animal,5] /= np.sqrt(l0/g)
        normalized_output[idx_animal,6] /= np.sqrt(l0/g)

    return normalized_speed, normalized_output

def normalize_contact_mode_sh3(speed_data, output_data, animal_data, animal_length):
    """
    Normalize the contact mode metrics based on the animal length
    """
    g, dt = 9.81, 1/120
    normalized_speed = copy.deepcopy(speed_data)
    normalized_output = copy.deepcopy(output_data)
    for animal in range(animal_length.shape[0]):
        idx_animal = np.where(animal_data==animal)[0]
        l0 = animal_length[animal,0]
        normalized_speed[idx_animal] /= np.sqrt(g*l0)
        normalized_output[idx_animal,2] /= np.sqrt(l0/g)
        normalized_output[idx_animal,3] = (normalized_output[idx_animal,3] * dt ) / (np.sqrt(l0/g))
        normalized_output[idx_animal,4] /= np.sqrt(l0/g)
        normalized_output[idx_animal,5] /= np.sqrt(l0/g)
        normalized_output[idx_animal,6] /= np.sqrt(l0/g)

    return normalized_speed, normalized_output

def normalize_contact_mode_verpeut(speed_data, output_data, animal_length):
    """
    Normalize the contact mode metrics based on the animal length
    """
    g, dt = 9.81, 1/120
    normalized_speed = copy.deepcopy(speed_data)
    normalized_output = copy.deepcopy(output_data)
    l0 = animal_length[0,0]
    normalized_speed /= np.sqrt(g*l0)
    normalized_output[:,2] /= np.sqrt(l0/g)
    normalized_output[:,3] = (normalized_output[:,3] * dt ) / (np.sqrt(l0/g))
    normalized_output[:,4] /= np.sqrt(l0/g)
    normalized_output[:,5] /= np.sqrt(l0/g)
    normalized_output[:,6] /= np.sqrt(l0/g)

    return normalized_speed, normalized_output



def evaluate_surprise(values, mu, sigma):
    """
    Evaluates the 1-d bayesian surprise
    """
    surprise_output = np.zeros((values.shape[0],values.shape[1]))
    for parameter in range(surprise_output.shape[1]):
        surprise_output[:,parameter] = -0.5*np.log(2*np.pi*sigma[parameter]**2) - ((values[:,parameter]-mu[parameter])**2/(2*sigma[parameter]**2))

    return np.sum(surprise_output,axis=1)


def zscore(value, mu, sigma):
    """
    zscore the value data
    """
    return (value-mu)/sigma


def individual_bl_verpeut(bl_data, animal_length):
    """
    Computes the individual values of the body-limb coordination
    """
    individual_bl_metrics = np.zeros((1,4))
    local_data = bl_data
    # Head oscillations 
    individual_bl_metrics[:,0] = np.nanmean(np.ptp(local_data[:,:,1],axis=1)) / animal_length[0,0]
    individual_bl_metrics[:,1] = np.nanmean(np.argmax(local_data[:,:,1],axis=1))

    # Tail oscillations 
    individual_bl_metrics[:,2] = np.nanmean(np.ptp(local_data[:,:,5],axis=1)) / animal_length[0,0]
    individual_bl_metrics[:,3] = np.nanmean(np.argmax(local_data[:,:,5],axis=1))

    return individual_bl_metrics



def individual_bl(bl_data, bl_animal, animal_length):
    """
    Computes the individual values of the body-limb coordination
    """
    individual_bl_metrics = np.zeros((len(np.unique(bl_animal)),4))
    for animal in range(individual_bl_metrics.shape[0]):
        idx_animal = np.where(bl_animal==animal)[0]
        local_data = bl_data[idx_animal,:,:]
        # Head oscillations 
        individual_bl_metrics[animal,0] = np.nanmean(np.ptp(local_data[:,:,1],axis=1)) / animal_length[animal,0]
        individual_bl_metrics[animal,1] = np.nanmean(np.argmax(local_data[:,:,1],axis=1))

        # Tail oscillations 
        individual_bl_metrics[animal,2] = np.nanmean(np.ptp(local_data[:,:,5],axis=1)) / animal_length[animal,0]
        individual_bl_metrics[animal,3] = np.nanmean(np.argmax(local_data[:,:,5],axis=1))

    return individual_bl_metrics


def individual_gross(gross_metrics, boot_option=False):
    """
    Computes the individual values of the gross metrics 
    """
    individual_gross_metrics = np.zeros((len(np.unique(gross_metrics[:,0])),3))
    if not boot_option:
        for animal in range(individual_gross_metrics.shape[0]):
            idx_animal = np.where(gross_metrics[:,0]==animal)[0]
            individual_gross_metrics[animal,:] = np.nanmedian(gross_metrics[idx_animal,1:],axis=0)
    else:
        for animal in range(individual_gross_metrics.shape[0]):
            idx_choice = np.where(gross_metrics[:,0]==animal)[0]
            idx_animal = np.random.choice(idx_choice, len(idx_choice), replace=True)
            individual_gross_metrics[animal,:] = np.nanmedian(gross_metrics[idx_animal,1:],axis=0)

    return individual_gross_metrics

def extract_high_level_metrics(input_raw, input_video, framerate=120):
    """
    Extracts the high level locomotion metrics
    Output is a nx4 matrix where the different columns contain
    - animal id 
    - bout-wise distance
    - bout-wise duration
    - bout-wise velocity
    """

    output_matrix = np.zeros((1,4))
    idx_nans = np.where((np.isnan(input_raw[0,0,:])))[0]
    for ii in tqdm(range(len(idx_nans)-1)):
        idx_begin, idx_end = idx_nans[ii]+1, idx_nans[ii+1]
        local_raw_data = input_raw[:,:,idx_begin:idx_end]
        local_duration = local_raw_data.shape[-1] / framerate
        local_distance = np.sqrt(np.square(local_raw_data[0,0,-1]-local_raw_data[0,0,0]) + np.square(local_raw_data[1,0,-1]-local_raw_data[1,0,0]))
        local_velocity = np.abs(np.nanmean(local_raw_data[2,0,:]))
        output_matrix = np.vstack((output_matrix, np.array([input_video[idx_begin][0],local_distance,local_duration,local_velocity])))

    return output_matrix[1:,:]  


def get_rsquare_matrix_feedback_verpeut(tot_input_list, tot_output_list, bool_hind, bool_lat):
    """
    Computes the rsquare matrix for the linear prediction of the foot contact location around the nominal 
    """


    rsquare_diagonal = np.zeros((1, 21))
    gains_diagonal = np.zeros((1,21,5))
    idx_nan = np.where(~np.isnan(tot_input_list[:,15,0]))[0]
    local_input = tot_input_list[:,:,4*bool_hind:4+4*bool_hind]
    local_output = tot_output_list[:,3*bool_hind+bool_lat]
    # Normalization of the inputs 
    local_input[:,:,1] = local_input[:,:,1] - np.nanmean(local_input[:,:,1],0)
    local_input[:,:,3] = local_input[:,:,3] - np.nanmean(local_input[:,:,3],0)
    tmp_vel = np.nanmean(local_input[:,:,2],1)
    if local_input.shape[0]==0:
        rsquare_diagonal = np.nan
        gains_diagonal = np.nan
        return rsquare_diagonal, gains_diagonal
    local_input[:,:,2] = local_input[:,:,2] - np.expand_dims(tmp_vel,-1)
    for line in range(local_input.shape[0]):
        xinput = np.arange(21)
        subjectlin = scipy.stats.linregress(xinput, local_input[line,:,0])
        local_input[line,:,0] = local_input[line,:,0] - (xinput*subjectlin.slope + subjectlin.intercept)
    # Normalization of the outputs
    if bool_lat:
        subjectlin = scipy.stats.linregress(tmp_vel, local_output)
        local_output = local_output - (tmp_vel * subjectlin.slope + subjectlin.intercept)
    else:
        local_output = local_output - np.nanmean(local_output)
    for time in range(21):
        if bool_lat:
            idx_plot = np.where(local_input[:,time,1]!=0)[0]
        else:
            idx_plot = np.where(local_input[:,time,1]!=0)[0]
        design_mat = np.hstack((np.ones((local_input[idx_plot].shape[0],1)),local_input[idx_plot,time,:]))
        if not bool_lat:
            design_mat_y = design_mat[:,[0,1,2,3,4]]
        else:
            design_mat_y = design_mat[:,[0,1,2,3,4]]
        if design_mat_y.shape[0]<10:
            rsquare_diagonal[:,time] = np.nan 
        else:
            a, b = multilinear_ols_rsquare_gains(design_mat_y, local_output[idx_plot])
            pred_output = b @ design_mat_y.T
            gains_diagonal[:,time,:] = b
            rsquare_diagonal[:,time] = a #multilinear_ols_rsquare(design_mat_y, local_output[idx_plot])

    return rsquare_diagonal, gains_diagonal

def normalize_phasor_metrics_verpeut(input_metrics, animal_length):
    """
    Normalizes the phasor metrics (only the speed)
    """
    g=9.81
    output_metrics = copy.deepcopy(input_metrics)
    local_length = animal_length[0,0]
    output_metrics[:,-1] = output_metrics[:,-1] / (np.sqrt(g*local_length))

    return output_metrics

def normalize_phasor_metrics(input_metrics, animal_length):
    """
    Normalizes the phasor metrics (only the speed)
    """
    g=9.81
    output_metrics = copy.deepcopy(input_metrics)
    n_animal = int(np.max(output_metrics[:,0]))+1
    for animal in range(n_animal):
        idx_animal = np.where(output_metrics[:,0]==animal)[0]
        local_length = animal_length[animal,0]
        output_metrics[idx_animal,-1] = output_metrics[idx_animal,-1] / (np.sqrt(g*local_length))

    return output_metrics

def normalize_metrics(input_metrics, animal_length, bool_klibaite=False):
    """
    This function normalizes the step-level metrics using Hof's definition
    """
    g = 9.81
    output_metrics = copy.deepcopy(input_metrics)
    n_animal = int(np.max(output_metrics[:,0]))+1
    if not bool_klibaite:
        for animal in range(n_animal):
            local_length = animal_length[animal,0]
            idx_animal = np.where(output_metrics[:,0]==animal)[0]
            output_metrics[idx_animal,3] = output_metrics[idx_animal,3] / (np.sqrt(local_length/g))
            output_metrics[idx_animal,4] = output_metrics[idx_animal,4] / local_length
            output_metrics[idx_animal,5] = output_metrics[idx_animal,5] / local_length
            output_metrics[idx_animal,6] = output_metrics[idx_animal,6] / (np.sqrt(g*local_length))
    else:
        for animal in range(n_animal):
            local_length = animal_length[animal,0]
            idx_animal = np.where(output_metrics[:,0]==animal)[0]
            output_metrics[idx_animal,3] = output_metrics[idx_animal,3] / (np.sqrt(local_length/g))
            output_metrics[idx_animal,4] = (0.001*output_metrics[idx_animal,4]) / local_length
            output_metrics[idx_animal,5] = (0.001*output_metrics[idx_animal,5]) / local_length
            output_metrics[idx_animal,6] = (0.001*output_metrics[idx_animal,6]) / (np.sqrt(g*local_length))

    return output_metrics




def get_rsquare_matrix_feedback(tot_animal, tot_input_list, tot_output_list, bool_hind, bool_lat):
    """
    Computes the rsquare matrix for the linear prediction of the foot contact location around the nominal 
    """


    n_animal = np.max(tot_animal).astype(int)+1
    rsquare_diagonal = np.zeros((n_animal, 21))
    gains_diagonal = np.zeros((n_animal,21,5))
    for animal in tqdm(range(n_animal)):
        idx_animal = np.where(tot_animal==animal)[0]
        idx_nan = np.where(~np.isnan(tot_input_list[idx_animal,15,0]))[0]
        local_input = tot_input_list[idx_animal[idx_nan],:,4*bool_hind:4+4*bool_hind]
        local_output = tot_output_list[idx_animal[idx_nan],3*bool_hind+bool_lat]
        # Normalization of the inputs 
        local_input[:,:,1] = local_input[:,:,1] - np.nanmean(local_input[:,:,1],0)
        local_input[:,:,3] = local_input[:,:,3] - np.nanmean(local_input[:,:,3],0)
        tmp_vel = np.nanmean(local_input[:,:,2],1)
        if local_input.shape[0]==0:
            rsquare_diagonal[animal,:] = np.nan
            gains_diagonal[animal,:] = np.nan
            continue
        local_input[:,:,2] = local_input[:,:,2] - np.expand_dims(tmp_vel,-1)
        for line in range(local_input.shape[0]):
            xinput = np.arange(21)
            subjectlin = scipy.stats.linregress(xinput, local_input[line,:,0])
            local_input[line,:,0] = local_input[line,:,0] - (xinput*subjectlin.slope + subjectlin.intercept)
        # Normalization of the outputs
        if bool_lat:
            subjectlin = scipy.stats.linregress(tmp_vel, local_output)
            local_output = local_output - (tmp_vel * subjectlin.slope + subjectlin.intercept)
        else:
            local_output = local_output - np.nanmean(local_output)
        for time in range(21):
            if bool_lat:
                idx_plot = np.where(local_input[:,time,1]!=0)[0]
            else:
                idx_plot = np.where(local_input[:,time,1]!=0)[0]
            design_mat = np.hstack((np.ones((local_input[idx_plot].shape[0],1)),local_input[idx_plot,time,:]))
            if not bool_lat:
                design_mat_y = design_mat[:,[0,1,2,3,4]]
            else:
                design_mat_y = design_mat[:,[0,1,2,3,4]]
            if design_mat_y.shape[0]<10:
                rsquare_diagonal[animal,time] = np.nan 
            else:
                a, b = multilinear_ols_rsquare_gains(design_mat_y, local_output[idx_plot])
                pred_output = b @ design_mat_y.T
                gains_diagonal[animal,time,:] = b
                rsquare_diagonal[animal,time] = a #multilinear_ols_rsquare(design_mat_y, local_output[idx_plot])
    
    return rsquare_diagonal, gains_diagonal


def multilinear_ols_rsquare_gains(X,y):
    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    yhat = X @ theta_hat
    rsquare = 1 - np.sum(np.square(yhat-y)) / np.sum(np.square(y))
    return rsquare, theta_hat

def multilinear_ols_rsquare(X,y):
    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
    yhat = X @ theta_hat
    rsquare = 1 - np.sum(np.square(yhat-y)) / np.sum(np.square(y))
    return rsquare

def get_rsquare_matrix_self(tot_animal, tot_input_self, tot_output_self, bool_hind, bool_lat):
    """
    Computes the rsquares matrix for the self prediction
    """
    n_animal = np.max(tot_animal).astype(int) + 1
    rsquare_diagonal = np.zeros((n_animal,21))
    for animal in range(n_animal):
        idx_animal = np.where(tot_animal==animal)[0]
        idx_nan = np.where(~np.isnan(tot_input_self[idx_animal,15,0]))[0]
        local_input = tot_input_self[idx_animal[idx_nan],:,4*bool_hind:4+4*bool_hind]
        local_output = tot_output_self[idx_animal[idx_nan],bool_lat+3*bool_hind] - np.nanmean(tot_output_self[idx_animal[idx_nan],bool_lat+3*bool_hind])
        for time in range(21):
            tmp_input_ = local_input[:,time,:]
            design_mat = np.hstack((np.ones((tmp_input_.shape[0],1)),tmp_input_))
            design_mat_y = design_mat
            if design_mat_y.shape[0]<10:
                rsquare_diagonal[animal,time] = np.nan 
            else:
                rsquare_diagonal[animal,time] = multilinear_ols_rsquare(design_mat_y, local_output)
    
    return rsquare_diagonal

def get_rsquare_matrix_self_verpeut(tot_input_self, tot_output_self, bool_hind, bool_lat):
    """
    Computes the rsquares matrix for the self prediction
    """
    rsquare_diagonal = np.zeros((1,21))
    idx_nan = np.where(~np.isnan(tot_input_self[:,15,0]))[0]
    local_input = tot_input_self[:,:,4*bool_hind:4+4*bool_hind]
    local_output = tot_output_self[:,bool_lat+3*bool_hind] - np.nanmean(tot_output_self[:,bool_lat+3*bool_hind])
    for time in range(21):
        tmp_input_ = local_input[:,time,:]
        design_mat = np.hstack((np.ones((tmp_input_.shape[0],1)),tmp_input_))
        design_mat_y = design_mat
        if design_mat_y.shape[0]<10:
            rsquare_diagonal[:,time] = np.nan 
        else:
            rsquare_diagonal[:,time] = multilinear_ols_rsquare(design_mat_y, local_output)
    
    return rsquare_diagonal


def plot_foot_placement_control_sh3(rsquares_body, rsquares_self, gains, labels, bool_plot=False, bool_save=False, figname=None):
    """
    Plots all the figures for the foot placement control
    """

    fig, axs = plt.subplots(1,2, figsize=(6,3), sharex=True, sharey=True)
    axs[0].spines[['top','right']].set_visible(False), axs[1].spines[['top','right']].set_visible(False)
    axs[0].plot(np.nanmedian(rsquares_body[labels==0,:], axis=0), color='k', lw=2)
    axs[0].plot(np.nanmedian(rsquares_self[labels==0,:], axis=0), color='b', lw=2)
    axs[0].fill_between(np.arange(21), np.nanpercentile(rsquares_body[labels==0,:],axis=0, q=25),np.nanpercentile(rsquares_body[labels==0,:],axis=0, q=75), color='k', alpha=0.5)
    axs[0].fill_between(np.arange(21), np.nanpercentile(rsquares_self[labels==0,:],axis=0, q=25),np.nanpercentile(rsquares_self[labels==0,:],axis=0, q=75), color='b', alpha=0.5)

    axs[1].plot(np.nanmedian(rsquares_body[labels==1,:], axis=0), color='k', lw=2)
    axs[1].plot(np.nanmedian(rsquares_self[labels==1,:], axis=0), color='b', lw=2)
    axs[1].fill_between(np.arange(21), np.nanpercentile(rsquares_body[labels==1,:],axis=0, q=25),np.nanpercentile(rsquares_body[labels==1,:],axis=0, q=75), color='k', alpha=0.5)
    axs[1].fill_between(np.arange(21), np.nanpercentile(rsquares_self[labels==1,:],axis=0, q=25),np.nanpercentile(rsquares_self[labels==1,:],axis=0, q=75), color='b', alpha=0.5)
    axs[0].set_xlabel('Relative gait fraction'), axs[1].set_xlabel('Relative gait fraction'), axs[0].set_ylabel('Explained variance')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_comparison_group2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_comparison_group2.svg'),bbox_inches='tight')

    diff = rsquares_body - rsquares_self

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(diff[labels==0,:], axis=0), color='k', lw=2)
    axs.plot(np.nanmedian(diff[labels==1,:], axis=0), color='r', lw=2)
    axs.fill_between(np.arange(21), np.nanpercentile(diff[labels==0,:], axis=0, q=25), np.nanpercentile(diff[labels==0,:], axis=0, q=75), color='k' ,alpha=0.5)
    axs.fill_between(np.arange(21), np.nanpercentile(diff[labels==1,:], axis=0, q=25), np.nanpercentile(diff[labels==1,:], axis=0, q=75), color='r' ,alpha=0.5)
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel(r'$\Delta$ explained variance')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_difference_group3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_difference_group3.svg'),bbox_inches='tight')


    # Feedback gains

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(gains[labels==0,:,2], axis=0), color='k', lw=2)
    axs.plot(np.nanmedian(gains[labels==1,:,2], axis=0), color='r', lw=2)
    axs.fill_between(np.arange(21), np.nanpercentile(gains[labels==0,:,2], axis=0, q=25), np.nanpercentile(gains[labels==0,:,2], axis=0, q=75), color='k' ,alpha=0.5)
    axs.fill_between(np.arange(21), np.nanpercentile(gains[labels==1,:,2], axis=0, q=25), np.nanpercentile(gains[labels==1,:,2], axis=0, q=75), color='r' ,alpha=0.5)
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Feedback gains')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_position_group3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_position_group3.svg'),bbox_inches='tight')


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(gains[labels==0,:,4], axis=0), color='k', lw=2)
    axs.plot(np.nanmedian(gains[labels==1,:,4], axis=0), color='r', lw=2)
    axs.fill_between(np.arange(21), np.nanpercentile(gains[labels==0,:,4], axis=0, q=25), np.nanpercentile(gains[labels==0,:,4], axis=0, q=75), color='k' ,alpha=0.5)
    axs.fill_between(np.arange(21), np.nanpercentile(gains[labels==1,:,4], axis=0, q=25), np.nanpercentile(gains[labels==1,:,4], axis=0, q=75), color='r' ,alpha=0.5)
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Feedback gains')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_velocity_group3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_velocity_group3.svg'),bbox_inches='tight')

    n_0 = len(np.where(labels==0)[0])
    n_1 = len(np.where(labels==1)[0])

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,n_0), gains[labels==0,15,2], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,n_1), gains[labels==1,15,2], color='r', s=5)
    axs.scatter(0,np.nanmedian(gains[labels==0,15,2]), color='k', s=20)
    axs.scatter(1,np.nanmedian(gains[labels==1,15,2]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(gains[labels==0,15,2], q=25), np.nanpercentile(gains[labels==0,15,2], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(gains[labels==1,15,2], q=25), np.nanpercentile(gains[labels==1,15,2], q=75)], color='r', lw=2)
    axs.set_ylabel('Feedback gains')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_ind_position_group3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_ind_position_group3.svg'),bbox_inches='tight')


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,n_0), gains[labels==0,15,4], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,n_1), gains[labels==1,15,4], color='r', s=5)
    axs.scatter(0,np.nanmedian(gains[labels==0,15,4]), color='k', s=20)
    axs.scatter(1,np.nanmedian(gains[labels==1,15,4]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(gains[labels==0,15,4], q=25), np.nanpercentile(gains[labels==0,15,4], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(gains[labels==1,15,4], q=25), np.nanpercentile(gains[labels==1,15,4], q=75)], color='r', lw=2)
    axs.set_ylabel('Feedback gains')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_ind_velocity_group3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_ind_velocity_group3.svg'),bbox_inches='tight')


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,n_0), np.nanmax(diff[labels==0,:],axis=1), color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,n_1), np.nanmax(diff[labels==1,:],axis=1), color='r', s=5)
    axs.scatter(0, np.nanmedian(np.nanmax(diff[labels==0,:],axis=1)), color='k', s=20)
    axs.scatter(1, np.nanmedian(np.nanmax(diff[labels==1,:],axis=1)), color='r', s=20)
    axs.plot([0,0], [np.nanpercentile(np.nanmax(diff[labels==0,:],axis=1), q=25), np.nanpercentile(np.nanmax(diff[labels==0,:],axis=1), q=75)], color='k', lw=2)
    axs.plot([1,1], [np.nanpercentile(np.nanmax(diff[labels==1,:],axis=1), q=25), np.nanpercentile(np.nanmax(diff[labels==1,:],axis=1), q=75)], color='r', lw=2)
    axs.set_ylabel('Control amplitude')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_control_amplitude_group3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_contorl_amplitude_group3.svg'),bbox_inches='tight')


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,n_0), rsquares_body[labels==0,0], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,n_1), rsquares_body[labels==1,0], color='r', s=5)
    axs.scatter(0, np.nanmedian(rsquares_body[labels==0,0]), color='k', s=20)
    axs.scatter(1, np.nanmedian(rsquares_body[labels==1,0]), color='r', s=20)
    axs.plot([0,0], [np.nanpercentile(rsquares_body[labels==0,0], q=25), np.nanpercentile(rsquares_body[labels==0,0], q=75)], color='k', lw=2)
    axs.plot([1,1], [np.nanpercentile(rsquares_body[labels==1,0], q=25), np.nanpercentile(rsquares_body[labels==1,0], q=75)], color='r', lw=2)
    axs.set_ylabel('feedforward control')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_ffwd_control_group3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_ffwd_control_group3.svg'),bbox_inches='tight')


    # Defining the metrics 
    local_matrix = np.concatenate((np.expand_dims(np.nanmax(diff,axis=1),-1), np.expand_dims(rsquares_body[:,0],-1), np.expand_dims(gains[:,15,4],-1)), axis=1)
    
    if bool_plot:
        plt.show()
    else:
        plt.close('all')

    return local_matrix

def plot_foot_placement_control_rett(list_rsquares_body, list_rsquares_self, list_fp_gains, bool_plot=False, bool_save=False, figname=None):
    """
    Plots all the figures for the foot placement control
    """
    cmap = cm.plasma(np.linspace(0,1,len(list_rsquares_body)))
    ###################
    ### Group level ###
    ###################

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_rsquares_body[0], axis=0), color='k', lw=2)
    axs.plot(np.nanmedian(list_rsquares_self[0], axis=0), color='b', lw=2)
    axs.fill_between(np.arange(21), np.nanpercentile(list_rsquares_body[0],axis=0,q=25), np.nanpercentile(list_rsquares_body[0],axis=0,q=75),color='k', alpha=0.5)
    axs.fill_between(np.arange(21), np.nanpercentile(list_rsquares_self[0],axis=0,q=25), np.nanpercentile(list_rsquares_self[0],axis=0,q=75),color='b', alpha=0.5)
    axs.set_ylim([-0.05,1.05])
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Explained variance')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_comparison_group0.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_comparison_group0.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_rsquares_body[1], axis=0), color='k', lw=2)
    axs.plot(np.nanmedian(list_rsquares_self[1], axis=0), color='b', lw=2)
    axs.fill_between(np.arange(21), np.nanpercentile(list_rsquares_body[1],axis=0,q=25), np.nanpercentile(list_rsquares_body[1],axis=0,q=75),color='k', alpha=0.5)
    axs.fill_between(np.arange(21), np.nanpercentile(list_rsquares_self[1],axis=0,q=25), np.nanpercentile(list_rsquares_self[1],axis=0,q=75),color='b', alpha=0.5)
    axs.set_ylim([-0.05,1.05])
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Explained variance')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_comparison_group1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_comparison_group1.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_rsquares_body[2], axis=0), color='k', lw=2)
    axs.plot(np.nanmedian(list_rsquares_self[2], axis=0), color='b', lw=2)
    axs.fill_between(np.arange(21), np.nanpercentile(list_rsquares_body[2],axis=0,q=25), np.nanpercentile(list_rsquares_body[2],axis=0,q=75),color='k', alpha=0.5)
    axs.fill_between(np.arange(21), np.nanpercentile(list_rsquares_self[2],axis=0,q=25), np.nanpercentile(list_rsquares_self[2],axis=0,q=75),color='b', alpha=0.5)
    axs.set_ylim([-0.05,1.05])
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Explained variance')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_comparison_group2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_comparison_group2.svg'),bbox_inches='tight')

    if len(list_rsquares_body)>3:

        fig, axs = plt.subplots(1,1,figsize=(3,3))
        axs.spines[['top','right']].set_visible(False)
        axs.plot(np.nanmedian(list_rsquares_body[3], axis=0), color='k', lw=2)
        axs.plot(np.nanmedian(list_rsquares_self[3], axis=0), color='b', lw=2)
        axs.fill_between(np.arange(21), np.nanpercentile(list_rsquares_body[3],axis=0,q=25), np.nanpercentile(list_rsquares_body[3],axis=0,q=75),color='k', alpha=0.5)
        axs.fill_between(np.arange(21), np.nanpercentile(list_rsquares_self[3],axis=0,q=25), np.nanpercentile(list_rsquares_self[3],axis=0,q=75),color='b', alpha=0.5)
        axs.set_ylim([-0.05,1.05])
        axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Explained variance')
        plt.tight_layout()
        if bool_save:
            fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_comparison_group3.png'),bbox_inches='tight')
            fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_comparison_group3.svg'),bbox_inches='tight')

    diff_list = [] 
    for group in range(len(list_rsquares_body)):
        diff_list.append(list_rsquares_body[group] - list_rsquares_self[group])


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(diff_list)):
        axs.plot(np.nanmedian(diff_list[group], axis=0), color=cmap[group], lw=2)
        axs.fill_between(np.arange(21), np.nanpercentile(diff_list[group], axis=0, q=25), np.nanpercentile(diff_list[group], axis=0, q=75), color=cmap[group] ,alpha=0.5)
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel(r'$\Delta$ explained variance')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_difference_all_groups.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_difference_all_groups.svg'),bbox_inches='tight')


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_fp_gains)):
        axs.plot(np.nanmedian(list_fp_gains[group][:,:,2], axis=0), color=cmap[group], lw=2)
        axs.fill_between(np.arange(21), np.nanpercentile(list_fp_gains[group][:,:,2], axis=0, q=25), np.nanpercentile(list_fp_gains[group][:,:,2], axis=0, q=75), color=cmap[group] ,alpha=0.5)
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Feedback gains')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_position_all_groups.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_position_all_groups.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_fp_gains)):
        axs.plot(np.nanmedian(list_fp_gains[group][:,:,4], axis=0), color=cmap[group], lw=2)
        axs.fill_between(np.arange(21), np.nanpercentile(list_fp_gains[group][:,:,4], axis=0, q=25), np.nanpercentile(list_fp_gains[group][:,:,4], axis=0, q=75), color=cmap[group] ,alpha=0.5)
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Feedback gains')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_velocity_all_groups.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_velocity_all_groups.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_fp_gains)):
        axs.scatter(np.random.uniform(group-0.2,group+0.2,list_fp_gains[group].shape[0]), list_fp_gains[group][:,15,2], color=cmap[group], s=5)
        axs.scatter(group,np.nanmedian(list_fp_gains[group][:,15,2]), color=cmap[group], s=20)
        axs.plot([group,group],[np.nanpercentile(list_fp_gains[group][:,15,2], q=25), np.nanpercentile(list_fp_gains[group][:,15,2], q=75)], color=cmap[group], lw=2)
    axs.set_ylabel('Feedback gains')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_ind_position_all_groups.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_ind_position_all_groups.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_fp_gains)):
        axs.scatter(np.random.uniform(group-0.2,group+0.2,list_fp_gains[group].shape[0]), list_fp_gains[group][:,15,4], color=cmap[group], s=5)
        axs.scatter(group,np.nanmedian(list_fp_gains[group][:,15,4]), color=cmap[group], s=20)
        axs.plot([group,group],[np.nanpercentile(list_fp_gains[group][:,15,4], q=25), np.nanpercentile(list_fp_gains[group][:,15,4], q=75)], color=cmap[group], lw=2)
    axs.set_ylabel('Feedback gains')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_ind_velocity_all_groups.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_ind_velocity_all_groups.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(diff_list)):
        axs.scatter(np.random.uniform(group-0.2,group+0.2,diff_list[group].shape[0]), np.nanmax(diff_list[group],axis=1), color=cmap[group], s=5)
        axs.scatter(group, np.nanmedian(np.nanmax(diff_list[group],axis=1)), color=cmap[group], s=20)
        axs.plot([group,group], [np.nanpercentile(np.nanmax(diff_list[group],axis=1), q=25), np.nanpercentile(np.nanmax(diff_list[group],axis=1), q=75)], color=cmap[group], lw=2)
    axs.set_ylabel('Control amplitude')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_control_amplitude_all_groups.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_contorl_amplitude_all_groups.svg'),bbox_inches='tight')


    # Saving the individual metrics 
    list_ind_metrics = []
    for group in range(len(list_rsquares_body)):
        n_animal = list_rsquares_body[group].shape[0]
        local_matrix = np.concatenate((np.expand_dims(np.nanmax(diff_list[group],axis=1),-1), np.expand_dims(list_rsquares_body[group][:,0],-1), np.expand_dims(list_fp_gains[group][:,15,2],-1), np.expand_dims(list_fp_gains[group][:,15,4],-1)), axis=1)
        list_ind_metrics.append(local_matrix)


    # Statistics for the foot placement behaviors
    diff_kruskall, ffwd_kruskall, pos_kruskall, vel_kruskall = [], [], [], []
    for ii in range(len(diff_list)):
        diff_kruskall.append(np.nanmax(diff_list[ii],axis=1))
        ffwd_kruskall.append(list_rsquares_body[ii][:,0])
        pos_kruskall.append(list_fp_gains[ii][:,15,2])
        vel_kruskall.append(list_fp_gains[ii][:,15,4])
    print('============================')
    print('Statistics control amplitude')
    print('============================')
    print(scipy.stats.kruskal(diff_kruskall[0], diff_kruskall[1], diff_kruskall[2]))
    print(spph.posthoc_dunn(diff_kruskall, p_adjust="fdr_bh"))

    print('=======================')
    print('Statistics ffwd control')
    print('=======================')
    print(scipy.stats.kruskal(ffwd_kruskall[0], ffwd_kruskall[1], ffwd_kruskall[2]))
    print(spph.posthoc_dunn(ffwd_kruskall, p_adjust="fdr_bh"))

    print('=======================')
    print('Statistics pos fb gains')
    print('=======================')
    print(scipy.stats.kruskal(pos_kruskall[0], pos_kruskall[1], pos_kruskall[2]))
    print(spph.posthoc_dunn(pos_kruskall, p_adjust="fdr_bh"))

    print('=======================')
    print('Statistics vel fb gains')
    print('=======================')
    print(scipy.stats.kruskal(vel_kruskall[0], vel_kruskall[1], vel_kruskall[2]))
    print(spph.posthoc_dunn(vel_kruskall, p_adjust="fdr_bh"))

    if bool_plot:
        plt.show()
    else:
        plt.close('all')

    return list_ind_metrics
    

def plot_foot_placement_control(list_rsquares_body, list_rsquares_self, list_fp_gains, bool_plot=False, bool_save=False, figname=None):
    """
    Plots all the figures for the foot placement control
    """

    ###################
    ### Group level ###
    ###################

    fig, axs = plt.subplots(1,2,figsize=(6,3), sharex=True, sharey=True)
    axs[0].spines[['top','right']].set_visible(False), axs[1].spines[['top','right']].set_visible(False)
    axs[0].plot(np.nanmedian(list_rsquares_body[0], axis=0), color='k', lw=2)
    axs[0].plot(np.nanmedian(list_rsquares_self[0], axis=0), color='b', lw=2)
    axs[0].fill_between(np.arange(21), np.nanpercentile(list_rsquares_body[0],axis=0,q=25), np.nanpercentile(list_rsquares_body[0],axis=0,q=75),color='k', alpha=0.5)
    axs[0].fill_between(np.arange(21), np.nanpercentile(list_rsquares_self[0],axis=0,q=25), np.nanpercentile(list_rsquares_self[0],axis=0,q=75),color='b', alpha=0.5)
    axs[0].set_ylim([-0.05,1.05])

    axs[1].plot(np.nanmedian(list_rsquares_body[2], axis=0), color='k', lw=2)
    axs[1].plot(np.nanmedian(list_rsquares_self[2], axis=0), color='b', lw=2)
    axs[1].fill_between(np.arange(21), np.nanpercentile(list_rsquares_body[2],axis=0,q=25), np.nanpercentile(list_rsquares_body[2],axis=0,q=75),color='k', alpha=0.5)
    axs[1].fill_between(np.arange(21), np.nanpercentile(list_rsquares_self[2],axis=0,q=25), np.nanpercentile(list_rsquares_self[2],axis=0,q=75),color='b', alpha=0.5)
    axs[0].set_xlabel('Relative gait fraction'), axs[1].set_xlabel('Relative gait fraction'), axs[0].set_ylabel('Explained variance')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_comparison_group1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_comparison_group1.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,2,figsize=(6,3), sharex=True, sharey=True)
    axs[0].spines[['top','right']].set_visible(False), axs[1].spines[['top','right']].set_visible(False)
    axs[0].plot(np.nanmedian(list_rsquares_body[0], axis=0), color='k', lw=2)
    axs[0].plot(np.nanmedian(list_rsquares_self[0], axis=0), color='b', lw=2)
    axs[0].fill_between(np.arange(21), np.nanpercentile(list_rsquares_body[0],axis=0,q=25), np.nanpercentile(list_rsquares_body[0],axis=0,q=75),color='k', alpha=0.5)
    axs[0].fill_between(np.arange(21), np.nanpercentile(list_rsquares_self[0],axis=0,q=25), np.nanpercentile(list_rsquares_self[0],axis=0,q=75),color='b', alpha=0.5)
    axs[0].set_ylim([-0.05,1.05])

    axs[1].plot(np.nanmedian(list_rsquares_body[2], axis=0), color='k', lw=2)
    axs[1].plot(np.nanmedian(list_rsquares_self[2], axis=0), color='b', lw=2)
    axs[1].fill_between(np.arange(21), np.nanpercentile(list_rsquares_body[2],axis=0,q=25), np.nanpercentile(list_rsquares_body[2],axis=0,q=75),color='k', alpha=0.5)
    axs[1].fill_between(np.arange(21), np.nanpercentile(list_rsquares_self[2],axis=0,q=25), np.nanpercentile(list_rsquares_self[2],axis=0,q=75),color='b', alpha=0.5)
    axs[0].set_xlabel('Relative gait fraction'), axs[1].set_xlabel('Relative gait fraction'), axs[0].set_ylabel('Explained variance')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_comparison_group2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_comparison_group2.svg'),bbox_inches='tight')

    diff_list = [] 
    for group in range(6):
        diff_list.append(list_rsquares_body[group] - list_rsquares_self[group])


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(diff_list[0], axis=0), color='k', lw=2)
    axs.plot(np.nanmedian(diff_list[2], axis=0), color='r', lw=2)
    axs.fill_between(np.arange(21), np.nanpercentile(diff_list[0], axis=0, q=25), np.nanpercentile(diff_list[0], axis=0, q=75), color='k' ,alpha=0.5)
    axs.fill_between(np.arange(21), np.nanpercentile(diff_list[2], axis=0, q=25), np.nanpercentile(diff_list[2], axis=0, q=75), color='r' ,alpha=0.5)
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel(r'$\Delta$ explained variance')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_difference_group1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_difference_group1.svg'),bbox_inches='tight')


    
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(diff_list[3], axis=0), color='k', lw=2)
    axs.plot(np.nanmedian(diff_list[5], axis=0), color='r', lw=2)
    axs.fill_between(np.arange(21), np.nanpercentile(diff_list[3], axis=0, q=25), np.nanpercentile(diff_list[3], axis=0, q=75), color='k' ,alpha=0.5)
    axs.fill_between(np.arange(21), np.nanpercentile(diff_list[5], axis=0, q=25), np.nanpercentile(diff_list[5], axis=0, q=75), color='r' ,alpha=0.5)
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel(r'$\Delta$ explained variance')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_difference_group2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_rsquare_difference_group2.svg'),bbox_inches='tight')


    # Feedback gains

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_fp_gains[0][:,:,2], axis=0), color='k', lw=2)
    axs.plot(np.nanmedian(list_fp_gains[2][:,:,2], axis=0), color='r', lw=2)
    axs.fill_between(np.arange(21), np.nanpercentile(list_fp_gains[0][:,:,2], axis=0, q=25), np.nanpercentile(list_fp_gains[0][:,:,2], axis=0, q=75), color='k' ,alpha=0.5)
    axs.fill_between(np.arange(21), np.nanpercentile(list_fp_gains[2][:,:,2], axis=0, q=25), np.nanpercentile(list_fp_gains[2][:,:,2], axis=0, q=75), color='r' ,alpha=0.5)
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Feedback gains')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_position_group1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_position_group1.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_fp_gains[3][:,:,2], axis=0), color='k', lw=2)
    axs.plot(np.nanmedian(list_fp_gains[5][:,:,2], axis=0), color='r', lw=2)
    axs.fill_between(np.arange(21), np.nanpercentile(list_fp_gains[3][:,:,2], axis=0, q=25), np.nanpercentile(list_fp_gains[3][:,:,2], axis=0, q=75), color='k' ,alpha=0.5)
    axs.fill_between(np.arange(21), np.nanpercentile(list_fp_gains[5][:,:,2], axis=0, q=25), np.nanpercentile(list_fp_gains[5][:,:,2], axis=0, q=75), color='r' ,alpha=0.5)
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Feedback gains')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_position_group2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_position_group2.svg'),bbox_inches='tight')


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_fp_gains[0][:,:,4], axis=0), color='k', lw=2)
    axs.plot(np.nanmedian(list_fp_gains[2][:,:,4], axis=0), color='r', lw=2)
    axs.fill_between(np.arange(21), np.nanpercentile(list_fp_gains[0][:,:,4], axis=0, q=25), np.nanpercentile(list_fp_gains[0][:,:,4], axis=0, q=75), color='k' ,alpha=0.5)
    axs.fill_between(np.arange(21), np.nanpercentile(list_fp_gains[2][:,:,4], axis=0, q=25), np.nanpercentile(list_fp_gains[2][:,:,4], axis=0, q=75), color='r' ,alpha=0.5)
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Feedback gains')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_velocity_group1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_velocity_group1.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_fp_gains[3][:,:,4], axis=0), color='k', lw=2)
    axs.plot(np.nanmedian(list_fp_gains[5][:,:,4], axis=0), color='r', lw=2)
    axs.fill_between(np.arange(21), np.nanpercentile(list_fp_gains[3][:,:,4], axis=0, q=25), np.nanpercentile(list_fp_gains[3][:,:,4], axis=0, q=75), color='k' ,alpha=0.5)
    axs.fill_between(np.arange(21), np.nanpercentile(list_fp_gains[5][:,:,4], axis=0, q=25), np.nanpercentile(list_fp_gains[5][:,:,4], axis=0, q=75), color='r' ,alpha=0.5)
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Feedback gains')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_velocity_group2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_velocity_group2.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_fp_gains[0].shape[0]), list_fp_gains[0][:,15,2], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_fp_gains[2].shape[0]), list_fp_gains[2][:,15,2], color='r', s=5)
    axs.scatter(0,np.nanmedian(list_fp_gains[0][:,15,2]), color='k', s=20)
    axs.scatter(1,np.nanmedian(list_fp_gains[2][:,15,2]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_fp_gains[0][:,15,2], q=25), np.nanpercentile(list_fp_gains[0][:,15,2], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_fp_gains[2][:,15,2], q=25), np.nanpercentile(list_fp_gains[2][:,15,2], q=75)], color='r', lw=2)
    axs.set_ylabel('Feedback gains')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_ind_position_group1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_ind_position_group1.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_fp_gains[0].shape[0]), list_fp_gains[0][:,15,4], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_fp_gains[2].shape[0]), list_fp_gains[2][:,15,4], color='r', s=5)
    axs.scatter(0,np.nanmedian(list_fp_gains[0][:,15,4]), color='k', s=20)
    axs.scatter(1,np.nanmedian(list_fp_gains[2][:,15,4]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_fp_gains[0][:,15,4], q=25), np.nanpercentile(list_fp_gains[0][:,15,4], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_fp_gains[2][:,15,4], q=25), np.nanpercentile(list_fp_gains[2][:,15,4], q=75)], color='r', lw=2)
    axs.set_ylabel('Feedback gains')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_ind_position_group2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_ind_position_group2.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_fp_gains[3].shape[0]), list_fp_gains[3][:,15,2], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_fp_gains[5].shape[0]), list_fp_gains[5][:,15,2], color='r', s=5)
    axs.scatter(0,np.nanmedian(list_fp_gains[3][:,15,2]), color='k', s=20)
    axs.scatter(1,np.nanmedian(list_fp_gains[5][:,15,2]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_fp_gains[3][:,15,2], q=25), np.nanpercentile(list_fp_gains[3][:,15,2], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_fp_gains[5][:,15,2], q=25), np.nanpercentile(list_fp_gains[5][:,15,2], q=75)], color='r', lw=2)
    axs.set_ylabel('Feedback gains')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_ind_velocity_group1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_ind_velocity_group1.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_fp_gains[3].shape[0]), list_fp_gains[3][:,15,4], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_fp_gains[5].shape[0]), list_fp_gains[5][:,15,4], color='r', s=5)
    axs.scatter(0,np.nanmedian(list_fp_gains[3][:,15,4]), color='k', s=20)
    axs.scatter(1,np.nanmedian(list_fp_gains[5][:,15,4]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_fp_gains[3][:,15,4], q=25), np.nanpercentile(list_fp_gains[3][:,15,4], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_fp_gains[5][:,15,4], q=25), np.nanpercentile(list_fp_gains[5][:,15,4], q=75)], color='r', lw=2)
    axs.set_ylabel('Feedback gains')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_ind_velocity_group2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_fbgains_ind_velocity_group2.svg'),bbox_inches='tight')
    
    # Control amplitude

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,diff_list[0].shape[0]), np.nanmax(diff_list[0],axis=1), color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,diff_list[2].shape[0]), np.nanmax(diff_list[2],axis=1), color='r', s=5)
    axs.scatter(0, np.nanmedian(np.nanmax(diff_list[0],axis=1)), color='k', s=20)
    axs.scatter(1, np.nanmedian(np.nanmax(diff_list[2],axis=1)), color='r', s=20)
    axs.plot([0,0], [np.nanpercentile(np.nanmax(diff_list[0],axis=1), q=25), np.nanpercentile(np.nanmax(diff_list[0],axis=1), q=75)], color='k', lw=2)
    axs.plot([1,1], [np.nanpercentile(np.nanmax(diff_list[2],axis=1), q=25), np.nanpercentile(np.nanmax(diff_list[2],axis=1), q=75)], color='r', lw=2)
    axs.set_ylabel('Control amplitude')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_control_amplitude_group1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_contorl_amplitude_group1.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,diff_list[3].shape[0]), np.nanmax(diff_list[3],axis=1), color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,diff_list[5].shape[0]), np.nanmax(diff_list[5],axis=1), color='r', s=5)
    axs.scatter(0, np.nanmedian(np.nanmax(diff_list[3],axis=1)), color='k', s=20)
    axs.scatter(1, np.nanmedian(np.nanmax(diff_list[5],axis=1)), color='r', s=20)
    axs.plot([0,0], [np.nanpercentile(np.nanmax(diff_list[3],axis=1), q=25), np.nanpercentile(np.nanmax(diff_list[3],axis=1), q=75)], color='k', lw=2)
    axs.plot([1,1], [np.nanpercentile(np.nanmax(diff_list[5],axis=1), q=25), np.nanpercentile(np.nanmax(diff_list[5],axis=1), q=75)], color='r', lw=2)
    axs.set_ylabel('Control amplitude')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_control_amplitude_group2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_control_amplitude_group2.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,diff_list[0].shape[0]), list_rsquares_body[0][:,0], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,diff_list[2].shape[0]), list_rsquares_body[2][:,0], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_rsquares_body[0][:,0]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_rsquares_body[2][:,0]), color='r', s=20)
    axs.plot([0,0], [np.nanpercentile(list_rsquares_body[0][:,0], q=25), np.nanpercentile(list_rsquares_body[0][:,0], q=75)], color='k', lw=2)
    axs.plot([1,1], [np.nanpercentile(list_rsquares_body[2][:,0], q=25), np.nanpercentile(list_rsquares_body[2][:,0], q=75)], color='r', lw=2)
    axs.set_ylabel('feedforward control')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_ffwd_control_group1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_ffwd_control_group1.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,diff_list[3].shape[0]), list_rsquares_body[3][:,0], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,diff_list[5].shape[0]), list_rsquares_body[5][:,0], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_rsquares_body[3][:,0]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_rsquares_body[5][:,0]), color='r', s=20)
    axs.plot([0,0], [np.nanpercentile(list_rsquares_body[3][:,0], q=25), np.nanpercentile(list_rsquares_body[3][:,0], q=75)], color='k', lw=2)
    axs.plot([1,1], [np.nanpercentile(list_rsquares_body[5][:,0], q=25), np.nanpercentile(list_rsquares_body[5][:,0], q=75)], color='r', lw=2)
    axs.set_ylabel('feedforward control')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_ffwd_control_group2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_ffwd_control_group2.svg'),bbox_inches='tight')


    # Saving the individual metrics 
    list_ind_metrics = []
    for group in range(6):
        n_animal = list_rsquares_body[group].shape[0]
        local_matrix = np.concatenate((np.expand_dims(np.nanmax(diff_list[group],axis=1),-1), np.expand_dims(list_rsquares_body[group][:,0],-1), np.expand_dims(list_fp_gains[group][:,15,4],-1)), axis=1)
        list_ind_metrics.append(local_matrix)
    
    if bool_plot:
        plt.show()
    else:
        plt.close('all')

    return list_ind_metrics


def plot_interlimb_coordination_rett(list_phasor_data, bins_centers, bool_plot=False, bool_save=False, figname=None):
    """
    Plots all the figures for the interlimb coordination metrics
    """
    cmap = cm.plasma(np.linspace(0,1,len(list_phasor_data)))
    ###################
    ### Group level ###
    ###################
    fig, axs = plt.subplots(1,1,figsize=(5,5), subplot_kw={"projection":"polar"})
    for group in range(len(list_phasor_data)):
        axs.plot(scipy.stats.circmean(list_phasor_data[group][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit'),bins_centers,color=cmap[group],lw=2,ls='--')
        axs.plot(scipy.stats.circmean(list_phasor_data[group][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit'),bins_centers,color=cmap[group],lw=2,ls=':')
        axs.plot(scipy.stats.circmean(list_phasor_data[group][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit'),bins_centers,color=cmap[group],lw=2,ls='-')
        axs.fill_betweenx(bins_centers,scipy.stats.circmean(list_phasor_data[group][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit')+scipy.stats.circstd(list_phasor_data[group][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit'),scipy.stats.circmean(list_phasor_data[group][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit')-scipy.stats.circstd(list_phasor_data[group][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit'),color=cmap[group],alpha=0.5)
        axs.fill_betweenx(bins_centers,scipy.stats.circmean(list_phasor_data[group][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit')+scipy.stats.circstd(list_phasor_data[group][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit'),scipy.stats.circmean(list_phasor_data[group][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit')-scipy.stats.circstd(list_phasor_data[group][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit'),color=cmap[group],alpha=0.5)
        axs.fill_betweenx(bins_centers,scipy.stats.circmean(list_phasor_data[group][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit')+scipy.stats.circstd(list_phasor_data[group][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit'),scipy.stats.circmean(list_phasor_data[group][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit')-scipy.stats.circstd(list_phasor_data[group][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit'),color=cmap[group],alpha=0.5)
    axs.set_ylim([0.15,1.05]), axs.set_xticks([0,np.pi/2,np.pi,3*np.pi/2]), axs.set_xticklabels(['0','0.25','0.5','0.75'])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_average_all_groups.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_average_all_groups.svg'),bbox_inches='tight')

    

    ########################
    ### Individual level ###
    ########################

    list_ind_metrics = []
    for group in range(len(list_phasor_data)):
        list_ind_metrics.append(np.nanmean(list_phasor_data[group]/(2*np.pi),axis=1))

    fig, axs = plt.subplots(1,3,figsize=(9,3))
    for leg in range(3):
        axs[leg].spines[['top','right']].set_visible(False)
        for group in range(len(list_phasor_data)):
            axs[leg].scatter(np.random.uniform(group-0.2,group+0.2, list_phasor_data[group].shape[0]), np.nanmean(list_phasor_data[group][:,:,leg]/(2*np.pi),axis=1), color=cmap[group], s=5)
            axs[leg].scatter(group, np.nanmedian(np.nanmean(list_phasor_data[group][:,:,leg]/(2*np.pi),1),0), color=cmap[group])
            axs[leg].plot([group,group],[np.nanpercentile(np.nanmean(list_phasor_data[group][:,:,leg]/(2*np.pi),1), axis=0, q=25), np.nanpercentile(np.nanmean(list_phasor_data[group][:,:,leg]/(2*np.pi),1), axis=0, q=75)], color=cmap[group], lw=2)
        axs[leg].set_xlim([-0.5,len(list_phasor_data)-0.5])
    axs[0].set_ylabel('Relative gait fraction')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_individual_all_groups.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_individual_all_groups.svg'),bbox_inches='tight')

    # Statistics individual level
    leg1_kruskall, leg2_kruskall, leg3_kruskall = [], [], []
    for ii in range(len(list_ind_metrics)):
        leg1_kruskall.append(list_ind_metrics[ii][:,0])
        leg2_kruskall.append(list_ind_metrics[ii][:,1])
        leg3_kruskall.append(list_ind_metrics[ii][:,2])
    print('================')
    print('Statistics leg 1')
    print('================')
    print(scipy.stats.kruskal(leg1_kruskall[0], leg1_kruskall[1], leg1_kruskall[2]))
    print(spph.posthoc_dunn(leg1_kruskall, p_adjust="fdr_bh"))

    print('================')
    print('Statistics leg 2')
    print('================')
    print(scipy.stats.kruskal(leg2_kruskall[0], leg2_kruskall[1], leg2_kruskall[2]))
    print(spph.posthoc_dunn(leg2_kruskall, p_adjust="fdr_bh"))

    print('================')
    print('Statistics leg 3')
    print('================')
    print(scipy.stats.kruskal(leg3_kruskall[0], leg3_kruskall[1], leg3_kruskall[2]))
    print(spph.posthoc_dunn(leg3_kruskall, p_adjust="fdr_bh"))

    if bool_plot:
        plt.show()
    else:
        plt.close('all')

    return list_ind_metrics

def plot_interlimb_coordination_sh3(list_phasor_data, bins_centers, labels, bool_plot=False, bool_save=False, figname=None):
    """
    Plots all the figures for the interlimb coordination metrics 
    """

    #################
    ## Group level ##
    #################

    fig, axs = plt.subplots(1,1,figsize=(5,5), subplot_kw={"projection":"polar"})
    axs.plot(scipy.stats.circmean(list_phasor_data[labels==0,:,0], low=0, high=2*np.pi, axis=0, nan_policy='omit'),bins_centers,color='k',ls='--',lw=2)
    axs.plot(scipy.stats.circmean(list_phasor_data[labels==0,:,1], low=0, high=2*np.pi, axis=0, nan_policy='omit'),bins_centers,color='k',ls=':',lw=2)
    axs.plot(scipy.stats.circmean(list_phasor_data[labels==0,:,2], low=-np.pi, high=np.pi, axis=0, nan_policy='omit'),bins_centers,color='k',ls='-',lw=2)
    axs.plot(scipy.stats.circmean(list_phasor_data[labels==1,:,0], low=0, high=2*np.pi, axis=0, nan_policy='omit'),bins_centers,color='r',ls='--',lw=2)
    axs.plot(scipy.stats.circmean(list_phasor_data[labels==1,:,1], low=0, high=2*np.pi, axis=0, nan_policy='omit'),bins_centers,color='r',ls=':',lw=2)
    axs.plot(scipy.stats.circmean(list_phasor_data[labels==1,:,2], low=-np.pi, high=np.pi, axis=0, nan_policy='omit'),bins_centers,color='r',ls='-',lw=2)
    axs.fill_betweenx(bins_centers, scipy.stats.circmean(list_phasor_data[labels==0,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit')+scipy.stats.circstd(list_phasor_data[labels==0,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit'),scipy.stats.circmean(list_phasor_data[labels==0,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit')-scipy.stats.circstd(list_phasor_data[labels==0,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit'),color='k',alpha=0.5)
    axs.fill_betweenx(bins_centers, scipy.stats.circmean(list_phasor_data[labels==0,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit')+scipy.stats.circstd(list_phasor_data[labels==0,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit'),scipy.stats.circmean(list_phasor_data[labels==0,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit')-scipy.stats.circstd(list_phasor_data[labels==0,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit'),color='k',alpha=0.5)
    axs.fill_betweenx(bins_centers, scipy.stats.circmean(list_phasor_data[labels==0,:,2],low=0,high=2*np.pi,axis=0,nan_policy='omit')+scipy.stats.circstd(list_phasor_data[labels==0,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit'),scipy.stats.circmean(list_phasor_data[labels==0,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit')-scipy.stats.circstd(list_phasor_data[labels==0,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit'),color='k',alpha=0.5)
    axs.fill_betweenx(bins_centers, scipy.stats.circmean(list_phasor_data[labels==1,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit')+scipy.stats.circstd(list_phasor_data[labels==1,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit'),scipy.stats.circmean(list_phasor_data[labels==1,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit')-scipy.stats.circstd(list_phasor_data[labels==1,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit'),color='r',alpha=0.5)
    axs.fill_betweenx(bins_centers, scipy.stats.circmean(list_phasor_data[labels==1,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit')+scipy.stats.circstd(list_phasor_data[labels==1,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit'),scipy.stats.circmean(list_phasor_data[labels==1,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit')-scipy.stats.circstd(list_phasor_data[labels==1,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit'),color='r',alpha=0.5)
    axs.fill_betweenx(bins_centers, scipy.stats.circmean(list_phasor_data[labels==1,:,2],low=0,high=2*np.pi,axis=0,nan_policy='omit')+scipy.stats.circstd(list_phasor_data[labels==1,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit'),scipy.stats.circmean(list_phasor_data[labels==1,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit')-scipy.stats.circstd(list_phasor_data[labels==1,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit'),color='r',alpha=0.5)
    axs.set_ylim([0.1, 0.3]), axs.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2]), axs.set_xticklabels(['0','0.25','0.5','0.75'])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_average_group3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_average_group3.svg'),bbox_inches='tight')    

    ######################
    ## Individual level ##
    ######################

    list_ind_metrics = np.nanmean(list_phasor_data/(2*np.pi),axis=1)
    n_0 = len(np.where(labels==0)[0])
    n_1 = len(np.where(labels==1)[0])
    fig, axs = plt.subplots(1,3,figsize=(9,3))
    for leg in range(3):
        axs[leg].spines[['top','right']].set_visible(False)
        axs[leg].scatter(np.random.uniform(-0.2,0.2, n_0), np.nanmean(list_phasor_data[labels==0,:,leg]/(2*np.pi),axis=1), color='k', s=5)
        axs[leg].scatter(np.random.uniform(0.8,1.2, n_1), np.nanmean(list_phasor_data[labels==1,:,leg]/(2*np.pi),axis=1), color='r', s=5)
        axs[leg].scatter(0, np.nanmedian(np.nanmean(list_phasor_data[labels==0,:,leg]/(2*np.pi),1),0), color='k')
        axs[leg].scatter(1, np.nanmedian(np.nanmean(list_phasor_data[labels==1,:,leg]/(2*np.pi),1),0), color='r')
        axs[leg].plot([0,0],[np.nanpercentile(np.nanmean(list_phasor_data[labels==0,:,leg]/(2*np.pi),1), axis=0, q=25), np.nanpercentile(np.nanmean(list_phasor_data[labels==0,:,leg]/(2*np.pi),1), axis=0, q=75)], color='k', lw=2)
        axs[leg].plot([1,1],[np.nanpercentile(np.nanmean(list_phasor_data[labels==1,:,leg]/(2*np.pi),1), axis=0, q=25), np.nanpercentile(np.nanmean(list_phasor_data[labels==1,:,leg]/(2*np.pi),1), axis=0, q=75)], color='r', lw=2)
        axs[leg].set_xlim([-0.5,1.5])
    axs[0].set_ylabel('Relative gait fraction')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_individual_group1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_individual_group1.svg'),bbox_inches='tight')


    if bool_plot:
        plt.show()
    else:
        plt.close('all')

    return list_ind_metrics


def plot_interlimb_coordination(list_phasor_data, bins_centers, bool_plot=False, bool_save=False, figname=None):
    """
    Plots all the figures for the interlimb coordination metrics
    """

    #################
    ## Group level ##
    #################

    fig, axs = plt.subplots(1,1,figsize=(5,5), subplot_kw={"projection":"polar"})
    axs.plot(scipy.stats.circmean(list_phasor_data[0][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit'),bins_centers,color='k',lw=2,ls='--')
    axs.plot(scipy.stats.circmean(list_phasor_data[0][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit'),bins_centers,color='k',lw=2,ls=':')
    axs.plot(scipy.stats.circmean(list_phasor_data[0][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit'),bins_centers,color='k',lw=2,ls='-')
    axs.plot(scipy.stats.circmean(list_phasor_data[2][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit'),bins_centers,color='r',lw=2,ls='--')
    axs.plot(scipy.stats.circmean(list_phasor_data[2][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit'),bins_centers,color='r',lw=2,ls=':')
    axs.plot(scipy.stats.circmean(list_phasor_data[2][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit'),bins_centers,color='r',lw=2,ls='-')
    axs.fill_betweenx(bins_centers,scipy.stats.circmean(list_phasor_data[0][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit')+scipy.stats.circstd(list_phasor_data[0][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit'),scipy.stats.circmean(list_phasor_data[0][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit')-scipy.stats.circstd(list_phasor_data[0][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit'),color='k',alpha=0.5)
    axs.fill_betweenx(bins_centers,scipy.stats.circmean(list_phasor_data[0][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit')+scipy.stats.circstd(list_phasor_data[0][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit'),scipy.stats.circmean(list_phasor_data[0][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit')-scipy.stats.circstd(list_phasor_data[0][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit'),color='k',alpha=0.5)
    axs.fill_betweenx(bins_centers,scipy.stats.circmean(list_phasor_data[0][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit')+scipy.stats.circstd(list_phasor_data[0][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit'),scipy.stats.circmean(list_phasor_data[0][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit')-scipy.stats.circstd(list_phasor_data[0][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit'),color='k',alpha=0.5)
    axs.fill_betweenx(bins_centers,scipy.stats.circmean(list_phasor_data[2][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit')+scipy.stats.circstd(list_phasor_data[2][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit'),scipy.stats.circmean(list_phasor_data[2][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit')-scipy.stats.circstd(list_phasor_data[2][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit'),color='r',alpha=0.5)
    axs.fill_betweenx(bins_centers,scipy.stats.circmean(list_phasor_data[2][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit')+scipy.stats.circstd(list_phasor_data[2][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit'),scipy.stats.circmean(list_phasor_data[2][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit')-scipy.stats.circstd(list_phasor_data[2][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit'),color='r',alpha=0.5)
    axs.fill_betweenx(bins_centers,scipy.stats.circmean(list_phasor_data[2][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit')+scipy.stats.circstd(list_phasor_data[2][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit'),scipy.stats.circmean(list_phasor_data[2][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit')-scipy.stats.circstd(list_phasor_data[2][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit'),color='r',alpha=0.5)
    axs.set_ylim([0.15,0.55]), axs.set_xticks([0,np.pi/2,np.pi,3*np.pi/2]), axs.set_xticklabels(['0','0.25','0.5','0.75'])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_average_group1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_average_group1.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(5,5), subplot_kw={"projection":"polar"})
    axs.plot(scipy.stats.circmean(list_phasor_data[3][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit'),bins_centers,color='k',lw=2,ls='--')
    axs.plot(scipy.stats.circmean(list_phasor_data[3][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit'),bins_centers,color='k',lw=2,ls=':')
    axs.plot(scipy.stats.circmean(list_phasor_data[3][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit'),bins_centers,color='k',lw=2,ls='-')
    axs.plot(scipy.stats.circmean(list_phasor_data[5][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit'),bins_centers,color='r',lw=2,ls='--')
    axs.plot(scipy.stats.circmean(list_phasor_data[5][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit'),bins_centers,color='r',lw=2,ls=':')
    axs.plot(scipy.stats.circmean(list_phasor_data[5][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit'),bins_centers,color='r',lw=2,ls='-')
    axs.fill_betweenx(bins_centers,scipy.stats.circmean(list_phasor_data[3][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit')+scipy.stats.circstd(list_phasor_data[3][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit'),scipy.stats.circmean(list_phasor_data[3][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit')-scipy.stats.circstd(list_phasor_data[3][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit'),color='k',alpha=0.5)
    axs.fill_betweenx(bins_centers,scipy.stats.circmean(list_phasor_data[3][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit')+scipy.stats.circstd(list_phasor_data[3][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit'),scipy.stats.circmean(list_phasor_data[3][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit')-scipy.stats.circstd(list_phasor_data[3][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit'),color='k',alpha=0.5)
    axs.fill_betweenx(bins_centers,scipy.stats.circmean(list_phasor_data[3][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit')+scipy.stats.circstd(list_phasor_data[3][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit'),scipy.stats.circmean(list_phasor_data[3][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit')-scipy.stats.circstd(list_phasor_data[3][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit'),color='k',alpha=0.5)
    axs.fill_betweenx(bins_centers,scipy.stats.circmean(list_phasor_data[5][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit')+scipy.stats.circstd(list_phasor_data[5][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit'),scipy.stats.circmean(list_phasor_data[5][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit')-scipy.stats.circstd(list_phasor_data[5][:,:,0],low=0,high=2*np.pi,axis=0,nan_policy='omit'),color='r',alpha=0.5)
    axs.fill_betweenx(bins_centers,scipy.stats.circmean(list_phasor_data[5][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit')+scipy.stats.circstd(list_phasor_data[5][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit'),scipy.stats.circmean(list_phasor_data[5][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit')-scipy.stats.circstd(list_phasor_data[5][:,:,1],low=0,high=2*np.pi,axis=0,nan_policy='omit'),color='r',alpha=0.5)
    axs.fill_betweenx(bins_centers,scipy.stats.circmean(list_phasor_data[5][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit')+scipy.stats.circstd(list_phasor_data[5][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit'),scipy.stats.circmean(list_phasor_data[5][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit')-scipy.stats.circstd(list_phasor_data[5][:,:,2],low=-np.pi,high=np.pi,axis=0,nan_policy='omit'),color='r',alpha=0.5)
    axs.set_ylim([0.15,0.55]), axs.set_xticks([0,np.pi/2,np.pi,3*np.pi/2]), axs.set_xticklabels(['0','0.25','0.5','0.75'])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_average_group2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_average_group2.svg'),bbox_inches='tight')

    ######################
    ## Individual level ##
    ######################

    list_ind_metrics = []
    for group in range(6):
        list_ind_metrics.append(np.nanmean(list_phasor_data[group]/(2*np.pi),axis=1))

    fig, axs = plt.subplots(1,3,figsize=(9,3))
    for leg in range(3):
        axs[leg].spines[['top','right']].set_visible(False)
        axs[leg].scatter(np.random.uniform(-0.2,0.2, list_phasor_data[0].shape[0]), np.nanmean(list_phasor_data[0][:,:,leg]/(2*np.pi),axis=1), color='k', s=5)
        axs[leg].scatter(np.random.uniform(0.8,1.2, list_phasor_data[2].shape[0]), np.nanmean(list_phasor_data[2][:,:,leg]/(2*np.pi),axis=1), color='r', s=5)
        axs[leg].scatter(0, np.nanmedian(np.nanmean(list_phasor_data[0][:,:,leg]/(2*np.pi),1),0), color='k')
        axs[leg].scatter(1, np.nanmedian(np.nanmean(list_phasor_data[2][:,:,leg]/(2*np.pi),1),0), color='r')
        axs[leg].plot([0,0],[np.nanpercentile(np.nanmean(list_phasor_data[0][:,:,leg]/(2*np.pi),1), axis=0, q=25), np.nanpercentile(np.nanmean(list_phasor_data[0][:,:,leg]/(2*np.pi),1), axis=0, q=75)], color='k', lw=2)
        axs[leg].plot([1,1],[np.nanpercentile(np.nanmean(list_phasor_data[2][:,:,leg]/(2*np.pi),1), axis=0, q=25), np.nanpercentile(np.nanmean(list_phasor_data[2][:,:,leg]/(2*np.pi),1), axis=0, q=75)], color='r', lw=2)
        axs[leg].set_xlim([-0.5,1.5])
    axs[0].set_ylabel('Relative gait fraction')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_individual_group1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_individual_group1.svg'),bbox_inches='tight')


    fig, axs = plt.subplots(1,3,figsize=(9,3))
    for leg in range(3):
        axs[leg].spines[['top','right']].set_visible(False)
        axs[leg].scatter(np.random.uniform(-0.2,0.2, list_phasor_data[3].shape[0]), np.nanmean(list_phasor_data[3][:,:,leg]/(2*np.pi),axis=1), color='k', s=5)
        axs[leg].scatter(np.random.uniform(0.8,1.2, list_phasor_data[5].shape[0]), np.nanmean(list_phasor_data[5][:,:,leg]/(2*np.pi),axis=1), color='r', s=5)
        axs[leg].scatter(0, np.nanmedian(np.nanmean(list_phasor_data[3][:,:,leg]/(2*np.pi),1),0), color='k')
        axs[leg].scatter(1, np.nanmedian(np.nanmean(list_phasor_data[5][:,:,leg]/(2*np.pi),1),0), color='r')
        axs[leg].plot([0,0],[np.nanpercentile(np.nanmean(list_phasor_data[3][:,:,leg]/(2*np.pi),1), axis=0, q=25), np.nanpercentile(np.nanmean(list_phasor_data[3][:,:,leg]/(2*np.pi),1), axis=0, q=75)], color='k', lw=2)
        axs[leg].plot([1,1],[np.nanpercentile(np.nanmean(list_phasor_data[5][:,:,leg]/(2*np.pi),1), axis=0, q=25), np.nanpercentile(np.nanmean(list_phasor_data[5][:,:,leg]/(2*np.pi),1), axis=0, q=75)], color='r', lw=2)
        axs[leg].set_xlim([-0.5,1.5]) 
    axs[0].set_ylabel('Relative gait fraction')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_individual_group2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_individual_group2.svg'),bbox_inches='tight')
    
    if bool_plot:
        plt.show()
    else:
        plt.close('all')

    return list_ind_metrics


def plot_gross_metrics_distribution_shank3(gross_metrics, labels, bool_save=False, bool_plot=False, figname=None):
    """
    Represent the speed / distance / timing distributions on a bout basis
    """

    # Speed 
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.hist(np.abs(gross_metrics[labels==0,3]), bins=20, color='k', alpha=0.5, density=True)
    axs.hist(np.abs(gross_metrics[labels==1,3]), bins=20, color='r', alpha=0.5, density=True)
    axs.axvline(np.nanmean(np.abs(gross_metrics[labels==0,3])), color='k', lw=2, ls='--')
    axs.axvline(np.nanmean(np.abs(gross_metrics[labels==1,3])), color='r', lw=2, ls='--')
    axs.set_xlabel('speed [*]'), axs.set_ylabel('distribution')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_speed_group3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_speed_group3.svg'),bbox_inches='tight')

    # Distance 
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.hist(np.abs(gross_metrics[labels==0,1]), bins=20, color='k', alpha=0.5, density=True)
    axs.hist(np.abs(gross_metrics[labels==1,1]), bins=20, color='r', alpha=0.5, density=True)
    axs.axvline(np.nanmean(np.abs(gross_metrics[labels==0,1])), color='k', lw=2, ls='--')
    axs.axvline(np.nanmean(np.abs(gross_metrics[labels==1,1])), color='r', lw=2, ls='--')
    axs.set_xlabel('distance [*]'), axs.set_ylabel('distribution')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_distance_group3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_distance_group3.svg'),bbox_inches='tight')

    # Duration 
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.hist(np.abs(gross_metrics[labels==0,2]), bins=20, color='k', alpha=0.5, density=True)
    axs.hist(np.abs(gross_metrics[labels==1,2]), bins=20, color='r', alpha=0.5, density=True)
    axs.axvline(np.nanmean(np.abs(gross_metrics[labels==0,2])), color='k', lw=2, ls='--')
    axs.axvline(np.nanmean(np.abs(gross_metrics[labels==1,2])), color='r', lw=2, ls='--')
    axs.set_xlabel('duration [*]'), axs.set_ylabel('distribution')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_duration_group3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_duration_group3.svg'),bbox_inches='tight')

    if bool_plot:
        plt.show()
    else:
        plt.close('all')

def plot_gross_metrics_distribution_rett(list_gross_metrics, bool_save=False, bool_plot=False, figname=None):
    """
    Represents the speed / distance / timing distributions on a bout basis
    """
    # Speed
    cmap = cm.plasma(np.linspace(0,1,len(list_gross_metrics)))
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_gross_metrics)):
        axs.hist(np.abs(list_gross_metrics[group][:,3]), bins=20, color=cmap[group], alpha=0.5, density=True)
        axs.axvline(np.nanmean(np.abs(list_gross_metrics[group][:,3])), color=cmap[group], lw=2, ls='--')
    axs.set_xlabel('speed [*]'), axs.set_ylabel('distribution')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_speed_all_groups.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_speed_all_groups.svg'),bbox_inches='tight')

    # Distance 
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_gross_metrics)):
        axs.hist(np.abs(list_gross_metrics[group][:,1]), bins=20, color=cmap[group], alpha=0.5, density=True)
        axs.axvline(np.nanmean(np.abs(list_gross_metrics[group][:,1])), color=cmap[group], lw=2, ls='--')
    axs.set_xlabel('distance [*]'), axs.set_ylabel('distribution')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_distance_all_groups.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_distance_all_groups.svg'),bbox_inches='tight')

    # Timing
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_gross_metrics)):
        axs.hist(np.abs(list_gross_metrics[group][:,2]), bins=20, color=cmap[group], alpha=0.5, density=True)
        axs.axvline(np.nanmean(np.abs(list_gross_metrics[group][:,2])), color=cmap[group], lw=2, ls='--')
    axs.set_xlabel('duration [*]'), axs.set_ylabel('distribution')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_duration_all_groups.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_duration_all_groups.svg'),bbox_inches='tight')

    if bool_plot:
        plt.show()
    else:
        plt.close()
        
def plot_gross_metrics_distribution(list_gross_metrics, bool_save=False, bool_plot=False, figname=None):
    
    """
    Represents the speed / distance / timing distributions on a bout basis
    """

    # Speed
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.hist(np.abs(list_gross_metrics[0][:,3]), bins=20, color='k', alpha=0.5, density=True)
    axs.hist(np.abs(list_gross_metrics[2][:,3]), bins=20, color='r', alpha=0.5, density=True)
    axs.axvline(np.nanmean(np.abs(list_gross_metrics[0][:,3])), color='k', lw=2, ls='--')
    axs.axvline(np.nanmean(np.abs(list_gross_metrics[2][:,3])), color='r', lw=2, ls='--')
    axs.set_xlabel('speed [*]'), axs.set_ylabel('distribution')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_speed_group1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_speed_group1.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.hist(np.abs(list_gross_metrics[3][:,3]), bins=20, color='k', alpha=0.5, density=True)
    axs.hist(np.abs(list_gross_metrics[5][:,3]), bins=20, color='r', alpha=0.5, density=True)
    axs.axvline(np.nanmean(np.abs(list_gross_metrics[3][:,3])), color='k', lw=2, ls='--')
    axs.axvline(np.nanmean(np.abs(list_gross_metrics[5][:,3])), color='r', lw=2, ls='--')
    axs.set_xlabel('speed [*]'), axs.set_ylabel('distribution')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_speed_group2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_speed_group2.svg'),bbox_inches='tight')

    # Distance
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.hist(np.abs(list_gross_metrics[0][:,1]), bins=20, color='k', alpha=0.5, density=True)
    axs.hist(np.abs(list_gross_metrics[2][:,1]), bins=20, color='r', alpha=0.5, density=True)
    axs.axvline(np.nanmean(np.abs(list_gross_metrics[0][:,1])), color='k', lw=2, ls='--')
    axs.axvline(np.nanmean(np.abs(list_gross_metrics[2][:,1])), color='r', lw=2, ls='--')
    axs.set_xlabel('distance [*]'), axs.set_ylabel('distribution')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_distance_group1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_distance_group1.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.hist(np.abs(list_gross_metrics[3][:,1]), bins=20, color='k', alpha=0.5, density=True)
    axs.hist(np.abs(list_gross_metrics[5][:,1]), bins=20, color='r', alpha=0.5, density=True)
    axs.axvline(np.nanmean(np.abs(list_gross_metrics[3][:,1])), color='k', lw=2, ls='--')
    axs.axvline(np.nanmean(np.abs(list_gross_metrics[5][:,1])), color='r', lw=2, ls='--')
    axs.set_xlabel('distance [*]'), axs.set_ylabel('distribution')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_distance_group2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_distance_group2.svg'),bbox_inches='tight')


    # Timing
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.hist(np.abs(list_gross_metrics[0][:,2]), bins=20, color='k', alpha=0.5, density=True)
    axs.hist(np.abs(list_gross_metrics[2][:,2]), bins=20, color='r', alpha=0.5, density=True)
    axs.axvline(np.nanmean(np.abs(list_gross_metrics[0][:,2])), color='k', lw=2, ls='--')
    axs.axvline(np.nanmean(np.abs(list_gross_metrics[2][:,2])), color='r', lw=2, ls='--')
    axs.set_xlabel('duration [*]'), axs.set_ylabel('distribution')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_duration_group1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_duration_group1.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.hist(np.abs(list_gross_metrics[3][:,2]), bins=20, color='k', alpha=0.5, density=True)
    axs.hist(np.abs(list_gross_metrics[5][:,2]), bins=20, color='r', alpha=0.5, density=True)
    axs.axvline(np.nanmean(np.abs(list_gross_metrics[3][:,2])), color='k', lw=2, ls='--')
    axs.axvline(np.nanmean(np.abs(list_gross_metrics[5][:,2])), color='r', lw=2, ls='--')
    axs.set_xlabel('duration [*]'), axs.set_ylabel('distribution')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_duration_group2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_duration_group2.svg'),bbox_inches='tight')


    if bool_plot:
        plt.show()
    else:
        plt.close('all')

def plot_gross_metrics_individual_shank3(list_gross_metrics, labels, bool_save=False, bool_plot=False, figname=None):
    """
    Plots the individual level gross metrics
    """
    n_0 = len(np.where(labels==0)[0])
    n_1 = len(np.where(labels==1)[0])
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,n_0), list_gross_metrics[labels==0,2],color='k', s=5, alpha=0.5)
    axs.scatter(np.random.uniform(0.8,1.2,n_1), list_gross_metrics[labels==1,2],color='r', s=5, alpha=0.5)
    axs.scatter(0, np.nanmedian(list_gross_metrics[labels==0,2]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_gross_metrics[labels==1,2]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_gross_metrics[labels==0,2],q=25), np.nanpercentile(list_gross_metrics[labels==0,2],q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_gross_metrics[labels==1,2],q=25), np.nanpercentile(list_gross_metrics[labels==1,2],q=75)], color='r', lw=2)
    axs.set_ylabel('speed [*]')
    axs.set_xlim([-0.5, 1.5])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_velocity_group3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_velocity_group3.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,n_0), list_gross_metrics[labels==0,0],color='k', s=5, alpha=0.5)
    axs.scatter(np.random.uniform(0.8,1.2,n_1), list_gross_metrics[labels==1,0],color='r', s=5, alpha=0.5)
    axs.scatter(0, np.nanmedian(list_gross_metrics[labels==0,0]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_gross_metrics[labels==1,0]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_gross_metrics[labels==0,0],q=25), np.nanpercentile(list_gross_metrics[labels==0,0],q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_gross_metrics[labels==1,0],q=25), np.nanpercentile(list_gross_metrics[labels==1,0],q=75)], color='r', lw=2)
    axs.set_ylabel('distance [*]')
    axs.set_xlim([-0.5, 1.5])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_distance_group3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_distance_group3.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,n_0), list_gross_metrics[labels==0,1],color='k', s=5, alpha=0.5)
    axs.scatter(np.random.uniform(0.8,1.2,n_1), list_gross_metrics[labels==1,1],color='r', s=5, alpha=0.5)
    axs.scatter(0, np.nanmedian(list_gross_metrics[labels==0,1]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_gross_metrics[labels==1,1]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_gross_metrics[labels==0,1],q=25), np.nanpercentile(list_gross_metrics[labels==0,1],q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_gross_metrics[labels==1,1],q=25), np.nanpercentile(list_gross_metrics[labels==1,1],q=75)], color='r', lw=2)
    axs.set_ylabel('duration [*]')
    axs.set_xlim([-0.5, 1.5])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_duration_group3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_duration_group3.svg'),bbox_inches='tight')

    if bool_plot:
        plt.show()
    else:
        plt.close('all')


def plot_gross_metrics_individual_rett(list_gross_metrics, bool_save=False, bool_plot=False, figname=None):
    """
    Plots the individual level gross metrics
    """
    cmap = cm.plasma(np.linspace(0,1,len(list_gross_metrics)))

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_gross_metrics)):
        axs.scatter(np.random.uniform(group-0.2,group+0.2,list_gross_metrics[group].shape[0]), list_gross_metrics[group][:,2], color=cmap[group], s=5, alpha=0.5)
        axs.scatter(group,np.nanmedian(list_gross_metrics[group][:,2]), color=cmap[group], s=20)
        axs.plot([group,group],[np.nanpercentile(list_gross_metrics[group][:,2], q=25), np.nanpercentile(list_gross_metrics[group][:,2], q=75)], color=cmap[group], lw=2)
    axs.set_ylabel('speed [*]')
    axs.set_xlim([-0.5,len(list_gross_metrics)-0.5])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_velocity_individuals.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_velocity_individuals.svg'),bbox_inches='tight')
    velocity_kruskall, distance_kruskall, duration_kruskall = [], [], []
    for ii in range(len(list_gross_metrics)):
        velocity_kruskall.append(list_gross_metrics[ii][:,2])
        distance_kruskall.append(list_gross_metrics[ii][:,0])
        duration_kruskall.append(list_gross_metrics[ii][:,1])
    print('=========================')
    print('Statistics gross velocity')
    print('=========================')
    print(scipy.stats.kruskal(velocity_kruskall[0], velocity_kruskall[1], velocity_kruskall[2]))
    print(spph.posthoc_dunn(velocity_kruskall, p_adjust="fdr_bh"))

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_gross_metrics)):
        axs.scatter(np.random.uniform(group-0.2,group+0.2,list_gross_metrics[group].shape[0]), list_gross_metrics[group][:,0], color=cmap[group], s=5, alpha=0.5)
        axs.scatter(group,np.nanmedian(list_gross_metrics[group][:,0]), color=cmap[group], s=20)
        axs.plot([group,group],[np.nanpercentile(list_gross_metrics[group][:,0], q=25), np.nanpercentile(list_gross_metrics[group][:,0], q=75)], color=cmap[group], lw=2)
    axs.set_ylabel('distance [*]')
    axs.set_xlim([-0.5,len(list_gross_metrics)-0.5])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_distance_individuals.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_distance_individuals.svg'),bbox_inches='tight')
    print('=========================')
    print('Statistics gross distance')
    print('=========================')
    print(scipy.stats.kruskal(distance_kruskall[0], distance_kruskall[1], distance_kruskall[2]))
    print(spph.posthoc_dunn(distance_kruskall, p_adjust="fdr_bh"))
  
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_gross_metrics)):
        axs.scatter(np.random.uniform(group-0.2,group+0.2,list_gross_metrics[group].shape[0]), list_gross_metrics[group][:,1], color=cmap[group], s=5, alpha=0.5)
        axs.scatter(group,np.nanmedian(list_gross_metrics[group][:,1]), color=cmap[group], s=20)
        axs.plot([group,group],[np.nanpercentile(list_gross_metrics[group][:,1], q=25), np.nanpercentile(list_gross_metrics[group][:,1], q=75)], color=cmap[group],lw=2)
    axs.set_ylabel('duration [*]')
    axs.set_xlim([-0.5,len(list_gross_metrics)-0.5])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_duration_individuals.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_duration_individuals.svg'),bbox_inches='tight')
    print('=========================')
    print('Statistics gross duration')
    print('=========================')
    print(scipy.stats.kruskal(duration_kruskall[0], duration_kruskall[1], duration_kruskall[2]))
    print(spph.posthoc_dunn(duration_kruskall, p_adjust="fdr_bh"))
   
    if bool_plot:
        plt.show()
    else:
        plt.close('all')


def plot_gross_metrics_individual(list_gross_metrics, bool_save=False, bool_plot=False, figname=None):
    """
    Plots the individual level gross metrics
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_gross_metrics[0].shape[0]), list_gross_metrics[0][:,2], color='k', s=5, alpha=0.5)
    axs.scatter(np.random.uniform(0.8,1.2,list_gross_metrics[2].shape[0]), list_gross_metrics[2][:,2], color='r', s=5, alpha=0.5)
    axs.scatter(0,np.nanmedian(list_gross_metrics[0][:,2]), color='k', s=20)
    axs.scatter(1,np.nanmedian(list_gross_metrics[2][:,2]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_gross_metrics[0][:,2], q=25), np.nanpercentile(list_gross_metrics[0][:,2], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_gross_metrics[2][:,2], q=25), np.nanpercentile(list_gross_metrics[2][:,2], q=75)], color='r', lw=2)
    axs.set_ylabel('speed [*]')
    axs.set_xlim([-0.5,1.5])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_velocity_group1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_velocity_group1.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_gross_metrics[3].shape[0]), list_gross_metrics[3][:,2], color='k', s=5, alpha=0.5)
    axs.scatter(np.random.uniform(0.8,1.2,list_gross_metrics[5].shape[0]), list_gross_metrics[5][:,2], color='r', s=5, alpha=0.5)
    axs.scatter(0,np.nanmedian(list_gross_metrics[3][:,2]), color='k', s=20)
    axs.scatter(1,np.nanmedian(list_gross_metrics[5][:,2]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_gross_metrics[3][:,2], q=25), np.nanpercentile(list_gross_metrics[3][:,2], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_gross_metrics[5][:,2], q=25), np.nanpercentile(list_gross_metrics[5][:,2], q=75)], color='r', lw=2)
    axs.set_ylabel('speed [*]')
    axs.set_xlim([-0.5,1.5])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_velocity_group2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_velocity_group2.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_gross_metrics[0].shape[0]), list_gross_metrics[0][:,0], color='k', s=5, alpha=0.5)
    axs.scatter(np.random.uniform(0.8,1.2,list_gross_metrics[2].shape[0]), list_gross_metrics[2][:,0], color='r', s=5, alpha=0.5)
    axs.scatter(0,np.nanmedian(list_gross_metrics[0][:,0]), color='k', s=20)
    axs.scatter(1,np.nanmedian(list_gross_metrics[2][:,0]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_gross_metrics[0][:,0], q=25), np.nanpercentile(list_gross_metrics[0][:,0], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_gross_metrics[2][:,0], q=25), np.nanpercentile(list_gross_metrics[2][:,0], q=75)], color='r', lw=2)
    axs.set_ylabel('distance [*]')
    axs.set_xlim([-0.5,1.5])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_distance_group1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_distance_group1.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_gross_metrics[3].shape[0]), list_gross_metrics[3][:,0], color='k', s=5, alpha=0.5)
    axs.scatter(np.random.uniform(0.8,1.2,list_gross_metrics[5].shape[0]), list_gross_metrics[5][:,0], color='r', s=5, alpha=0.5)
    axs.scatter(0,np.nanmedian(list_gross_metrics[3][:,0]), color='k', s=20)
    axs.scatter(1,np.nanmedian(list_gross_metrics[5][:,0]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_gross_metrics[3][:,0], q=25), np.nanpercentile(list_gross_metrics[3][:,0], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_gross_metrics[5][:,0], q=25), np.nanpercentile(list_gross_metrics[5][:,0], q=75)], color='r', lw=2)
    axs.set_ylabel('distance[*]')
    axs.set_xlim([-0.5,1.5])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_distance_group2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_distance_group2.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_gross_metrics[0].shape[0]), list_gross_metrics[0][:,1], color='k', s=5, alpha=0.5)
    axs.scatter(np.random.uniform(0.8,1.2,list_gross_metrics[2].shape[0]), list_gross_metrics[2][:,1], color='r', s=5, alpha=0.5)
    axs.scatter(0,np.nanmedian(list_gross_metrics[0][:,1]), color='k', s=20)
    axs.scatter(1,np.nanmedian(list_gross_metrics[2][:,1]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_gross_metrics[0][:,1], q=25), np.nanpercentile(list_gross_metrics[0][:,1], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_gross_metrics[2][:,1], q=25), np.nanpercentile(list_gross_metrics[2][:,1], q=75)], color='r', lw=2)
    axs.set_ylabel('duration [*]')
    axs.set_xlim([-0.5,1.5])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_duration_group1.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_duration_group1.svg'),bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_gross_metrics[3].shape[0]), list_gross_metrics[3][:,1], color='k', s=5, alpha=0.5)
    axs.scatter(np.random.uniform(0.8,1.2,list_gross_metrics[5].shape[0]), list_gross_metrics[5][:,1], color='r', s=5, alpha=0.5)
    axs.scatter(0,np.nanmedian(list_gross_metrics[3][:,1]), color='k', s=20)
    axs.scatter(1,np.nanmedian(list_gross_metrics[5][:,1]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_gross_metrics[3][:,1], q=25), np.nanpercentile(list_gross_metrics[3][:,1], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_gross_metrics[5][:,1], q=25), np.nanpercentile(list_gross_metrics[5][:,1], q=75)], color='r', lw=2)
    axs.set_ylabel('duration[*]')
    axs.set_xlim([-0.5,1.5])
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_duration_group2.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_duration_group2.svg'),bbox_inches='tight')

    if bool_plot:
        plt.show()
    else:
        plt.close('all')


def plot_bl_coordination_individual_sh3(list_bl_coord, bl_labels, bool_save=False, bool_plot=False, figname=None):
    """
    Represents the individual body limb coordination metrics (peak to peak amplitude and timing)
    """

    n_0 = len(np.where(bl_labels==0)[0])
    n_1 = len(np.where(bl_labels==1)[0])

    # Head measures
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,n_0), list_bl_coord[bl_labels==0,0],color='k',s=5)
    axs.scatter(np.random.uniform(0.8,1.2,n_1), list_bl_coord[bl_labels==1,0],color='r',s=5)
    axs.scatter(0, np.nanmedian(list_bl_coord[bl_labels==0,0]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_bl_coord[bl_labels==1,0]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_bl_coord[bl_labels==0,0],q=25), np.nanpercentile(list_bl_coord[bl_labels==0,0],q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_bl_coord[bl_labels==1,0],q=25), np.nanpercentile(list_bl_coord[bl_labels==1,0],q=75)], color='r', lw=2)
    axs.set_ylabel('Head p2p amplitude')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_group3_head_p2p.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_group3_head_p2p.svg'), bbox_inches='tight')


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,n_0), list_bl_coord[bl_labels==0,1],color='k',s=5)
    axs.scatter(np.random.uniform(0.8,1.2,n_1), list_bl_coord[bl_labels==1,1],color='r',s=5)
    axs.scatter(0, np.nanmedian(list_bl_coord[bl_labels==0,1]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_bl_coord[bl_labels==1,1]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_bl_coord[bl_labels==0,1],q=25), np.nanpercentile(list_bl_coord[bl_labels==0,1],q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_bl_coord[bl_labels==1,1],q=25), np.nanpercentile(list_bl_coord[bl_labels==1,1],q=75)], color='r', lw=2)
    axs.set_ylabel('Head p2p amplitude')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_group3_head_phase.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_group3_head_phase.svg'), bbox_inches='tight')

    # Tail measures 
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,n_0), list_bl_coord[bl_labels==0,2],color='k',s=5)
    axs.scatter(np.random.uniform(0.8,1.2,n_1), list_bl_coord[bl_labels==1,2],color='r',s=5)
    axs.scatter(0, np.nanmedian(list_bl_coord[bl_labels==0,2]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_bl_coord[bl_labels==1,2]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_bl_coord[bl_labels==0,2],q=25), np.nanpercentile(list_bl_coord[bl_labels==0,2],q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_bl_coord[bl_labels==1,2],q=25), np.nanpercentile(list_bl_coord[bl_labels==1,2],q=75)], color='r', lw=2)
    axs.set_ylabel('Head p2p amplitude')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_group3_tail_p2p.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_group3_tail_p2p.svg'), bbox_inches='tight')


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,n_0), list_bl_coord[bl_labels==0,3],color='k',s=5)
    axs.scatter(np.random.uniform(0.8,1.2,n_1), list_bl_coord[bl_labels==1,3],color='r',s=5)
    axs.scatter(0, np.nanmedian(list_bl_coord[bl_labels==0,3]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_bl_coord[bl_labels==1,3]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_bl_coord[bl_labels==0,3],q=25), np.nanpercentile(list_bl_coord[bl_labels==0,3],q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_bl_coord[bl_labels==1,3],q=25), np.nanpercentile(list_bl_coord[bl_labels==1,3],q=75)], color='r', lw=2)
    axs.set_ylabel('Head p2p amplitude')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_group3_tail_phase.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_group3_tail_phase.svg'), bbox_inches='tight')

    if bool_plot:
        plt.show()
    else:
        plt.close('all')

def plot_bl_coordination_individual_rett(list_bl_coord, list_bl_animal, bool_save=False, bool_plot=False, figname=None):
    """
    Represents the individual body limb coordination metrics (peak to peak amplitude and timing)
    """

    cmap = cm.plasma(np.linspace(0,1,len(list_bl_coord)))
    # Head measures
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_bl_coord)):
        axs.scatter(np.random.uniform(group-0.2,group+0.2,list_bl_coord[group].shape[0]), list_bl_coord[group][:,0], color=cmap[group], s=5)
        axs.scatter(group, np.nanmedian(list_bl_coord[group][:,0]), color=cmap[group], s=20)
        axs.plot([group,group], [np.nanpercentile(list_bl_coord[group][:,0], q=25), np.nanpercentile(list_bl_coord[group][:,0], q=75)], color=cmap[group], lw=2)
    axs.set_ylabel('Head p2p amplitude')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_all_groups_head_p2p.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_all_groups_head_p2p.svg'), bbox_inches='tight')

    headp2p_kruskall, headphase_kruskall = [], []
    tailp2p_kruskall, tailphase_kruskall = [], []
    for ii in range(len(list_bl_coord)):
        headp2p_kruskall.append(list_bl_coord[ii][:,0])
        headphase_kruskall.append(list_bl_coord[ii][:,1])
        tailp2p_kruskall.append(list_bl_coord[ii][:,2])
        tailphase_kruskall.append(list_bl_coord[ii][:,3])
    print('===================')
    print('Statistics head p2p')
    print('===================')
    print(scipy.stats.kruskal(headp2p_kruskall[0], headp2p_kruskall[1], headp2p_kruskall[2]))
    print(spph.posthoc_dunn(headp2p_kruskall, p_adjust="fdr_bh"))

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_bl_coord)):
        axs.scatter(np.random.uniform(group-0.2,group+0.2,list_bl_coord[group].shape[0]), list_bl_coord[group][:,1], color=cmap[group], s=5)
        axs.scatter(group, np.nanmedian(list_bl_coord[group][:,1]), color=cmap[group], s=20)
        axs.plot([group,group], [np.nanpercentile(list_bl_coord[group][:,1], q=25), np.nanpercentile(list_bl_coord[group][:,1], q=75)], color=cmap[group], lw=2)
    axs.set_ylabel('Head phase')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_all_groups_head_phase.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_all_groups_head_phase.svg'), bbox_inches='tight')

    print('=====================')
    print('Statistics head phase')
    print('=====================')
    print(scipy.stats.kruskal(headphase_kruskall[0], headphase_kruskall[1], headphase_kruskall[2]))
    print(spph.posthoc_dunn(headphase_kruskall, p_adjust="fdr_bh"))

    # Tail measures
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_bl_coord)):
        axs.scatter(np.random.uniform(group-0.2,group+0.2,list_bl_coord[group].shape[0]), list_bl_coord[group][:,2], color=cmap[group], s=5)
        axs.scatter(group, np.nanmedian(list_bl_coord[group][:,2]), color=cmap[group], s=20)
        axs.plot([group,group], [np.nanpercentile(list_bl_coord[group][:,2], q=25), np.nanpercentile(list_bl_coord[group][:,2], q=75)], color=cmap[group], lw=2)
    axs.set_ylabel('Tail p2p amplitude')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_all_groups_tail_p2p.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_all_groups_tail_p2p.svg'), bbox_inches='tight')

    print('===================')
    print('Statistics tail p2p')
    print('===================')
    print(scipy.stats.kruskal(tailp2p_kruskall[0], tailp2p_kruskall[1], tailp2p_kruskall[2]))
    print(spph.posthoc_dunn(tailp2p_kruskall, p_adjust="fdr_bh"))

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_bl_coord)):
        axs.scatter(np.random.uniform(group-0.2,group+0.2,list_bl_coord[group].shape[0]), list_bl_coord[group][:,3], color=cmap[group], s=5)
        axs.scatter(group, np.nanmedian(list_bl_coord[group][:,3]), color=cmap[group], s=20)
        axs.plot([group,group], [np.nanpercentile(list_bl_coord[group][:,3], q=25), np.nanpercentile(list_bl_coord[group][:,3], q=75)], color=cmap[group], lw=2)
    axs.set_ylabel('Tail phase')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_all_groups_tail_phase.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_all_groups_tail_phase.svg'), bbox_inches='tight')

    print('=====================')
    print('Statistics tail phase')
    print('=====================')
    print(scipy.stats.kruskal(tailphase_kruskall[0], tailphase_kruskall[1], tailphase_kruskall[2]))
    print(spph.posthoc_dunn(tailphase_kruskall, p_adjust="fdr_bh"))

    if bool_plot:
        plt.show()
    else:
        plt.close('all')


def plot_bl_coordination_individual(list_bl_coord, list_bl_animal, bool_save=False, bool_plot=False, figname=None):
    """
    Represents the individual body limb coordination metrics (peak to peak amplitude and timing)
    """

    # GROUP !

    # Head measures
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_bl_coord[0].shape[0]), list_bl_coord[0][:,0], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_bl_coord[2].shape[0]), list_bl_coord[2][:,0], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_bl_coord[0][:,0]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_bl_coord[2][:,0]), color='r', s=20)
    axs.plot([0,0], [np.nanpercentile(list_bl_coord[0][:,0], q=25), np.nanpercentile(list_bl_coord[0][:,0], q=75)], color='k', lw=2)
    axs.plot([1,1], [np.nanpercentile(list_bl_coord[2][:,0], q=25), np.nanpercentile(list_bl_coord[2][:,0], q=75)], color='r', lw=2)
    axs.set_ylabel('Head p2p amplitude')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_group1_head_p2p.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_group1_head_p2p.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_bl_coord[0].shape[0]), list_bl_coord[0][:,1], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_bl_coord[2].shape[0]), list_bl_coord[2][:,1], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_bl_coord[0][:,1]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_bl_coord[2][:,1]), color='r', s=20)
    axs.plot([0,0], [np.nanpercentile(list_bl_coord[0][:,1], q=25), np.nanpercentile(list_bl_coord[0][:,1], q=75)], color='k', lw=2)
    axs.plot([1,1], [np.nanpercentile(list_bl_coord[2][:,1], q=25), np.nanpercentile(list_bl_coord[2][:,1], q=75)], color='r', lw=2)
    axs.set_ylabel('Head phase')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_group1_head_phase.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_group1_head_phase.svg'), bbox_inches='tight')

    # Tail measures
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_bl_coord[0].shape[0]), list_bl_coord[0][:,2], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_bl_coord[2].shape[0]), list_bl_coord[2][:,2], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_bl_coord[0][:,2]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_bl_coord[2][:,2]), color='r', s=20)
    axs.plot([0,0], [np.nanpercentile(list_bl_coord[0][:,2], q=25), np.nanpercentile(list_bl_coord[0][:,2], q=75)], color='k', lw=2)
    axs.plot([1,1], [np.nanpercentile(list_bl_coord[2][:,2], q=25), np.nanpercentile(list_bl_coord[2][:,2], q=75)], color='r', lw=2)
    axs.set_ylabel('Tail p2p amplitude')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_group1_tail_p2p.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_group1_tail_p2p.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_bl_coord[0].shape[0]), list_bl_coord[0][:,3], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_bl_coord[2].shape[0]), list_bl_coord[2][:,3], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_bl_coord[0][:,3]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_bl_coord[2][:,3]), color='r', s=20)
    axs.plot([0,0], [np.nanpercentile(list_bl_coord[0][:,3], q=25), np.nanpercentile(list_bl_coord[0][:,3], q=75)], color='k', lw=2)
    axs.plot([1,1], [np.nanpercentile(list_bl_coord[2][:,3], q=25), np.nanpercentile(list_bl_coord[2][:,3], q=75)], color='r', lw=2)
    axs.set_ylabel('Tail phase')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_group1_tail_phase.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_group1_tail_phase.svg'), bbox_inches='tight')

    # GROUPE 2

    # Head measures
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_bl_coord[3].shape[0]), list_bl_coord[3][:,0], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_bl_coord[5].shape[0]), list_bl_coord[5][:,0], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_bl_coord[3][:,0]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_bl_coord[5][:,0]), color='r', s=20)
    axs.plot([0,0], [np.nanpercentile(list_bl_coord[3][:,0], q=25), np.nanpercentile(list_bl_coord[3][:,0], q=75)], color='k', lw=2)
    axs.plot([1,1], [np.nanpercentile(list_bl_coord[5][:,0], q=25), np.nanpercentile(list_bl_coord[5][:,0], q=75)], color='r', lw=2)
    axs.set_ylabel('Head p2p amplitude')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_group2_head_p2p.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_group2_head_p2p.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_bl_coord[3].shape[0]), list_bl_coord[3][:,1], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_bl_coord[5].shape[0]), list_bl_coord[5][:,1], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_bl_coord[3][:,1]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_bl_coord[5][:,1]), color='r', s=20)
    axs.plot([0,0], [np.nanpercentile(list_bl_coord[3][:,1], q=25), np.nanpercentile(list_bl_coord[3][:,1], q=75)], color='k', lw=2)
    axs.plot([1,1], [np.nanpercentile(list_bl_coord[5][:,1], q=25), np.nanpercentile(list_bl_coord[5][:,1], q=75)], color='r', lw=2)
    axs.set_ylabel('Head phase')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_group2_head_phase.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_group2_head_phase.svg'), bbox_inches='tight')

    # Tail measures
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_bl_coord[3].shape[0]), list_bl_coord[3][:,2], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_bl_coord[5].shape[0]), list_bl_coord[5][:,2], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_bl_coord[3][:,2]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_bl_coord[5][:,2]), color='r', s=20)
    axs.plot([0,0], [np.nanpercentile(list_bl_coord[3][:,2], q=25), np.nanpercentile(list_bl_coord[3][:,2], q=75)], color='k', lw=2)
    axs.plot([1,1], [np.nanpercentile(list_bl_coord[5][:,2], q=25), np.nanpercentile(list_bl_coord[5][:,2], q=75)], color='r', lw=2)
    axs.set_ylabel('Tail p2p amplitude')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_group2_tail_p2p.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_group2_tail_p2p.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_bl_coord[3].shape[0]), list_bl_coord[3][:,3], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_bl_coord[5].shape[0]), list_bl_coord[5][:,3], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_bl_coord[3][:,3]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_bl_coord[5][:,3]), color='r', s=20)
    axs.plot([0,0], [np.nanpercentile(list_bl_coord[3][:,3], q=25), np.nanpercentile(list_bl_coord[3][:,3], q=75)], color='k', lw=2)
    axs.plot([1,1], [np.nanpercentile(list_bl_coord[5][:,3], q=25), np.nanpercentile(list_bl_coord[5][:,3], q=75)], color='r', lw=2)
    axs.set_ylabel('Tail phase')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_group2_tail_phase.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_group2_tail_phase.svg'), bbox_inches='tight')


    if bool_plot:
        plt.show()
    else:
        plt.close('all')

def plot_bl_coordination_group_sh3(list_bl_coord, bl_labels, bool_save=False, bool_plot=False, figname=None):
    """
    Represents the body limb coordination information
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_bl_coord[bl_labels==0,:,5], axis=0), color='k', lw=2)
    axs.plot(np.nanmedian(list_bl_coord[bl_labels==1,:,5], axis=0), color='r', lw=2)
    axs.fill_between(np.arange(11), np.nanpercentile(list_bl_coord[bl_labels==0,:,5],axis=0,q=25), np.nanpercentile(list_bl_coord[bl_labels==0,:,5],axis=0,q=75), color='k', alpha=0.5)
    axs.fill_between(np.arange(11), np.nanpercentile(list_bl_coord[bl_labels==1,:,5],axis=0,q=25), np.nanpercentile(list_bl_coord[bl_labels==1,:,5],axis=0,q=75), color='r', alpha=0.5)
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Lateral deviation')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_group3_tail_oscillations.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_group3_tail_oscillations.svg'), bbox_inches='tight')


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_bl_coord[bl_labels==0,:,1], axis=0), color='k', lw=2)
    axs.plot(np.nanmedian(list_bl_coord[bl_labels==1,:,1], axis=0), color='r', lw=2)
    axs.fill_between(np.arange(11), np.nanpercentile(list_bl_coord[bl_labels==0,:,1],axis=0,q=25), np.nanpercentile(list_bl_coord[bl_labels==0,:,1],axis=0,q=75), color='k', alpha=0.5)
    axs.fill_between(np.arange(11), np.nanpercentile(list_bl_coord[bl_labels==1,:,1],axis=0,q=25), np.nanpercentile(list_bl_coord[bl_labels==1,:,1],axis=0,q=75), color='r', alpha=0.5)
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Lateral deviation')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_group3_head_oscillations.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_group3_head_oscillations.svg'), bbox_inches='tight')


    if bool_plot:
        plt.show()
    else:
        plt.close('all')

def plot_bl_coordination_group_rett(list_bl_coord, list_bl_animal, bool_save=False, bool_plot=False, figname=None):
    """
    Represents the body limb coordination information
    """

    cmap = cm.plasma(np.linspace(0,1,len(list_bl_coord)))
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_bl_coord)):
        axs.plot(np.nanmedian(list_bl_coord[group][:,:,5],axis=0), color=cmap[group],lw=2)
        axs.fill_between(np.arange(11), np.nanpercentile(list_bl_coord[group][:,:,5], axis=0, q=25), np.nanpercentile(list_bl_coord[group][:,:,5], axis=0, q=75), color=cmap[group], alpha=0.5)
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Lateral deviation')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_all_groups_tail_oscillations.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_all_groups_tail_oscillations.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_bl_coord)):
        axs.plot(np.nanmedian(list_bl_coord[group][:,:,1],axis=0), color=cmap[group],lw=2)
        axs.fill_between(np.arange(11), np.nanpercentile(list_bl_coord[group][:,:,1], axis=0, q=25), np.nanpercentile(list_bl_coord[group][:,:,1], axis=0, q=75), color=cmap[group], alpha=0.5)
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Lateral deviation')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_all_groups_head_oscillations.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_all_groups_head_oscillations.svg'), bbox_inches='tight')

    
    if bool_plot:
        plt.show()
    else:
        plt.close('all')


def plot_bl_coordination_group(list_bl_coord, list_bl_animal, bool_save=False, bool_plot=False, figname=None):
    """
    Represents the body limb coordination information
    """


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_bl_coord[0][:,:,5],axis=0), color='k',lw=2)
    axs.plot(np.nanmedian(list_bl_coord[2][:,:,5],axis=0), color='r',lw=2)
    axs.fill_between(np.arange(11), np.nanpercentile(list_bl_coord[0][:,:,5], axis=0, q=25), np.nanpercentile(list_bl_coord[0][:,:,5], axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(np.arange(11), np.nanpercentile(list_bl_coord[2][:,:,5], axis=0, q=25), np.nanpercentile(list_bl_coord[2][:,:,5], axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Lateral deviation')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_group1_tail_oscillations.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_group1_tail_oscillations.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_bl_coord[0][:,:,1],axis=0), color='k',lw=2)
    axs.plot(np.nanmedian(list_bl_coord[2][:,:,1],axis=0), color='r',lw=2)
    axs.fill_between(np.arange(11), np.nanpercentile(list_bl_coord[0][:,:,1], axis=0, q=25), np.nanpercentile(list_bl_coord[0][:,:,1], axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(np.arange(11), np.nanpercentile(list_bl_coord[2][:,:,1], axis=0, q=25), np.nanpercentile(list_bl_coord[2][:,:,1], axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Lateral deviation')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_group1_head_oscillations.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_group1_head_oscillations.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_bl_coord[3][:,:,5],axis=0), color='k',lw=2)
    axs.plot(np.nanmedian(list_bl_coord[5][:,:,5],axis=0), color='r',lw=2)
    axs.fill_between(np.arange(11), np.nanpercentile(list_bl_coord[3][:,:,5], axis=0, q=25), np.nanpercentile(list_bl_coord[3][:,:,5], axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(np.arange(11), np.nanpercentile(list_bl_coord[5][:,:,5], axis=0, q=25), np.nanpercentile(list_bl_coord[5][:,:,5], axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Lateral deviation')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_group2_tail_oscillations.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_group2_tail_oscillations.svg'), bbox_inches='tight')


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(np.nanmedian(list_bl_coord[3][:,:,1],axis=0), color='k',lw=2)
    axs.plot(np.nanmedian(list_bl_coord[5][:,:,1],axis=0), color='r',lw=2)
    axs.fill_between(np.arange(11), np.nanpercentile(list_bl_coord[3][:,:,1], axis=0, q=25), np.nanpercentile(list_bl_coord[3][:,:,1], axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(np.arange(11), np.nanpercentile(list_bl_coord[5][:,:,1], axis=0, q=25), np.nanpercentile(list_bl_coord[5][:,:,1], axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('Relative gait fraction'), axs.set_ylabel('Lateral deviation')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_group2_head_oscillations.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_group2_head_oscillations.svg'), bbox_inches='tight')
    
    if bool_plot:
        plt.show()
    else:
        plt.close('all')


def plot_step_length_metrics_sh3(stride_length, bins_centers, labels, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the step lengths metrics
    """
    n_0 = len(np.where(labels==0)[0])
    n_1 = len(np.where(labels==1)[0])
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers, np.nanmedian(stride_length[labels==0,:],axis=0), color='k')
    axs.plot(bins_centers, np.nanmedian(stride_length[labels==1,:],axis=0), color='r')
    axs.fill_between(bins_centers, np.nanpercentile(stride_length[labels==0,:],axis=0, q=25), np.nanpercentile(stride_length[labels==0,:],axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(stride_length[labels==1,:],axis=0, q=25), np.nanpercentile(stride_length[labels==1,:],axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('stride length [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_stride_length_group3.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_stride_length_group3.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,n_0), stride_length[labels==0,1], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,n_1), stride_length[labels==1,1], color='r', s=5)
    axs.scatter(0, np.nanmean(stride_length[labels==0,1]), color='k', s=20)
    axs.scatter(1, np.nanmean(stride_length[labels==1,1]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(stride_length[labels==0,1],axis=0,q=25), np.nanpercentile(stride_length[labels==0,1],axis=0,q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(stride_length[labels==1,1],axis=0,q=25), np.nanpercentile(stride_length[labels==1,1],axis=0,q=75)], color='r', lw=2)
    axs.set_xlim([-0.5, 1.5]), axs.set_ylabel('stride length [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_stride_length_group3.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_stride_length_group3.svg'), bbox_inches='tight')

    if bool_plot:
        plt.show()
    else:
        plt.close('all')


def plot_step_duration_metrics_sh3(stride_duration, bins_centers, labels, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the step lengths metrics
    """
    n_0 = len(np.where(labels==0)[0])
    n_1 = len(np.where(labels==1)[0])
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers, np.nanmedian(stride_duration[labels==0,:],axis=0), color='k')
    axs.plot(bins_centers, np.nanmedian(stride_duration[labels==1,:],axis=0), color='r')
    axs.fill_between(bins_centers, np.nanpercentile(stride_duration[labels==0,:],axis=0, q=25), np.nanpercentile(stride_duration[labels==0,:],axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(stride_duration[labels==1,:],axis=0, q=25), np.nanpercentile(stride_duration[labels==1,:],axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('stride duration [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_stride_duration_group3.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_stride_duration_group3.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,n_0), stride_duration[labels==0,1], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,n_1), stride_duration[labels==1,1], color='r', s=5)
    axs.scatter(0, np.nanmean(stride_duration[labels==0,1]), color='k', s=20)
    axs.scatter(1, np.nanmean(stride_duration[labels==1,1]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(stride_duration[labels==0,1],axis=0,q=25), np.nanpercentile(stride_duration[labels==0,1],axis=0,q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(stride_duration[labels==1,1],axis=0,q=25), np.nanpercentile(stride_duration[labels==1,1],axis=0,q=75)], color='r', lw=2)
    axs.set_xlim([-0.5, 1.5]), axs.set_ylabel('stride duration [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_stride_duration_group3.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_stride_duration_group3.svg'), bbox_inches='tight')

    if bool_plot:
        plt.show()
    else:
        plt.close('all')


def plot_step_length_metrics_rett(list_metrics, bins_centers, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the step lengths metrics 
    """
    cmap = cm.plasma(np.linspace(0,1,len(list_metrics)))
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_metrics)):
        axs.plot(bins_centers,np.nanmedian(list_metrics[group],axis=0), color=cmap[group])
        axs.fill_between(bins_centers, np.nanpercentile(list_metrics[group],axis=0, q=25), np.nanpercentile(list_metrics[group],axis=0, q=75), color=cmap[group], alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('stride length [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_stride_length_all_groups.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_stride_length_all_groups.svg'), bbox_inches='tight')


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_metrics)):
        axs.scatter(np.random.uniform(group-0.2,group+0.2,list_metrics[group].shape[0]), list_metrics[group][:,1], color=cmap[group], s=5)
        axs.scatter(group, np.nanmean(list_metrics[group][:,1]), color=cmap[group], s=20)
        axs.plot([group,group],[np.nanpercentile(list_metrics[group][:,1],axis=0,q=25), np.nanpercentile(list_metrics[group][:,1],axis=0,q=75)], color=cmap[group], lw=2)
    axs.set_xlim([-0.5,len(list_metrics)-0.5]), axs.set_ylabel('stride length [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_stride_length_all_groups.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_stride_length_all_groups.svg'), bbox_inches='tight')

    length_kruskall = []
    for ii in range(len(list_metrics)):
        length_kruskall.append(list_metrics[ii][:,1])
    print('======================')
    print('Statistics step length')
    print('======================')
    print(scipy.stats.kruskal(length_kruskall[0], length_kruskall[1], length_kruskall[2]))
    print(spph.posthoc_dunn(length_kruskall, p_adjust="fdr_bh"))
    

    if bool_plot:
        plt.show()
    else:
        plt.close('all')

def plot_step_width_metrics_rett(list_front, list_hind, bins_centers, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the step widths metrics 
    """
    cmap = cm.plasma(np.linspace(0,1,len(list_front)))
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_front)):
        axs.plot(bins_centers,np.nanmedian(list_front[group],axis=0), color=cmap[group])
        axs.fill_between(bins_centers, np.nanpercentile(list_front[group],axis=0, q=25), np.nanpercentile(list_front[group],axis=0, q=75), color=cmap[group], alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('front step width [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_front_width_all_groups.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_front_width_all_groups.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_front)):
        axs.plot(bins_centers,np.nanmedian(list_hind[group],axis=0), color=cmap[group])
        axs.fill_between(bins_centers, np.nanpercentile(list_hind[group],axis=0, q=25), np.nanpercentile(list_hind[group],axis=0, q=75), color=cmap[group], alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('hind step width [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_hind_width_all_groups.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_hind_width_all_groups.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_front)):
        axs.scatter(np.random.uniform(group-0.2,group+0.2,list_front[group].shape[0]), np.nanmean(list_front[group],axis=1),color=cmap[group], s=5)
        axs.scatter(group, np.nanmedian(np.nanmean(list_front[group],axis=1)), color=cmap[group], s=20)
        axs.plot([group,group],[np.nanpercentile(np.nanmean(list_front[group],axis=1), q=25), np.nanpercentile(np.nanmean(list_front[group],axis=1), q=75)], color=cmap[group], lw=2)
    axs.set_ylabel('front step width [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_front_width_all_groups.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_front_width_all_groups.svg'), bbox_inches='tight')

    front_kruskall = []
    for ii in range(len(list_front)):
        front_kruskall.append(np.nanmean(list_front[ii],axis=1))
    print('======================')
    print('Statistics front width')
    print('======================')
    print(scipy.stats.kruskal(front_kruskall[0], front_kruskall[1], front_kruskall[2]))
    print(spph.posthoc_dunn(front_kruskall, p_adjust="fdr_bh"))
    

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_front)):
        axs.scatter(np.random.uniform(group-0.2,group+0.2,list_hind[group].shape[0]), np.nanmean(list_hind[group],axis=1),color=cmap[group], s=5)
        axs.scatter(group, np.nanmedian(np.nanmean(list_hind[group],axis=1)), color=cmap[group], s=20)
        axs.plot([group,group],[np.nanpercentile(np.nanmean(list_hind[group],axis=1), q=25), np.nanpercentile(np.nanmean(list_hind[group],axis=1), q=75)], color=cmap[group], lw=2)
    axs.set_ylabel('hind step width [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_hind_width_all_groups.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_hind_width_all_groups.svg'), bbox_inches='tight')

    hind_kruskall = []
    for ii in range(len(list_hind)):
        hind_kruskall.append(np.nanmean(list_hind[ii],axis=1))
    print('=====================')
    print('Statistics hind width')
    print('=====================')
    print(scipy.stats.kruskal(hind_kruskall[0], hind_kruskall[1], hind_kruskall[2]))
    print(spph.posthoc_dunn(hind_kruskall, p_adjust="fdr_bh"))

    if bool_plot:
        plt.show()
    else:
        plt.close('all')
    

def plot_step_duration_metrics_rett(list_metrics, bins_centers, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the step lengths metrics 

    """
    cmap = cm.plasma(np.linspace(0,1,len(list_metrics)))
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_metrics)):
        axs.plot(bins_centers,np.nanmedian(list_metrics[group],axis=0), color=cmap[group])
        axs.fill_between(bins_centers, np.nanpercentile(list_metrics[group],axis=0, q=25), np.nanpercentile(list_metrics[group],axis=0, q=75), color=cmap[group], alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('stride duration [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_stride_duration_all_groups.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_stride_duration_all_groups.svg'), bbox_inches='tight')


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_metrics)):
        axs.scatter(np.random.uniform(group-0.2,group+0.2,list_metrics[group].shape[0]), list_metrics[group][:,1], color=cmap[group], s=5)
        axs.scatter(group, np.nanmean(list_metrics[group][:,1]), color=cmap[group], s=20)
        axs.plot([group,group],[np.nanpercentile(list_metrics[group][:,1],axis=0,q=25), np.nanpercentile(list_metrics[group][:,1],axis=0,q=75)], color=cmap[group], lw=2)
    axs.set_xlim([-0.5,len(list_metrics)-0.5]), axs.set_ylabel('Stride duration [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_stride_duration_all_groups.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_stride_duration_all_groups.svg'), bbox_inches='tight')

    duration_kruskall = []
    for ii in range(len(list_metrics)):
        duration_kruskall.append(list_metrics[ii][:,1])
    print('========================')
    print('Statistics step duration')
    print('========================')
    print(scipy.stats.kruskal(duration_kruskall[0], duration_kruskall[1], duration_kruskall[2]))
    print(spph.posthoc_dunn(duration_kruskall, p_adjust="fdr_bh"))

    if bool_plot:
        plt.show()
    else:
        plt.close('all')


def plot_step_length_metrics(list_metrics, bins_centers, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the step lengths metrics 
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers,np.nanmedian(list_metrics[0],axis=0), color='k')
    axs.plot(bins_centers,np.nanmedian(list_metrics[2],axis=0), color='r')
    axs.fill_between(bins_centers, np.nanpercentile(list_metrics[0],axis=0, q=25), np.nanpercentile(list_metrics[0],axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(list_metrics[2],axis=0, q=25), np.nanpercentile(list_metrics[2],axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('stride length [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_stride_length_group1.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_stride_length_group1.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers,np.nanmedian(list_metrics[3],axis=0), color='k')
    axs.plot(bins_centers,np.nanmedian(list_metrics[5],axis=0), color='r')
    axs.fill_between(bins_centers, np.nanpercentile(list_metrics[3],axis=0, q=25), np.nanpercentile(list_metrics[3],axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(list_metrics[5],axis=0, q=25), np.nanpercentile(list_metrics[5],axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('stride length [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_stride_length_group2.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_stride_length_group2.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_metrics[0].shape[0]), list_metrics[0][:,1], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_metrics[2].shape[0]), list_metrics[2][:,1], color='r', s=5)
    axs.scatter(0, np.nanmean(list_metrics[0][:,1]), color='k', s=20)
    axs.scatter(1, np.nanmean(list_metrics[2][:,1]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_metrics[0][:,1],axis=0,q=25), np.nanpercentile(list_metrics[0][:,1],axis=0,q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_metrics[2][:,1],axis=0,q=25), np.nanpercentile(list_metrics[2][:,1],axis=0,q=75)], color='r', lw=2)
    axs.set_xlim([-0.5,1.5]), axs.set_ylabel('stride length [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_stride_length_group1.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_stride_length_group1.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_metrics[3].shape[0]), list_metrics[3][:,1], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_metrics[5].shape[0]), list_metrics[5][:,1], color='r', s=5)
    axs.scatter(0, np.nanmean(list_metrics[3][:,1]), color='k', s=20)
    axs.scatter(1, np.nanmean(list_metrics[5][:,1]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_metrics[3][:,1],axis=0,q=25), np.nanpercentile(list_metrics[3][:,1],axis=0,q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_metrics[5][:,1],axis=0,q=25), np.nanpercentile(list_metrics[5][:,1],axis=0,q=75)], color='r', lw=2)
    axs.set_xlim([-0.5,1.5]), axs.set_ylabel('Stride length [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_stride_length_group2.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_stride_length_group2.svg'), bbox_inches='tight')

    if bool_plot:
        plt.show()
    else:
        plt.close('all')

def plot_step_duration_metrics(list_metrics, bins_centers, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the step lengths metrics 

    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers,np.nanmedian(list_metrics[0],axis=0), color='k')
    axs.plot(bins_centers,np.nanmedian(list_metrics[2],axis=0), color='r')
    axs.fill_between(bins_centers, np.nanpercentile(list_metrics[0],axis=0, q=25), np.nanpercentile(list_metrics[0],axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(list_metrics[2],axis=0, q=25), np.nanpercentile(list_metrics[2],axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('stride duration [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_stride_duration_group2.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_stride_duration_group2.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers,np.nanmedian(list_metrics[3],axis=0), color='k')
    axs.plot(bins_centers,np.nanmedian(list_metrics[5],axis=0), color='r')
    axs.fill_between(bins_centers, np.nanpercentile(list_metrics[3],axis=0, q=25), np.nanpercentile(list_metrics[3],axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(list_metrics[5],axis=0, q=25), np.nanpercentile(list_metrics[5],axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('stride duration [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_stride_duration_group2.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_stride_duration_group2.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_metrics[0].shape[0]), list_metrics[0][:,1], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_metrics[2].shape[0]), list_metrics[2][:,1], color='r', s=5)
    axs.scatter(0, np.nanmean(list_metrics[0][:,1]), color='k', s=20)
    axs.scatter(1, np.nanmean(list_metrics[2][:,1]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_metrics[0][:,1],axis=0,q=25), np.nanpercentile(list_metrics[0][:,1],axis=0,q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_metrics[2][:,1],axis=0,q=25), np.nanpercentile(list_metrics[2][:,1],axis=0,q=75)], color='r', lw=2)
    axs.set_xlim([-0.5,1.5]), axs.set_ylabel('Stride duration [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_stride_duration_group2.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_stride_duration_group2.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_metrics[3].shape[0]), list_metrics[3][:,1], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_metrics[5].shape[0]), list_metrics[5][:,1], color='r', s=5)
    axs.scatter(0, np.nanmean(list_metrics[3][:,1]), color='k', s=20)
    axs.scatter(1, np.nanmean(list_metrics[5][:,1]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_metrics[3][:,1],axis=0,q=25), np.nanpercentile(list_metrics[3][:,1],axis=0,q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_metrics[5][:,1],axis=0,q=25), np.nanpercentile(list_metrics[5][:,1],axis=0,q=75)], color='r', lw=2)
    axs.set_xlim([-0.5,1.5]), axs.set_ylabel('Stride duration [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_stride_duration_group2.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_stride_duration_group2.svg'), bbox_inches='tight')

    if bool_plot:
        plt.show()
    else:
        plt.close('all')


def plot_step_width_metrics_sh3(matrix_front, matrix_hind, bins_centers, labels, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the step widths metrics
    """
    n_0 = len(np.where(labels==0)[0])
    n_1 = len(np.where(labels==1)[0])
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers, np.nanmedian(matrix_front[labels==0,:],axis=0), color='k')
    axs.plot(bins_centers, np.nanmedian(matrix_front[labels==1,:],axis=0), color='r')
    axs.fill_between(bins_centers, np.nanpercentile(matrix_front[labels==0,:],axis=0,q=25), np.nanpercentile(matrix_front[labels==0,:],axis=0,q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(matrix_front[labels==1,:],axis=0,q=25), np.nanpercentile(matrix_front[labels==1,:],axis=0,q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('front step width [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_front_width_group3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path, f'{figname}_front_width_group3.svg'),bbox_inches='tight')
    
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers, np.nanmedian(matrix_hind[labels==0,:],axis=0), color='k')
    axs.plot(bins_centers, np.nanmedian(matrix_hind[labels==1,:],axis=0), color='r')
    axs.fill_between(bins_centers, np.nanpercentile(matrix_hind[labels==0,:],axis=0,q=25), np.nanpercentile(matrix_hind[labels==0,:],axis=0,q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(matrix_hind[labels==1,:],axis=0,q=25), np.nanpercentile(matrix_hind[labels==1,:],axis=0,q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('front step width [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_hind_width_group3.png'),bbox_inches='tight')
        fig.savefig(os.path.join(figure_path, f'{figname}_hind_width_group3.svg'),bbox_inches='tight')
    

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,n_0), np.nanmean(matrix_front[labels==0,:],axis=1),color='k', s=5)
    axs.scatter(np.random.uniform( 0.8,1.2,n_1), np.nanmean(matrix_front[labels==1,:],axis=1),color='r', s=5)
    axs.scatter(0, np.nanmedian(np.nanmean(matrix_front[labels==0,:],axis=1)), color='k', s=20)
    axs.scatter(1, np.nanmedian(np.nanmean(matrix_front[labels==1,:],axis=1)), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(np.nanmean(matrix_front[labels==0,:],axis=1), q=25), np.nanpercentile(np.nanmean(matrix_front[labels==0,:],axis=1), q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(np.nanmean(matrix_front[labels==1,:],axis=1), q=25), np.nanpercentile(np.nanmean(matrix_front[labels==1,:],axis=1), q=75)], color='r', lw=2)
    axs.set_ylabel('front step width [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_front_width_group1.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_front_width_group1.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,n_0), np.nanmean(matrix_hind[labels==0,:],axis=1),color='k', s=5)
    axs.scatter(np.random.uniform( 0.8,1.2,n_1), np.nanmean(matrix_hind[labels==1,:],axis=1),color='r', s=5)
    axs.scatter(0, np.nanmedian(np.nanmean(matrix_hind[labels==0,:],axis=1)), color='k', s=20)
    axs.scatter(1, np.nanmedian(np.nanmean(matrix_hind[labels==1,:],axis=1)), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(np.nanmean(matrix_hind[labels==0,:],axis=1), q=25), np.nanpercentile(np.nanmean(matrix_hind[labels==0,:],axis=1), q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(np.nanmean(matrix_hind[labels==1,:],axis=1), q=25), np.nanpercentile(np.nanmean(matrix_hind[labels==1,:],axis=1), q=75)], color='r', lw=2)
    axs.set_ylabel('front step width [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_hind_width_group1.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_hind_width_group1.svg'), bbox_inches='tight')

    if bool_plot:
        plt.show()
    else:
        plt.close('all')


def plot_step_width_metrics(list_front, list_hind, bins_centers, bool_plot=False, bool_save=False, figname=None):
    """
    Plots the step widths metrics 
    """
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers,np.nanmedian(list_front[0],axis=0), color='k')
    axs.plot(bins_centers,np.nanmedian(list_front[2],axis=0), color='r')
    axs.fill_between(bins_centers, np.nanpercentile(list_front[0],axis=0, q=25), np.nanpercentile(list_front[0],axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(list_front[2],axis=0, q=25), np.nanpercentile(list_front[2],axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('front step width [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_front_width_group1.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_front_width_group1.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers,np.nanmedian(list_hind[0],axis=0), color='k')
    axs.plot(bins_centers,np.nanmedian(list_hind[2],axis=0), color='r')
    axs.fill_between(bins_centers, np.nanpercentile(list_hind[0],axis=0, q=25), np.nanpercentile(list_hind[0],axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(list_hind[2],axis=0, q=25), np.nanpercentile(list_hind[2],axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('hind step width [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_hind_width_group1.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_hind_width_group1.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers,np.nanmedian(list_front[3],axis=0), color='k')
    axs.plot(bins_centers,np.nanmedian(list_front[5],axis=0), color='r')
    axs.fill_between(bins_centers, np.nanpercentile(list_front[3],axis=0, q=25), np.nanpercentile(list_front[3],axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(list_front[5],axis=0, q=25), np.nanpercentile(list_front[5],axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('front step width [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_front_width_group2.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_front_width_group2.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers,np.nanmedian(list_hind[3],axis=0), color='k')
    axs.plot(bins_centers,np.nanmedian(list_hind[5],axis=0), color='r')
    axs.fill_between(bins_centers, np.nanpercentile(list_hind[3],axis=0, q=25), np.nanpercentile(list_hind[3],axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(list_hind[5],axis=0, q=25), np.nanpercentile(list_hind[5],axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('hind step width [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_hind_width_group2.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_hind_width_group2.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_front[0].shape[0]), np.nanmean(list_front[0],axis=1),color='k', s=5)
    axs.scatter(np.random.uniform( 0.8,1.2,list_front[2].shape[0]), np.nanmean(list_front[2],axis=1),color='r', s=5)
    axs.scatter(0, np.nanmedian(np.nanmean(list_front[0],axis=1)), color='k', s=20)
    axs.scatter(1, np.nanmedian(np.nanmean(list_front[2],axis=1)), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(np.nanmean(list_front[0],axis=1), q=25), np.nanpercentile(np.nanmean(list_front[0],axis=1), q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(np.nanmean(list_front[2],axis=1), q=25), np.nanpercentile(np.nanmean(list_front[2],axis=1), q=75)], color='r', lw=2)
    axs.set_ylabel('front step width [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_front_width_group1.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_front_width_group1.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_hind[0].shape[0]), np.nanmean(list_hind[0],axis=1),color='k', s=5)
    axs.scatter(np.random.uniform( 0.8,1.2,list_hind[2].shape[0]), np.nanmean(list_hind[2],axis=1),color='r', s=5)
    axs.scatter(0, np.nanmedian(np.nanmean(list_hind[0],axis=1)), color='k', s=20)
    axs.scatter(1, np.nanmedian(np.nanmean(list_hind[2],axis=1)), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(np.nanmean(list_hind[0],axis=1), q=25), np.nanpercentile(np.nanmean(list_hind[0],axis=1), q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(np.nanmean(list_hind[2],axis=1), q=25), np.nanpercentile(np.nanmean(list_hind[2],axis=1), q=75)], color='r', lw=2)
    axs.set_ylabel('hind step width [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_hind_width_group1.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_hind_width_group1.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_front[3].shape[0]), np.nanmean(list_front[3],axis=1),color='k', s=5)
    axs.scatter(np.random.uniform( 0.8,1.2,list_front[5].shape[0]), np.nanmean(list_front[5],axis=1),color='r', s=5)
    axs.scatter(0, np.nanmedian(np.nanmean(list_front[3],axis=1)), color='k', s=20)
    axs.scatter(1, np.nanmedian(np.nanmean(list_front[5],axis=1)), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(np.nanmean(list_front[3],axis=1), q=25), np.nanpercentile(np.nanmean(list_front[3],axis=1), q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(np.nanmean(list_front[5],axis=1), q=25), np.nanpercentile(np.nanmean(list_front[5],axis=1), q=75)], color='r', lw=2)
    axs.set_ylabel('front step width [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_front_width_group2.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_front_width_group2.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_hind[3].shape[0]), np.nanmean(list_hind[3],axis=1),color='k', s=5)
    axs.scatter(np.random.uniform( 0.8,1.2,list_hind[5].shape[0]), np.nanmean(list_hind[5],axis=1),color='r', s=5)
    axs.scatter(0, np.nanmedian(np.nanmean(list_hind[3],axis=1)), color='k', s=20)
    axs.scatter(1, np.nanmedian(np.nanmean(list_hind[5],axis=1)), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(np.nanmean(list_hind[3],axis=1), q=25), np.nanpercentile(np.nanmean(list_hind[3],axis=1), q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(np.nanmean(list_hind[5],axis=1), q=25), np.nanpercentile(np.nanmean(list_hind[5],axis=1), q=75)], color='r', lw=2)
    axs.set_ylabel('hind step width [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_hind_width_group2.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_ind_hind_width_group2.svg'), bbox_inches='tight')


    if bool_plot:
        plt.show()
    else:
        plt.close('all')

def plot_lr_step_duration_rett(list_left, list_right, bins_centers, bool_plot=False, bool_save=False, figname=None):
    """
    Plot the left-right asymmetry in step duration
    """
    cmap = cm.plasma(np.linspace(0,1,len(list_left)))
    ratio_list = []
    for group in range(len(list_left)):
        ratio_list.append(list_left[group][:,2] / list_right[group][:,2])

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(ratio_list)):
        axs.scatter(np.random.uniform(group-0.2,group+0.2,list_left[group].shape[0]), ratio_list[group], color=cmap[group], s=5)
        axs.scatter(group, np.nanmedian(ratio_list[group]), color=cmap[group], s=20)
        axs.plot([group,group],[np.nanpercentile(ratio_list[group], q=25), np.nanpercentile(ratio_list[group], q=75)], color=cmap[group], lw=2)
    axs.set_ylabel('left-right ratio')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_symmetry_duration_all_groups.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_symmetry_duration_all_groups.svg'), bbox_inches='tight')

    duration_lr_kruskall = []
    for ii in range(len(ratio_list)):
        duration_lr_kruskall.append(ratio_list[ii])
    print('==================================')
    print('Statistics step duration asymmetry')
    print('==================================')
    print(scipy.stats.kruskal(duration_lr_kruskall[0], duration_lr_kruskall[1], duration_lr_kruskall[2]))
    print(spph.posthoc_dunn(duration_lr_kruskall, p_adjust="fdr_bh"))


    if bool_plot:
        plt.show()
    else:
        plt.close('all')


def plot_lr_step_duration(list_left, list_right, bins_centers, bool_plot=False, bool_save=False, figname=None):
    """
    Plot the left-right asymmetry in step duration
    """

    ratio_0 = list_left[0][:,2] / list_right[0][:,2]
    ratio_2 = list_left[2][:,2] / list_right[2][:,2]

    ratio_3 = list_left[3][:,2] / list_right[3][:,2]
    ratio_5 = list_left[5][:,2] / list_right[5][:,2]

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_left[0].shape[0]), ratio_0, color='k', s=5)
    axs.scatter(np.random.uniform( 0.8,1.2,list_left[2].shape[0]), ratio_2, color='r', s=5)
    axs.scatter(0, np.nanmedian(ratio_0), color='k', s=20)
    axs.scatter(1, np.nanmedian(ratio_2), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(ratio_0, q=25), np.nanpercentile(ratio_0, q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(ratio_2, q=25), np.nanpercentile(ratio_2, q=75)], color='r', lw=2)
    axs.set_ylabel('left-right ratio')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_symmetry_duration_group1.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_symmetry_duration_group1.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_left[3].shape[0]), ratio_3, color='k', s=5)
    axs.scatter(np.random.uniform( 0.8,1.2,list_left[5].shape[0]), ratio_5, color='r', s=5)
    axs.scatter(0, np.nanmedian(ratio_3), color='k', s=20)
    axs.scatter(1, np.nanmedian(ratio_5), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(ratio_3, q=25), np.nanpercentile(ratio_3, q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(ratio_5, q=25), np.nanpercentile(ratio_5, q=75)], color='r', lw=2)
    axs.set_ylabel('left-right ratio')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_symmetry_duration_group2.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_symmetry_duration_group2.svg'), bbox_inches='tight')

    # fig, axs = plt.subplots(1,1,figsize=(3,3))
    # axs.spines[['top','right']].set_visible(False)
    # axs.plot(bins_centers, np.nanmedian(list_left[0],axis=0), color='k', lw=2)
    # axs.plot(bins_centers, np.nanmedian(list_left[2],axis=0), color='r', lw=2)
    # axs.plot(bins_centers, np.nanmedian(list_right[0],axis=0), color='k', lw=2, ls=':')
    # axs.plot(bins_centers, np.nanmedian(list_right[2],axis=0), color='r', lw=2, ls=':')
    # axs.set_xlabel('speed [*]'), axs.set_ylabel('step duration [*]')
    # plt.tight_layout()

    # fig, axs = plt.subplots(1,1,figsize=(3,3))
    # axs.spines[['top','right']].set_visible(False)
    # axs.plot(bins_centers, np.nanmedian(list_left[3],axis=0), color='k', lw=2)
    # axs.plot(bins_centers, np.nanmedian(list_left[5],axis=0), color='r', lw=2)
    # axs.plot(bins_centers, np.nanmedian(list_right[3],axis=0), color='k', lw=2, ls=':')
    # axs.plot(bins_centers, np.nanmedian(list_right[5],axis=0), color='r', lw=2, ls=':')
    # axs.set_xlabel('speed [*]'), axs.set_ylabel('step duration [*]')
    # plt.tight_layout()

    if bool_plot:
        plt.show()
    else:
        plt.close('all')

def plot_lr_step_length_sh3(matrix_left, matrix_right, bins_centers, labels, bool_plot=False, bool_save=False, figname=False):
    """
    Plots the left-right asymetry in step length
    """
    n_0 = len(np.where(labels==0)[0])
    n_1 = len(np.where(labels==1)[0])
    ratio_0 = np.nanmean(matrix_left[labels==0,:] / matrix_right[labels==0,:], axis=1)
    ratio_1 = np.nanmean(matrix_left[labels==1,:] / matrix_right[labels==1,:], axis=1)     

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,n_0), ratio_0, color='k', s=5)
    axs.scatter(np.random.uniform( 0.8,1.2,n_1), ratio_1, color='r', s=5)
    axs.scatter(0, np.nanmedian(ratio_0), color='k', s=20)
    axs.scatter(1, np.nanmedian(ratio_1), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(ratio_0, q=25), np.nanpercentile(ratio_0, q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(ratio_1, q=25), np.nanpercentile(ratio_1, q=75)], color='r', lw=2)
    axs.set_ylabel('left-right ratio')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_symmetry_length_group3.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_symmetry_length_group3.svg'), bbox_inches='tight')
 
    if bool_plot:
        plt.show()
    else:
        plt.close('all')

def plot_lr_step_duration_sh3(matrix_left, matrix_right, bins_centers, labels, bool_plot=False, bool_save=False, figname=False):
    """
    Plots the left-right asymetry in step length
    """
    n_0 = len(np.where(labels==0)[0])
    n_1 = len(np.where(labels==1)[0])
    ratio_0 = np.nanmean(matrix_left[labels==0,:] / matrix_right[labels==0,:], axis=1)
    ratio_1 = np.nanmean(matrix_left[labels==1,:] / matrix_right[labels==1,:], axis=1)     

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,n_0), ratio_0, color='k', s=5)
    axs.scatter(np.random.uniform( 0.8,1.2,n_1), ratio_1, color='r', s=5)
    axs.scatter(0, np.nanmedian(ratio_0), color='k', s=20)
    axs.scatter(1, np.nanmedian(ratio_1), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(ratio_0, q=25), np.nanpercentile(ratio_0, q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(ratio_1, q=25), np.nanpercentile(ratio_1, q=75)], color='r', lw=2)
    axs.set_ylabel('left-right ratio')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_symmetry_duration_group3.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_symmetry_duration_group3.svg'), bbox_inches='tight')
 
    if bool_plot:
        plt.show()
    else:
        plt.close('all')

def plot_lr_step_length_rett(list_left, list_right, bins_centers, bool_plot=False, bool_save=False, figname=None):
    """
    Plot the left-right asymmetry in step duration
    """
    cmap = cm.plasma(np.linspace(0,1,len(list_left)))
    ratio_list = []
    for group in range(len(list_left)):
        ratio_list.append(np.nanmean(list_left[group]/list_right[group],axis=1))

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(ratio_list)):
        axs.scatter(np.random.uniform(group-0.2,group+0.2,list_left[group].shape[0]), ratio_list[group], color=cmap[group], s=5)
        axs.scatter(group, np.nanmedian(ratio_list[group]), color=cmap[group], s=20)
        axs.plot([group,group],[np.nanpercentile(ratio_list[group], q=25), np.nanpercentile(ratio_list[group], q=75)], color=cmap[group], lw=2)
    axs.set_ylabel('left-right ratio')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_symmetry_length_all_groups.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_symmetry_length_all_groups.svg'), bbox_inches='tight')

    length_lr_kruskall = []
    for ii in range(len(list_left)):
        length_lr_kruskall.append(ratio_list[ii])
    print('================================')
    print('Statistics step length asymmetry')
    print('================================')
    print(scipy.stats.kruskal(length_lr_kruskall[0], length_lr_kruskall[1], length_lr_kruskall[2]))
    print(spph.posthoc_dunn(length_lr_kruskall, p_adjust="fdr_bh"))

    if bool_plot:
        plt.show()
    else:
        plt.close('all')

def plot_lr_step_length(list_left, list_right, bins_centers, bool_plot=False, bool_save=False, figname=None):
    """
    Plot the left-right asymmetry in step duration
    """

    ratio_0 = np.nanmean(list_left[0] / list_right[0],axis=1)
    ratio_2 = np.nanmean(list_left[2] / list_right[2],axis=1)

    ratio_3 = np.nanmean(list_left[3] / list_right[3],axis=1)
    ratio_5 = np.nanmean(list_left[5] / list_right[5],axis=1)


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_left[0].shape[0]), ratio_0, color='k', s=5)
    axs.scatter(np.random.uniform( 0.8,1.2,list_left[2].shape[0]), ratio_2, color='r', s=5)
    axs.scatter(0, np.nanmedian(ratio_0), color='k', s=20)
    axs.scatter(1, np.nanmedian(ratio_2), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(ratio_0, q=25), np.nanpercentile(ratio_0, q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(ratio_2, q=25), np.nanpercentile(ratio_2, q=75)], color='r', lw=2)
    axs.set_ylabel('left-right ratio')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_symmetry_length_group1.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_symmetry_length_group1.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_left[3].shape[0]), ratio_3, color='k', s=5)
    axs.scatter(np.random.uniform( 0.8,1.2,list_left[5].shape[0]), ratio_5, color='r', s=5)
    axs.scatter(0, np.nanmedian(ratio_3), color='k', s=20)
    axs.scatter(1, np.nanmedian(ratio_5), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(ratio_3, q=25), np.nanpercentile(ratio_3, q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(ratio_5, q=25), np.nanpercentile(ratio_5, q=75)], color='r', lw=2)
    axs.set_ylabel('left-right ratio')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_symmetry_length_group2.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_symmetry_length_group2.svg'), bbox_inches='tight')


    # fig, axs = plt.subplots(1,1,figsize=(3,3))
    # axs.spines[['top','right']].set_visible(False)
    # axs.plot(bins_centers, np.nanmedian(list_left[0],axis=0), color='k', lw=2)
    # axs.plot(bins_centers, np.nanmedian(list_left[2],axis=0), color='r', lw=2)
    # axs.plot(bins_centers, np.nanmedian(list_right[0],axis=0), color='k', lw=2, ls=':')
    # axs.plot(bins_centers, np.nanmedian(list_right[2],axis=0), color='r', lw=2, ls=':')
    # axs.set_xlabel('speed [*]'), axs.set_ylabel('step length [*]')
    # plt.tight_layout()


    # fig, axs = plt.subplots(1,1,figsize=(3,3))
    # axs.spines[['top','right']].set_visible(False)
    # axs.plot(bins_centers, np.nanmedian(list_left[3],axis=0), color='k', lw=2)
    # axs.plot(bins_centers, np.nanmedian(list_left[5],axis=0), color='r', lw=2)
    # axs.plot(bins_centers, np.nanmedian(list_right[3],axis=0), color='k', lw=2, ls=':')
    # axs.plot(bins_centers, np.nanmedian(list_right[5],axis=0), color='r', lw=2, ls=':')
    # axs.set_xlabel('speed [*]'), axs.set_ylabel('step length [*]')
    # plt.tight_layout()

    if bool_plot:
        plt.show()
    else:
        plt.close('all')


def plot_contact_mode_sh3(list_duration, list_contact_mode, bins_centers, labels, bool_plot=False, bool_save=False, figname=None):
    """
    Represents the information relative to the contact mode
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers, np.nanmedian(list_duration[labels==0,:],axis=0), color='k',lw=2)
    axs.plot(bins_centers, np.nanmedian(list_duration[labels==1,:],axis=0), color='r',lw=2)
    axs.fill_between(bins_centers, np.nanpercentile(list_duration[labels==0,:], axis=0, q=25), np.nanpercentile(list_duration[labels==0,:], axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(list_duration[labels==1,:], axis=0, q=25), np.nanpercentile(list_duration[labels==1,:], axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('stance duration [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_stance_duration_group3.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_stance_duration_group3.svg'))
    

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers, np.nanmedian(list_contact_mode[labels==0,:,2],axis=0), color='k',lw=2)
    axs.plot(bins_centers, np.nanmedian(list_contact_mode[labels==1,:,2],axis=0), color='r',lw=2)
    axs.fill_between(bins_centers, np.nanpercentile(list_contact_mode[labels==0,:,2], axis=0, q=25), np.nanpercentile(list_contact_mode[labels==0,:,2], axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(list_contact_mode[labels==1,:,2], axis=0, q=25), np.nanpercentile(list_contact_mode[labels==1,:,2], axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('diagonal contact [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_diagonal_contact_group3.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_diagonal_contact_group3.svg'))
    
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers, np.nanmedian(list_contact_mode[labels==0,:,3],axis=0), color='k',lw=2)
    axs.plot(bins_centers, np.nanmedian(list_contact_mode[labels==1,:,3],axis=0), color='r',lw=2)
    axs.fill_between(bins_centers, np.nanpercentile(list_contact_mode[labels==0,:,3], axis=0, q=25), np.nanpercentile(list_contact_mode[labels==0,:,3], axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(list_contact_mode[labels==1,:,3], axis=0, q=25), np.nanpercentile(list_contact_mode[labels==1,:,3], axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('other double [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_other_double_group3.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_other_double_group3.svg'))
    
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers, np.nanmedian(list_contact_mode[labels==0,:,4],axis=0), color='k',lw=2)
    axs.plot(bins_centers, np.nanmedian(list_contact_mode[labels==1,:,4],axis=0), color='r',lw=2)
    axs.fill_between(bins_centers, np.nanpercentile(list_contact_mode[labels==0,:,4], axis=0, q=25), np.nanpercentile(list_contact_mode[labels==0,:,4], axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(list_contact_mode[labels==1,:,4], axis=0, q=25), np.nanpercentile(list_contact_mode[labels==1,:,4], axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('triple contact [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_triple_contact_group3.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_triple_contact_group3.svg'))

    labels_plot = ['0 paw','1 paw','2 diago','2 others','3 paws','4 paws']
    import matplotlib.cm as cm
    cmap = cm.gray(np.linspace(0,0.5,6))
    fig, axs = plt.subplots(1,2,figsize=(6,3),sharex=True, sharey=True)
    axs[0].spines[['top','right']].set_visible(False)
    axs[1].spines[['top','right']].set_visible(False)
    axs[0].stackplot(bins_centers, np.nanmean(list_contact_mode[labels==0,:,:],axis=0).T, colors=cmap, labels=labels_plot)
    axs[1].stackplot(bins_centers, np.nanmean(list_contact_mode[labels==1,:,:],axis=0).T, colors=cmap)
    axs[0].set_ylabel('proportion'), axs[0].set_xlabel('speed [*]'), axs[1].set_xlabel('speed [*]')
    axs[0].legend(frameon=False)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_contact_modes_group3.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_contact_modes_group3.svg'))

    n_0 = len(np.where(labels==0)[0])
    n_1 = len(np.where(labels==1)[0])

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,n_0), list_duration[labels==0,1], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,n_1), list_duration[labels==1,1], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_duration[labels==0,1]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_duration[labels==1,1]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_duration[labels==0,1], q=25), np.nanpercentile(list_duration[labels==0,1], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_duration[labels==1,1], q=25), np.nanpercentile(list_duration[labels==1,1], q=75)], color='r', lw=2)
    axs.set_ylabel('stance duration [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_stance_duration_ind_group3.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_stance_duration_ind_group3.svg'), bbox_inches='tight')


    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,n_0), list_contact_mode[labels==0,1,2], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,n_1), list_contact_mode[labels==1,1,2], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_contact_mode[labels==0,1,2]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_contact_mode[labels==1,1,2]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_contact_mode[labels==0,1,2], q=25), np.nanpercentile(list_contact_mode[labels==0,1,2], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_contact_mode[labels==1,1,2], q=25), np.nanpercentile(list_contact_mode[labels==1,1,2], q=75)], color='r', lw=2)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('diagonal contact [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_diagonal_contact_ind_group3.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_diagonal_contact_ind_group3.svg'))

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,n_0), list_contact_mode[labels==0,1,3], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,n_1), list_contact_mode[labels==1,1,3], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_contact_mode[labels==0,1,3]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_contact_mode[labels==1,1,3]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_contact_mode[labels==0,1,3], q=25), np.nanpercentile(list_contact_mode[labels==0,1,3], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_contact_mode[labels==1,1,3], q=25), np.nanpercentile(list_contact_mode[labels==1,1,3], q=75)], color='r', lw=2)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('other double [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_other_double_ind_group3.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_other_double_ind_group3.svg'))

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,n_0), list_contact_mode[labels==0,1,4], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,n_1), list_contact_mode[labels==1,1,4], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_contact_mode[labels==0,1,4]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_contact_mode[labels==1,1,4]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_contact_mode[labels==0,1,4], q=25), np.nanpercentile(list_contact_mode[labels==0,1,4], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_contact_mode[labels==1,1,4], q=25), np.nanpercentile(list_contact_mode[labels==1,1,4], q=75)], color='r', lw=2)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('triple contact [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_triple_contact_ind_group3.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_triple_contact_ind_group3.svg'))

    if bool_plot:
        plt.show()
    else:
        plt.close('all')

    
    tmp_matrix = np.concatenate((np.expand_dims(list_duration[:,1],-1), np.expand_dims(list_contact_mode[:,1,2],-1), np.expand_dims(list_contact_mode[:,1,3],-1), np.expand_dims(list_contact_mode[:,1,4],-1)),axis=1)
    return tmp_matrix

def plot_contact_mode(list_duration, list_contact_mode, bins_centers, bool_plot=False, bool_save=False, figname=None):
    """
    Represents the information relative to the contact mode 
    """

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers, np.nanmedian(list_duration[0],axis=0), color='k',lw=2)
    axs.plot(bins_centers, np.nanmedian(list_duration[2],axis=0), color='r',lw=2)
    axs.fill_between(bins_centers, np.nanpercentile(list_duration[0], axis=0, q=25), np.nanpercentile(list_duration[0], axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(list_duration[2], axis=0, q=25), np.nanpercentile(list_duration[2], axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('stance duration [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_stance_duration_group1.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_stance_duration_group1.svg'))
    
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers, np.nanmedian(list_duration[3],axis=0), color='k',lw=2)
    axs.plot(bins_centers, np.nanmedian(list_duration[5],axis=0), color='r',lw=2)
    axs.fill_between(bins_centers, np.nanpercentile(list_duration[3], axis=0, q=25), np.nanpercentile(list_duration[3], axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(list_duration[5], axis=0, q=25), np.nanpercentile(list_duration[5], axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('stance duration [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_stance_duration_group2.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_stance_duration_group2.svg'))

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers, np.nanmedian(list_contact_mode[0][:,:,2],axis=0), color='k',lw=2)
    axs.plot(bins_centers, np.nanmedian(list_contact_mode[2][:,:,2],axis=0), color='r',lw=2)
    axs.fill_between(bins_centers, np.nanpercentile(list_contact_mode[0][:,:,2], axis=0, q=25), np.nanpercentile(list_contact_mode[0][:,:,2], axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(list_contact_mode[2][:,:,2], axis=0, q=25), np.nanpercentile(list_contact_mode[2][:,:,2], axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('diagonal contact [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_diagonal_contact_group1.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_diagonal_contact_group1.svg'))
    
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers, np.nanmedian(list_contact_mode[3][:,:,2],axis=0), color='k',lw=2)
    axs.plot(bins_centers, np.nanmedian(list_contact_mode[5][:,:,2],axis=0), color='r',lw=2)
    axs.fill_between(bins_centers, np.nanpercentile(list_contact_mode[3][:,:,2], axis=0, q=25), np.nanpercentile(list_contact_mode[3][:,:,2], axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(list_contact_mode[5][:,:,2], axis=0, q=25), np.nanpercentile(list_contact_mode[5][:,:,2], axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('diagonal contact [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_diagonal_contact_group2.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_diagonal_contact_group2.svg'))

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers, np.nanmedian(list_contact_mode[0][:,:,3],axis=0), color='k',lw=2)
    axs.plot(bins_centers, np.nanmedian(list_contact_mode[2][:,:,3],axis=0), color='r',lw=2)
    axs.fill_between(bins_centers, np.nanpercentile(list_contact_mode[0][:,:,3], axis=0, q=25), np.nanpercentile(list_contact_mode[0][:,:,3], axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(list_contact_mode[2][:,:,3], axis=0, q=25), np.nanpercentile(list_contact_mode[2][:,:,3], axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('other double [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_other_double_group1.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_other_double_group1.svg'))
    
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers, np.nanmedian(list_contact_mode[3][:,:,3],axis=0), color='k',lw=2)
    axs.plot(bins_centers, np.nanmedian(list_contact_mode[5][:,:,3],axis=0), color='r',lw=2)
    axs.fill_between(bins_centers, np.nanpercentile(list_contact_mode[3][:,:,3], axis=0, q=25), np.nanpercentile(list_contact_mode[3][:,:,3], axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(list_contact_mode[5][:,:,3], axis=0, q=25), np.nanpercentile(list_contact_mode[5][:,:,3], axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('other double [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_other_double_group2.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_other_double_group2.svg'))

    import matplotlib.cm as cm
    cmap = cm.gray(np.linspace(0,0.5,6))

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers, np.nanmedian(list_contact_mode[0][:,:,4],axis=0), color='k',lw=2)
    axs.plot(bins_centers, np.nanmedian(list_contact_mode[2][:,:,4],axis=0), color='r',lw=2)
    axs.fill_between(bins_centers, np.nanpercentile(list_contact_mode[0][:,:,4], axis=0, q=25), np.nanpercentile(list_contact_mode[0][:,:,4], axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(list_contact_mode[2][:,:,4], axis=0, q=25), np.nanpercentile(list_contact_mode[2][:,:,4], axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('triple contact [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_triple_contact_group1.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_triple_contact_group1.svg'))
    
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.plot(bins_centers, np.nanmedian(list_contact_mode[3][:,:,4],axis=0), color='k',lw=2)
    axs.plot(bins_centers, np.nanmedian(list_contact_mode[5][:,:,4],axis=0), color='r',lw=2)
    axs.fill_between(bins_centers, np.nanpercentile(list_contact_mode[3][:,:,4], axis=0, q=25), np.nanpercentile(list_contact_mode[3][:,:,4], axis=0, q=75), color='k', alpha=0.5)
    axs.fill_between(bins_centers, np.nanpercentile(list_contact_mode[5][:,:,4], axis=0, q=25), np.nanpercentile(list_contact_mode[5][:,:,4], axis=0, q=75), color='r', alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('triple contact [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_triple_contact_group2.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_triple_contact_group2.svg'))

    labels = ['0 paw','1 paw','2 diago','2 others','3 paws','4 paws']

    fig, axs = plt.subplots(1,2,figsize=(6,3),sharex=True, sharey=True)
    axs[0].spines[['top','right']].set_visible(False)
    axs[1].spines[['top','right']].set_visible(False)
    axs[0].stackplot(bins_centers, np.nanmean(list_contact_mode[0],axis=0).T, colors=cmap, labels=labels)
    axs[1].stackplot(bins_centers, np.nanmean(list_contact_mode[2],axis=0).T, colors=cmap)
    axs[0].set_ylabel('proportion'), axs[0].set_xlabel('speed [*]'), axs[1].set_xlabel('speed [*]')
    axs[0].legend(frameon=False)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_contact_modes_group1.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_contact_modes_group1.svg'))

    fig, axs = plt.subplots(1,2,figsize=(6,3),sharex=True, sharey=True)
    axs[0].spines[['top','right']].set_visible(False)
    axs[1].spines[['top','right']].set_visible(False)
    axs[0].stackplot(bins_centers, np.nanmean(list_contact_mode[3],axis=0).T, colors=cmap, labels=labels)
    axs[1].stackplot(bins_centers, np.nanmean(list_contact_mode[5],axis=0).T, colors=cmap)
    axs[0].set_ylabel('proportion'), axs[0].set_xlabel('speed [*]'), axs[1].set_xlabel('speed [*]')
    axs[0].legend(frameon=False)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_contact_modes_group2.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_contact_modes_group2.svg'))


    # Adding the individual plots ...

    list_tmp_metrics = []
    for group in range(6):
        tmp_matrix = np.concatenate((np.expand_dims(list_duration[group][:,1],-1), np.expand_dims(list_contact_mode[group][:,1,2],-1), np.expand_dims(list_contact_mode[group][:,1,3],-1), np.expand_dims(list_contact_mode[group][:,1,4],-1)),axis=1)
        list_tmp_metrics.append(tmp_matrix)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_duration[0].shape[0]), list_duration[0][:,1], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_duration[2].shape[0]), list_duration[2][:,1], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_duration[0][:,1]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_duration[2][:,1]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_duration[0][:,1], q=25), np.nanpercentile(list_duration[0][:,1], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_duration[2][:,1], q=25), np.nanpercentile(list_duration[2][:,1], q=75)], color='r', lw=2)
    axs.set_ylabel('stance duration [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_stance_duration_ind_group1.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_stance_duration_ind_group1.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_duration[3].shape[0]), list_duration[3][:,1], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_duration[5].shape[0]), list_duration[5][:,1], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_duration[3][:,1]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_duration[5][:,1]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_duration[3][:,1], q=25), np.nanpercentile(list_duration[3][:,1], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_duration[5][:,1], q=25), np.nanpercentile(list_duration[5][:,1], q=75)], color='r', lw=2)
    axs.set_ylabel('stance duration [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_stance_duration_ind_group2.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_stance_duration_ind_group2.svg'), bbox_inches='tight')

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_contact_mode[0].shape[0]), list_contact_mode[0][:,1,2], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_contact_mode[2].shape[0]), list_contact_mode[2][:,1,2], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_contact_mode[0][:,1,2]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_contact_mode[2][:,1,2]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_contact_mode[0][:,1,2], q=25), np.nanpercentile(list_contact_mode[0][:,1,2], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_contact_mode[2][:,1,2], q=25), np.nanpercentile(list_contact_mode[2][:,1,2], q=75)], color='r', lw=2)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('diagonal contact [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_diagonal_contact_ind_group1.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_diagonal_contact_ind_group1.svg'))

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_contact_mode[3].shape[0]), list_contact_mode[3][:,1,2], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_contact_mode[5].shape[0]), list_contact_mode[5][:,1,2], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_contact_mode[3][:,1,2]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_contact_mode[5][:,1,2]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_contact_mode[3][:,1,2], q=25), np.nanpercentile(list_contact_mode[3][:,1,2], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_contact_mode[5][:,1,2], q=25), np.nanpercentile(list_contact_mode[5][:,1,2], q=75)], color='r', lw=2)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('diagonal contact [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_diagonal_contact_ind_group2.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_diagonal_contact_ind_group2.svg'))

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_contact_mode[0].shape[0]), list_contact_mode[0][:,1,3], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_contact_mode[2].shape[0]), list_contact_mode[2][:,1,3], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_contact_mode[0][:,1,3]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_contact_mode[2][:,1,3]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_contact_mode[0][:,1,3], q=25), np.nanpercentile(list_contact_mode[0][:,1,3], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_contact_mode[2][:,1,3], q=25), np.nanpercentile(list_contact_mode[2][:,1,3], q=75)], color='r', lw=2)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('other double [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_other_contact_ind_group1.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_other_contact_ind_group1.svg'))

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_contact_mode[3].shape[0]), list_contact_mode[3][:,1,3], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_contact_mode[5].shape[0]), list_contact_mode[5][:,1,3], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_contact_mode[3][:,1,3]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_contact_mode[5][:,1,3]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_contact_mode[3][:,1,3], q=25), np.nanpercentile(list_contact_mode[3][:,1,3], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_contact_mode[5][:,1,3], q=25), np.nanpercentile(list_contact_mode[5][:,1,3], q=75)], color='r', lw=2)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('other double [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_other_contact_ind_group2.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_other_contact_ind_group2.svg'))

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_contact_mode[0].shape[0]), list_contact_mode[0][:,1,4], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_contact_mode[2].shape[0]), list_contact_mode[2][:,1,4], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_contact_mode[0][:,1,4]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_contact_mode[2][:,1,4]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_contact_mode[0][:,1,4], q=25), np.nanpercentile(list_contact_mode[0][:,1,4], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_contact_mode[2][:,1,4], q=25), np.nanpercentile(list_contact_mode[2][:,1,4], q=75)], color='r', lw=2)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('triple contact [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_triple_contact_ind_group1.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_triple_contact_ind_group1.svg'))

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    axs.scatter(np.random.uniform(-0.2,0.2,list_contact_mode[3].shape[0]), list_contact_mode[3][:,1,4], color='k', s=5)
    axs.scatter(np.random.uniform(0.8,1.2,list_contact_mode[5].shape[0]), list_contact_mode[5][:,1,4], color='r', s=5)
    axs.scatter(0, np.nanmedian(list_contact_mode[3][:,1,4]), color='k', s=20)
    axs.scatter(1, np.nanmedian(list_contact_mode[5][:,1,4]), color='r', s=20)
    axs.plot([0,0],[np.nanpercentile(list_contact_mode[3][:,1,4], q=25), np.nanpercentile(list_contact_mode[3][:,1,4], q=75)], color='k', lw=2)
    axs.plot([1,1],[np.nanpercentile(list_contact_mode[5][:,1,4], q=25), np.nanpercentile(list_contact_mode[5][:,1,4], q=75)], color='r', lw=2)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('triple contact [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_triple_contact_ind_group2.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_triple_contact_ind_group2.svg'))
    
    if bool_plot:
        plt.show()
    else:
        plt.close('all')

    return list_tmp_metrics


def plot_contact_mode_rett(list_duration, list_contact_mode, bins_centers, bool_plot=False, bool_save=False, figname=None):
    """
    Represents the information relative to the contact mode 
    """
    cmap = cm.plasma(np.linspace(0,1,len(list_duration)))
    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_duration)):
        axs.plot(bins_centers, np.nanmedian(list_duration[group],axis=0), color=cmap[group],lw=2)
        axs.fill_between(bins_centers, np.nanpercentile(list_duration[group], axis=0, q=25), np.nanpercentile(list_duration[group], axis=0, q=75), color=cmap[group], alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('stance duration [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_stance_duration_all_groups.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_stance_duration_all_groups.svg'))
    

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_contact_mode)):
        axs.plot(bins_centers, np.nanmedian(list_contact_mode[group][:,:,2],axis=0), color=cmap[group],lw=2)
        axs.fill_between(bins_centers, np.nanpercentile(list_contact_mode[group][:,:,2], axis=0, q=25), np.nanpercentile(list_contact_mode[group][:,:,2], axis=0, q=75), color=cmap[group], alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('diagonal contact [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_diagonal_contact_all_groups.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_diagonal_contact_all_groups.svg'))
    

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_contact_mode)):
        axs.plot(bins_centers, np.nanmedian(list_contact_mode[group][:,:,3],axis=0), color=cmap[group],lw=2)
        axs.fill_between(bins_centers, np.nanpercentile(list_contact_mode[group][:,:,3], axis=0, q=25), np.nanpercentile(list_contact_mode[group][:,:,3], axis=0, q=75), color=cmap[group], alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('other double [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_other_double_all_groups.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_other_double_all_groups.svg'))
    
    

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_contact_mode)):
        axs.plot(bins_centers, np.nanmedian(list_contact_mode[group][:,:,4],axis=0), color=cmap[group],lw=2)
        axs.fill_between(bins_centers, np.nanpercentile(list_contact_mode[group][:,:,4], axis=0, q=25), np.nanpercentile(list_contact_mode[group][:,:,4], axis=0, q=75), color=cmap[group], alpha=0.5)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('triple contact [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_triple_contact_all_groups.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_triple_contact_all_groups.svg'))

    labels = ['0 paw','1 paw','2 diago','2 others','3 paws','4 paws']
    cmap = cm.gray(np.linspace(0,0.5,6))
    fig, axs = plt.subplots(1,2,figsize=(6,3),sharex=True, sharey=True)
    axs[0].spines[['top','right']].set_visible(False)
    axs[1].spines[['top','right']].set_visible(False)
    axs[0].stackplot(bins_centers, np.nanmean(list_contact_mode[0],axis=0).T, colors=cmap, labels=labels)
    axs[1].stackplot(bins_centers, np.nanmean(list_contact_mode[1],axis=0).T, colors=cmap)
    axs[0].set_ylabel('proportion'), axs[0].set_xlabel('speed [*]'), axs[1].set_xlabel('speed [*]')
    axs[0].legend(frameon=False)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_contact_modes_group1.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_contact_modes_group1.svg'))

    fig, axs = plt.subplots(1,2,figsize=(6,3),sharex=True, sharey=True)
    axs[0].spines[['top','right']].set_visible(False)
    axs[1].spines[['top','right']].set_visible(False)
    axs[0].stackplot(bins_centers, np.nanmean(list_contact_mode[0],axis=0).T, colors=cmap, labels=labels)
    axs[1].stackplot(bins_centers, np.nanmean(list_contact_mode[2],axis=0).T, colors=cmap)
    axs[0].set_ylabel('proportion'), axs[0].set_xlabel('speed [*]'), axs[1].set_xlabel('speed [*]')
    axs[0].legend(frameon=False)
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_contact_modes_group2.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_contact_modes_group2.svg'))

    if len(list_contact_mode)>3:
        fig, axs = plt.subplots(1,2,figsize=(6,3),sharex=True, sharey=True)
        axs[0].spines[['top','right']].set_visible(False)
        axs[1].spines[['top','right']].set_visible(False)
        axs[0].stackplot(bins_centers, np.nanmean(list_contact_mode[0],axis=0).T, colors=cmap, labels=labels)
        axs[1].stackplot(bins_centers, np.nanmean(list_contact_mode[3],axis=0).T, colors=cmap)
        axs[0].set_ylabel('proportion'), axs[0].set_xlabel('speed [*]'), axs[1].set_xlabel('speed [*]')
        axs[0].legend(frameon=False)
        plt.tight_layout()
        if bool_save:
            fig.savefig(os.path.join(figure_path, f'{figname}_contact_modes_group3.png'))
            fig.savefig(os.path.join(figure_path, f'{figname}_contact_modes_group3.svg'))


    # Adding the individual plots ...
    cmap = cm.plasma(np.linspace(0,1,len(list_duration)))
    list_tmp_metrics = []
    for group in range(len(list_duration)):
        tmp_matrix = np.concatenate((np.expand_dims(list_duration[group][:,1],-1), np.expand_dims(list_contact_mode[group][:,1,2],-1), np.expand_dims(list_contact_mode[group][:,1,3],-1), np.expand_dims(list_contact_mode[group][:,1,4],-1)),axis=1)
        list_tmp_metrics.append(tmp_matrix)

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_duration)):
        axs.scatter(np.random.uniform(group-0.2,group+0.2,list_duration[group].shape[0]), list_duration[group][:,1], color=cmap[group], s=5)
        axs.scatter(group, np.nanmedian(list_duration[group][:,1]), color=cmap[group], s=20)
        axs.plot([group,group],[np.nanpercentile(list_duration[group][:,1], q=25), np.nanpercentile(list_duration[group][:,1], q=75)], color=cmap[group], lw=2)
    axs.set_ylabel('stance duration [*]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path,f'{figname}_stance_duration_ind_all_groups.png'), bbox_inches='tight')
        fig.savefig(os.path.join(figure_path,f'{figname}_stance_duration_ind_all_groups.svg'), bbox_inches='tight')

    stance_duration_kruskall = []
    for ii in range(len(list_duration)):
        stance_duration_kruskall.append(list_duration[ii][:,1])
    print('==========================')
    print('Statistics stance duration')
    print('==========================')
    print(scipy.stats.kruskal(stance_duration_kruskall[0], stance_duration_kruskall[1], stance_duration_kruskall[2]))
    print(spph.posthoc_dunn(stance_duration_kruskall, p_adjust="fdr_bh"))

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_duration)):
        axs.scatter(np.random.uniform(group-0.2,group+0.2,list_contact_mode[group].shape[0]), list_contact_mode[group][:,1,2], color=cmap[group], s=5)
        axs.scatter(group, np.nanmedian(list_contact_mode[group][:,1,2]), color=cmap[group], s=20)
        axs.plot([group,group],[np.nanpercentile(list_contact_mode[group][:,1,2], q=25), np.nanpercentile(list_contact_mode[group][:,1,2], q=75)], color=cmap[group], lw=2)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('diagonal contact [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_diagonal_contact_ind_all_groups.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_diagonal_contact_ind_all_groups.svg'))

    diagonal_contact_kruskall = []
    for ii in range(len(list_contact_mode)):
        diagonal_contact_kruskall.append(list_contact_mode[ii][:,1,2])
    print('===========================')
    print('Statistics diagonal contact')
    print('===========================')
    print(scipy.stats.kruskal(diagonal_contact_kruskall[0], diagonal_contact_kruskall[1], diagonal_contact_kruskall[2]))
    print(spph.posthoc_dunn(diagonal_contact_kruskall, p_adjust="fdr_bh"))

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_duration)):
        axs.scatter(np.random.uniform(group-0.2,group+0.2,list_contact_mode[group].shape[0]), list_contact_mode[group][:,1,3], color=cmap[group], s=5)
        axs.scatter(group, np.nanmedian(list_contact_mode[group][:,1,3]), color=cmap[group], s=20)
        axs.plot([group,group],[np.nanpercentile(list_contact_mode[group][:,1,3], q=25), np.nanpercentile(list_contact_mode[group][:,1,3], q=75)], color=cmap[group], lw=2)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('other double [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_other_contact_ind_all_groups.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_other_contact_ind_all_groups.svg'))

    other_contact_kruskall = []
    for ii in range(len(list_contact_mode)):
        other_contact_kruskall.append(list_contact_mode[ii][:,1,3])
    print('========================')
    print('Statistics other contact')
    print('========================')
    print(scipy.stats.kruskal(other_contact_kruskall[0], other_contact_kruskall[1], other_contact_kruskall[2]))
    print(spph.posthoc_dunn(other_contact_kruskall, p_adjust="fdr_bh"))

    fig, axs = plt.subplots(1,1,figsize=(3,3))
    axs.spines[['top','right']].set_visible(False)
    for group in range(len(list_duration)):
        axs.scatter(np.random.uniform(group-0.2,group+0.2,list_contact_mode[group].shape[0]), list_contact_mode[group][:,1,4], color=cmap[group], s=5)
        axs.scatter(group, np.nanmedian(list_contact_mode[group][:,1,4]), color=cmap[group], s=20)
        axs.plot([group,group],[np.nanpercentile(list_contact_mode[group][:,1,4], q=25), np.nanpercentile(list_contact_mode[group][:,1,4], q=75)], color=cmap[group], lw=2)
    axs.set_xlabel('speed [*]'), axs.set_ylabel('triple contact [%]')
    plt.tight_layout()
    if bool_save:
        fig.savefig(os.path.join(figure_path, f'{figname}_triple_contact_ind_all_groups.png'))
        fig.savefig(os.path.join(figure_path, f'{figname}_triple_contact_ind_all_groups.svg'))

    triple_contact_kruskall = []
    for ii in range(len(list_contact_mode)):
        triple_contact_kruskall.append(list_contact_mode[ii][:,1,4])
    print('=========================')
    print('Statistics triple contact')
    print('=========================')
    print(scipy.stats.kruskal(triple_contact_kruskall[0], triple_contact_kruskall[1], triple_contact_kruskall[2]))
    print(spph.posthoc_dunn(triple_contact_kruskall, p_adjust="fdr_bh"))
    
    if bool_plot:
        plt.show()
    else:
        plt.close('all')

    return list_tmp_metrics
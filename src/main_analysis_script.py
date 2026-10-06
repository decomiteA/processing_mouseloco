import os, sys, copy
import numpy as np
import pickle 
import scipy.stats 
import warnings
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from sklearn.decomposition import PCA 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils.utils_mouse_analysis import *
warnings.filterwarnings('ignore')

# Toggle those booleans if you want to have the figures generated and/or saved 
bool_save, bool_plot = False, True
# Change the following line to save the statistical analysis to a new text file
with open(os.path.join("..","results","metrics","mice","female_rett","test_stats.txt"), "w") as f:
    original_stdout = sys.stdout  
    sys.stdout = f   

    # Change the following line to match the different cohorts you want to analyse
    str_population = ["WildType","R270X","ABE"]
    n_group = len(str_population)
    # Change this path to the data you want to analyse (ie the output of the processing script)
    input_path = os.path.join(os.getcwd(),'..','datasets','mice','test_code','io_processed')
    list_gross_metrics, list_bl_animal, list_bl_coord = [], [], []
    list_animal_length, list_phase_metrics, list_metrics = [], [], []
    list_animal_fr, list_input_fr_body, list_input_fr_self, list_output_fr = [], [], [], []
    list_stride_animal, list_stride_input, list_stride_output, list_stride_velocity = [], [], [], []
    # Change the line below to match the number of animal per cohort your are analysing ...
    labels = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3])
    for group in str_population:
        local_path = os.path.join(input_path, group)
        

        # Animal length 
        animal_length = np.load(os.path.join(local_path,'animal_length.npy'))
        list_animal_length.append(animal_length)
        n_animal = animal_length.shape[0]

        # Gross metrics 
        list_gross_metrics.append(normalize_gross(np.load(os.path.join(local_path,'gross_locomotion_metrics.npy')),animal_length))

        # Body limb coordination
        body_limb_coordination = np.load(os.path.join(local_path,'body_limb_coordination.npy'))
        idx_flip = np.where(np.nanmean(body_limb_coordination[:,:,2],axis=1)<0)[0]
        flipped_coordination = copy.deepcopy(body_limb_coordination)
        flipped_coordination[idx_flip,:,:] = - flipped_coordination[idx_flip,:,:]
        for row in range(flipped_coordination.shape[0]):
            flipped_coordination[row,:,:] = flipped_coordination[row,:,:] - np.nanmean(flipped_coordination[row,:,:],axis=0)
        body_limb_animal = np.load(os.path.join(local_path,'body_limb_animal.npy'))
        list_bl_animal.append(body_limb_animal)
        list_bl_coord.append(flipped_coordination)

        # Interlimb coordination and stride metrics
        stride_level_data_input = np.load(os.path.join(local_path,'stride_level_data_input.npy'))
        stride_level_data_output = np.load(os.path.join(local_path,'stride_level_data_output.npy'))
        stride_level_data_animal = np.load(os.path.join(local_path,'stride_level_data_animal.npy'))
        tmp_speed = np.nanmean(stride_level_data_input[:,:,2],axis=1)
        normalized_speed, normalized_output = normalize_contact_mode_sh3(tmp_speed, stride_level_data_output, stride_level_data_animal, animal_length)
        list_stride_animal.append(stride_level_data_animal)
        list_stride_input.append(stride_level_data_input)
        list_stride_output.append(normalized_output)
        list_stride_velocity.append(normalized_speed)
        
        with open(os.path.join(local_path,'phasor_metrics_leg_0.pkl'),'rb') as f:
            phasor_metrics_leg_0 = pickle.load(f)
        with open(os.path.join(local_path,'phasor_metrics_leg_1.pkl'),'rb') as f:
            phasor_metrics_leg_1 = pickle.load(f)
        with open(os.path.join(local_path,'phasor_metrics_leg_2.pkl'),'rb') as f:
            phasor_metrics_leg_2 = pickle.load(f)
        with open(os.path.join(local_path,'phasor_metrics_leg_3.pkl'),'rb') as f:
            phasor_metrics_leg_3 = pickle.load(f)

        list_phase_metrics.append(normalize_phasor_metrics(phasor_metrics_leg_0, animal_length))


        # Foot placement control metrics

        list_animal_fr.append(np.load(os.path.join(local_path,'step_level_data_animal.npy')))
        list_input_fr_body.append(np.load(os.path.join(local_path,'step_level_data_input.npy')))
        list_input_fr_self.append(np.load(os.path.join(local_path,'step_level_data_input_self.npy')))
        list_output_fr.append(np.load(os.path.join(local_path ,'step_level_data_output.npy')))


        # Velocity dependent metrics
        with open(os.path.join(local_path,'list_metrics.pkl'),'rb') as f:
            tmp_metrics = pickle.load(f)
        list_metrics.append(normalize_metrics(tmp_metrics, animal_length, bool_klibaite=False))

        
    #####################
    ### Gross Metrics ###
    #####################
    list_individual_gross = []
    for group in range(n_group):
        list_individual_gross.append(individual_gross(list_gross_metrics[group]))

    plot_gross_metrics_distribution_rett(list_gross_metrics, bool_plot=bool_plot, bool_save=bool_save,figname="gross")
    plot_gross_metrics_individual_rett(list_individual_gross, bool_save=bool_save, bool_plot=bool_plot, figname="gross")

    metrics_gross = np.concatenate((list_individual_gross[0], list_individual_gross[1], list_individual_gross[2]),axis=0)


    #############################
    ### Bodylimb coordination ###
    #############################
    list_bl_flipped_coord = []
    for group in range(n_group):
        local_data = list_bl_coord[group]
        for row in range(local_data.shape[0]):
            local_data[row,:,:] = local_data[row,:,:] - np.nanmean(local_data[row,:,:],axis=0)
        list_bl_flipped_coord.append(local_data)

    list_individual_bl = []
    for group in range(n_group):
        list_individual_bl.append(individual_bl(list_bl_flipped_coord[group],list_bl_animal[group], list_animal_length[group]))


    plot_bl_coordination_group_rett(list_bl_flipped_coord, list_bl_animal, bool_save=bool_save, bool_plot=bool_plot,figname="bl_coord")
    plot_bl_coordination_individual_rett(list_individual_bl, list_bl_animal, bool_save=bool_save, bool_plot=bool_plot,figname="bl_coord")

    metrics_bl = np.concatenate((list_individual_bl[0], list_individual_bl[1], list_individual_bl[2]),axis=0)

    ##############################
    ### Interlimb coordination ###
    ##############################

    bins_limits = np.linspace(0.2,0.8,6)
    bins_centers = np.diff(bins_limits)[0]/2 + bins_limits[:-1]
    list_phasor_data = []
    for group in range(n_group):
        matrix_relative_time = np.zeros((len(np.unique(list_phase_metrics[group][:,0])), len(bins_centers),3))
        for animal in range(matrix_relative_time.shape[0]):
            local_phase_data = list_phase_metrics[group][list_phase_metrics[group][:,0]==animal,:]
            for bin in range(len(bins_centers)):
                idx_local = np.where((np.abs(local_phase_data[:,-1])>bins_limits[bin]) & (np.abs(local_phase_data[:,-1])<bins_limits[bin+1]))[0]
                matrix_relative_time[animal,bin,0] = scipy.stats.circmean(local_phase_data[idx_local,3]/local_phase_data[idx_local,2], low=0, high=1, nan_policy='omit')
                matrix_relative_time[animal,bin,1] = scipy.stats.circmean(local_phase_data[idx_local,4]/local_phase_data[idx_local,2], low=0, high=1, nan_policy='omit')
                matrix_relative_time[animal,bin,2] = scipy.stats.circmean(local_phase_data[idx_local,5]/local_phase_data[idx_local,2], low=-0.5, high=0.5, nan_policy='omit')
        matrix_relative_time *= 2*np.pi
        list_phasor_data.append(matrix_relative_time)


    interlimb_metrics = plot_interlimb_coordination_rett(list_phasor_data, bins_centers, bool_plot=bool_plot, bool_save=bool_save,figname="interlimb")

    metrics_interlimb = np.concatenate((interlimb_metrics[0],interlimb_metrics[1],interlimb_metrics[2]),axis=0)

    ##############################
    ### Foot placement control ###
    ##############################



    list_rsquare_body_lateral, list_rsquares_self_lateral, list_gains_lateral = [], [], []

    for group in range(n_group):
        rsquare_lateral, gains_lateral = get_rsquare_matrix_feedback(list_animal_fr[group], list_input_fr_body[group], list_output_fr[group], bool_hind=False, bool_lat=True)
        rsquare_body = get_rsquare_matrix_self(list_animal_fr[group], list_input_fr_self[group], list_output_fr[group], bool_hind=False, bool_lat=True)
        list_rsquare_body_lateral.append(rsquare_lateral)
        list_gains_lateral.append(gains_lateral)
        list_rsquares_self_lateral.append(rsquare_body)


    list_fp_metrics = plot_foot_placement_control_rett(list_rsquare_body_lateral, list_rsquares_self_lateral, list_gains_lateral, bool_plot=bool_plot, bool_save=bool_save, figname="foot_placement")
    metrics_fp_control = np.concatenate((list_fp_metrics[0],list_fp_metrics[1],list_fp_metrics[2]),axis=0)
    ##################################
    ### Velocity-dependent metrics ###
    ##################################
    list_stride_length, list_stride_duration, list_front_width, list_hind_width = [], [], [], []
    for group in range(n_group):
        matrix_stride_length, matrix_stride_duration = np.zeros((len(np.unique(list_metrics[group][:,0])), len(bins_centers))), np.zeros((len(np.unique(list_metrics[group][:,0])), len(bins_centers)))
        matrix_front_width, matrix_hind_width = np.zeros((len(np.unique(list_metrics[group][:,0])), len(bins_centers))), np.zeros((len(np.unique(list_metrics[group][:,0])), len(bins_centers)))
        for animal in range(matrix_stride_length.shape[0]):
            local_metrics_data = list_metrics[group][list_metrics[group][:,0]==animal,:]
            for bin in range(len(bins_centers)):
                idx_local = np.where((np.abs(local_metrics_data[:,6])>bins_limits[bin]) & (np.abs(local_metrics_data[:,6])<bins_limits[bin+1]) & (local_metrics_data[:,1]==0) & (local_metrics_data[:,2]==0))[0]
                matrix_stride_length[animal, bin] = np.nanmean(np.abs(local_metrics_data[idx_local,4]))
                matrix_stride_duration[animal, bin] = np.nanmean(local_metrics_data[idx_local,3])

                idx_front = np.where((np.abs(local_metrics_data[:,6])>bins_limits[bin]) & (np.abs(local_metrics_data[:,6])<bins_limits[bin+1]) & (((local_metrics_data[:,1]==1) & (local_metrics_data[:,2]==0)) | ((local_metrics_data[:,1]==0) & (local_metrics_data[:,2]==1))))[0]
                idx_hind = np.where((np.abs(local_metrics_data[:,6])>bins_limits[bin]) & (np.abs(local_metrics_data[:,6])<bins_limits[bin+1]) & (((local_metrics_data[:,1]==3) & (local_metrics_data[:,2]==2)) | ((local_metrics_data[:,1]==2) & (local_metrics_data[:,2]==3))))[0]
                matrix_front_width[animal,bin] = np.nanmean(np.abs(local_metrics_data[idx_front,5]))
                matrix_hind_width[animal,bin] = np.nanmean(np.abs(local_metrics_data[idx_hind,5]))
        list_stride_length.append(matrix_stride_length)
        list_stride_duration.append(matrix_stride_duration)
        list_front_width.append(matrix_front_width)
        list_hind_width.append(matrix_hind_width)

    plot_step_length_metrics_rett(list_stride_length, bins_centers, bool_plot=bool_plot, bool_save=bool_save,figname="length")
    plot_step_duration_metrics_rett(list_stride_duration, bins_centers, bool_plot=bool_plot, bool_save=bool_save,figname="duration")
    plot_step_width_metrics_rett(list_front_width, list_hind_width, bins_centers, bool_plot=bool_plot, bool_save=bool_save,figname="width")

    list_veldp_metrics = [] 
    for group in range(n_group):
        tmp_matrix = np.concatenate((np.expand_dims(list_stride_length[group][:,1],-1), np.expand_dims(list_stride_duration[group][:,1],-1), np.expand_dims(np.nanmean(list_front_width[group],axis=1),-1), np.expand_dims(np.nanmean(list_hind_width[group],axis=1),-1)),axis=1)
        list_veldp_metrics.append(tmp_matrix)

    metrics_vld = np.concatenate((list_veldp_metrics[0], list_veldp_metrics[1], list_veldp_metrics[2]),axis=0)

    ############################
    ### Left-right asymmetry ###
    ############################

    list_left_step_length, list_right_step_length = [], []
    list_left_step_duration, list_right_step_duration = [], [] 
    for group in range(n_group):
        matrix_left_length, matrix_right_length = np.zeros((len(np.unique(list_metrics[group][:,0])), len(bins_centers))), np.zeros((len(np.unique(list_metrics[group][:,0])), len(bins_centers)))
        matrix_left_duration, matrix_right_duration = np.zeros((len(np.unique(list_metrics[group][:,0])), len(bins_centers))), np.zeros((len(np.unique(list_metrics[group][:,0])), len(bins_centers)))
        for animal in range(matrix_left_length.shape[0]):
            local_metrics_data = list_metrics[group][list_metrics[group][:,0]==animal,:]
            for bin in range(len(bins_centers)):
                idx_right = np.where((np.abs(local_metrics_data[:,6])>bins_limits[bin]) & (np.abs(local_metrics_data[:,6])<bins_limits[bin+1]) & (local_metrics_data[:,1]==2) & (local_metrics_data[:,2]==0))[0]
                idx_left = np.where((np.abs(local_metrics_data[:,6])>bins_limits[bin]) & (np.abs(local_metrics_data[:,6])<bins_limits[bin+1]) & (local_metrics_data[:,1]==3) & (local_metrics_data[:,2]==1))[0]

                matrix_left_length[animal, bin] = np.nanmean(np.abs(local_metrics_data[idx_left,4]))
                matrix_right_length[animal, bin] = np.nanmean(np.abs(local_metrics_data[idx_right,4]))
                matrix_left_duration[animal, bin] = np.nanmean(np.abs(local_metrics_data[idx_left,3]))
                matrix_right_duration[animal, bin] = np.nanmean(np.abs(local_metrics_data[idx_right,3]))

        list_left_step_length.append(matrix_left_length), list_left_step_duration.append(matrix_left_duration)
        list_right_step_length.append(matrix_right_length), list_right_step_duration.append(matrix_right_duration)

    plot_lr_step_length_rett(list_left_step_length, list_right_step_length, bins_centers, bool_plot=bool_plot, bool_save=bool_save, figname="length_asymmetry")
    plot_lr_step_duration_rett(list_left_step_duration, list_right_step_duration, bins_centers, bool_plot=bool_plot, bool_save=bool_save, figname="duration_asymmetry")


    metrics_ratio_length = np.concatenate((list_left_step_length[0]/list_right_step_length[0], list_left_step_length[1]/list_right_step_length[1], list_left_step_length[2]/list_right_step_length[2]),axis=0)[:,2]
    metrics_ratio_duration = np.concatenate((list_left_step_duration[0]/list_right_step_duration[0], list_left_step_duration[1]/list_right_step_duration[1], list_left_step_duration[2]/list_right_step_duration[2]),axis=0)[:,2]

    metrics_symmetry = np.concatenate((np.expand_dims(metrics_ratio_length,-1), np.expand_dims(metrics_ratio_duration, -1)),axis=1)


    ####################
    ### Contact mode ###
    ####################
    list_stance_duration = []
    list_contact_mode = []
    for group in range(n_group):
        matrix_stance_duration = np.zeros((len(np.unique(list_stride_animal[group])), len(bins_centers)))
        matrix_contact_mode = np.zeros((len(np.unique(list_stride_animal[group])), len(bins_centers), 6))
        for animal in range(matrix_stance_duration.shape[0]):
            local_speed_data = list_stride_velocity[group][list_stride_animal[group]==animal]
            local_metrics_data = list_stride_output[group][list_stride_animal[group]==animal,:]
            for bin in range(len(bins_centers)):
                idx_local = np.where((np.abs(local_speed_data)>bins_limits[bin]) & (np.abs(local_speed_data)<bins_limits[bin+1]))[0]
                matrix_stance_duration[animal,bin] = np.nanmean(local_metrics_data[idx_local,3])
                for ii in range(matrix_contact_mode.shape[-1]):
                    matrix_contact_mode[animal,bin,ii] = np.nanmean(local_metrics_data[idx_local,7+ii])
        list_stance_duration.append(matrix_stance_duration)
        list_contact_mode.append(matrix_contact_mode)

    list_contact_metrics = plot_contact_mode_rett(list_stance_duration, list_contact_mode, bins_centers, bool_plot=bool_plot, bool_save=bool_save, figname="contact_mode")
    metrics_contact_mode = np.concatenate((list_contact_metrics[0], list_contact_metrics[1], list_contact_metrics[2]),axis=0)

    total_metrics_matrix = np.concatenate((metrics_gross, metrics_bl, metrics_interlimb, metrics_vld, metrics_fp_control, metrics_symmetry, metrics_contact_mode),axis=1)
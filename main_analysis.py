"""
Main analysis script for the network-energy project.

Structure should be:
1. Load data
2. Calculate FC with Pearson correlation
3. Calculate network metrics from FC
4. Calculate FC with information theory
5. Calculate network metrics from information theory
6. Compare with PET (include spin nulls)

"""

#%%
import time
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.connectome import ConnectivityMeasure
import seaborn as sns
import bct as brainconn
from glob import glob
from nilearn import datasets
from scipy import stats

from src.utils import read_atlases_timeseries, plot_histogram, plot_matrix
from src.calculators import Calculators
from sklearn.decomposition import PCA

from neuromaps.nulls import alexander_bloch
from neuromaps.images import dlabel_to_gifti
from neuromaps.stats import compare_images

#%% Varibles to change
atlas = 'Schaefer400'
plot = False
sub_task = 'rest'

base_path = os.getcwd() + '/data/tum_data'

#%%
# Create subdirectories if they don't exist

if not os.path.exists(os.path.join(base_path, 'concatenated_time-series')):
    print('Creating concatenated_time-series folder.')
    os.makedirs(os.path.join(base_path, 'concatenated_time-series'))

if not os.path.exists(os.path.join(base_path, 'fc_matrices')):
    print('Creating fc_matrices folder.')
    os.makedirs(os.path.join(base_path, 'fc_matrices'))

if not os.path.exists(os.path.join(base_path, 'network-metrics')):
    print('Creating network-metrics folder.')
    os.makedirs(os.path.join(base_path, 'network-metrics'))

if not os.path.exists(os.path.join(base_path, 'information-theory_outputs')):
    print('Creating information-theory_outputs folder.')
    os.makedirs(os.path.join(base_path, 'information-theory_outputs'))


#%%
npy_timeseries_path = os.path.join(base_path, atlas)

files = glob(os.path.join(npy_timeseries_path, f"sub-*_desc-roi_timeseries_{atlas}.npy"))
files.sort()
print(files)

#%% Make FC matrix and load PET data  
pet_list = []

for file in files: 
    print(f'Calculating FC for {file}.')

    sub = file.split('/')[-1].split('_')[0].split('-')[-1]
    roi_time_series_cort, pet = read_atlases_timeseries(atlas, sub, sub_task, base_path)
    pet_list.append(pet)

# is it necessary to use ConnectivityMeasure? Can't we use np.corrcoef?
    FC = ConnectivityMeasure(kind='correlation')
    FC = FC.fit_transform([roi_time_series_cort])[0]

    if plot:
        # plot the correlation matrix
        plt.imshow(FC, interpolation='nearest', cmap='RdBu_r')
        plt.colorbar()
        plt.title(f'FC sub : {sub}')
        plt.show()

pet_avg = np.mean(pet_list, axis=0)

#%%
################################################################################
#################### NETWORK METRICS ###########################################
################################################################################ 

#%%

#load file paths
sub_files = glob(os.path.join(base_path,"fc_matrices/sub-*_matrix.csv"))
print(len(sub_files))
print(sub_files)

# load matrices into list
corr_list =[]

for sub in sub_files:
    corr = pd.read_csv(sub, header=None)
    corr = corr.to_numpy()
    corr_list.append(corr)
# %%

# Set diagonal to zero 
adj_wei_list = []
for corr in corr_list:
    
    adj_wei = corr - np.eye(corr.shape[0])
# why are we zscoring with arctanh? Can we keep it in the -1 to 1 range?
    adj_wei = np.arctanh(adj_wei)
    adj_wei_list.append(adj_wei)
    
    if plot:
        # Plot a histogram of the correlation strengths
        plt.hist(adj_wei.ravel(), bins=50)
        plt.show()

# %%

# threshold fc matrix & plot figs 
adj_bin_list = []
adj_filt_list = []
fc_threshold = 0.17

for adj_wei, corr in zip(adj_wei_list, corr_list):
    adj_wei_abs = np.abs(adj_wei)
    adj_wei_abs_threshold = brainconn.utils.threshold_proportional(adj_wei, fc_threshold)
    adj_bin = brainconn.utils.binarize(adj_wei_abs_threshold)
    adj_filt = np.multiply(corr, adj_bin)
    adj_bin_list.append(adj_bin)
    adj_filt_list.append(adj_filt)

    if plot:
    
        plt.hist(adj_filt.ravel(), bins=50)
        plt.xlim(.4,1)
        plt.ylim(0, 1000)
        plt.title('Histogram of adj_filt')
        plt.show()

        # Look at binary adjacency matrix
        plt.imshow(adj_bin)
        plt.colorbar()
        plt.title('Binary adjacency matrix')
        plt.show()

        # Plot original and filter correlation matrices
        plt.imshow(adj_wei, interpolation='nearest', cmap='RdBu_r')
        plt.colorbar()
        plt.title('Original Correlation Matrix')
        plt.show()

        plt.imshow(adj_filt, interpolation='nearest', cmap='RdBu_r')
        plt.colorbar()
        plt.title('Thresholded Correlation Matrix')
        plt.show()

# %%

# If the labels file doesn't exist in the base_path directory, create it
if not os.path.exists(os.path.join(base_path, 'network_labels.csv')):
    # Create labels for the 400 ROI network
    print('Creating network_labels.csv file.')

    atlas = datasets.fetch_atlas_schaefer_2018(
                                n_rois=200, 
                                yeo_networks=7,
                                resolution_mm=1)

    roi_labels = atlas.labels

    roi_labels = [row.tobytes().decode('UTF-8') for row in roi_labels]

    nets_dict = {
                "_Vis_" : 0,
                "_SomMot_" : 1,
                "_DorsAttn_" : 2,
                "_SalVentAttn_" : 3,
                "_Limbic_" : 4,
                "_Cont_" : 5,
                "_Default_" : 6
                }

    # Create a list that has the nets_dict keys in the order of the roi_labels
    nets_list = []
    for roi in roi_labels:
        for net in nets_dict.keys():
            if net in roi:
                nets_list.append(nets_dict[net])
                break

    # save the nets_list as a csv file in the base_path
    np.savetxt(f'{base_path}/network_labels.csv', nets_list, delimiter=',', fmt='%s')

labels = pd.read_csv(base_path+"/network_labels.csv", header=None)
labels = [i[0] for i in labels.to_numpy()]
labels = np.reshape(np.array(labels),(len(labels),-1))


#%%
metrics_df = pd.DataFrame()

for adj_filt in adj_filt_list:

    part_und = brainconn.participation_coef(adj_filt, 
                                            labels, 
                                            degree='undirected')
    degr_und = brainconn.degree.degrees_und(adj_filt)
    str_und = brainconn.degree.strengths_und(adj_filt)
    eig_und = brainconn.centrality.eigenvector_centrality_und(adj_filt)
    clust_und = brainconn.clustering_coef_wu(adj_filt)

    metrics = {'participation_coef': [part_und],
               'degree': [degr_und],
               'strength': [str_und],
               'eigen': [eig_und],
               'clustering': [clust_und]}
    
    metrics = pd.DataFrame(metrics)
    metrics_df = pd.concat((metrics_df,metrics), 
                           axis=0, 
                           ignore_index=True)
    
part = np.array(metrics_df['participation_coef'].to_list())
deg = np.array(metrics_df['degree'].to_list())
stre = np.array(metrics_df['strength'].to_list())
clust = np.array(metrics_df['clustering'].to_list())
eig = np.array(metrics_df['eigen'].to_list())

part = np.mean(part,axis=0)
deg = np.mean(deg,axis=0)
stre = np.mean(stre,axis=0)
clust = np.mean(clust,axis=0)
eig = np.mean(eig,axis=0)

metrics_avg_df = pd.DataFrame({'participation_coef': part,
                               'degree': deg,
                               'strength': stre,
                               'clustering': clust,
                               'eigen': eig})

# %%

# TODO check that this is okay

# From the metrics_avg_df, per row, calculate the principal component of the 5 metrics and save it in a column PC1:

# create a PCA object with 5 components
pca = PCA(n_components=5)

# fit the PCA model to the metrics data 
# and transform the metrics data to the principal component space
pcs = pca.fit_transform(metrics_avg_df)

# add the PC columns to the metrics_avg_df DataFrame
for i in range(pcs.shape[1]):
    metrics_avg_df[f'PC{i+1}'] = pcs[:, i]


metrics_avg_df['pet'] = pet_avg

# put the labels as a column in the df metrics_avg_df
metrics_avg_df['Network Labels'] = labels

nets_names = {
            0: 'Visual',
            1: 'Somatomotor',
            2: 'Dorsal Attention',
            3: 'Ventral Attention', # "_SalVentAttn_" : 
            4: 'Limbic',
            5: 'Control',
            6: 'Default'
            }
# make a new column network_names that has the network names from the dictionary mapping nets_names
metrics_avg_df['Network Names'] = metrics_avg_df['Network Labels'].map(nets_names)

metrics_avg_df.to_csv(os.path.join(base_path,'network-metrics/metrics_avg_fz.csv'))
#%%
################################################################################
#################### INFORMATION THEORY METRICS ################################
################################################################################ 

timeseries_path = os.path.join(base_path, 'concatenated_time-series')

# Get all the files that are in the timeseries_path
files = os.listdir(timeseries_path)
files.sort()
print(files)

#%%
# Set the variables
n_surrogates = 2
estimator = 'Gaussian' # Kraskov

# Run the Calculators
cal = Calculators()
cal.activate()
cal.mi_init(estimator)
cal.te_init(estimator)

#%%

for metric in ['transfer-entropy', 'mutual-information']:
    print(f'Calculating {metric} using {estimator} estimator.')

    for file in files: 
        print(f'Calculating {metric} for {file}.')  

        # Read in the data but don't put the first row as column names:
        data = pd.read_csv(os.path.join(timeseries_path, file), header=None)
        data = data.to_numpy()

        # TODO maybe timepoints = 1452 will not be the case for all subjects
        # 1452 rows for timepoints and 200 columns for regions
        regions = data.shape[1]
        timepoints = data.shape[0]
        # assert data.shape == (timepoints, regions)

        # Append the regions and timepoints for the subject in a df also with the file name using iloc:
        # TODO

        metric_pathname = os.path.join(base_path, 
                                        'information-theory_outputs', 
                                        file.split('_')[0]+f'_{estimator}_{metric}.csv')
        
        metric_corrected_pathname = os.path.join(base_path,
                                                'information-theory_outputs',
                                                file.split('_')[0]+f'_{estimator}_{metric}_corrected.csv')
        
        pvalue_pathname = os.path.join(base_path,
                                        'information-theory_outputs',
                                        file.split('_')[0]+f'_{estimator}_{metric}_pvalue.csv')
        
        surrogate_dist_pathname = os.path.join(base_path,
                                                'information-theory_outputs',
                                                file.split('_')[0]+f'_{estimator}_{metric}_surrogate_dist.csv')
        

        # Check if the metric file already exists and if yes skip the calculation
        if os.path.isfile(metric_pathname):
            print(f'{metric} file already exists. Skipping calculation and reading existing files.')

            # Read in the metric file : regular and corrected
            r_matrix = pd.read_csv(metric_pathname)
            r_matrix = r_matrix.to_numpy()

            r_matrix_corrected = pd.read_csv(metric_corrected_pathname)
            r_matrix_corrected = r_matrix_corrected.to_numpy()

        else:
            print(f'{metric} file does not exist. Calculating now.')
            # Run the calculator for each brain region pair
            
            r_matrix = np.zeros((regions, regions))
            r_matrix_corrected = np.zeros((regions, regions))
            pvalue = np.zeros((regions, regions))
            surrogate_dist = np.zeros((regions, regions, n_surrogates))

            # start a time calculation
            start = time.time()

            for i in range(regions):
                if metric == 'mutual-information':
                    for j in range(i):
                            r_matrix[i,j], pvalue[i,j], surrogate_dist[i,j,:] = cal.mi_calc(data[:,i], 
                                                                                            data[:,j], 
                                                                                            surrogates=n_surrogates)
                            # Calculate a mean of the surrogate distribution 
                            # and take this mean 2D matrix from the r_matrix
                            r_matrix_corrected[i,j] = r_matrix[i,j] - np.mean(surrogate_dist[i,j,:])

                elif metric == 'transfer-entropy':
                    for j in range(regions):
                        if i != j:
                            r_matrix[i,j], pvalue[i,j], surrogate_dist[i,j,:] = cal.te_calc(data[:,i], 
                                                                                            data[:,j], 
                                                                                            surrogates=n_surrogates)
                            r_matrix_corrected[i,j] = r_matrix[i,j] - np.mean(surrogate_dist[i,j,:])

            if estimator == 'Gaussian':
                # set diagonal to 0 (otherwise MI is 1)
                np.fill_diagonal(r_matrix, 0)
                np.fill_diagonal(r_matrix_corrected, 0)
            
            # end the time calculation
            end = time.time()
            print(f'Time elapsed: {end-start}')

            # Save the df to csv of the metric
            df = pd.DataFrame(r_matrix)
            df.to_csv(metric_pathname, index=False)

            df_corrected = pd.DataFrame(r_matrix_corrected)
            df_corrected.to_csv(metric_corrected_pathname, index=False)

            df_pvalue = pd.DataFrame(pvalue)
            df_pvalue.to_csv(pvalue_pathname, index=False)

            # Todo, fix this
            # df_surrogate_dist = pd.DataFrame(surrogate_dist)
            # df_surrogate_dist.to_csv(surrogate_dist_pathname, index=False)

        if plot:     
            # Plot and save the histogram of the metric
            plot_histogram(matrix=r_matrix, 
                        estimator=estimator, 
                        metric=metric, 
                        description='')
            plot_histogram(matrix=r_matrix_corrected, 
                        estimator=estimator, 
                        metric=metric, 
                        description='corrected')

            # Plot and save the matrix of metric
            plot_matrix(matrix=r_matrix, 
                        estimator=estimator, 
                        metric=metric, 
                        description='')
            plot_matrix(matrix=r_matrix_corrected, 
                        estimator=estimator, 
                        metric=metric, 
                        description='corrected')        

        if metric == 'mutual-information':
            # Remove the upper triangle of the r_matrix matrix
            r_matrix_lower_triangle = np.tril(r_matrix, k=-1)

            if plot: plot_matrix(matrix=r_matrix_lower_triangle, 
                        estimator=estimator, 
                        metric=metric,
                        description='lower triangle')        

            r_matrix_corrected_lower_triangle = np.tril(r_matrix_corrected, k=-1)
            if plot: plot_matrix(matrix=r_matrix_corrected_lower_triangle, 
                        estimator=estimator, 
                        metric=metric,
                        description='corrected lower triangle')

            # Sum the lower triangle per region of interest
            r_matrix_sum = np.sum(r_matrix_lower_triangle, axis=0)
            if plot: plot_histogram(matrix=r_matrix_sum, 
                           estimator=estimator, 
                           metric=metric, 
                           description='lower triangle sum')

            r_matrix_corrected_sum = np.sum(r_matrix_corrected_lower_triangle, axis=0)
            if plot: plot_histogram(matrix=r_matrix_corrected_sum, 
                           estimator=estimator, 
                           metric=metric, 
                           description='corrected lower triangle sum')

            # Save the sum of the lower triangle to a csv
            df_sum = pd.DataFrame(r_matrix_sum)
            df_sum.to_csv(os.path.join(base_path,
                                        'information-theory_outputs',
                                        file.split('_')[0]+f'_{estimator}_{metric}_sum.csv'),
                            index=False)
            
            df_corrected_sum = pd.DataFrame(r_matrix_corrected_sum)
            df_corrected_sum.to_csv(os.path.join(base_path,
                                            'information-theory_outputs',
                                            file.split('_')[0]+f'_{estimator}_{metric}_corrected_sum.csv'),
                                index=False)

#%%
################################################################################
#################### MULTIMODAL COMBINATION ####################################
################################################################################ 

# Spin nulls
atlas_surf = dlabel_to_gifti(os.path.dirname(os.getcwd()) + '/schaefer_2018/Schaefer2018_400_7N_space-fsLR_den-32k.dlabel.nii')
nulls = alexander_bloch(metrics_avg_df['pet'], atlas='fsLR', density='32k', parcellation=atlas_surf,
                        n_perm=10000, seed=0)

#%% Network metrics
metrics = ['participation_coef', 
           'degree', 
           'strength', 
            'clustering', 'eigen', 
           'PC1', 'PC2', 'PC3', 'PC4', 'PC5']

for metric in metrics:
    print(f'Plotting {metric} vs PET.')
    xlabel = f'Node {metric}'

    r, p = stats.pearsonr(metrics_avg_df[metric], metrics_avg_df['pet'])
    r, p_spin = compare_images(metrics_avg_df['pet'], metrics_avg_df[metric], nulls=nulls)

    ax = sns.jointplot(x=metric, y='pet', data=metrics_avg_df, 
                       kind='reg', scatter_kws={'s': 10})
    plt.xlabel(xlabel)
    plt.ylabel('PET mean CMRglc')
    ax.fig.suptitle(f'r2 = {r**2:.2f} \n$p_{{uncorrected}}$ = {p:.4f}, $p_{{spin}}$ = {p_spin:.4f}', y=1.05)
    plt.show()

    ax = sns.jointplot(x=metric, y='pet', data=metrics_avg_df, 
                    hue='Network Names',
                    kind='scatter', 
                    palette='colorblind',)
    plt.xlabel(xlabel)
    plt.ylabel('PET mean CMRglc')
    ax.fig.suptitle(f'r2 = {r**2:.2f} \n$p_{{uncorrected}}$ = {p:.4f}, $p_{{spin}}$ = {p_spin:.4f}', y=1.05)
    plt.show()
    
# %% Information theory metrics

metrics = ['te', 'mi']
group_info_theory_path = os.path.join('./output-group/')

for metric in metrics:
    print(f'Plotting {metric} vs PET.')
    xlabel = f'Node {metric}'
    
    metric_path = os.path.join(group_info_theory_path, f'{metric}.npy')
    metrics_avg_df[metric] = np.load(metric_path)
    
    
    r, p = stats.pearsonr(metrics_avg_df[metric], metrics_avg_df['pet'])
    r, p_spin = compare_images(metrics_avg_df[metric], metrics_avg_df['pet'], nulls=nulls)

    ax = sns.jointplot(x=metric, y='pet', data=metrics_avg_df, 
                       kind='reg', scatter_kws={'s': 10})
    plt.xlabel(xlabel)
    plt.ylabel('PET mean CMRglc')
    ax.fig.suptitle(f'r2 = {r**2:.2f} \n$p_{{uncorrected}}$ = {p:.4f}, $p_{{spin}}$ = {p_spin:.4f}', y=1.05)
    plt.show()

    ax = sns.jointplot(x=metric, y='pet', data=metrics_avg_df, 
                    hue='Network Names',
                    kind='scatter', 
                    palette='colorblind',)
    plt.xlabel(xlabel)
    plt.ylabel('PET mean CMRglc')
    ax.fig.suptitle(f'r2 = {r**2:.2f} \n$p_{{uncorrected}}$ = {p:.4f}, $p_{{spin}}$ = {p_spin:.4f}', y=1.05)
    plt.show()





def read_atlases_timeseries(atlas, sub, sub_task, base_path):
    """
    sub-003_task-rest_bold_desc-roi_timeseries_Schaefer400.npy
    sub-003_task-rest_pet_desc-meanCMRglc_Schaefer400.npy
    """
    
    roi_time_series_name = os.path.join(base_path, 
            f'{atlas}/sub-{sub}_task-{sub_task}_bold_desc-roi_timeseries_{atlas}.npy')
    roi_time_series_cort = np.load(roi_time_series_name, allow_pickle=True)
    
    pet_name = os.path.join(base_path,
            f'{atlas}/sub-{sub}_task-{sub_task}_pet_desc-*glc_{atlas}.npy')
    pet_name = glob(pet_name)[0]
    pet = np.load(pet_name, allow_pickle=True)

    return roi_time_series_cort, pet

def plot_histogram(file=None, matrix=None, estimator='Gaussian', metric='', description=''):

    plt.figure()
    plt.hist(matrix.flatten())#, bins=100)  
    plt.xlabel(f'{metric} (without the diagonal) {description}')
    plt.ylabel('Count')
    plt.title(f'{estimator} estimator')
    
    if description != '':
        description = '_' + description
        description = description.replace(' ', '_')
    metric = metric.replace(' ', '-')

    # Save the histogram
    hist_pathname = os.path.join(base_path, 
                                 'information-theory_outputs', 
                                 file.split('_')[0]+f'_{estimator}-estimator{description}_{metric}-histogram.png')
    plt.savefig(hist_pathname)
    plt.close()

def plot_matrix(file=None, matrix=None, estimator='Gaussian', metric='', description=''):
    # Plot the matrix of mutual information
    plt.figure()
    plt.matshow(matrix)
    plt.colorbar()
    plt.title(f'{metric} using {estimator} estimator {description}')
    plt.xlabel('Regions of Interest')
    plt.ylabel('Regions of Interest')

    if description != '':
        description = '_' + description
        description = description.replace(' ', '_')
    
    # Save the plot
    plot_pathname = os.path.join(base_path, 
                                 'information-theory_outputs', 
                                 file.split('_')[0]+f'_{estimator}-estimator{description}_{metric}-matrix.png')
    plt.savefig(plot_pathname)
    plt.close()
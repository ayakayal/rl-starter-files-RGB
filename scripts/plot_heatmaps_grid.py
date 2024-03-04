import csv
import numpy as np
import matplotlib.pyplot as plt
import ast
import os

def process_row(row_number,rows_saved,grid_rows,grid_cols):
    keys=rows_saved[0]
# Convert each string to a tuple
    keys = [ast.literal_eval(item) for item in keys]
    values= rows_saved[row_number]
    values=[float(item) for item in values]
    N = sum(values)
    matrix_dict = np.zeros((grid_rows, grid_cols))
    for key, value in zip(keys, values):
            if key[0] < grid_cols and key[1] < grid_rows:
                matrix_dict[key[1],key[0]]= value/N
           
    #clip high count values for better visualisation of lower values 
    for i in range(grid_rows):
        for j in range(grid_cols):
            if matrix_dict[i,j] > 0.05:
                matrix_dict[i,j]=0.05
               
    max_matrix_dict= 0.05
    return matrix_dict,max_matrix_dict #return new_matrix for RedBlue

def plot_heatmap(matrix_dict, Vmax, ax):
    # Plot the heatmap
    cax = ax.imshow(matrix_dict, cmap='inferno', interpolation='nearest', vmin=0, vmax=Vmax)
    # Add color bar
    #plt.colorbar(cax, ax=ax, label='Value')
    # Turn off the tick labels
    ax.set_xticks([])
    ax.set_yticks([])


def read_csv_and_retrieve(file_name,grid_rows,grid_cols,row_number,RGB):
    current_directory= os.getcwd()
    if RGB=='True':
        file_path = os.path.join(current_directory,'final_heatmaps_Desktop', 'RGB',f'{file_name}')
    else:
        file_path = os.path.join(current_directory,'final_heatmaps_Desktop', 'NonRGB',f'{file_name}')
        
    with open(file_path,'r') as csv_file:
        reader = csv.reader(csv_file)
        rows_saved=[r for r in reader]
        matrix_dict1,max_matrix_dict1=process_row(row_number,rows_saved,grid_rows,grid_cols)
    return matrix_dict1,max_matrix_dict1


def heatmap_grid(filename_algos, grid_rows, grid_cols, row_numbers,fig_name,RGB):
    num_rows = len(row_numbers)
    num_cols = len(filename_algos)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows), squeeze=False)

    # Common row of titles
    if RGB=='True':
        titles = ["PPO", "SimHash", "ICM", "Max Entropy", "DIAYN-pretraining", "DIAYN-finetuning"]
    else:
         titles = ["PPO", "State Count", "ICM", "Max Entropy", "DIAYN-pretraining", "DIAYN-finetuning"]

    for i, title in enumerate(titles):
        axes[0, i].set_title(title)
        axes[0, i].title.set_size(30) 

    for i, filename in enumerate(filename_algos):
        for j, row_number in enumerate(row_numbers):
            matrix_dict, max_matrix_dict = read_csv_and_retrieve(filename, grid_rows, grid_cols, row_number,RGB)
            ax = axes[j, i]
            plot_heatmap(matrix_dict, max_matrix_dict, ax)
            ax_row = axes[j, :]  # Access all Axes in the current row
            ax_row[0].set_ylabel(f"T{j+1}", size=30)  # Set title on the first Axes of the row

    plt.tight_layout()

    # Add common color bar
    cbar = plt.colorbar(axes[0, 0].images[0], ax=axes.ravel().tolist(), fraction=0.03, pad=0.04)
    cbar.ax.tick_params(labelsize=25) 
    # Access the axes object of the colorbar
    cax = cbar.ax

    # Set the label position to the left
    cax.yaxis.set_label_position('left')
 

    # Set tick positions to the left
    cax.yaxis.set_ticks_position('left')
 

    plt.savefig(fig_name)
    plt.close()



###########################################################################################
filename_algos = ['state_MiniGrid-Empty-16x16-v0_ppo_seed10005_ir0.0_ent0.0005.csv',
                   'state_MiniGrid-Empty-16x16-v0_ppo_simhash2_seed10005_ir0.005_ent0.0005.csv',
                   'state_MiniGrid-Empty-16x16-v0_ppo_icm_alain_seed10005_ir0.05_ent0.0005.csv', 
                   'state_MiniGrid-Empty-16x16-v0_ppo_entropy_seed10005_ir0.0005_ent0.0005.csv',
                   'state_MiniGrid-Empty-16x16-v0_ppo_diayn_seed10005_ir0.01_ent0.0005_sk10_dis0.0003.csv',
                   'state_MiniGrid-Empty-16x16-v0_ppo_diayn_seed10005_ir0.0_ent0.0005_sk10_dis0.0003.csv']
row_numbers = [1, 5, 97]
grid_rows = 16
grid_cols = 16
fig_name='Empty_RGB.png'
RGB='True'

#################################################################################################
filename_algos = ['state_MiniGrid-DoorKey-8x8-v0_ppo_seed10005_ir0.0_ent0.0005_beaker.csv',
                   'state_MiniGrid-DoorKey-8x8-v0_ppo_simhash2_seed10005_ir0.005_ent0.0005.csv',
                   'state_MiniGrid-DoorKey-8x8-v0_ppo_icm_alain_seed10005_ir0.05_ent0.0005.csv', 
                   'state_MiniGrid-DoorKey-8x8-v0_ppo_entropy_seed10005_ir0.0005_ent0.0005.csv',
                   'state_MiniGrid-DoorKey-8x8-v0_ppo_diayn_seed10005_ir0.01_ent0.0005_sk10_dis0.0003.csv',
                   'state_MiniGrid-DoorKey-8x8-v0_ppo_diayn_seed10005_ir0.0_ent0.0005_sk10_dis0.0003.csv']
row_numbers = [1, 5, 97]
grid_rows = 8
grid_cols = 8
fig_name='DoorKey_RGB.png'
RGB='True'
#change threshold to 0.1 and max to 0.1
#heatmap_grid(filename_algos, grid_rows, grid_cols, row_numbers,fig_name,RGB)
######################################################################################################
filename_algos = ['state_MiniGrid-RedBlueDoors-8x8-v0_ppo_seed10005_ir0.0_ent0.0005.csv',
                   'state_MiniGrid-RedBlueDoors-8x8-v0_ppo_simhash2_seed10005_ir0.005_ent0.0005.csv',
                   'state_MiniGrid-RedBlueDoors-8x8-v0_ppo_icm_alain_seed10005_ir0.05_ent0.0005.csv', 
                   'state_MiniGrid-RedBlueDoors-8x8-v0_ppo_entropy_seed10005_ir0.0005_ent0.0005.csv',
                   'state_MiniGrid-RedBlueDoors-8x8-v0_ppo_diayn_seed10005_ir0.01_ent0.0005_sk10_dis0.0003.csv',
                   'state_MiniGrid-RedBlueDoors-8x8-v0_ppo_diayn_seed10005_ir0.0_ent0.0005_sk10_dis0.0003.csv']
row_numbers = [1, 5, 97]
grid_rows = 8
grid_cols = 16
fig_name='RedBlueDoors_RGB.png'
RGB='True'
#heatmap_grid(filename_algos, grid_rows, grid_cols, row_numbers,fig_name,RGB)
#######################################################################################################
filename_algos = ['state_MiniGrid-FourRooms-v0_ppo_seed10005_ir0.0_ent0.0005_beaker.csv',
                   'state_MiniGrid-FourRooms-v0_ppo_simhash2_seed10005_ir0.005_ent0.0005.csv',
                   'state_MiniGrid-FourRooms-v0_ppo_icm_alain_seed10005_ir0.05_ent0.0005_beaker.csv', 
                   'state_MiniGrid-FourRooms-v0_ppo_entropy_seed10005_ir0.0005_ent0.0005_beaker.csv',
                   'state_MiniGrid-FourRooms-v0_ppo_diayn_seed10005_ir0.01_ent0.0005_sk10_dis0.0003_corrected.csv',
                   'state_MiniGrid-FourRooms-v0_ppo_diayn_seed10005_ir0_ent0.0005_sk10_dis0.0003_corrected.csv']
row_numbers = [1, 5, 97]
grid_rows = 19
grid_cols = 19
fig_name='FouRooms_RGB.png'
RGB='True'
#heatmap_grid(filename_algos, grid_rows, grid_cols, row_numbers,fig_name,RGB)
######################################################################################################
filename_algos = ['state_MiniGrid-Empty-16x16-v0_ppo_seed10005_ir0.0_ent0.0005.csv',
                   'state_MiniGrid-Empty-16x16-v0_ppo_state_count_seed10005_ir0.005_ent0.0005.csv',
                   'state_MiniGrid-Empty-16x16-v0_ppo_icm_alain_seed10005_ir0.05_ent0.0005.csv', 
                   'state_MiniGrid-Empty-16x16-v0_ppo_entropy_seed10005_ir0.0005_ent0.0005.csv',
                   'state_MiniGrid-Empty-16x16-v0_ppo_diayn_seed10005_ir0.01_ent0.0005_sk10_dis0.0003.csv',
                   'state_MiniGrid-Empty-16x16-v0_ppo_diayn_seed10005_ir0.0_ent0.0005_sk10_dis0.0003.csv']
row_numbers = [1, 5, 97]
grid_rows = 16
grid_cols = 16
fig_name='Empty_NonRGB.png'
RGB='False'
#heatmap_grid(filename_algos, grid_rows, grid_cols, row_numbers,fig_name,RGB)
###################################################################################
filename_algos = ['state_MiniGrid-DoorKey-16x16-v0_ppo_seed10005_ir0.0_ent0.0005.csv',
                   'state_MiniGrid-DoorKey-16x16-v0_ppo_state_count_seed10005_ir0.005_ent0.0005.csv',
                   'state_MiniGrid-DoorKey-16x16-v0_ppo_icm_alain_seed10005_ir0.05_ent0.0005.csv', 
                   'state_MiniGrid-DoorKey-16x16-v0_ppo_entropy_seed10005_ir0.0005_ent0.0005.csv',
                   'state_MiniGrid-DoorKey-16x16-v0_ppo_diayn_seed10005_ir0.01_ent0.0005_sk10_dis0.0003.csv',
                   'state_MiniGrid-DoorKey-16x16-v0_ppo_diayn_seed10005_ir0.0_ent0.0005_sk10_dis0.0003.csv']
row_numbers = [1, 5, 97]
grid_rows = 16
grid_cols = 16
fig_name='DoorKey_NonRGB.png'
RGB='False'
#heatmap_grid(filename_algos, grid_rows, grid_cols, row_numbers,fig_name,RGB)
########################################################################################
filename_algos = ['state_MiniGrid-RedBlueDoors-8x8-v0_ppo_seed10005_ir0.0_ent0.0005.csv',
                   'state_MiniGrid-RedBlueDoors-8x8-v0_ppo_state_count_seed10005_ir0.005_ent0.0005.csv',
                   'state_MiniGrid-RedBlueDoors-8x8-v0_ppo_icm_alain_seed10005_ir0.05_ent0.0005.csv', 
                   'state_MiniGrid-RedBlueDoors-8x8-v0_ppo_entropy_seed10005_ir0.0005_ent0.0005.csv',
                   'state_MiniGrid-RedBlueDoors-8x8-v0_ppo_diayn_seed1_ir0.01_ent0.0005_sk10_dis0.0003.csv',
                   'state_MiniGrid-RedBlueDoors-8x8-v0_ppo_diayn_seed10005_ir0.0_ent0.0005_sk10_dis0.0003.csv']
row_numbers = [1, 5, 97]
grid_rows = 8
grid_cols = 16
fig_name='RedBlueDoors_NonRGB.png'
RGB='False'
#heatmap_grid(filename_algos, grid_rows, grid_cols, row_numbers,fig_name,RGB)
##########################################################################################
filename_algos = ['state_MiniGrid-FourRooms-v0_ppo_seed10005_ir0.0_ent0.0005.csv',
                   'state_MiniGrid-FourRooms-v0_ppo_state_count_seed10005_ir0.005_ent0.0005.csv',
                   'state_MiniGrid-FourRooms-v0_ppo_icm_alain_seed10005_ir0.05_ent0.0005.csv', 
                   'state_MiniGrid-FourRooms-v0_ppo_entropy_seed10005_ir0.0005_ent0.0005.csv',
                   'state_MiniGrid-FourRooms-v0_ppo_diayn_seed10005_ir0.01_ent0.0005_sk10_dis0.0003.csv',
                   'state_MiniGrid-FourRooms-v0_ppo_diayn_seed10005_ir0.0_ent0.0005_sk10_dis0.0003.csv']
row_numbers = [1, 5, 97]
grid_rows = 19
grid_cols = 19
fig_name='FouRooms_NonRGB.png'
RGB='False'
heatmap_grid(filename_algos, grid_rows, grid_cols, row_numbers,fig_name,RGB)
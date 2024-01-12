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
            matrix_dict[key[1],key[0]]= value/N
            print('dict before',key[1],' ',key[0],' ',matrix_dict[key[1],key[0]])
    if clip==1:  
         for i in range(grid_rows):
              for j in range(grid_cols):
                if matrix_dict[j,i] > 0.05:
                    matrix_dict[j,i]=0.05
                print('dict after',j,' ',i,' ',matrix_dict[j,i]) 
                                
         
    # print(matrix_dict[1,10]*N)
    max_matrix_dict= np.max(matrix_dict)
    print('max',max_matrix_dict)
    #max_matrix_dict= 0.05
    # # # #add this for RedBlue only because I did mistake: 16x16 csv file saved
    #new_matrix= np.zeros((8, 16))
    #uncomment for Doorkey 8x8
    #new_matrix= np.zeros((8, 8))
    
    #uncomment for RedlueDoors Only
    # for i in range(8):
    #     for j in range(16):
    #     # Copy valid elements based on the specified condition
        
    #         new_matrix[i, j] = matrix_dict[i, j]
    #uncomment for Doorkey
    # for i in range(8):
    #     for j in range(8):
    #         new_matrix[i, j] = matrix_dict[i, j]

    return matrix_dict,max_matrix_dict #return new_matrix for RedBlue
def plot_heatmap(matrix_dict,Vmax,filename):
    fig, ax = plt.subplots()
    # Plot the heatmap
    cax = ax.imshow(matrix_dict, cmap='inferno', interpolation='nearest',vmin=0, vmax=Vmax) # you can choose other color maps like 'coolwarm', 'Blues', etc.
    # Add color bar
    cbar = plt.colorbar(cax, label='Value')
    # plt.title(' Heatmap in Minigrid')
    #plt.xlabel('Grid X Coordinate')
    #plt.ylabel('Grid Y Coordinate')
    # Turn off the tick labels
    ax.set_xticks([])  # Turn off x-axis tick labels
    ax.set_yticks([])  # Turn off y-axis tick labels
    # Save the plot to a file 
    plt.savefig(filename, format='png', dpi=300)




def read_csv_and_retrieve(file_name,grid_rows,grid_cols):
    current_directory= os.getcwd()
    file_path = os.path.join(current_directory,'heatmaps', f'{file_name}')
    with open(file_path,'r') as csv_file:
        reader = csv.reader(csv_file)
        rows_saved=[r for r in reader]
        matrix_dict1,max_matrix_dict1=process_row(240,rows_saved,grid_rows,grid_cols)
    return matrix_dict1,max_matrix_dict1


#filenames_algos=['state_MiniGrid-FourRooms-v0_ppo_state_count_seed10005_ir0.005_ent0.0005.csv','state_MiniGrid-FourRooms-v0_ppo_diayn_seed10005_ir0.01_ent0.0005_sk10_dis0.0003.csv','state_MiniGrid-FourRooms-v0_ppo_icm_alain_seed10005_ir0.05_ent0.0005.csv','state_MiniGrid-FourRooms-v0_ppo_entropy_seed10005_ir0.0005_ent0.0005.csv']
#filenames_algos=['state_MiniGrid-FourRooms-v0_ppo_entropy_seed10005_ir0.0005_ent0.0005.csv']
#filenames_algos=['state_MiniGrid-DoorKey-16x16-v0_ppo_state_count_seed10005_ir0.005_ent0.0005.csv','state_MiniGrid-DoorKey-16x16-v0_ppo_icm_alain_seed10005_ir0.05_ent0.0005.csv','state_MiniGrid-DoorKey-16x16-v0_ppo_diayn_seed10005_ir0.01_ent0.0005_sk10_dis0.0003.csv','state_MiniGrid-DoorKey-16x16-v0_ppo_entropy_seed10005_ir0.0005_ent0.0005.csv']
#filenames_algos=['state_MiniGrid-RedBlueDoors-8x8-v0_ppo_state_count_seed10005_ir0.005_ent0.0005.csv','state_MiniGrid-RedBlueDoors-8x8-v0_ppo_icm_alain_seed10005_ir0.05_ent0.0005.csv','state_MiniGrid-RedBlueDoors-8x8-v0_ppo_diayn_seed10005_ir0.01_ent0.0005_sk10_dis0.0003.csv','state_MiniGrid-RedBlueDoors-8x8-v0_ppo_entropy_seed10005_ir0.0005_ent0.0005.csv']
filenames_algos=['state_MiniGrid-FourRooms-v0_ppo_diayn_seed10_ir0.01_ent0.0005_sk10_dis0.0003_corrected.csv']
global_max=0
grid_rows=19
grid_cols=19
clip=1
# for filename in filenames_algos:
#         matrix_dict,max_matrix_dict=read_csv_and_retrieve(filename,grid_rows,grid_cols)
#         if max_matrix_dict>global_max:
#              global_max=max_matrix_dict

for filename in filenames_algos:
    matrix_dict,max_matrix_dict=read_csv_and_retrieve(filename,grid_rows,grid_cols)       
    plot_heatmap(matrix_dict,max_matrix_dict,f'/home/rmapkay/rl-starter-files-RGB/rl-starter-files/heatmaps/{filename}_t240.png')  #it was global max
        

#I added this
import sys
import os
import csv
sys.path.append('/home/rmapkay/rl-starter-files')
import argparse

import time
import datetime
import torch_ac
import tensorboardX


import utils
from utils import device
from model import ACModel
import torch
import wandb
from array2gif import write_gif
import numpy as np
import matplotlib.pyplot as plt


# Add the parent directory of train.py to the system path

# Parse arguments

parser = argparse.ArgumentParser()

# General parameters

parser.add_argument("--gpu", type=int, default=0,
                    help="GPU number to use")

parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--folder-name", default=None,
                    help="name of the folder inside storage")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")

# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="learning rate (default: 0.001)") #I changed it to 0.0001 bcz it was the best with ICM
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")
parser.add_argument("--ir-coef", type=float, default=0,
                    help="intrinsic reward coefficient (default: 0.01)")
parser.add_argument("--num-skills", type=int, default=10,
                    help="number of skills for DIAYN only (default:4)")

parser.add_argument("--disc-lr", type=float, default=0.0003,
                    help="for DIAYN only")
parser.add_argument("--singleton-env", default='False',
                    help="singleton environment training (default: False)")

parser.add_argument("--RGB", default='False',
                    help="using RGB observations")
parser.add_argument("--pretraining", default='False',
                    help="If we are pretraining with intrinsic rewards only")
parser.add_argument("--pretrained-model-name", default=None,
                    help="directoy of the pretrained model to use instead of starting from scratch")
parser.add_argument("--folder-name-pretrained-model", default=None,
                    help="name of the folder containing the pretrained model that I would like to load")
parser.add_argument("--save-heatmaps", default='False',
                    help="saving heatmaps of state visitation and intrinsic rewards while training")

args = parser.parse_args()


def log_dict_csv(dict,is_first_log=False,default_model_name='dict'):
    file_path = os.path.join('heatmaps', f'{default_model_name}.csv')
    #print('file path',file_path)
    with open(file_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if is_first_log:
            keys = [key for key, _ in dict.items()]
            writer.writerow(keys)
        values = [value for _ , value in dict.items()]
        
        # Write values from the second dictionary
        writer.writerow(values)



# def save_heatmap_matrix(state_visitation_counts,heatmap_plot):
    

# # Define the CSV file path
#     csv_file_path = f"state_visitation_counts_{heatmap_plot}.csv"

#     # Open the CSV file in write mode
#     with open(csv_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
        
#         # Write the header row
#         writer.writerow(["State", "Visitation Count"])
        
#         # Write the data from the dictionary
#         for state, count in state_visitation_counts.items():
#             writer.writerow([state, count])


# def plot_heatmap(state_visitation,heatmap_plot): #state_visitation is the matrix to plot
#     grid_rows, grid_cols = 8,8 #change according to your grid size
#     N = sum(state_visitation.values())
#     # Create a matrix from the state visitation dictionary
#     state_visitation_matrix = np.zeros((grid_rows, grid_cols))
#     for (row, col), count in state_visitation.items():
#         state_visitation_matrix[col, row] = count/N
#     fig, ax = plt.subplots()
#     # Plot the heatmap
#     cax = ax.imshow(state_visitation_matrix, cmap='inferno', interpolation='nearest',vmin=0, vmax=0.5) # you can choose other color maps like 'coolwarm', 'Blues', etc.
#     # Add color bar
#     cbar = plt.colorbar(cax, label='Visitation Count')
#     plt.title('State Visitation Heatmap in Minigrid')
#     plt.xlabel('Grid X Coordinate')
#     plt.ylabel('Grid Y Coordinate')
#     # Turn off the tick labels
#     ax.set_xticks([])  # Turn off x-axis tick labels
#     ax.set_yticks([])  # Turn off y-axis tick labels
#     # Save the plot to a file 
#     plt.savefig(f'heatmap_DoorKey_DIAYN_repeat_{heatmap_plot}.png', format='png', dpi=300)

# def plot_all_heatmaps(heatmap_time_step): #state_visitation is the matrix to plot
#     #print('heatmap_time_step',heatmap_time_step)
#     grid_rows, grid_cols = 16,16 #change according to your grid size
#     max_value=0
#     for i in range(len(heatmap_time_step)):
#         state_visitation= heatmap_time_step[i]
#         N = sum(state_visitation.values())
#         # Create a matrix from the state visitation dictionary
#         state_visitation_matrix = np.zeros((grid_rows, grid_cols))
#         for (row, col), count in state_visitation.items():
#             state_visitation_matrix[col, row] = count/N
#         if np.max(state_visitation_matrix)>max_value:
#             max_value = np.max(state_visitation_matrix)
#     for j in range(len(heatmap_time_step)):
#         state_visitation= heatmap_time_step[j]
#         N = sum(state_visitation.values())
#         # Create a matrix from the state visitation dictionary
#         state_visitation_matrix = np.zeros((grid_rows, grid_cols))
#         for (row, col), count in state_visitation.items():
#             state_visitation_matrix[col, row] = count/N
#         fig, ax = plt.subplots()
#         # Plot the heatmap
#         cax = ax.imshow(state_visitation_matrix, cmap='inferno', interpolation='nearest',vmin=0, vmax=max_value) # you can choose other color maps like 'coolwarm', 'Blues', etc.
#         # Add color bar
#         cbar = plt.colorbar(cax, label='Visitation Count')
#         plt.title('State Visitation Heatmap in Minigrid')
#         plt.xlabel('Grid X Coordinate')
#         plt.ylabel('Grid Y Coordinate')
#         # Turn off the tick labels
#         ax.set_xticks([])  # Turn off x-axis tick labels
#         ax.set_yticks([])  # Turn off y-axis tick labels
#         # Save the plot to a file 
#         plt.savefig(f'heatmap_Doorkey_16_entropy_{j}.png', format='png', dpi=300)

def main():
    
   
    torch.cuda.set_device(args.gpu) 
    args.mem = args.recurrence > 1

    # Set run dir

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    #default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"
    if args.algo=="ppo_diayn" or args.algo=="ppo_diayn_v2":
        default_model_name=f"{args.env}_{args.algo}_seed{args.seed}_ir{args.ir_coef}_ent{args.entropy_coef}_sk{args.num_skills}_dis{args.disc_lr}"
    else:
        default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_ir{args.ir_coef}_ent{args.entropy_coef}"

    model_name = args.model or default_model_name
    #model_dir = utils.get_model_dir(model_name)
    model_dir = utils.get_model_dir_folder(args.folder_name,model_name)
    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    txt_logger.info(f"Device: {device}\n")

    # Load environments

    if args.singleton_env == 'False': 
        envs = []
        for i in range(args.procs):
            envs.append(utils.make_env(args.env, args.seed + 10000 * i, args.RGB))
    else:
        envs = []
        for i in range(args.procs):
            env = utils.singleton_make_env(args.env, 10005, args.RGB)
            envs.append(env)
        
    txt_logger.info("Environments loaded\n")

    # Load training status

    try:
        print('model dir',model_dir)
        status = utils.get_status(model_dir)
        print('status of old model loaded',status.keys())
        
    except OSError:
        status = {"num_frames": 0, "update": 0}
        print('status',status)
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor

    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    #print("envs[0].observation_space",envs[0].observation_space)
    #print('obs space', obs_space)
    #print('pre process obs space', preprocess_obss)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model
    if args.algo=="ppo_diayn" or args.algo=="ppo_diayn_v2":
        use_diayn=True
        args.save_interval=1
    else:
        use_diayn=False
    # print('use_diayn',use_diayn)
    acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text, use_diayn,args.num_skills,args.RGB)
    if "model_state" in status:
            acmodel.load_state_dict(status["model_state"])
            print('acmodel of old model loaded')
 
    # in case of DIAYN, this loads the pretrained model (AC weights only)
    if args.pretrained_model_name!= None:
        pretrained_model_dir = utils.get_model_dir_folder(args.folder_name_pretrained_model,args.pretrained_model_name)
        print('pretrained_model_dir',pretrained_model_dir)
        try:
            acmodel.load_state_dict(utils.get_model_state(pretrained_model_dir))
            print("Pretrained model state loaded successfully.")
            status = utils.get_status(pretrained_model_dir)
            print('status from pretrained model',status.keys())
          
        except Exception as e:
             print(f"Error loading pretraining model state: {str(e)}")

        # print(utils.get_status(model_dir))
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    # Load algo

    if args.algo == "a2c":
        algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_alpha, args.optim_eps, preprocess_obss)
        #print("frames per proc",args.frames_per_proc)
    elif args.algo == "ppo":
        algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, args.singleton_env,args.RGB, preprocess_obss)
    
    elif args.algo=="ppo_diayn":
        algo= torch_ac.PPOAlgoDIAYN(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, args.singleton_env,args.RGB,preprocess_obss,args.ir_coef, args.num_skills, args.disc_lr, args.pretraining)
    elif args.algo=="ppo_diayn_v2":
        algo= torch_ac.PPOAlgoDIAYNV2(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size,args.singleton_env,args.RGB, preprocess_obss,args.ir_coef, args.num_skills, args.disc_lr)
    elif args.algo=="ppo_state_count":
        algo= torch_ac.PPOAlgoStateCount(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size,args.singleton_env, args.RGB, preprocess_obss,args.ir_coef)
    elif args.algo=="ppo_entropy":
        algo= torch_ac.PPOAlgoEntropy(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, args.singleton_env, args.RGB, preprocess_obss,args.ir_coef)
    elif args.algo=="ppo_icm":
        algo= torch_ac.PPOAlgoICM(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size,args.RGB, preprocess_obss,args.ir_coef)
    elif args.algo=="ppo_icm_alain":
        algo= torch_ac.PPOAlgoICMAlain(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, args.singleton_env,args.RGB,preprocess_obss,args.ir_coef)
    
    elif args.algo=="ppo_pseudocount":
        algo= torch_ac.PPOAlgoPseudocount(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size,args.singleton_env,args.RGB, preprocess_obss,args.ir_coef)
    elif args.algo=="ppo_simhash":
        algo= torch_ac.PPOAlgoStateHash(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size,args.singleton_env,args.RGB, preprocess_obss,args.ir_coef)
    
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")
 # SAVE, in case u need to resume training, the observation, position dictionaries, and 1st/2nd/3rd reward found for all algos except DIAYN (we want to restart the dicts btw pretraining and testing) 
    if args.algo !="ppo_diayn":  
       
        algo.total_frames = status ["num_frames"]   
        # print('algo.total_frames', algo.total_frames)  
        if "found_reward" in status:
            algo.found_reward=status["found_reward"] 
        # print(" algo.found_reward", algo.found_reward)
        if "obs_dict" in status:
            algo.train_state_count= status["obs_dict"]
           
        if "pos_dict" in status:
            algo.state_visitation_pos=status["pos_dict"]
   # SAVE hash table for simhash only in case u need to resume training 
    if args.algo == "ppo_simhash":
        if "hash_dict" in status:
            algo.hash_function.hash= status ["hash_dict"]
    if args.algo == "ppo_icm_alain":
        
        if "embedding_network" in status:
         
            algo.im_module.state_embedding.load_state_dict(status["embedding_network"])
            algo.im_module.forward_dynamics.load_state_dict(status["forward_dynamics"])
            algo.im_module.inverse_dynamics.load_state_dict(status["inverse_dynamics"])
            algo.im_module.optimizer_state_embedding.load_state_dict(status["state_embedding_optimizer"])
            algo.im_module.optimizer_forward_dynamics.load_state_dict(status["forward_dynamics_optimizer"])
            algo.im_module.optimizer_inverse_dynamics.load_state_dict(status["inverse_dynamics_optimizer"])
           

        
    #log into wandb
    run = wandb.init(
        # Set the project where this run will be logged
        project="test_RGB", #test_RGB
        name= model_name, #it was default_model_name
        tags=[args.env, args.algo],
        # Track hyperparameters and run metadata
        config=vars(args)
      
        )
    

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()
    heatmap_plot=0
    heatmap_time_step={}
    is_first_log=True
    while num_frames < args.frames:

        # Update model parameters
        update_start_time = time.time()
        exps, logs1, stacked_frames = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])
            state_coverage=logs["state_coverage"] # partial obs coverage
            state_coverage_position=logs["state_coverage_position"] #(x,y) state coverage
            frame_first_reward=logs["frame_first_reward"]
            frame_second_reward=logs["frame_second_reward"]
            frame_third_reward=logs["frame_third_reward"]
            intrinsic_reward_per_frame=logs["reward_int_per_frame"]

            #wandb.log({"return_per_episode":  return_per_episode['mean']})

            # if args.algo=="ppo_diayn_debug":
            #     discriminator_loss= logs["discriminator_loss"]
            #     header = ["update", "frames", "FPS", "duration","discriminator_loss"]
            #     data = [update, num_frames, fps, duration, discriminator_loss]
            # else:
            header = ["update", "frames", "FPS", "duration","state_coverage","state_coverage_position","frame_first_reward","frame_second_reward","frame_third_reward","intrinsic_reward_per_frame"]
            data = [update, num_frames, fps, duration, state_coverage,state_coverage_position,frame_first_reward,frame_second_reward,frame_third_reward,intrinsic_reward_per_frame]

            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
        
            if args.algo== "ppo_state_count" or args.algo== "ppo_pseudocount" or args.algo=="ppo_simhash" or args.algo=="ppo_entropy":
                return_intrinsic_per_episode = utils.synthesize(logs["return_int_per_episode"])
                header += ["return_intrinsic_" + key for key in return_intrinsic_per_episode.keys()]
                data += return_intrinsic_per_episode.values()
                txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | SC {} | SCP {} | FR {} | SR {}| TR {}| r_int_frame {:.4f}| rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f} | r_int:μσmM {:.2f} {:.2f} {:.2f} {:.2f}"
                .format(*data))
            elif args.algo=="ppo_icm_alain":
                header += ["forward_dynamics_loss","inverse_dynamics_loss"]
                data += [logs["forward_dynamics_loss"],logs["inverse_dynamics_loss"]]
                return_intrinsic_per_episode = utils.synthesize(logs["return_int_per_episode"])
                header += ["return_intrinsic_" + key for key in return_intrinsic_per_episode.keys()]
                data += return_intrinsic_per_episode.values()
                txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | SC {} | SCP {} | FR {} | SR {}| TR {}| r_int_frame {:.4f}|  rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f} | FDL {:.3f} | IDL {:.3f}| r_int:μσmM {:.2f} {:.2f} {:.2f} {:.2f}"
                .format(*data))

            elif args.algo=="ppo_diayn" or args.algo=="ppo_diayn_v2":
                header += ["discriminator_loss"]
                data += [logs["discriminator_loss"]]
                return_intrinsic_per_episode = utils.synthesize(logs["return_int_per_episode"])
                header += ["return_intrinsic_" + key for key in return_intrinsic_per_episode.keys()]
                data += return_intrinsic_per_episode.values()
                header += ["KL_div"]
                data += [logs["KL_div"]]
                txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | SC {} | SCP {} | FR {} | SR {} | TR {} | r_int_frame {:.4f}| rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f} | DL {:.3f} | r_int:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | kl_div {:.5f}"
                .format(*data))
            else:

                txt_logger.info(    
                        "U {} | F {:06} | FPS {:04.0f} | D {} | SC {} | SCP {} | FR {} | SR {} | TR {}| r_int_frame {:.4f}|  rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                        .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()
            #log everything to wandb
            wandb.log(dict(zip(header, data)))
            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            for field, value in zip(header, data):
                tb_writer.add_scalar(field, value, num_frames)

        # Save status
    
        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict(), "obs_dict":algo.train_state_count, "pos_dict":algo.state_visitation_pos, "found_reward":algo.found_reward}
           

            if args.algo == "ppo_simhash":
                status["hash_dict"]= algo.hash_function.hash
                # print(' algo.hash_function.hash', algo.hash_function.hash)
            if args.algo == "ppo_icm_alain":
                status["embedding_network"]= algo.im_module.state_embedding.state_dict()
                status["forward_dynamics"]= algo.im_module.forward_dynamics.state_dict()
                status["inverse_dynamics"]= algo.im_module.inverse_dynamics.state_dict()
                status["state_embedding_optimizer"]=algo.im_module.optimizer_state_embedding.state_dict()
                status["forward_dynamics_optimizer"]=algo.im_module.optimizer_forward_dynamics.state_dict()
                status["inverse_dynamics_optimizer"]=algo.im_module.optimizer_inverse_dynamics.state_dict()
                # print('status of icm models loaded')


            # if args.algo=="a2c_state_count": #it means we are running algo state count
            #     print('yes algo state count')
            #     state_dict=algo.pass_models_parameters()
            #     status = {"num_frames": num_frames, "update": update,
            #           "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict(),"state_count":state_dict}
            # if args.algo=="a2c_state_count_single": #it means we are running algo state count
            #     print('yes algo state count single')
            #     state_dict=algo.pass_models_parameters()
            #     status = {"num_frames": num_frames, "update": update,
            #         "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict(),"state_count":state_dict}
            # if args.algo=="a2c_icm_inverse":
            #     print('yes algo icm inverse')
            #     embedding_network,forward_dynamics_model,inverse_dynamics_model,state_embedding_optimizer,forward_dynamics_optimizer,inverse_dynamics_optimizer= algo.pass_models_parameters()
            #     status = {"num_frames": num_frames, "update": update,
            #           "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict(),"embedding_network":embedding_network.state_dict(),"forward_dynamics_model":forward_dynamics_model.state_dict(),"inverse_dynamics_model":inverse_dynamics_model.state_dict(),
            #           "state_embedding_optimizer":state_embedding_optimizer.state_dict(),"forward_dynamics_optimizer":forward_dynamics_optimizer.state_dict(),"inverse_dynamics_optimizer":inverse_dynamics_optimizer.state_dict()}
            # if args.algo=="a2c_icm":
            #     print('yes algo icm random')
            #     embedding_network,forward_dynamics_model,forward_dynamics_optimizer=algo.pass_models_parameters()

            #     status = {"num_frames": num_frames, "update": update,
            #             "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict(),"embedding_network":embedding_network.state_dict(),"forward_dynamics_model":forward_dynamics_model.state_dict(),
            #             "forward_dynamics_optimizer":forward_dynamics_optimizer.state_dict()}
                
            

            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")

#uncomment if you would like to plot heatmaps
    #     if num_frames % 2048==0 and num_frames<=1000000: #198656
    #         heatmap_time_step[heatmap_plot]=logs["state_visitation_pos"]
    #         save_heatmap_matrix(logs["state_visitation_pos"],heatmap_plot)
    #         heatmap_plot+=1
    #         #plot_heatmap(logs["state_visitation_pos"],heatmap_plot)
    # plot_all_heatmaps(heatmap_time_step)

    #uncomment for final heatmap
            
        if args.save_heatmaps!= 'False':
            if num_frames % (2048*50)==0: #save every 50 updates
                save_state_dict=logs["state_visitation_pos"]
                save_ir_dict=logs["ir_dict"]
                print('saving heatmaps')
                log_dict_csv(save_state_dict,is_first_log,f'state_{model_name}')
                log_dict_csv(save_ir_dict,is_first_log,f'ir_{model_name}')
                is_first_log=False
            
            
  

#uncomment if you would like to save videos during training
    # print("Saving gif... ", end="")
    # write_gif(np.array(stacked_frames), "icm_training"+".gif", fps=1/0.1)
    # print("Done.")

if __name__ == "__main__":
    main()
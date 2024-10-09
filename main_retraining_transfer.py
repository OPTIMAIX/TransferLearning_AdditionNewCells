from __future__ import absolute_import, division, print_function

import tensorflow as tf
import csv
from pandas import DataFrame
import pandas as pd
import numpy as np
import os
import tempfile
import zipfile
import shutil
import statistics as st

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from environment_5G_multipleBS import environment_5G_multipleBS
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import policy_step



from BS import BS
from BS_controller import BS_controller
from common import *
from transfer_weights import transfer_weights


tf.compat.v1.enable_v2_behavior()

tf.version.VERSION

def main_retraining_transfer():

  full_path = os.path.realpath(__file__)
  path, filename = os.path.split(full_path)
  os.chdir(path)

  #HYPERPARAMETERS DQN
  initial_collect_steps = 100  # @param {type:"integer"} 
  collect_steps_per_iteration = 1  # @param {type:"integer"}
  replay_buffer_max_length = 10000000  # @param {type:"integer"}
  checkpoint_steps=5e4

  max_steps=5e6

  batch_size = 512  # @param {type:"integer"}
  learning_rate = 1e-4  # @param {type:"number"}
  log_interval = 10  # @param {type:"integer"}

  n_eval = 1  # @param {type:"integer"}
  eval_interval = 1e3  # @param {type:"integer"}

  fc_layer_params = (100,) #neural network params

  dir_act = os.getcwd()

  #5G scenario parameters

  #Common parameters for agents for N and N'=N+1 cells
  k_tenants=2
  Nt=65
  Seff=5
  PRB_B=360e3
  Ct=Nt*Seff*PRB_B #Ct per cell
  pos_actions=[-3,0,3]
  n_rep=5
  chunk_size=5e3
  version='v1'

  #Parameters for the OLD agent for N cells
  n_BS_old=1 #Number of BS

  #Parameters for NEW agent for N'=N+1 cells
  n_BS_new=2 #Number of BS
  Ct_allcells_new=Ct*n_BS_new #Total capacity in the scenario

  #Indicate name of previously trained policies as checkpointer
  output_name='results_BS'+str(n_BS_old)+'_to_BS'+str(n_BS_new)+'_low_numbers_transf_'+version
  file_name_checkpointer_load=[]

  file_name_checkpointer_load.append('./results_BS1_non_transf_low_num_v2/exported_cp_T_1_eval_12') #Tenant 1
  file_name_checkpointer_load.append('./results_BS1_non_transf_low_num_v2/exported_cp_T_2_eval_12') #Tenant 2

  temp_file_name=['Temp_file_BS'+str(n_BS_old)+'_to_BS9'+str(n_BS_new)+'_low_numbers_transf_T1_'+version,'Temp_file_BS'+str(n_BS_old)+'_to_BS9'+str(n_BS_new)+'_low_numbers_transf_T2_'+version]
  checkpoint_save_name=['checkpoint_BS'+str(n_BS_old)+'_to_BS9'+str(n_BS_new)+'_low_numbers_transf_T1_'+version,'checkpoint_BS'+str(n_BS_old)+'_to_BS9'+str(n_BS_new)+'_low_numbers_transf_T2_'+version]

  print('Loading Training dataset...')

  #Import traffic from csv for training
  data_train = pd.read_csv("training_file.csv",sep=';',header=None, chunksize=chunk_size) #Obtain data in chunks
  data_train_firstchunk=next(data_train)

  O_k_n_train, SAGBR_train, MCBR_train=obtain_data_from_dataset(data_train_firstchunk,n_rep,Ct_allcells_new, Ct)

  print('Loading Evaluation dataset...')

  #Import traffic from csv for evaluation 
  data_eval = pd.read_csv("evaluation_file.csv",sep=',',header=None) #Obtain the whole file (shorter)

  O_k_n_eval, SAGBR_eval, MCBR_eval=obtain_data_from_dataset(data_eval,n_rep,Ct_allcells_new, Ct)

  #Create folder to store output results
  os.mkdir(output_name)
  path='./'+output_name+'/'

  tempdir =[os.getenv(temp_file_name[0], tempfile.gettempdir()),os.getenv(temp_file_name[0], tempfile.gettempdir())] #REQ CHECKPOINT

  #Create csv file for convergence evaluation
  file_name_conv=path+output_name+'_convergence.csv'
  convg_data={}
  for k in range(k_tenants):
    convg_data['Avg Reward T'+str(k+1)]=[]
  convg_data['Avg total Reward']=[]
  df_exp=DataFrame(convg_data,columns=[*convg_data])
  df_exp.to_csv(file_name_conv, index=None, header=True, sep=',',mode='w',encoding='utf-8-sig')

  #Create csv file for loss export
  file_name_loss_results=path+output_name+'loss_results.csv'
  loss_headers={}
  loss_headers['Iteration']=[]
  for k in range(k_tenants):
    loss_headers['Loss T'+str(k+1)]=[]
  df_exp=DataFrame(loss_headers,columns=[*loss_headers])
  df_exp.to_csv(file_name_loss_results, index=None, header=True, sep=',',mode='w',encoding='utf-8-sig')

  #Create objects for the OLD agent to be able to read the policy saved as checkpointer

  print('Creating objects for training...')

  #BS controller for training
  bs_controller_old=BS_controller(k_tenants,n_BS_old,Nt,Seff,PRB_B,Ct,SAGBR_train,MCBR_train,O_k_n_train)

  #Environments for traininig
  env_tenants_py_old=[]
  env_tenants_tf_old=[]
  q_network_tenants_old=[]
  optimizer_tenants_old=[]
  global_step_tenants_old=[] #CHECKPOINTER
  agents_tenants_old=[]

  print('Generating environments...')
  for k in range(k_tenants):
    env_tenants_py_old.append(environment_5G_multipleBS(k,bs_controller_old,pos_actions))

  #Generate tf environment for training
  for k in range(k_tenants):
    env_tenants_tf_old.append(tf_py_environment.TFPyEnvironment(env_tenants_py_old[k]))
    env_tenants_tf_old[k].reset() #Initialise

  print('Creating agents, polices and reply buffers...')

  #Create the agent,random policy and replay buffer for each of the tenants
  
  for k in range(k_tenants):
    #Neural network
    q_network_tenants_old.append(q_network.QNetwork(
        env_tenants_tf_old[k].observation_spec(),
        env_tenants_tf_old[k].action_spec(),
        fc_layer_params=fc_layer_params))

    optimizer_tenants_old.append(tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate))

    global_step_tenants_old.append(tf.compat.v1.train.get_or_create_global_step())#REQ CHECKPOINT

    agents_tenants_old.append(dqn_agent.DqnAgent(
        env_tenants_tf_old[k].time_step_spec(),
        env_tenants_tf_old[k].action_spec(),
        q_network=q_network_tenants_old[k],
        optimizer=optimizer_tenants_old[k],
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=global_step_tenants_old[k]))
    agents_tenants_old[-1].initialize() #Initialise last agent

  #Restore checkpointer per tenant
    with zipfile.ZipFile(file_name_checkpointer_load[k]+'.zip', 'r') as zip_ref:
        zip_ref.extractall(file_name_checkpointer_load[k])

    checkpoint=tf.train.Checkpoint(agent=agents_tenants_old[k])
    latest_checkpoint = tf.train.latest_checkpoint(file_name_checkpointer_load[k])
    checkpoint_load_status = checkpoint.restore(latest_checkpoint)
    checkpoint_load_status.initialize_or_restore()


  #Create elements for the NEW agent for N'=N+1 cells

  #BS controller for training
  bs_controller=BS_controller(k_tenants,n_BS_new,Nt,Seff,PRB_B,Ct,SAGBR_train,MCBR_train,O_k_n_train)

  #Environments for traininig
  env_tenants_py=[]
  env_tenants_tf=[]
  q_network_tenants=[]
  optimizer_tenants=[]
  agents_tenants=[]
  random_policy_tenants=[]
  replay_buffer_tenants=[]
  train_checkpointer=[]
  checkpoint_dir=[]
  global_step_tenants=[] #CHECKPOINTER

  print('Generating environments...')
  for k in range(k_tenants):
    env_tenants_py.append(environment_5G_multipleBS(k,bs_controller,pos_actions))

  #Generate tf environment for training
  for k in range(k_tenants):
    env_tenants_tf.append(tf_py_environment.TFPyEnvironment(env_tenants_py[k]))
    env_tenants_tf[k].reset() #Initialise

  #BS controller for evaluation
  bs_controller_eval=BS_controller(k_tenants,n_BS_new,Nt,Seff,PRB_B,Ct,SAGBR_eval,MCBR_eval,O_k_n_eval)

  #Generate environments for evaluation
  env_tenants_py_eval=[]
  env_tenants_tf_eval=[]
  for k in range(k_tenants):
    env_tenants_py_eval.append(environment_5G_multipleBS(k,bs_controller_eval,pos_actions))

  #Generate tf environment for evaluation
  for k in range(k_tenants):
    env_tenants_tf_eval.append(tf_py_environment.TFPyEnvironment(env_tenants_py_eval[k]))
    env_tenants_tf_eval[k].reset() #Initialise

  #Create the agent,random policy and replay buffer for each of the tenants
  print('Creating agents, polices and reply buffers...')
  for k in range(k_tenants):

    #Neural network
    q_network_tenants.append(q_network.QNetwork(
        env_tenants_tf[k].observation_spec(),
        env_tenants_tf[k].action_spec(),
        fc_layer_params=fc_layer_params))

    optimizer_tenants.append(tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate))

    global_step_tenants.append(tf.compat.v1.train.get_or_create_global_step())#REQ CHECKPOINT

    agents_tenants.append(dqn_agent.DqnAgent(
        env_tenants_tf[k].time_step_spec(),
        env_tenants_tf[k].action_spec(),
        q_network=q_network_tenants[k],
        optimizer=optimizer_tenants[k],
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=global_step_tenants[k]))

    agents_tenants[-1].initialize() #Initialise last agent

    random_policy_tenants.append(random_tf_policy.RandomTFPolicy(env_tenants_tf[k].time_step_spec(),
                                                  env_tenants_tf[k].action_spec()))

    replay_buffer_tenants.append(tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agents_tenants[k].collect_data_spec,
        batch_size=env_tenants_tf[k].batch_size,
        max_length=replay_buffer_max_length))

    #CheKkpoint initialization per tenant
    checkpoint_dir.append(os.path.join(tempdir[k], checkpoint_save_name[k]))
    train_checkpointer.append(common.Checkpointer(
        ckpt_dir=checkpoint_dir[k],
        max_to_keep=1,
        agent=agents_tenants[k],
        policy=agents_tenants[k].policy,
        replay_buffer=replay_buffer_tenants[k],
        global_step=global_step_tenants[k]
    ))

  #Transfer weights from the old agent to the new agent for all tenants
  transfer_weights(agents_tenants,agents_tenants_old,k_tenants,n_BS_new,pos_actions)

  #Perform new training of the transferred policy
  print("Collecting initial data...")

  collect_data_MA(env_tenants_tf, random_policy_tenants, replay_buffer_tenants, bs_controller, steps=initial_collect_steps)

  #Reset the train step
  for k in range(k_tenants):
    agents_tenants[k].train_step_counter.assign(0)

  #Initial evaluation
  evaluation_num=0
  eval_polices=[]
  for k in range(k_tenants):
    eval_polices.append(agents_tenants[k].policy) #Generate greedy policy to evaluate

  file_name=path+output_name+'_eval_'+str(evaluation_num)
  evaluation(k_tenants,n_BS_new,env_tenants_tf_eval,eval_polices,file_name,n_eval) #evaluate
  evaluation_num+=1

  dataset=[]
  iterator=[]
  chunk_number=1

  for k in range(k_tenants):
    dataset.append(replay_buffer_tenants[k].as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3))

    iterator.append(iter(dataset[-1]))

  for k in range(k_tenants):
    train_checkpointer[k].save(global_step_tenants[k])
    checkpoint_zip_filename = create_zip_file(checkpoint_dir[k], os.path.join(path, 'exported_cp_T_'+str(k+1)+'_eval_'+str(0)))



  #Training
  train_loss=[[] for _ in range(k_tenants)]

  print("Regular operation and training...")
  for it in range(bs_controller.time_step,int(max_steps)):
    if it%log_interval==0:
      print('Iteration: ',it)

    #Keep reading chunks from file
    if bs_controller.time_step%(chunk_size*n_rep)==0: 
      data_train_chunk=next(data_train)
      O_k_n_train, SAGBR_train, MCBR_train=obtain_data_from_dataset(data_train_chunk,n_rep,Ct_allcells_new, Ct)
      bs_controller.import_data_chunk(O_k_n_train,SAGBR_train,MCBR_train)
      chunk_number+=1
      print('Chunk '+str(chunk_number)+' imported...')

    #Generate collecting polices for the iteration 
    collecting_polices=[]
    for k in range(k_tenants):
      collecting_polices.append(agents_tenants[k].collect_policy)

    # Collect a few steps using collect_policy and save to the replay buffer.
    for _ in range(collect_steps_per_iteration):
      collect_step_MA(env_tenants_tf, collecting_polices, replay_buffer_tenants, bs_controller)

    # Sample a batch of data from the buffer and update the agent's network.
    
    for k in range(k_tenants):
      experience, unused_info = next(iterator[k])
      train_loss[k] = agents_tenants[k].train(experience).loss
      step = agents_tenants[k].train_step_counter.numpy()

    if (it-1) %log_interval ==0:
      export_loss_results(file_name_loss_results,it-initial_collect_steps,train_loss,k_tenants)

    if it % eval_interval == 0: #Eval every eval_interval iterations
      eval_polices=[]
      for k in range(k_tenants):
        eval_polices.append(agents_tenants[k].policy) #Generate greedy policy to evaluate

      file_name=path+output_name+'_eval_'+str(evaluation_num)
      evaluation(k_tenants,n_BS_new,env_tenants_tf_eval,eval_polices,file_name,n_eval)
      evaluation_num+=1

    if it % checkpoint_steps == 0:
      for k in range(k_tenants):
        train_checkpointer[k].save(global_step_tenants[k])
        checkpoint_zip_filename = create_zip_file(checkpoint_dir[k], os.path.join(path, 'exported_cp_T_'+str(k+1)+'_eval_'+str(evaluation_num)))

      
      
  #Export final evaluation
  eval_polices=[]
  for k in range(k_tenants):
    eval_polices.append(agents_tenants[k].policy) #Generate greedy policy to evaluate

  file_name=path+output_name+'_eval_'+str(evaluation_num)
  evaluation(k_tenants,n_BS_new,env_tenants_tf_eval,eval_polices,file_name,n_eval)

  for k in range(k_tenants):
    train_checkpointer[k].save(global_step_tenants[k])
    checkpoint_zip_filename = create_zip_file(checkpoint_dir[k], os.path.join(path, 'exported_cp_T_'+str(k+1)+'_eval_final'))


if __name__ == '__main__':
  main_retraining_transfer()












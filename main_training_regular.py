from __future__ import absolute_import, division, print_function


import tensorflow as tf
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




from BS import BS
from BS_controller import BS_controller
from common import *

def main_training_regular():

  tf.compat.v1.enable_v2_behavior()

  tf.version.VERSION

  full_path = os.path.realpath(__file__)
  path, filename = os.path.split(full_path)
  os.chdir(path)

  #HYPERPARAMETERS DQN

  initial_collect_steps = 50  # @param {type:"integer"} 
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

  k_tenants=2
  n_BS=1
  Nt=65
  Seff=5
  PRB_B=360e3
  Ct=Nt*Seff*PRB_B #Ct a la cell
  Ct_allcells=Ct*n_BS
  SAGBR=[Ct_allcells*0.6, Ct_allcells*0.4]
  MABR=np.multiply([0.8,0.8],Ct)
  pos_actions=[-3,0,3]
  n_rep=5
  chunk_size=5e3
  version='v2'

  #Specify file names for input and output
  output_name='results_BS'+str(n_BS)+'_non_transf_low_num_'+version
  checkpoint_name=['checkpoint_BS'+str(n_BS)+'_T1_low_num_'+version,'checkpoint_BS'+str(n_BS)+'_T2_low_num_'+version]
  temp_file_name=['Temp_file_BS'+str(n_BS)+'_T1_low_num_'+version,'Temp_file_BS'+str(n_BS)+'_T2_low_num_'+version]
  file_training="training_file.csv"
  file_evaluation="evaluation_file.csv"


  print('Loading Training dataset...')

  #Import traffic from csv for training
  data_train = pd.read_csv(file_training,sep=';',header=None, chunksize=chunk_size) #Obtain data in chunks
  data_train_firstchunk=next(data_train)

  O_k_n_train, SAGBR_train, MCBR_train=obtain_data_from_dataset(data_train_firstchunk,n_rep,Ct_allcells, Ct)

  print('Loading Evaluation dataset...')

  #Import traffic from csv for evaluation final
  data_eval = pd.read_csv(file_evaluation,sep=',',header=None) #Obtain the whole file (shorter)

  O_k_n_eval, SAGBR_eval, MCBR_eval=obtain_data_from_dataset(data_eval,n_rep,Ct_allcells, Ct)

  #Create folder to store output results
  os.mkdir(output_name)
  path='./'+output_name+'/'

  tempdir =[os.getenv(temp_file_name[0], tempfile.gettempdir()),os.getenv(temp_file_name[0], tempfile.gettempdir())] #REQ CHECKPOINT


  #Create csv file for convergence evaluation
  file_name_conv=path+output_name+'_convergence.csv'
  convg_data={}
  for k in range(k_tenants):
    convg_data['Avg Reward T'+str(k+1)]=[]
  convg_data['Avg Total Reward']=[]
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

  #INITIALIZATION of environment and agents
  print('Creating objects for training...')

  #BS controller for training
  bs_controller=BS_controller(k_tenants,n_BS,Nt,Seff,PRB_B,Ct,SAGBR_train,MCBR_train,O_k_n_train)

  #Environments for traininig
  env_tenants_py=[]
  env_tenants_tf=[]
  q_network_tenants=[]
  optimizer_tenants=[]
  global_step_tenants=[] #CHECKPOINTER
  agents_tenants=[]
  random_policy_tenants=[]
  replay_buffer_tenants=[]
  train_checkpointer=[]
  checkpoint_dir=[]

  print('Generating environments...')
  for k in range(k_tenants):
    env_tenants_py.append(environment_5G_multipleBS(k,bs_controller,pos_actions))

  #Generate tf environment for training
  for k in range(k_tenants):
    env_tenants_tf.append(tf_py_environment.TFPyEnvironment(env_tenants_py[k]))
    env_tenants_tf[k].reset() #Initialise

  #BS controller for evaluation
  bs_controller_eval=BS_controller(k_tenants,n_BS,Nt,Seff,PRB_B,Ct,SAGBR_eval,MCBR_eval,O_k_n_eval)

  #Generate environments for evaluation
  env_tenants_py_eval=[]
  env_tenants_tf_eval=[]
  for k in range(k_tenants):
    env_tenants_py_eval.append(environment_5G_multipleBS(k,bs_controller_eval,pos_actions))

  #Generate tf environment for evaluation
  for k in range(k_tenants):
    env_tenants_tf_eval.append(tf_py_environment.TFPyEnvironment(env_tenants_py_eval[k]))
    env_tenants_tf_eval[k].reset() #Initialise

  print('Creating agents, polices and reply buffers...')
  #Create the agent,random policy and replay buffer for each of the tenants
  for k in range(k_tenants):

    #Neural network
    q_network_tenants.append(q_network.QNetwork(
        env_tenants_tf[k].observation_spec(),
        env_tenants_tf[k].action_spec(),
        fc_layer_params=fc_layer_params))

    optimizer_tenants.append(tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate))

    global_step_tenants.append(tf.compat.v1.train.get_or_create_global_step())#REQ CHECKPOINT

    #global_step = tf.compat.v1.train.get_or_create_global_step() 

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

    #Checkpoint initialization 
    checkpoint_dir.append(os.path.join(tempdir[k], checkpoint_name[k]))
    train_checkpointer.append(common.Checkpointer(
        ckpt_dir=checkpoint_dir[k],
        max_to_keep=1,
        agent=agents_tenants[k],
        policy=agents_tenants[k].policy,
        replay_buffer=replay_buffer_tenants[k],
        global_step=global_step_tenants[k]
    ))

  #Initialize agents
  print("Collecting initial data...")

  collect_data_MA(env_tenants_tf, random_policy_tenants, replay_buffer_tenants, bs_controller, steps=initial_collect_steps)

  # Reset the train step
  for k in range(k_tenants):
    agents_tenants[k].train_step_counter.assign(0)

  #Initial evaluation
  evaluation_num=0
  eval_polices=[]
  for k in range(k_tenants):
    eval_polices.append(agents_tenants[k].policy) #Generate greedy policy to evaluate

  file_name=path+output_name+'_eval_'+str(evaluation_num)
  evaluation(k_tenants,n_BS,env_tenants_tf_eval,eval_polices,file_name,n_eval) #evaluate
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

    # Keep reading chunks from file
    if bs_controller.time_step%(chunk_size*n_rep)==0: 
      data_train_chunk=next(data_train)
      O_k_n_train, SAGBR_train, MCBR_train=obtain_data_from_dataset(data_train_chunk,n_rep,Ct_allcells, Ct)
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
      export_loss_results(file_name_loss_results,it-initial_collect_steps*3,train_loss,k_tenants)

    if it % eval_interval == 0: #Evaluate every eval_interval iterations
      eval_polices=[]
      for k in range(k_tenants):
        eval_polices.append(agents_tenants[k].policy) #Generate greedy policy to evaluate

      
      file_name=path+output_name+'_eval_'+str(evaluation_num)
      evaluation(k_tenants,n_BS,env_tenants_tf_eval,eval_polices,file_name,n_eval)
      evaluation_num+=1

    if it % checkpoint_steps == 0:
      for k in range(k_tenants):
        train_checkpointer[k].save(global_step_tenants[k])
        checkpoint_zip_filename = create_zip_file(checkpoint_dir[k], os.path.join(path, 'exported_cp_T_'+str(k+1)+'_eval_'+str(int(it/checkpoint_steps))))

      
  #Export final evaluation anc checkpoint
  eval_polices=[]
  for k in range(k_tenants):
    eval_polices.append(agents_tenants[k].policy) #Generate greedy policy to evaluate
    
  file_name=path+output_name+'_eval_'+str(evaluation_num)
  evaluation(k_tenants,n_BS,env_tenants_tf_eval,eval_polices,file_name,n_eval)

  for k in range(k_tenants):
    train_checkpointer[k].save(global_step_tenants[k])
    checkpoint_zip_filename = create_zip_file(checkpoint_dir[k], os.path.join(path, 'exported_cp_T_'+str(k+1)+'_eval_final'))


if __name__ == '__main__':
  main_training_regular()

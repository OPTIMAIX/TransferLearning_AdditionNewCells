
def transfer_weights(agents_tenants,agents_tenants_old,k_tenants,n_BS_new,pos_actions):
  print('Transferring weights...')
  for k in range(k_tenants):
    #Input layer
    num_layers=int(len(agents_tenants[k]._q_network.weights)/2) #Dividit per dos perquè hi ha el bias (crec que no s'utilitza)

    #Copy of pG_k components input nodes input weigths n=0...n_BS_new-2 (Range nBS_new-1 perque arriba fins a un menys del límit)
    for n in range(0,n_BS_new-1):
      agents_tenants[k]._q_network.weights[0][n].assign(agents_tenants_old[k]._q_network.weights[0][n])
      agents_tenants[k]._target_q_network.weights[0][n].assign(agents_tenants_old[k]._target_q_network.weights[0][n])
    
    #Copy of capacity share components input nodes weights n=n_BS_new...2*n_BS_new-2
    for n in range(n_BS_new,2*n_BS_new-1):
      agents_tenants[k]._q_network.weights[0][n].assign(agents_tenants_old[k]._q_network.weights[0][n-1])
      agents_tenants[k]._target_q_network.weights[0][n].assign(agents_tenants_old[k]._target_q_network.weights[0][n-1])

    #Copy of av_share_k components input nodes weights n=2*n_BS_new...3*n_BS_new-2
    for n in range(2*n_BS_new,3*n_BS_new-1):
      agents_tenants[k]._q_network.weights[0][n].assign(agents_tenants_old[k]._q_network.weights[0][n-2])
      agents_tenants[k]._target_q_network.weights[0][n].assign(agents_tenants_old[k]._target_q_network.weights[0][n-2])

    #Copy of av_pG_k components input nodes weights n=3*n_BS_new...4*n_BS_new-2
    for n in range(3*n_BS_new,4*n_BS_new-1):
      agents_tenants[k]._q_network.weights[0][n].assign(agents_tenants_old[k]._q_network.weights[0][n-3])
      agents_tenants[k]._target_q_network.weights[0][n].assign(agents_tenants_old[k]._target_q_network.weights[0][n-3])

    #Copy of MCBR components input nodes weights n=4*n_BS_new...5*n_BS_new-2
    for n in range(4*n_BS_new,5*n_BS_new-1):
      agents_tenants[k]._q_network.weights[0][n].assign(agents_tenants_old[k]._q_network.weights[0][n-4])
      agents_tenants[k]._target_q_network.weights[0][n].assign(agents_tenants_old[k]._target_q_network.weights[0][n-4])

    #Copy of SAGBRk and SAGBR_others components input nodes weights n=6*n_BS_new and n=6*n_BS_new+1
    agents_tenants[k]._q_network.weights[0][5*n_BS_new].assign(agents_tenants_old[k]._q_network.weights[0][5*(n_BS_new-1)])
    agents_tenants[k]._q_network.weights[0][5*n_BS_new+1].assign(agents_tenants_old[k]._q_network.weights[0][5*(n_BS_new-1)+1])
    agents_tenants[k]._target_q_network.weights[0][5*n_BS_new].assign(agents_tenants_old[k]._target_q_network.weights[0][5*(n_BS_new-1)])
    agents_tenants[k]._target_q_network.weights[0][5*n_BS_new+1].assign(agents_tenants_old[k]._target_q_network.weights[0][5*(n_BS_new-1)+1])   

    #Hidden layers
    num_hidden_weights=num_layers-2

    for h in range(1,num_hidden_weights):
      num_source_nodes=agents_tenants[k]._q_network.weights[2*h].shape[0]

      for s in range(0,num_source_nodes):
          agents_tenants[k]._q_network.weights[2*h][s].assign(agents_tenants_old[k]._q_network.weights[2*h][s])
          agents_tenants[k]._target_q_network.weights[2*h][s].assign(agents_tenants_old[k]._target_q_network.weights[2*h][s])

    #Output layer
    for n in range(agents_tenants[k]._q_network.weights[2*num_layers-2].shape[0]): #For each of the source nodes to each of the output nodes
      w=[]
      for a_old in range(agents_tenants_old[k]._q_network.weights[2*num_layers-2].shape[1]):
        for _ in range(len(pos_actions)):
          w.append(agents_tenants_old[k]._q_network.weights[2*num_layers-2][n][a_old].numpy())
      agents_tenants[k]._q_network.weights[2*num_layers-2][n].assign(w) #This assignation needs to be performed in this way since single elements cannot be assigned with a new value. 
      agents_tenants[k]._target_q_network.weights[2*num_layers-2][n].assign(w)
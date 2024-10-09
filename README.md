## Transfer Reinforcement Learning (TRL) approach for DQN-MARL Capacity sharing solution

This Github repository includes the code of a **Transfer Reinforcement Learning approach designed for the [DQN-MARL capacity sharing solution](https://ieeexplore.ieee.org/abstract/document/9497684)**.  
 
The DQN-MARL capacity sharing solution allows dynamically distributing the capacity in multiple RAN nodes among multiple tenants, each of them provided with a RAN slice. The capacity sharing is performed so that the traffic demands and Service Level Agreement (SLA) of the different tenant are satisfied and the resources in the different RAN nodes are efficiently used. A common operation for operators is to make changes of the network topology by deploying new cells, increasing the capacity. This operation involves the re-training of the DQN-MARL capacity sharing solution, since the traffic in the area will change and dimensions of the deep neural networks in the solution depends on the number of cells. This re-training can take a lot of time as it involves acquiring a large number of action/reward experiences with the new environment. The TRL approach included in this Github allows accelerating this re-training by leveraging the [inter-task mapping](https://dl.acm.org/doi/10.1145/1329125.1329170) technique, which allows transferring the weights of a previously trained policy for *N* cells to train a policy for *N'*=*N*+1 cells. 

## Contents
This folder contains the following python scripts: 

- [**main_training_regular.py**](./main_training_regular.py): Main function for performing the training of the DQN MARL capacity sharing from scratch. 
- [**main_retraining_transfer.py**](./main_training_regular.py): Main function for performing the re-training of the DQN MARL capacity sharing by employing the TRL approach. 
- [**transfer_weights.py**](./transfer_weights.py): Function that allows transfering the weights of a policy trained for *N* cells to a policy that will be trained for *N'*=*N*+1. 
- [**environment_5G_multipleBS.py**](./environment_5G_multipleBS.py): Python environment based on py_environment.PyEnvironment. 
- [**BS_controller.py**](./BS_controller.py): BS_controller class included in the DQN-MARL capacity sharing solution.
- [**BS.py**](./BS.py): BS class included in the DQN-MARL capacity sharing solution.
- [**common.py**](./common.py): Python script with different functions for supporting the training and evaluation operations.

### Notes: 
- The developed code relies on the TF-Agents library.
- To support transfer learning, policies need to be saved according to the checkpointer feature of the TF-Agents library (PolicySaver not supported). 

## Contributors
The solution has been designed and developed by [Mobile Communications Research Group (GRCM)](https://grcm.tsc.upc.edu/en) of the Department of Signal Theory and Communications (TSC) of Universitat Politècnica de Catalunya (UPC). Within the group, we would like to recognize the following individuals for their code contributions, discussions, and other work to make the DQN-MARL capacity sharing library:
- Irene Vilà.
- Jordi Pérez-Romero.
- Oriol Sallent.


# GruRL-HVAC_Ctrl
This repo stores the code used for [Link to BuildSys Paper], along with a [demonstration](agent/Demo.ipynb) in an EnergyPlus model. 

### Install Related Packages** 
The following two packages were used. Install following their documentation.    
- [Gym-Eplus](https://github.com/zhangzhizza/Gym-Eplus). 
..* This package is an OpenGym AI wrapper for EnergyPlus. 
..* The demonstration in this repo uses EnergyPlus version 8.6, but the Gym-plus package is applicable to any EnergyPlus version 8.x.  
- [mpc.torch](https://github.com/locuslab/mpc.pytorch)
..* This package is a fast and differentiable model predictive control solver for PyTorch.

To be confirmed: What else need to be installed other than PyTorch?

### Set up Simulation Environments**
- Read the documentation of [Gym-Eplus](https://github.com/zhangzhizza/Gym-Eplus) on setting up simulation environments. 
- Place the model and weather files in the *eplus_env* folder under the corresponding location in the Gym-Eplus folder. 
- Register the environments following this table. A *__init__.py* for registeration is included. But, check that it matches your own file placement.  
| **Environment Name** |**Model File (\*.idf)**|**Configuration File (\*.cfg)**|**Weather File (\*.epw)**| **Description**|
|:----------------|:---------------|:--------|:-----------|:--------------------------------|
|**5Zone-sim_TMY2-v0**|5Zone_Default.idf|variables_Default.cfg|pittsburgh_TMY2.epw|Baseline EnergyPlus control; Used in offline pretraining for expert demonstration|
|**5Zone-control_TMY3-v0**|5Zone_Control.idf|variables_Control.cfg|pittsburgh_TMY3.epw|Gru-RL control; Used in online learning|
| **5Zone-sim_TMY3-v0**   | 5Zone_Default.idf|variables_Default.cfg|pittsburgh_TMY3.epw|Baseline EnergyPlus control; Used for performance benchmark|

### Overview
An example is provided [here](agent/Demo.ipynb), which provides more details. 
 
For **Offline Pretraining**
```
$ python Imit_EP.py
```

For **Online Learning**
```
$ python PPO_MPC_EP.py
```






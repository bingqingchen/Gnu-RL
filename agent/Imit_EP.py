import os
import sys

# Assign env_path to be the file path where Gym-Eplus is located.
env_path = os.path.abspath(os.path.join(__file__, '..', '..', '..'))
sys.path.insert(0, env_path)
mpc_path = os.path.abspath(os.path.join(__file__,'..', '..'))
sys.path.insert(0, mpc_path)

import argparse
from numpy import genfromtxt
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

import gym
import eplus_env

from mpc import mpc
from mpc.mpc import QuadCost, LinDx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import make_dict, R_func

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE

parser = argparse.ArgumentParser(description='GruRL-Imitation Learning')
parser.add_argument('--gamma', type=float, default=0.98, metavar='G',
                    help='discount factor (default: 0.98)')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument('--T', type=int, default=12, metavar='N',
                    help='Planning Horizon (default: 12)')
parser.add_argument('--step', type=int, default=300*3, metavar='N',
                    help='Time Step in Simulation, Unit in Seconds (default: 900)') # 15 Minutes Now!
parser.add_argument('--eta', type=int, default=4,
                    help='Hyper Parameter for Balancing Comfort and Energy')
parser.add_argument('--save_name', type=str, default='rl',
                    help='save name')
args = parser.parse_args()
torch.manual_seed(args.seed)

# Modify here: Outputs from EnergyPlus; Match the variables.cfg file.
obs_name = ["Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction", "Diff. Solar Rad.", "Direct Solar Rad.", "Htg SP", "Clg SP", "Indoor Temp.", "Indoor Temp. Setpoint", "PPD", "Occupancy Flag", "Coil Power", "HVAC Power", "Sys In Temp.", "Sys In Mdot", "OA Temp.", "OA Mdot", "MA Temp.", "MA Mdot", "Sys Out Temp.", "Sys Out Mdot"]

# Modify here: Change based on the specific control problem
state_name = ["Indoor Temp."]
dist_name = ["Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction", "Diff. Solar Rad.", "Direct Solar Rad.", "Occupancy Flag"]
ctrl_name = ["SA Temp Setpoimt"]
target_name = ["Indoor Temp. Setpoint"]

n_state = len(state_name)
n_ctrl = len(ctrl_name)
n_dist = len(dist_name)

eta = [0.1, args.eta] # eta: Weight for comfort during unoccupied and occupied mode
step = args.step # step: Timestep; Unit in seconds
T = args.T # T: Number of timesteps in the planning horizon
tol_eps = 90 # tol_eps: Total number of episodes; Each episode is a natural day

# Read Information on Weather, Occupancy, and Target Setpoint
obs = pd.read_pickle("results/Dist-TMY2.pkl")
target = obs[target_name]
disturbance = obs[dist_name]
# Min-Max Normalization
disturbance = (disturbance - disturbance.min())/(disturbance.max() -disturbance.min())

# Note: Only the expert has access to the environment
class Expert():
    def __init__(self, n_state, n_ctrl, n_dist, env_name):
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.n_dist = n_dist
        
        self.env = gym.make(env_name)
        
        self.start_time = None
        self.cur_time = None
    
    def reset(self):
        timeStep, obs, isTerminal = self.env.reset()
        self.start_time = pd.datetime(year = self.env.start_year, month = self.env.start_mon, day = self.env.start_day)
        self.cur_time = self.start_time
        obs_dict = make_dict(obs_name, obs)
        x_init = torch.tensor([obs_dict[state] for state in state_name]).unsqueeze(0).double() # 1 x n_state
        return x_init, self.cur_time
    
    def forward(self, x_init):
        # Using EnergyPlus default control strategy;
        action = ()
        timeStep, obs, isTerminal = self.env.step(action)
        obs_dict = make_dict(obs_name, obs)
        next_state = torch.tensor([obs_dict[state] for state in state_name]).unsqueeze(0).double()
        self.cur_time = self.start_time + pd.Timedelta(seconds = timeStep)
        # The action is the difference between Supply Air Temp. and Mixed Air Temp., i.e. the amount of heating from the heating coil.
        action = obs_dict["Sys Out Temp."]-obs_dict["MA Temp."]
        reward = R_func(obs_dict, action, eta)
        return next_state, torch.tensor([action]).double(), reward, obs_dict

class Learner():
    def __init__(self, n_state, n_ctrl, n_dist, disturbance, target, u_upper, u_lower):
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.n_dist = n_dist
        self.disturbance = disturbance
        self.target = target
        
        # My Initial Guess
        self.F_hat = torch.ones((self.n_state, self.n_state+self.n_ctrl))
        self.F_hat[0, 0] = 0.9
        self.F_hat[0, 1] = 0.3
        self.F_hat = self.F_hat.double().requires_grad_()
        
        self.Bd_hat = np.random.rand(self.n_state, self.n_dist)
        self.Bd_hat = torch.tensor(self.Bd_hat).requires_grad_()
        
        self.optimizer = optim.Adam([self.F_hat, self.Bd_hat], lr=1e-3)
    
        self.u_lower = u_lower * torch.ones(T, 1, n_ctrl).double()
        self.u_upper = u_upper * torch.ones(T, 1, n_ctrl).double()
    
    def Cost_function(self, cur_time):
        diag = torch.zeros(T, self.n_state + self.n_ctrl)
        occupied = self.disturbance["Occupancy Flag"][cur_time:cur_time + pd.Timedelta(seconds = (T-1) * step)]
        eta_w_flag = torch.tensor([eta[int(flag)] for flag in occupied]).unsqueeze(1).double() # Tx1
        diag[:, :n_state] = eta_w_flag
        diag[:, n_state:] = 0.001
        
        C = []
        for i in range(T):
            C.append(torch.diag(diag[i]))
        C = torch.stack(C).unsqueeze(1) # T x 1 x (m+n) x (m+n)
        
        x_target = self.target[cur_time : cur_time + pd.Timedelta(seconds = (T-1) * step)] # in pd.Series
        x_target = torch.tensor(np.array(x_target))
        
        c = torch.zeros(T, self.n_state+self.n_ctrl) # T x (m+n)
        c[:, :n_state] = -eta_w_flag*x_target
        c[:, n_state:] = 1 # L1-norm now! Check
        
        c = c.unsqueeze(1) # T x 1 x (m+n)
        return C, c
    
    def forward(self, x_init, C, c, d):
        ft = torch.mm(self.Bd_hat, d).transpose(0, 1) # T-1 x n_state
        ft = ft.unsqueeze(1) # T-1 x 1 x n_state
    
        x_pred, u_pred, _ = mpc.MPC(n_state=n_state,
                                    n_ctrl=n_ctrl,
                                    T=T,
                                    u_lower = self.u_lower,
                                    u_upper = self.u_upper,
                                    lqr_iter=20,
                                    verbose=0,
                                    exit_unconverged=False,
                                    )(x_init.double(), QuadCost(C.double(), c.double()),
                                      LinDx(self.F_hat.repeat(T-1, 1, 1, 1), None))
        
        return x_pred[1, 0, :], u_pred[0, 0, :]
    
    def predict(self, x_init, action, cur_time):
        dt = np.array(self.disturbance.loc[cur_time]) # n_dist
        dt = torch.tensor(dt).unsqueeze(1) # n_dist x 1
        ft = torch.mm(self.Bd_hat, dt) # n_state x 1
        tau = torch.stack([x_init, action]) # (n_state + n_ctrl) x 1
        next_state  = torch.mm(self.F_hat, tau) + ft # n_state x 1
        return next_state
                                                                                            
    def update_parameters(self, x_true, u_true, x_pred, u_pred):
        # Every thing in T x Dim.
        state_loss = torch.mean((x_true.double() - x_pred)**2)
        action_loss = torch.mean((u_true.double() - u_pred)**2)
        
        # Note: Put more weight on predicting state than predicting action.
        traj_loss = 10*state_loss + action_loss
        print("From state {}, From action {}".format(state_loss, action_loss))
        self.optimizer.zero_grad()
        traj_loss.backward()
        self.optimizer.step()
        print(self.F_hat)
        print(self.Bd_hat)



def main():
    dir = 'results'
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    perf = []
    n_step = 96 # n_step: Number of Steps per Day
    
    timeStamp = []
    record = []
    record_name =["Learner nState", "Expert nState", "Learner action", "Expert action"]
    
    expert = Expert(n_state, n_ctrl, n_dist, '5Zone-sim_TMY2-v0')
    x_init, cur_time = expert.reset()
    
    u_upper = 5
    u_lower = 0
    
    learner = Learner(n_state, n_ctrl, n_dist, disturbance, target, u_upper, u_lower)
                                                                                            
    for i_episode in range(tol_eps):#  # Per Day
        x_true = []
        u_true = []
        x_pred = []
        u_pred = []
        
        # With re-planning
        for j in range(n_step):
            dt = np.array(disturbance[cur_time : cur_time + pd.Timedelta(seconds = (T-2) * step)]) # T-1 x n_dist
            dt = torch.tensor(dt).transpose(0, 1) # n_dist x T-1
            C, c = learner.Cost_function(cur_time)
            
            exp_state, exp_action, exp_reward, new_obs_dict = expert.forward(x_init) # T
            _, learner_action = learner.forward(x_init, C, c, dt)
            # Note: Learner predicts the next state based on expert's action.
            learner_state = learner.predict(x_init.squeeze(0), exp_action, cur_time)

            x_true.append(exp_state.squeeze(0))
            x_pred.append(learner_state.squeeze(0))
            u_true.append(exp_action)
            u_pred.append(learner_action)
            
            # Sanity Check:
            print("{} Predicted State: {}, Real State: {}, Predicted Action:{}, Real Action:{}".format(cur_time, learner_state.item(),  exp_state.item(), learner_action.item(), exp_action.item()))
            # Save for Record:
            timeStamp.append(cur_time)
            record.append([learner_state.item(), exp_state.item(), learner_action.item(), exp_action.item()])

            cur_time = expert.cur_time
            x_init = exp_state
            obs_dict = new_obs_dict
        
        x_true = torch.stack(x_true)
        u_true = torch.stack(u_true)
        x_pred = torch.stack(x_pred)
        u_pred = torch.stack(u_pred)

        learner.update_parameters(x_true, u_true, x_pred, u_pred)

        record_df = pd.DataFrame(np.array(record), index = np.array(timeStamp), columns = record_name)
        record_df.to_pickle("results/Imit_"+args.save_name+".pkl")

if __name__ == '__main__':
    main()

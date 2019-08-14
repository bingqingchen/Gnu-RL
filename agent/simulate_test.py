import gym
import eplus_env

import pandas as pd
import pickle
import numpy as np

from utils import make_dict

# Create Environment
# Environment "5Zone-sim-v0" is used for demonstration. Follow the documentation of 'Gym-Eplus' to set up additional EnergyPlus simulation environment.
env = gym.make('5Zone-control_TMY2-v0');

# Modify here: Change the observations and actions based on the specific control problem
obs_name = ["Outdoor Temp.", "Outdoor RH", "Wind Speed", "Wind Direction", "Diff. Solar Rad.", "Direct Solar Rad.", "Htg SP", "Clg SP", "Indoor Temp.", "Indoor Temp. Setpoint", "PPD", "Occupancy Flag", "Coil Power", "HVAC Power", "Sys In Temp.", "Sys In Mdot", "OA Temp.", "OA Mdot", "MA Temp.", "MA Mdot", "Sys Out Temp.", "Sys Out Mdot"]
# ctrl_name = []

# Reset the env (creat the EnergyPlus subprocess)
timeStep, obs, isTerminal = env.reset();
obs_dict = make_dict(obs_name, obs)
start_time = pd.datetime(year = env.start_year, month = env.start_mon, day = env.start_day)
print(start_time)

timeStamp = [start_time]
observations = [obs]
actions = []

for i in range(91*96):
    # Using EnergyPlus default control strategy;
    action = (20, )

    timeStep, obs, isTerminal = env.step(action)
    obs_dict = make_dict(obs_name, obs)
    cur_time = start_time + pd.Timedelta(seconds = timeStep)
    
    print("{}:  Sys In: {:.2f}({:.2f})-> OA: {:.2f}({:.2f})-> MA: {:.2f}({:.2f})-> Sys Out: {:.2f}({:.2f})-> Zone Temp: {:.2f}".format(cur_time,  obs_dict["Sys In Temp."], obs_dict["Sys In Mdot"],obs_dict["OA Temp."], obs_dict["OA Mdot"],
                                                    obs_dict["MA Temp."], obs_dict["MA Mdot"], obs_dict["Sys Out Temp."], obs_dict["Sys Out Mdot"],
                                                    obs_dict["Indoor Temp."]))

    timeStamp.append(cur_time)
    observations.append(obs)
    #actions.append(action)

# Save Observations
#obs_df = pd.DataFrame(np.array(observations), index = np.array(timeStamp), columns = obs_name)
#obs_df.to_pickle("results/Sim-Msultizone_Average.pkl")
#action_df = pd.DataFrame(np.array(actions), index = np.array(timeStamp[:-1]), columns = ctrl_name)
#action_df.to_pickle("results/action_TMY3_baseline.pkl")
#print("Saved!")

env.end_env() # Safe termination of the environment after use.

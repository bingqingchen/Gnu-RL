# Helper Functions
def make_dict(obs_name, obs):
    zipbObj = zip(obs_name, obs)
    return dict(zipbObj)

def R_func(obs_dict, action, eta):
    reward = - 0.5 * eta[int(obs_dict["Occupancy Flag"])] * (obs_dict["Indoor Temp."] - obs_dict["Indoor Temp. Setpoint"])**2 - action
    return reward#.item()
